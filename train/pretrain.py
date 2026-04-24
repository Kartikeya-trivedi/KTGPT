"""
Pretraining Loop
================

Two-phase pretraining for KT_GPT:
  Phase 1 (1B tokens):  lr=3e-4, FineWeb-Edu + DCLM  → stability + fluency
  Phase 2 (20B tokens): lr=1e-4, Stack v2 + FineWeb + DCLM + Math → intelligence

Features:
  - AdamW with cosine LR schedule + linear warmup
  - Gradient accumulation for large effective batch sizes on A10G
  - bf16 mixed precision
  - Checkpoint save/resume (model + optimizer + scheduler + step + router biases)
  - W&B logging: loss, grad_norm, lr, tokens/sec, router entropy, expert load CV
  - Automatic resume from latest checkpoint on restart
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import KTGPTConfig
from model.model import KTGPT


# ═══════════════════════════════════════════════════════════════════════
#  Training Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """All training hyperparameters.

    Use TrainConfig.phase1() or .phase2() class methods for preset configs.
    Batch size in tokens = micro_batch_size * seq_len * grad_accum_steps.
    """

    # ── Phase identity ───────────────────────────────────────────────
    phase: int = 1
    total_tokens: int = 1_000_000_000       # 1B for Phase 1

    # ── Optimization ─────────────────────────────────────────────────
    lr: float = 3e-4
    min_lr: float = 1e-5                    # cosine schedule floor
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 64              # effective batch = micro * accum

    # ── Batch / sequence ─────────────────────────────────────────────
    micro_batch_size: int = 2               # per-GPU micro batch
    seq_len: int = 4096

    # ── Logging & checkpointing ──────────────────────────────────────
    log_every: int = 10                     # log to W&B every N steps
    checkpoint_every: int = 500             # save checkpoint every N steps
    checkpoint_dir: str = "/checkpoints"    # Modal volume mount point
    wandb_project: str = "kt-gpt"
    wandb_run_name: Optional[str] = None

    # ── Data ─────────────────────────────────────────────────────────
    num_workers: int = 2
    seed: int = 42

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed per optimizer step (across all grad accum steps)."""
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps

    @property
    def total_steps(self) -> int:
        """Total optimizer steps for this phase."""
        return self.total_tokens // self.tokens_per_step

    @classmethod
    def phase1(cls) -> TrainConfig:
        """Phase 1: 1B tokens, high LR, FineWeb-Edu + DCLM."""
        return cls(
            phase=1,
            total_tokens=1_000_000_000,
            lr=3e-4,
            warmup_steps=2000,
            wandb_run_name="phase1-1B",
        )

    @classmethod
    def phase2(cls) -> TrainConfig:
        """Phase 2: 20B tokens, lower LR, full data mix."""
        return cls(
            phase=2,
            total_tokens=20_000_000_000,
            lr=1e-4,
            warmup_steps=2000,
            wandb_run_name="phase2-20B",
        )


# ═══════════════════════════════════════════════════════════════════════
#  Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════

def cosine_lr_schedule(
    step: int,
    total_steps: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Cosine annealing with linear warmup.

    Returns the learning rate multiplier for the given step.
    """
    if step < warmup_steps:
        # Linear warmup
        return lr * (step + 1) / warmup_steps

    if step >= total_steps:
        return min_lr

    # Cosine decay from lr to min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════
#  Router Metrics
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_router_entropy(model: KTGPT) -> list[float]:
    """Compute routing entropy per layer from the latest router states.

    Entropy H = -sum(p * log(p)) where p is the normalized expert load
    distribution. High entropy = balanced routing, low = collapse risk.
    """
    entropies: list[float] = []
    for layer in model.layers:
        bias = layer.ffn.router.expert_bias
        # Use softmax of biases as a proxy for routing distribution
        probs = F.softmax(bias.float(), dim=0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)
    return entropies


@torch.no_grad()
def compute_expert_load_cv(expert_counts: torch.Tensor) -> float:
    """Coefficient of variation of expert token counts.

    CV = std / mean.  CV ≈ 0 means perfectly balanced load.
    CV > 1 means severe imbalance / potential collapse.
    """
    if expert_counts.sum() == 0:
        return 0.0
    mean = expert_counts.float().mean()
    std = expert_counts.float().std()
    return (std / (mean + 1e-10)).item()


# ═══════════════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════════════

class Trainer:
    """Complete training loop for KT_GPT.

    Handles:
      - Mixed precision (bf16) forward/backward
      - Gradient accumulation across micro-batches
      - Cosine LR schedule with warmup
      - Periodic checkpointing (model + optimizer + scheduler + step)
      - W&B metric logging (loss, lr, grad_norm, tokens/sec, router metrics)
      - Automatic resume from latest checkpoint
    """

    def __init__(
        self,
        model: KTGPT,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.global_step = 0
        self.tokens_seen = 0

        # Optimizer — exclude router biases (updated separately)
        param_groups = self._build_param_groups()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            fused=True if device.type == "cuda" else False,
        )

        # GradScaler for mixed precision (bf16 doesn't strictly need it,
        # but it's a safety net for edge cases)
        self.scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

        # W&B
        self.wandb_run = None

    def _build_param_groups(self) -> list[dict]:
        """Separate weight-decay and no-weight-decay parameter groups.

        Convention: don't apply weight decay to norms, biases, or embeddings.
        Router expert_bias is excluded entirely (requires_grad=False).
        """
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _update_lr(self) -> float:
        """Set the learning rate for the current step and return it."""
        lr = cosine_lr_schedule(
            step=self.global_step,
            total_steps=self.config.total_steps,
            lr=self.config.lr,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ── Checkpointing ────────────────────────────────────────────────

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Save a full checkpoint to the checkpoint directory.

        Saves: model state, optimizer state, step count, tokens seen,
        and all router biases (float32).
        """
        tag = tag or f"step_{self.global_step}"
        ckpt_dir = Path(self.config.checkpoint_dir) / f"phase{self.config.phase}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{tag}.pt"

        # Collect router biases separately (they're float32)
        router_biases = {
            f"layer_{i}": layer.ffn.router.expert_bias.data.clone()
            for i, layer in enumerate(self.model.layers)
        }

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "router_biases": router_biases,
            "config": {
                "phase": self.config.phase,
                "total_tokens": self.config.total_tokens,
                "lr": self.config.lr,
            },
        }
        torch.save(checkpoint, path)

        # Also save a "latest" symlink/copy for easy resume
        latest_path = ckpt_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        print(f"[CKPT] Saved checkpoint: {path} (step={self.global_step})")
        return str(path)

    def load_checkpoint(self, path: Optional[str] = None) -> bool:
        """Load a checkpoint. If path is None, try to load 'latest.pt'.

        Returns True if a checkpoint was loaded, False otherwise.
        """
        if path is None:
            ckpt_dir = Path(self.config.checkpoint_dir) / f"phase{self.config.phase}"
            latest = ckpt_dir / "latest.pt"
            if not latest.exists():
                print("[CKPT] No checkpoint found, starting from scratch")
                return False
            path = str(latest)

        print(f"[CKPT] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]

        # Restore router biases
        for i, layer in enumerate(self.model.layers):
            key = f"layer_{i}"
            if key in checkpoint.get("router_biases", {}):
                layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])

        print(f"[CKPT] Resumed from step={self.global_step}, tokens={self.tokens_seen}")
        return True

    def load_phase1_checkpoint(self) -> bool:
        """Load the Phase 1 final checkpoint for Phase 2 continuation.

        Loads model weights only (resets optimizer and scheduler for Phase 2).
        """
        ckpt_dir = Path(self.config.checkpoint_dir) / "phase1"
        latest = ckpt_dir / "latest.pt"
        if not latest.exists():
            print("[CKPT] No Phase 1 checkpoint found!")
            return False

        print(f"[CKPT] Loading Phase 1 model weights for Phase 2...")
        checkpoint = torch.load(str(latest), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])

        # Restore router biases from Phase 1
        for i, layer in enumerate(self.model.layers):
            key = f"layer_{i}"
            if key in checkpoint.get("router_biases", {}):
                layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])

        print(f"[CKPT] Loaded Phase 1 model (trained on {checkpoint['tokens_seen']} tokens)")
        return True

    # ── Training ─────────────────────────────────────────────────────

    def train(self, dataloader) -> None:
        """Main training loop.

        Runs until total_tokens is consumed or dataloader is exhausted.
        Handles gradient accumulation, LR scheduling, logging, and
        periodic checkpointing.
        """
        import wandb

        # Initialize W&B
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "phase": self.config.phase,
                    "total_tokens": self.config.total_tokens,
                    "lr": self.config.lr,
                    "micro_batch": self.config.micro_batch_size,
                    "seq_len": self.config.seq_len,
                    "grad_accum": self.config.grad_accum_steps,
                    "tokens_per_step": self.config.tokens_per_step,
                    "total_steps": self.config.total_steps,
                },
            )

        self.model.train()
        accum_loss = 0.0
        accum_count = 0
        step_start = time.time()

        print(f"\n{'='*60}")
        print(f"  Phase {self.config.phase} Training")
        print(f"  Total tokens: {self.config.total_tokens / 1e9:.1f}B")
        print(f"  Total steps:  {self.config.total_steps}")
        print(f"  Tokens/step:  {self.config.tokens_per_step:,}")
        print(f"  Resuming from step {self.global_step}")
        print(f"{'='*60}\n")

        micro_step = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)      # (B, S)
            labels = batch["labels"].to(self.device)             # (B, S)

            # ── Forward + loss ───────────────────────────────────
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, _ = self.model(input_ids)
                # Shift: predict next token (logits[:, :-1] vs labels[:, 1:])
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                )
                # Scale loss by grad accumulation
                scaled_loss = loss / self.config.grad_accum_steps

            # ── Backward ─────────────────────────────────────────
            self.scaler.scale(scaled_loss).backward()

            accum_loss += loss.detach().item()
            accum_count += 1
            micro_step += 1
            self.tokens_seen += input_ids.numel()

            # ── Optimizer step (every grad_accum_steps) ──────────
            if micro_step % self.config.grad_accum_steps == 0:
                # Unscale for grad clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                ).item()

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Update learning rate
                current_lr = self._update_lr()

                # Average loss over accumulation steps
                avg_loss = accum_loss / max(accum_count, 1)

                self.global_step += 1

                # ── Logging ──────────────────────────────────────
                if self.global_step % self.config.log_every == 0:
                    elapsed = time.time() - step_start
                    tokens_per_sec = (
                        self.config.tokens_per_step * self.config.log_every / elapsed
                    )

                    # Router metrics
                    entropies = compute_router_entropy(self.model)
                    avg_entropy = sum(entropies) / len(entropies)

                    log_dict = {
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm,
                        "train/lr": current_lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_seen": self.tokens_seen,
                        "train/step": self.global_step,
                        "router/avg_entropy": avg_entropy,
                    }

                    # Per-layer entropy (every 10th layer to avoid clutter)
                    for i in range(0, len(entropies), max(1, len(entropies) // 6)):
                        log_dict[f"router/entropy_L{i}"] = entropies[i]

                    wandb.log(log_dict, step=self.global_step)

                    print(
                        f"[step {self.global_step:>6d}/{self.config.total_steps}] "
                        f"loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                        f"gnorm={grad_norm:.2f}  tok/s={tokens_per_sec:.0f}  "
                        f"entropy={avg_entropy:.2f}"
                    )
                    step_start = time.time()

                # ── Checkpoint ───────────────────────────────────
                if self.global_step % self.config.checkpoint_every == 0:
                    self.save_checkpoint()

                # Reset accumulators
                accum_loss = 0.0
                accum_count = 0

                # ── Termination ──────────────────────────────────
                if self.global_step >= self.config.total_steps:
                    print(f"\nReached {self.config.total_steps} steps. Training complete.")
                    break

        # Final checkpoint
        self.save_checkpoint(tag="final")
        if self.wandb_run:
            wandb.finish()
        print(f"\nPhase {self.config.phase} complete. "
              f"Total tokens: {self.tokens_seen:,}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point (for local testing / direct runs)
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run training locally (primarily for dry-run testing)."""
    import argparse

    parser = argparse.ArgumentParser(description="KT_GPT Pretraining")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 steps with tiny batch for validation")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model_config = KTGPTConfig()
    model = KTGPT(model_config).to(device=device, dtype=torch.bfloat16)

    # Training config
    train_config = TrainConfig.phase1() if args.phase == 1 else TrainConfig.phase2()
    train_config.checkpoint_dir = args.checkpoint_dir

    if args.dry_run:
        # Override for quick local validation
        train_config.total_tokens = train_config.tokens_per_step * 2
        train_config.log_every = 1
        train_config.checkpoint_every = 999999
        train_config.num_workers = 0
        print("[DRY RUN] Running 2 steps for validation...\n")

    # Data
    from data.mix import create_dataloader
    dataloader = create_dataloader(
        phase=args.phase,
        batch_size=train_config.micro_batch_size,
        seq_len=train_config.seq_len,
        total_tokens=train_config.total_tokens,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )

    # Trainer
    trainer = Trainer(model=model, config=train_config, device=device)

    # Resume or load Phase 1 weights for Phase 2
    if args.phase == 2:
        if not trainer.load_checkpoint():
            # No Phase 2 checkpoint — try loading Phase 1 model
            trainer.load_phase1_checkpoint()
    else:
        trainer.load_checkpoint()

    trainer.train(dataloader)


if __name__ == "__main__":
    main()
