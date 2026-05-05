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
    grad_accum_steps: int = 1               # Adjusted down to 1 since micro_batch is 8
    compile: bool = False                   # Disable by default for Phase 1 stability

    # ── Batch / sequence ─────────────────────────────────────────────
    micro_batch_size: int = 8               # Base Phase 1 batch size
    seq_len: int = 4096                     # Base Phase 1 sequence length

    # ── Logging & checkpointing ──────────────────────────────────────
    log_every: int = 10                     # log to W&B every N steps
    checkpoint_every: int = 500             # save checkpoint every N steps
    checkpoint_dir: str = "/checkpoints"    # Modal volume mount point
    wandb_project: str = "kt-gpt"
    wandb_run_name: Optional[str] = None
    phase1_checkpoint_path: Optional[str] = None

    # ── Data ─────────────────────────────────────────────────────────
    num_workers: int = 8
    seed: int = 42

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed per optimizer step (across all grad accum steps and GPUs)."""
        import os
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return self.micro_batch_size * self.seq_len * self.grad_accum_steps * world_size

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
        """Phase 2: 5B tokens, lower LR, full data mix."""
        return cls(
            phase=2,
            total_tokens=5_000_000_000,
            lr=1e-4,
            warmup_steps=200,               # Reduced from 2000 because total steps is now only ~2384
            wandb_run_name="phase2-5B",
            micro_batch_size=8,             # Halved to 8 to fit batched MoE activation memory
            grad_accum_steps=32,            # Effective batch: 8*2048*32*4GPUs = ~2.1M tokens/step
            seq_len=2048,                   # Halved to 2048 to slash attention compute by 4x
            num_workers=16,                 # Max out IO to prevent dataloader bottlenecking
            compile=False,                  # Disabled — MoE dynamic routing causes recompilation, no speedup
            phase1_checkpoint_path="/checkpoints/phase1/final.pt",
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
        is_ddp: bool = False,
        global_rank: int = 0,
        local_rank: int = 0,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.is_ddp = is_ddp
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.global_step = 0
        self.tokens_seen = 0

        # Optimizer — exclude router biases (updated separately)
        param_groups = self._build_param_groups()
        # fused AdamW requires PyTorch >= 2.1 and CUDA
        use_fused = (
            device.type == "cuda"
            and hasattr(torch.optim, '_multi_tensor')
            or int(torch.__version__.split('.')[0]) >= 2
        )
        try:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                fused=use_fused and device.type == "cuda",
            )
        except RuntimeError:
            # Fallback if fused is not supported
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                fused=False,
            )

        # GradScaler is disabled — bf16 has the same dynamic range as fp32,
        # so loss scaling is unnecessary and actually errors on bf16 grads.
        # Keeping the scaler object (enabled=False) lets all .scale()/.step()
        # calls act as no-ops without changing the training loop structure.
        self.scaler = torch.amp.GradScaler(device.type, enabled=False)
        
        # Wrap in DDP if needed
        if self.is_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank]
            )

        # Compile model to fuse kernels and drastically reduce memory bandwidth
        if self.config.compile:
            print("🚀 Compiling model via torch.compile()... (this takes 1-2 mins on first step)")
            # Use default inductor backend
            self.model = torch.compile(self.model)

        # W&B
        self.wandb_run = None

    @property
    def raw_model(self) -> KTGPT:
        return self.model.module if self.is_ddp and hasattr(self.model, 'module') else self.model

    def _build_param_groups(self) -> list[dict]:
        """Separate weight-decay and no-weight-decay parameter groups.

        Convention: don't apply weight decay to norms, biases, or embeddings.
        Router expert_bias is excluded entirely (requires_grad=False).
        """
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, param in self.raw_model.named_parameters():
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

    @torch.no_grad()
    def _get_expert_load(self) -> torch.Tensor:
        """Get approximate expert load distribution from router biases.

        Aggregates the bias values across all layers. Higher bias = expert
        was underloaded (bias got increased), lower bias = overloaded.
        We convert to approximate counts: experts with lower bias received
        more tokens.
        """
        num_experts = self.raw_model.config.num_routed_experts
        total_load = torch.zeros(num_experts, device=self.device)
        for layer in self.raw_model.layers:
            bias = layer.ffn.router.expert_bias
            # Invert bias: lower bias → more tokens → higher "load"
            # Normalize so all values are positive
            load = -bias + bias.max() + 1.0
            total_load += load.to(self.device)
        return total_load

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
            for i, layer in enumerate(self.raw_model.layers)
        }

        checkpoint = {
            "model": self.raw_model.state_dict(),
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
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except (RuntimeError, EOFError) as e:
            print(f"[CKPT] WARNING: Checkpoint corrupted or incomplete ({e}). Starting from scratch.")
            return False

        self.raw_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]

        # Restore router biases
        for i, layer in enumerate(self.raw_model.layers):
            key = f"layer_{i}"
            if key in checkpoint.get("router_biases", {}):
                layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])

        print(f"[CKPT] Resumed from step={self.global_step}, tokens={self.tokens_seen}")
        return True

    def load_phase1_checkpoint(self) -> bool:
        """Load the Phase 1 final checkpoint for Phase 2 continuation.

        Loads model weights only (resets optimizer and scheduler for Phase 2).
        """
        if self.config.phase1_checkpoint_path:
            latest = Path(self.config.phase1_checkpoint_path)
        else:
            ckpt_dir = Path(self.config.checkpoint_dir) / "phase1"
            latest = ckpt_dir / "latest.pt"

        if not latest.exists():
            print(f"[CKPT] No Phase 1 checkpoint found at {latest}!")
            return False

        print(f"[CKPT] Loading Phase 1 model weights for Phase 2 from {latest}...")
        checkpoint = torch.load(str(latest), map_location=self.device, weights_only=False)

        # Handle torch.compile() key prefix mismatch:
        # Phase 1 saves keys like "embed.weight" but a compiled model expects
        # "_orig_mod.embed.weight". Remap if needed.
        state_dict = checkpoint["model"]
        
        # --- Convert Phase 1 ModuleList experts to Phase 2 Batched Experts ---
        new_state_dict = {}
        # Collect expert weights across all layers
        num_layers = self.raw_model.config.num_layers
        layer_experts_gate = {i: [] for i in range(num_layers)}
        layer_experts_up = {i: [] for i in range(num_layers)}
        layer_experts_down = {i: [] for i in range(num_layers)}
        
        for k, v in state_dict.items():
            if "routed_experts" in k:
                # k looks like: "layers.0.ffn.routed_experts.5.gate_proj.weight"
                parts = k.split(".")
                layer_idx = int(parts[1])
                expert_idx = int(parts[4])
                proj_type = parts[5] # gate_proj, up_proj, down_proj
                
                # Make sure the list is big enough (since dict iteration is unordered)
                if proj_type == "gate_proj":
                    while len(layer_experts_gate[layer_idx]) <= expert_idx:
                        layer_experts_gate[layer_idx].append(None)
                    layer_experts_gate[layer_idx][expert_idx] = v
                elif proj_type == "up_proj":
                    while len(layer_experts_up[layer_idx]) <= expert_idx:
                        layer_experts_up[layer_idx].append(None)
                    layer_experts_up[layer_idx][expert_idx] = v
                elif proj_type == "down_proj":
                    while len(layer_experts_down[layer_idx]) <= expert_idx:
                        layer_experts_down[layer_idx].append(None)
                    layer_experts_down[layer_idx][expert_idx] = v
            else:
                new_state_dict[k] = v
                
        # Stack the collected expert weights
        for i in range(num_layers):
            if layer_experts_gate[i] and layer_experts_gate[i][0] is not None:
                new_state_dict[f"layers.{i}.ffn.expert_gate_weight"] = torch.stack(layer_experts_gate[i])
                new_state_dict[f"layers.{i}.ffn.expert_up_weight"] = torch.stack(layer_experts_up[i])
                new_state_dict[f"layers.{i}.ffn.expert_down_weight"] = torch.stack(layer_experts_down[i])
                
        state_dict = new_state_dict
        # ---------------------------------------------------------------------

        model_keys = set(self.raw_model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        needs_orig_prefix = (
            any(k.startswith("_orig_mod.") for k in model_keys)
            and not any(k.startswith("_orig_mod.") for k in ckpt_keys)
        )
        needs_strip_prefix = (
            not any(k.startswith("_orig_mod.") for k in model_keys)
            and any(k.startswith("_orig_mod.") for k in ckpt_keys)
        )
        if needs_orig_prefix:
            print("[CKPT] Adding _orig_mod. prefix to checkpoint keys for compiled model")
            state_dict = {"_orig_mod." + k: v for k, v in state_dict.items()}
        elif needs_strip_prefix:
            print("[CKPT] Stripping _orig_mod. prefix from checkpoint keys")
            state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

        self.raw_model.load_state_dict(state_dict)

        # Restore router biases from Phase 1
        # Access layers through raw KTGPT model (unwrap compile + DDP)
        unwrapped = self.raw_model
        if hasattr(unwrapped, '_orig_mod'):
            unwrapped = unwrapped._orig_mod
        for i, layer in enumerate(unwrapped.layers):
            key = f"layer_{i}"
            if key in checkpoint.get("router_biases", {}):
                layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])

        print(f"[CKPT] Loaded Phase 1 model (trained on {checkpoint.get('tokens_seen', 'unknown')} tokens)")
        print("[CKPT] Note: Optimizer state and LR scheduler are reset for Phase 2.")
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
        if self.wandb_run is None and self.global_rank == 0:
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

        # Accumulate expert counts across micro-batches for load metrics
        num_routed = self.raw_model.config.num_routed_experts
        accum_expert_counts = torch.zeros(num_routed, device=self.device)

        if self.global_rank == 0:
            print(f"\n{'='*60}")
            print(f"  Phase {self.config.phase} Training")
            print(f"  Total tokens: {self.config.total_tokens / 1e9:.1f}B")
            print(f"  Total steps:  {self.config.total_steps}")
            print(f"  Tokens/step:  {self.config.tokens_per_step:,}")
            print(f"  Resuming from step {self.global_step}")
            print(f"{'='*60}\n")

        from tqdm import tqdm
        
        micro_step = 0
        pbar = None
        if self.global_rank == 0:
            pbar = tqdm(
                total=self.config.total_steps,
                initial=self.global_step,
                desc=f"Phase {self.config.phase} Steps",
                dynamic_ncols=True
            )
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)      # (B, S)
            labels = batch["labels"].to(self.device)             # (B, S)

            # ── Forward + loss ───────────────────────────────────
            import contextlib
            is_accumulating = (micro_step + 1) % self.config.grad_accum_steps != 0
            sync_context = self.model.no_sync() if self.is_ddp and is_accumulating else contextlib.nullcontext()

            with sync_context:
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

            import os
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            accum_loss += loss.detach().item()
            accum_count += 1
            micro_step += 1
            self.tokens_seen += input_ids.numel() * world_size

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
                if pbar: pbar.update(1)

                # ── Logging ──────────────────────────────────────
                if self.global_step % self.config.log_every == 0:
                    elapsed = time.time() - step_start
                    tokens_per_sec = (
                        self.config.tokens_per_step * self.config.log_every / elapsed
                    )

                    # Router metrics
                    entropies = compute_router_entropy(self.raw_model)
                    avg_entropy = sum(entropies) / len(entropies)
                    
                    bias_stds = [layer.ffn.router.expert_bias.float().std().item() for layer in self.raw_model.layers]
                    avg_bias_std = sum(bias_stds) / len(bias_stds)

                    # Expert load metrics — run a quick forward to get counts
                    expert_counts = self._get_expert_load()
                    expert_cv = compute_expert_load_cv(expert_counts)

                    # DDP Sync: Average metrics across all GPUs
                    if self.is_ddp:
                        import torch.distributed as dist
                        metrics = torch.tensor([avg_loss, tokens_per_sec, avg_entropy, expert_cv, avg_bias_std], device=self.device)
                        dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                        avg_loss, tokens_per_sec, avg_entropy, expert_cv, avg_bias_std = metrics.tolist()

                    if self.global_rank == 0:
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/lr": current_lr,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/tokens_seen": self.tokens_seen,
                            "train/step": self.global_step,
                            "router/avg_entropy": avg_entropy,
                            "router/expert_load_cv": expert_cv,
                            "router/avg_bias_std": avg_bias_std,
                        }

                        # Per-layer entropy (every 10th layer to avoid clutter)
                        for i in range(0, len(entropies), max(1, len(entropies) // 6)):
                            log_dict[f"router/entropy_L{i}"] = entropies[i]

                        # Per-expert token counts (log top/bottom 5 + full histogram)
                        if expert_counts.sum() > 0:
                            sorted_counts, sorted_idx = expert_counts.sort(descending=True)
                            log_dict["router/expert_max_load"] = sorted_counts[0].item()
                            log_dict["router/expert_min_load"] = sorted_counts[-1].item()
                            # Log a few representative experts
                            for rank, ei in enumerate(sorted_idx[:3].tolist()):
                                log_dict[f"router/top{rank+1}_expert_{ei}"] = sorted_counts[rank].item()

                        # GPU memory tracking
                        if self.device.type == "cuda":
                            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                            current_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                            log_dict["system/gpu_peak_memory_mb"] = peak_mb
                            log_dict["system/gpu_current_memory_mb"] = current_mb
                            log_dict["system/gpu_memory_utilization"] = (
                                peak_mb / (torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2))
                            )

                        if self.wandb_run:
                            wandb.log(log_dict, step=self.global_step)

                        # Enhanced console output
                        mem_str = ""
                        if self.device.type == "cuda":
                            mem_str = f"  mem={peak_mb:.0f}MB"

                        pbar.write(
                            f"[step {self.global_step:>6d}/{self.config.total_steps}] "
                            f"loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                            f"gnorm={grad_norm:.2f}  tok/s={tokens_per_sec:.0f}  "
                            f"entropy={avg_entropy:.2f}  cv={expert_cv:.3f}"
                            f"{mem_str}"
                        )

                # ── Checkpoint ───────────────────────────────────
                if self.global_step % self.config.checkpoint_every == 0 and self.global_rank == 0:
                    self.save_checkpoint()

                # Reset timer AFTER checkpointing so it doesn't ruin tok/s metric
                if self.global_step % self.config.log_every == 0:
                    step_start = time.time()

                # Reset accumulators
                accum_loss = 0.0
                accum_count = 0
                accum_expert_counts.zero_()

                # ── Termination ──────────────────────────────────
                if self.global_step >= self.config.total_steps:
                    if pbar: pbar.write(f"\nReached {self.config.total_steps} steps. Training complete.")
                    break

        if pbar: pbar.close()

        # Final checkpoint
        if self.global_rank == 0:
            self.save_checkpoint(tag="final")
        if self.wandb_run:
            wandb.finish()
        if self.global_rank == 0:
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

    # Enable TF32 for massive speedup on Ampere (A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # DDP Initialization
    import os
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        global_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training config
    train_config = TrainConfig.phase1() if args.phase == 1 else TrainConfig.phase2()
    train_config.checkpoint_dir = args.checkpoint_dir

    # Model config
    model_config = KTGPTConfig()
    model = KTGPT(model_config).to(device=device, dtype=torch.bfloat16)

    # NOTE: torch.compile() is handled inside Trainer.__init__ via config.compile.
    # Do NOT compile here — double-compiling causes _orig_mod key prefix mismatch
    # when loading Phase 1 checkpoints into Phase 2.



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
    trainer = Trainer(
        model=model, 
        config=train_config, 
        device=device,
        is_ddp=is_ddp,
        global_rank=global_rank,
        local_rank=local_rank
    )

    # Resume or load Phase 1 weights for Phase 2
    if args.phase == 2:
        if not trainer.load_checkpoint():
            # No Phase 2 checkpoint — try loading Phase 1 model
            trainer.load_phase1_checkpoint()
    else:
        trainer.load_checkpoint()

    trainer.train(dataloader)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
