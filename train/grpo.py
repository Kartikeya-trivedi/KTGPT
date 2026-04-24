"""
GRPO Reinforcement Learning
============================

Group Relative Policy Optimization for KT_GPT.

Simplified GRPO:
  1. For each problem, generate G=4 candidate outputs
  2. Execute each against hidden test suite → binary reward (0/1)
  3. Compute advantages: reward - mean(group_rewards)
  4. Policy gradient loss on positive-advantage outputs only
  5. KL penalty against the SFT reference model

This is the final training stage: SFT model → GRPO model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import KTGPTConfig
from model.model import KTGPT
from data.filter import execute_with_tests


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    group_size: int = 4           # candidates per problem
    lr: float = 1e-6              # very low LR
    kl_coef: float = 0.05         # KL penalty weight
    max_new_tokens: int = 2048
    temperature: float = 0.8
    grad_clip: float = 1.0
    max_steps: int = 1000
    log_every: int = 10
    checkpoint_every: int = 200
    checkpoint_dir: str = "./checkpoints/grpo"
    wandb_project: str = "kt-gpt"
    execution_timeout: int = 10


class GRPOTrainer:
    """GRPO trainer for code generation improvement.

    Uses the SFT model as both the policy (trainable) and the reference
    (frozen copy for KL penalty). Generates multiple solutions per problem,
    rewards correct ones, and does policy gradient with advantages.
    """

    def __init__(
        self,
        model: KTGPT,
        tokenizer,
        config: GRPOConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Freeze a copy of the SFT model as reference for KL penalty
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            betas=(0.9, 0.95),
        )

        self.global_step = 0

    @torch.no_grad()
    def _generate(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a completion and return (full_ids, log_probs).

        Returns the token IDs and per-token log probabilities under the
        current policy for the generated tokens.
        """
        self.model.eval()
        generated = prompt_ids.clone()

        for _ in range(self.config.max_new_tokens):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, _ = self.model(generated)

            next_logits = logits[:, -1, :] / self.config.temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        self.model.train()
        return generated

    def _compute_log_probs(
        self,
        model: KTGPT,
        input_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for the generated portion."""
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids)

        # Only compute log probs for generated tokens (after prompt)
        gen_logits = logits[:, prompt_len - 1:-1, :]  # shifted
        gen_targets = input_ids[:, prompt_len:]

        log_probs = F.log_softmax(gen_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=gen_targets.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    def _compute_rewards(
        self,
        completions: list[str],
        tests: str,
    ) -> list[float]:
        """Execute completions against test cases, return binary rewards."""
        rewards = []
        for completion in completions:
            passed, _ = execute_with_tests(
                completion, tests, timeout=self.config.execution_timeout
            )
            rewards.append(1.0 if passed else 0.0)
        return rewards

    def train_step(
        self,
        problem: str,
        tests: str,
    ) -> dict:
        """Single GRPO training step for one problem.

        1. Generate G candidates
        2. Compute rewards (execute tests)
        3. Compute advantages (reward - mean)
        4. Policy gradient with KL penalty on positive-advantage samples

        Returns metrics dict.
        """
        prompt = f"Solve the following problem:\n\n{problem}\n\nSolution:\n"
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.shape[1]

        # Generate G candidates
        all_ids: list[torch.Tensor] = []
        completions: list[str] = []

        for _ in range(self.config.group_size):
            gen_ids = self._generate(prompt_ids)
            all_ids.append(gen_ids)
            gen_text = self.tokenizer.decode(
                gen_ids[0][prompt_len:], skip_special_tokens=True
            )
            completions.append(gen_text)

        # Compute rewards
        rewards = self._compute_rewards(completions, tests)
        mean_reward = sum(rewards) / len(rewards)

        # Compute advantages
        advantages = [r - mean_reward for r in rewards]

        # Only train on positive-advantage samples
        total_loss = torch.tensor(0.0, device=self.device)
        num_positive = 0

        for i, (gen_ids, advantage) in enumerate(zip(all_ids, advantages)):
            if advantage <= 0:
                continue

            num_positive += 1

            # Policy log probs
            policy_log_probs = self._compute_log_probs(
                self.model, gen_ids, prompt_len
            )

            # Reference log probs (for KL penalty)
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(
                    self.ref_model, gen_ids, prompt_len
                )

            # KL divergence penalty: KL(policy || ref) per token
            kl_div = (policy_log_probs - ref_log_probs).mean()

            # Policy gradient loss: -advantage * sum(log_probs) + kl_penalty
            pg_loss = -advantage * policy_log_probs.sum()
            kl_loss = self.config.kl_coef * kl_div

            total_loss = total_loss + pg_loss + kl_loss

        if num_positive > 0:
            total_loss = total_loss / num_positive
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1

        return {
            "loss": total_loss.item() if num_positive > 0 else 0.0,
            "mean_reward": mean_reward,
            "num_positive": num_positive,
            "group_size": self.config.group_size,
        }

    def train(self, problems: list[dict]) -> None:
        """Full GRPO training loop over a set of problems.

        Args:
            problems: List of dicts with 'description' and 'tests' keys
        """
        import wandb
        wandb.init(project=self.config.wandb_project, name="grpo")

        print(f"\n[GRPO] Starting training: {len(problems)} problems, "
              f"max {self.config.max_steps} steps")

        for step in range(min(self.config.max_steps, len(problems))):
            prob = problems[step % len(problems)]
            metrics = self.train_step(
                prob.get("description", ""),
                prob.get("tests", ""),
            )

            if self.global_step % self.config.log_every == 0:
                print(f"[GRPO step {self.global_step}] "
                      f"reward={metrics['mean_reward']:.2f} "
                      f"loss={metrics['loss']:.4f} "
                      f"positive={metrics['num_positive']}/{metrics['group_size']}")
                wandb.log({f"grpo/{k}": v for k, v in metrics.items()},
                         step=self.global_step)

            if self.global_step % self.config.checkpoint_every == 0:
                from pathlib import Path
                ckpt_dir = Path(self.config.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": self.model.state_dict(), "step": self.global_step},
                    ckpt_dir / f"step_{self.global_step}.pt",
                )

        wandb.finish()
        print(f"\n[GRPO] Training complete. {self.global_step} steps.")
