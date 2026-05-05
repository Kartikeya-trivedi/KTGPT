from __future__ import annotations
"""
GRPO Reinforcement Learning for KT_GPT (Phase 4)
=================================================

Implements the DeepSeek Group Relative Policy Optimization (GRPO) algorithm 
for reasoning tasks and tool-use optimization.
"""

import json

import os
import sys

# Add project root to sys.path for local module resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import re
import copy
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.config import KTGPTConfig
from model.model import KTGPT
from train.pretrain import TrainConfig


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GRPOConfig(TrainConfig):
    phase: int = 4
    lr: float = 1e-6              # low LR for RL
    min_lr: float = 1e-7
    warmup_steps: int = 50
    weight_decay: float = 0.01
    
    micro_batch_size: int = 1     # prompts per step
    group_size: int = 4           # G factor (candidates per prompt)
    max_new_tokens: int = 256     # faster generation for RL
    temperature: float = 0.8
    num_epochs: int = 1
    
    # GRPO Specific
    clip_range: float = 0.2       # PPO clipping epsilon
    kl_coef: float = 0.01         # Start with low KL penalty
    grad_clip: float = 1.0
    
    log_every: int = 1            # See every step
    checkpoint_every: int = 100
    checkpoint_dir: str = "/checkpoints/grpo"
    wandb_run_name: str = "grpo-gsm8k"


# ═══════════════════════════════════════════════════════════════════════
#  Data & Rewards (Adapted from Colab)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text: return ""
    answer = text.split("<answer>")[-1]
    if "</answer>" in answer:
        answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str:
    if "####" not in text: return ""
    return text.split("####")[1].strip()

class GSM8KDataset(Dataset):
    def __init__(self, data_path: str):
        import json
        print(f"[GRPO] Loading GSM8K dataset from {data_path}...")
        
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))
                
        print(f"[GRPO] Loaded {len(self.samples)} samples from volume.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return {
        "prompt": [b["prompt"] for b in batch],
        "answer": [b["answer"] for b in batch]
    }

# --- Tool Execution (Internal for Rewards) ---
def _execute_reward_tool(text: str) -> str:
    """Parses a tool call and returns the simulated result for rewarding."""
    try:
        if "<tool_call>" not in text: return ""
        tc_start = text.find("<tool_call>") + len("<tool_call>")
        tc_end = text.find("</tool_call>")
        if tc_end == -1: return ""
        
        call_json = json.loads(text[tc_start:tc_end])
        name = call_json.get("name")
        args = call_json.get("arguments", {})
        
        if name == "calculator":
            expr = args.get("expression", "")
            # Very strict safety for reward-loop execution
            allowed = set("0123456789+-*/.(). ")
            if all(c in allowed for c in expr):
                return str(eval(expr))
            return "ERROR_INVALID_EXPR"
        return f"ERROR_UNKNOWN_TOOL_{name}"
    except Exception:
        return "ERROR_PARSE"

# --- Reward Functions ---
def correctness_reward(extracted_responses: List[str], ground_truth: str) -> List[float]:
    return [2.0 if r == ground_truth else 0.0 for r in extracted_responses]

def int_reward(extracted_responses: List[str]) -> List[float]:
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def format_reward(responses: List[str]) -> List[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]

def xmlcount_reward(responses: List[str]) -> List[float]:
    rewards = []
    for text in responses:
        count = 0.0
        if "<reasoning>\n" in text: count += 0.125
        if "\n</reasoning>\n" in text: count += 0.125
        if "\n<answer>\n" in text: count += 0.125
        if "\n</answer>" in text: count += 0.125
        rewards.append(count)
    return rewards

def tool_reward(responses: List[str], ground_truth: str) -> List[float]:
    """Rewards correct tool-use format and correct final answer after tool usage."""
    rewards = []
    for text in responses:
        r = 0.0
        # 1. Format Reward
        if "<tool_call>" in text and "</tool_call>" in text:
            r += 0.5
            
            # 2. Logic Check: Does it actually try to solve the math?
            # We can check if the model's final <answer> matches ground truth
            # after it made a tool call.
            ans = extract_xml_answer(text)
            if ans == ground_truth:
                r += 2.0  # Big boost for successful tool-assisted reasoning
        
        # 3. Penalty for mental math (forcing calculator usage)
        if any(c in "0123456789" for c in text) and "<tool_call>" not in text:
            # Curriculum: start with soft penalty, ramp up later
            r -= 0.1
            
        # 4. Penalty for empty answers with tools
        if "<tool_call>" in text and not extract_xml_answer(text):
            r -= 1.0
            
        rewards.append(r)
    return rewards

def calculate_group_rewards(generated_texts: List[str], ground_truth: str) -> torch.Tensor:
    extracted = [extract_xml_answer(r) for r in generated_texts]
    
    # 1. Correctness & Integer
    r_corr = correctness_reward(extracted, ground_truth)
    r_int = int_reward(extracted)
    
    # 2. Format & XML Structure (+0.1)
    r_fmt = format_reward(generated_texts) # 0.5 if full pattern
    r_xml = xmlcount_reward(generated_texts) # up to 0.5 for tags
    
    # 3. Tool Reward (Curriculum: start soft)
    r_tool = tool_reward(generated_texts, ground_truth)
    
    # 4. Numeric Signal (+0.3 if any number is present, even if wrong)
    # This helps the model avoid "empty" responses.
    r_num = [0.3 if any(c.isdigit() for c in e) else 0.0 for e in extracted]
    
    # 5. Length Penalty (-0.001 per token)
    # Prevents max-length collapse
    r_len = [-0.001 * len(t.split()) for t in generated_texts]
    
    total = [
        c + i + (f * 0.2) + (x * 0.2) + t + n + l 
        for c, i, f, x, t, n, l in zip(r_corr, r_int, r_fmt, r_xml, r_tool, r_num, r_len)
    ]
    return torch.tensor(total, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
#  GRPO Trainer
# ═══════════════════════════════════════════════════════════════════════

class GRPOTrainer:
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

        print("[GRPO] Setting up Reference Model...")
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        self.global_step = 0
        self.wandb_run = None

    @torch.no_grad()
    def _generate_group(self, prompt_ids: List[int]) -> torch.Tensor:
        """Generate G completions for a prompt."""
        self.model.eval()
        G = self.config.group_size
        
        # Expand prompt to (G, prompt_len)
        input_ids = torch.tensor([prompt_ids] * G, dtype=torch.long, device=self.device)
        generated = input_ids.clone()
        
        for _ in range(self.config.max_new_tokens):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits, _ = self.model(generated)

            next_logits = logits[:, -1, :] / self.config.temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Optimization: Stop if all sequences emitted EOS
            if (generated == self.tokenizer.eos_token_id).sum(-1).bool().all():
                break

        self.model.train()
        return generated

    def _compute_log_probs(self, model: KTGPT, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Compute per-token log probs for the generated sequence."""
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids)

        gen_logits = logits[:, prompt_len - 1:-1, :]  # shifted logits
        gen_targets = input_ids[:, prompt_len:]      # shifted targets

        log_probs = F.log_softmax(gen_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=gen_targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs

    def train_step(self, prompt_ids: List[int], answer: str) -> dict:
        prompt_len = len(prompt_ids)

        # 1. Rollout
        with torch.no_grad():
            gen_ids = self._generate_group(prompt_ids)
            
        # 2. Pre-compute log_old_pi (Sampling Baseline)
        # This is CRITICAL for PPO correctness
        with torch.no_grad():
            old_log_probs = self._compute_log_probs(self.model, gen_ids, prompt_len)
            ref_log_probs = self._compute_log_probs(self.ref_model, gen_ids, prompt_len)

        # Decode only the generated portion for rewards
        gen_tokens = gen_ids[:, prompt_len:]
        completions = [self.tokenizer.decode(t, skip_special_tokens=True) for t in gen_tokens]
        
        # 3. Rewards & Advantages
        rewards = calculate_group_rewards(completions, answer).to(self.device)
        mean_reward = rewards.mean()
        
        # Safe Advantage Normalization
        if rewards.std() < 1e-6:
            advantages = torch.zeros_like(rewards)
        else:
            advantages = (rewards - mean_reward) / (rewards.std() + 1e-8)

        # 4. PPO Update
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_kl = 0.0

        # We compute grads for the whole group in one pass if possible, 
        # but here we follow the existing per-completion loop for clarity.
        for i in range(self.config.group_size):
            # Current Policy log probs (with grad)
            pi_log_probs = self._compute_log_probs(self.model, gen_ids[i:i+1], prompt_len)
            
            # PPO Ratio (pi / old_pi)
            ratio = torch.exp(pi_log_probs - old_log_probs[i:i+1])
            
            # Surrogate Loss (Clipped)
            adv = advantages[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean() # Mean over sequence
            
            # Unbiased KL Estimator (DeepSeek Style)
            # kl = exp(log_ref - log_pi) - (log_ref - log_pi) - 1
            log_ratio = ref_log_probs[i:i+1] - pi_log_probs
            kl_div = torch.exp(log_ratio) - log_ratio - 1
            kl_div = torch.clamp(kl_div, max=10.0) # Stability clamp
            kl_loss = self.config.kl_coef * kl_div.mean()
            
            loss = (policy_loss + kl_loss) / self.config.group_size
            loss.backward()
            
            total_loss += loss.item()
            total_kl += kl_div.mean().item() / self.config.group_size

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.global_step += 1
        
        return {
            "loss": total_loss,
            "mean_reward": mean_reward.item(),
            "max_reward": rewards.max().item(),
            "kl_div": total_kl,
            "completion_len": gen_tokens.shape[1]
        }

    def train(self, dataloader) -> None:
        try:
            import wandb
            self.wandb_run = wandb.init(project="kt-gpt", name=self.config.wandb_run_name)
        except Exception:
            pass

        print(f"\n[GRPO] Starting training on GSM8K...")

        for i, batch in enumerate(dataloader):
            if i >= 500: # Speed limit for research
                print("[GRPO] Reached 500 step limit. Training complete.")
                break
                
            # We process 1 prompt at a time (micro_batch_size=1)
            prompt_text = batch["prompt"][0]
            answer = batch["answer"][0]
            
            # Tokenize on the fly
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            
            metrics = self.train_step(prompt_ids, answer)

            if self.global_step % self.config.log_every == 0:
                print(f"[GRPO {self.global_step}] "
                      f"R_mean={metrics['mean_reward']:.2f} | R_max={metrics['max_reward']:.2f} | "
                      f"Loss={metrics['loss']:.3f} | KL={metrics['kl_div']:.3f} | Len={metrics['completion_len']}")
                
                if self.wandb_run:
                    wandb.log({f"grpo/{k}": v for k, v in metrics.items()}, step=self.global_step)

            if self.global_step % self.config.checkpoint_every == 0:
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                path = os.path.join(self.config.checkpoint_dir, f"step_{self.global_step}.pt")
                torch.save({"model": self.model.state_dict(), "step": self.global_step}, path)
                print(f"[GRPO] Saved {path}")

        if self.wandb_run:
            wandb.finish()


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry
# ═══════════════════════════════════════════════════════════════════════

def run_grpo(checkpoint_path: str, output_dir: str = "/checkpoints/grpo") -> None:
    from data.mix import get_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[GRPO] Loading SFT Policy from {checkpoint_path}...")
    model_config = KTGPTConfig()
    model = KTGPT(model_config).to(device=device, dtype=torch.bfloat16)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    tokenizer = get_tokenizer()
    config = GRPOConfig()
    config.checkpoint_dir = output_dir

    dataset = GSM8KDataset(data_path="/checkpoints/data/pipeline/stage3_grpo_prompts.jsonl")
    dataloader = DataLoader(dataset, batch_size=config.micro_batch_size, shuffle=True, collate_fn=collate_fn)

    trainer = GRPOTrainer(model, tokenizer, config, device)
    trainer.train(dataloader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to final SFT checkpoint")
    parser.add_argument("--output_dir", type=str, default="/checkpoints/grpo")
    args = parser.parse_args()
    
    run_grpo(args.checkpoint, args.output_dir)
