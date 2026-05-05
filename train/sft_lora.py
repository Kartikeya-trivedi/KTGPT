from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from data.mix import get_tokenizer
from model.config import KTGPTConfig
from model.lora import LoRAConfig, inject_lora, lora_state_dict, trainable_parameter_count, merge_lora_linears
from model.model import KTGPT
from train.pretrain import TrainConfig, Trainer


@dataclass
class LoRASFTConfig(TrainConfig):
    phase: int = 3
    lr: float = 2e-4
    min_lr: float = 2e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    grad_accum_steps: int = 8
    micro_batch_size: int = 4
    seq_len: int = 512
    num_epochs: int = 2
    checkpoint_every: int = 500
    wandb_run_name: str = "sft-lora-final"
    sft_data_path: str = "data/lora_final.jsonl"


# Alpaca response separator — must match the template in build_lora_dataset.py
_RESPONSE_SPLIT = "### Response:\n"


class SupervisedPairDataset(Dataset):
    """
    Reads the Alpaca 4-field JSONL format produced by build_lora_dataset.py.

    Each row has:
        text        — full formatted string (prompt + response), used for tokenisation
        instruction — task directive           (kept for reference / filtering)
        input       — user data / context      (kept for reference / filtering)
        output      — expected model response  (kept for reference / filtering)

    Label masking strategy:
        Everything up to and including "### Response:\\n" → -100  (prompt, not trained on)
        Everything after                                 → token ids (trained on)
        Padding tokens                                   → -100

    This guarantees that loss is computed ONLY on the model's output,
    and that the exact same text the model sees at inference is what
    it was trained on.
    """

    def __init__(self, path: str, tokenizer, seq_len: int) -> None:
        self.seq_len   = seq_len
        self.tokenizer = tokenizer
        self.rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # Accept both the new 4-field format and old prompt/response format
                if "text" in row and "output" in row:
                    self.rows.append({"text": row["text"], "output": row["output"]})
                elif "prompt" in row and "response" in row:
                    # Fallback: reconstruct Alpaca text from flat fields
                    text = (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        f"### Instruction:\n{row['prompt'].strip()}\n\n"
                        f"### Response:\n{row['response'].strip()}"
                    )
                    self.rows.append({"text": text, "output": row["response"].strip()})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row  = self.rows[idx]
        text = row["text"]

        # Tokenise the full string (BOS prepended)
        full_ids = (
            [self.tokenizer.bos_token_id]
            + self.tokenizer.encode(text, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        # Find where the response starts so we can mask the prompt
        # We locate "### Response:\n" in the raw text, then count tokens
        split_pos = text.find(_RESPONSE_SPLIT)
        if split_pos != -1:
            prompt_part = text[: split_pos + len(_RESPONSE_SPLIT)]
            prompt_token_count = (
                1  # BOS
                + len(self.tokenizer.encode(prompt_part, add_special_tokens=False))
            )
        else:
            # Fallback: mask nothing (treat whole sequence as response)
            prompt_token_count = 1

        labels = [-100] * prompt_token_count + full_ids[prompt_token_count:]

        # Truncate to seq_len
        ids    = full_ids[:self.seq_len]
        labels = labels[:self.seq_len]

        # Pad to seq_len
        pad_id = self.tokenizer.pad_token_id or 0
        pad    = self.seq_len - len(ids)
        if pad > 0:
            ids.extend([pad_id] * pad)
            labels.extend([-100] * pad)

        return {
            "input_ids": torch.tensor(ids,    dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
        }


# NOTE: MixedSFTDataset / GenericTextDataset removed.
# The Alpaca portion of lora_final.jsonl already provides general language
# capability — mixing raw wikitext at training time just dilutes the
# behaviour-alignment signal and wastes GPU time.


def run_lora_sft(
    checkpoint_path: str,
    sft_data_path: str = "data/lora_final.jsonl",
    output_dir: str = "checkpoints/sft_lora",
    num_epochs: int = 2,
    seq_len: int = 512,
    lr: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    config = KTGPTConfig()
    config.max_seq_len = 2048
    model = KTGPT(config).to(device=device, dtype=torch.bfloat16)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    for i, layer in enumerate(model.layers):
        key = f"layer_{i}"
        if key in checkpoint.get("router_biases", {}):
            layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])

    inject_lora(model, LoRAConfig(r=lora_r, alpha=lora_alpha, dropout=lora_dropout))
    model.to(device=device, dtype=torch.bfloat16)  # Move new LoRA params to GPU
    trainable, total = trainable_parameter_count(model)
    print(f"[LoRA] trainable={trainable:,} / total={total:,} ({100.0*trainable/total:.3f}%)")

    tokenizer = get_tokenizer()
    # Pure instruction dataset — Alpaca inside already handles general language.
    dataset = SupervisedPairDataset(sft_data_path, tokenizer=tokenizer, seq_len=seq_len)
    print(f"[LoRA-SFT] Dataset size: {len(dataset):,} samples")

    train_cfg = LoRASFTConfig()
    train_cfg.sft_data_path = sft_data_path
    train_cfg.checkpoint_dir = output_dir
    train_cfg.num_epochs = num_epochs
    train_cfg.seq_len = seq_len
    train_cfg.lr = lr
    train_cfg.min_lr = lr * 0.1
    train_cfg.total_tokens = len(dataset) * seq_len * num_epochs

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    trainer = Trainer(
        model=model,
        config=train_cfg,
        device=device,
        is_ddp=is_ddp,
        global_rank=global_rank,
        local_rank=local_rank,
    )

    class _EpochRepeater:
        def __init__(self, dl, epochs: int):
            self.dl = dl
            self.epochs = epochs

        def __iter__(self):
            for epoch in range(self.epochs):
                print(f"\n[LoRA-SFT] Epoch {epoch + 1}/{self.epochs}")
                yield from self.dl

    trainer.train(_EpochRepeater(dataloader, num_epochs))
    
    # Save isolated LoRA weights
    lora_ckpt_path = Path(output_dir) / "phase3" / "lora_only.pt"
    lora_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": lora_state_dict(model)}, lora_ckpt_path)
    print(f"[LoRA-SFT] Saved isolated LoRA weights to {lora_ckpt_path}")

    # Merge LoRA back into base weights and save unified checkpoint
    print("[LoRA-SFT] Merging LoRA weights into base model...")
    merge_lora_linears(model)
    
    merged_ckpt_path = Path(output_dir) / "phase3" / "final.pt"
    
    # Save unified weights and the router biases
    save_data = {
        "model": model.state_dict(),
        "router_biases": {
            f"layer_{i}": layer.ffn.router.expert_bias.data.cpu() 
            for i, layer in enumerate(model.layers)
        }
    }
    torch.save(save_data, merged_ckpt_path)
    print(f"[LoRA-SFT] Saved MERGED weights to {merged_ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KT_GPT LoRA SFT")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint path")
    parser.add_argument("--data", default="data/lora_final.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    run_lora_sft(
        checkpoint_path=args.checkpoint,
        sft_data_path=args.data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        seq_len=args.seq_len,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()
