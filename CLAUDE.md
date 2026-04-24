# CLAUDE.md — KT_GPT Project Context

> This file provides context for AI coding assistants across sessions.
> Last updated: 2024-04-24

---

## Project Overview

**KT_GPT** is a ~1B parameter decoder-only Mixture-of-Experts language model implemented from scratch in PyTorch. The goal is an ablation study on MoE at small scale for code generation, targeting a research paper.

- **Total parameters**: 992.68M (~1B)
- **Active parameters/token**: 141.12M
- **Sparsity**: 85.8%
- **Previously named**: MiniMoE / mini-moe (directory still named `mini-moe`)

---

## Architecture (Config C)

| Parameter | Value |
|---|---|
| hidden_dim | 704 |
| num_layers | 36 |
| num_heads | 11 (head_dim=64) |
| q_lora_rank | 352 (= hidden_dim / 2) |
| kv_lora_rank | 128 |
| qk_nope_dim / qk_rope_dim | 32 / 32 |
| v_head_dim | 64 |
| expert_ffn_dim | 320 |
| num_routed_experts | 37 |
| num_shared_experts | 1 |
| top_k | 2 |
| vocab_size | 32,000 |
| max_seq_len | 4,096 |

### Attention: MLA (Multi-head Latent Attention)
Implemented from DeepSeek-V2/V3. Compresses KV cache from 1408 values → 160 values per token per layer (~9x). The key innovation is caching a low-rank `kv_compressed` latent instead of full K/V per head.

### MoE: Bias-based load balancing (DeepSeek-V3 style)
- 1 shared expert (always active) + 37 routed experts (top-2 selected)
- Router: softmax scores + float32 bias for selection (bias doesn't affect weighting)
- Bias updated online: `bias[overloaded] -= 0.001`, `bias[underloaded] += 0.001`
- No auxiliary loss in training loss — cleaner gradients

### Activation: SwiGLU
Each expert: gate_proj + up_proj + down_proj (3 linear layers, no bias)

---

## Key Classes & Imports

```python
from model.config import KTGPTConfig      # All hyperparameters
from model.model import KTGPT             # Full model
from model.model import KTGPTBlock        # Single transformer layer
from model.model import MLAAttention      # MLA attention module
from model.model import MoEFFN            # MoE feed-forward layer
from model.model import MoERouter         # Top-k router with bias LB
from model.model import SwiGLUExpert      # Single SwiGLU expert
from model.model import RMSNorm           # RMS normalization
from model.model import RotaryEmbedding   # RoPE

from train.pretrain import Trainer, TrainConfig
from data.mix import create_dataloader, PackedDataset
from data.mix import PHASE_1_SOURCES, PHASE_2_SOURCES
```

---

## Training Pipeline

### Two-Phase Pretraining

| | Phase 1 | Phase 2 |
|---|---|---|
| **Tokens** | 1B | 20B |
| **LR** | 3e-4 | 1e-4 |
| **Steps** | 1,907 | 38,146 |
| **Tok/step** | 524,288 | 524,288 |
| **Data** | 60% FineWeb-Edu + 40% DCLM | 40% Stack v2 + 30% FineWeb + 20% DCLM + 10% Math |
| **Warmup** | 2,000 | 2,000 |

### Post-Training
- **SFT**: `train/sft.py` — fine-tune on synthetic JSONL data (lr=5e-5, 1-2 epochs)
- **GRPO**: `train/grpo.py` — RL with binary code execution rewards (lr=1e-6)

### Running on Modal
```bash
modal run modal_train.py --phase 1   # Phase 1
modal run modal_train.py --phase 2   # Phase 2 (auto-loads Phase 1 weights)
```

Secrets needed: `wandb-secret`, `hf-secret`
GPU: A10G (24GB), 10hr timeout, auto-resume from checkpoint

---

## Data Pipeline

- **Tokenizer**: `mistralai/Mistral-7B-v0.1` (32k vocab, open access, no HF token needed)
- **Packing**: Documents concatenated with BOS/EOS boundaries into fixed 4096-token chunks
- **Streaming**: Never loads full dataset into RAM
- **Code filtering**: `data/filter.py` — AST syntax check + sandboxed execution
- **Synthetic data**: `data/synth.py` — R1-style: generate candidates → execute tests → dedup → JSONL

---

## Important Design Decisions

1. **Weight tying**: `lm_head.weight = embed.weight` — counted once in param total
2. **Router bias in float32**: Even when model runs in bf16, router expert_bias stays float32 for numerical stability
3. **No auxiliary loss**: Bias-based LB instead (Model D ablation uses aux loss for comparison)
4. **Output projection scaling**: `1/sqrt(2 * num_layers)` on attn out_proj and expert down_proj for training stability
5. **Gradient accumulation**: 64 micro-steps × 2 batch × 4096 seq = 524K tokens/step
6. **Cosine LR**: With linear warmup (2000 steps), floor at 1e-5

---

## File Structure

```
mini-moe/                          (project = KT_GPT)
├── modal_train.py                 Modal A10G launcher
├── model/
│   ├── config.py                  KTGPTConfig (all hyperparameters)
│   └── model.py                   KTGPT + all submodules
├── data/
│   ├── mix.py                     Streaming data pipeline
│   ├── filter.py                  Code filtering
│   └── synth.py                   Synthetic data gen
├── train/
│   ├── pretrain.py                2-phase training loop
│   ├── sft.py                     Supervised fine-tuning
│   └── grpo.py                    GRPO RL
├── eval/
│   ├── run_eval.py                HumanEval, MBPP benchmarks
│   └── routing_analysis.py        Expert analysis (scaffold)
└── scripts/
    ├── smoke_test.py              End-to-end verification
    ├── count_params.py            Param count tool
    └── sweep_config.py            Config sweep tool
```

---

## Current Status (April 2024)

### ✅ Complete
- Full model architecture (MLA attention + MoE FFN + bias LB)
- Config verified: 992.68M total, 141.12M active, 85.8% sparsity
- Smoke test passing (forward/backward/gradients/router bias)
- Data pipeline (streaming tokenization + packing + phase configs)
- Training loop (2-phase pretraining with checkpointing + W&B)
- Modal deployment (A10G, persistent volume, auto-resume)
- SFT, GRPO, eval, filter, synth modules
- Project renamed MiniMoE → KT_GPT

### 🔜 Next Steps
1. Set up Modal secrets and run Phase 1 training
2. Monitor loss/entropy/gradients on W&B
3. Run eval after Phase 1
4. Launch Phase 2 (auto-loads Phase 1 weights)

### ⏳ Future
- Routing analysis (expert specialization heatmaps)
- Ablation models (A, B, D)
- Paper writing

---

## Smoke Test

Run to verify everything works:
```bash
.venv\Scripts\python -m scripts.smoke_test
```

Expected output: All checks passed (forward pass, backward pass, gradients, router bias updates, weight tying, param count).

---

## Common Gotchas

- The directory is still named `mini-moe` but all code uses `KT_GPT`/`KTGPT`/`kt-gpt`
- Router bias is `requires_grad=False` — it's updated manually in `MoERouter.update_bias()`, not by the optimizer
- With 37 routed experts and small batches, some experts get zero tokens — this is normal MoE behavior
- The `configs/*.yaml` files use old names and are for reference only — the Python dataclass `KTGPTConfig` is the source of truth
- `model/transformer.py` and `model/attention.py` are legacy scaffolds — all code lives in `model/model.py`
