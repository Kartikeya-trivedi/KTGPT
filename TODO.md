# KT_GPT — Master TODO

> **Project renamed from Mini-MoE to KT_GPT (April 2024)**

---

## Phase 0: Project Setup & Infrastructure

### Environment
- [x] Python 3.13 with `uv` package manager
- [x] `pyproject.toml` for dependency management
- [x] Virtual environment (`.venv/`)
- [ ] Create `.gitignore` (checkpoints/, wandb/, data/raw/, __pycache__/, *.pt)
- [ ] Init git repo, push to GitHub

### Modal Setup
- [x] `modal_train.py` — A10G GPU, 10hr timeout, persistent checkpoint volume
- [x] Secrets: `wandb-secret` (WANDB_API_KEY), `hf-secret` (HF_TOKEN)
- [x] Phase 1 + Phase 2 entry points with auto-resume
- [ ] Test with a dummy run on Modal (allocate GPU, verify volume mounts)

### Weights & Biases
- [x] W&B project name: `kt-gpt`
- [ ] Set up dashboard with key metric panels (loss, entropy, grad_norm, lr)

---

## Phase 1: Model Architecture (`model/`) ✅ COMPLETE

### `model/config.py` — KTGPTConfig ✅
- [x] Dataclass with all hyperparameters
- [x] Config C: `h=704, L=36, ffn=320, 37 routed + 1 shared, top-2`
- [x] Parameter count helpers: total_params(), active_params(), verify_param_count()
- [x] **Verified: 992.68M total, 141.12M active, 85.8% sparsity**

### `model/model.py` — Full Architecture ✅
- [x] `RMSNorm` — Root Mean Square normalization (no bias)
- [x] `RotaryEmbedding` + `apply_rotary_emb` — RoPE for MLA
- [x] `MLAAttention` — Multi-head Latent Attention (DeepSeek-V2/V3 style)
  - [x] Low-rank query compression (q_lora_rank=352)
  - [x] KV cache compression (kv_lora_rank=128 → ~9x reduction)
  - [x] Separate nope/rope head dimensions
  - [x] F.scaled_dot_product_attention (SDPA) for GPU-optimized attention
- [x] `SwiGLUExpert` — SwiGLU FFN (gate + up + down projections)
- [x] `MoERouter` — Bias-based load balancing (DeepSeek-V3 style)
  - [x] Top-2 routing with softmax scores
  - [x] Bias added for selection only, not weighting
  - [x] Online bias updates (float32, requires_grad=False)
- [x] `MoEFFN` — 1 shared expert + 37 routed experts
- [x] `KTGPTBlock` — MLA attention + MoE FFN with pre-norm residuals
- [x] `KTGPT` — Full model with weight-tied embeddings
  - [x] Output projection scaling by 1/sqrt(2*num_layers)
  - [x] N(0, 0.02) weight initialization

### Verification ✅
- [x] Smoke test: forward/backward pass, gradient flow, output shape
- [x] Weight tying verified (embed == lm_head)
- [x] Router bias update verified during training mode
- [x] 95% routed experts receive gradients (5% idle expected with small batch)

---

## Phase 2: Data Pipeline (`data/`) ✅ COMPLETE

### `data/mix.py` — Dataset Mixing & Loading ✅
- [x] `DataSourceConfig` dataclass for per-source settings
- [x] Phase 1 mix: 60% FineWeb-Edu (score ≥ 4) + 40% DCLM-Baseline
- [x] Phase 2 mix: 40% Stack v2 + 30% FineWeb-Edu + 20% DCLM + 10% OpenWebMath
- [x] `PackedDataset(IterableDataset)` — streaming tokenization + document packing
  - [x] BOS/EOS boundaries between documents
  - [x] Fixed-length seq_len=4096 chunks
  - [x] No RAM accumulation (streaming)
- [x] `create_dataloader()` factory with phase selection
- [x] Tokenizer: `mistralai/Mistral-7B-v0.1` (32k vocab, open access)

### `data/filter.py` — Execution-Based Filtering ✅
- [x] `is_valid_python()` — AST syntax check
- [x] `execute_with_tests()` — sandboxed subprocess with timeout
- [x] `filter_code_samples()` — batch filtering with stats tracking
- [x] `filter_dataset_streaming()` — streaming filter for HF datasets
- [x] Safety: isolated subprocess, no network, memory-limited

### `data/synth.py` — Synthetic Data Generation ✅
- [x] `SynthConfig` — generation parameters (G=8, temp=0.8, timeout)
- [x] `SyntheticDataGenerator` — R1-style generation loop
  - [x] Multi-candidate generation with temperature sampling
  - [x] Execution-based validation against test cases
  - [x] Hash-based deduplication
  - [x] JSONL output format: `[PROBLEM] → [REASONING] → [CODE] → [TESTS]`

---

## Phase 3: Training Pipeline (`train/`) ✅ COMPLETE

### `train/pretrain.py` — Pretraining Loop ✅
- [x] `TrainConfig` dataclass with phase presets
  - [x] Phase 1: 1B tokens, lr=3e-4, 1,907 steps, 524K tok/step
  - [x] Phase 2: 20B tokens, lr=1e-4, 38,146 steps, 524K tok/step
- [x] `cosine_lr_schedule()` — cosine annealing with linear warmup
- [x] `compute_router_entropy()` — per-layer routing entropy
- [x] `compute_expert_load_cv()` — coefficient of variation for load balance
- [x] `Trainer` class:
  - [x] AdamW optimizer (β1=0.9, β2=0.95, fused on GPU)
  - [x] Weight decay groups (exclude norms, biases, embeddings)
  - [x] bf16 mixed precision with GradScaler
  - [x] Gradient accumulation (64 micro-steps)
  - [x] Gradient clipping (max_norm=1.0)
  - [x] W&B logging: loss, grad_norm, lr, tokens/sec, router entropy/CV
  - [x] Checkpoint save/load (model + optimizer + scaler + step + router biases)
  - [x] Auto-resume from latest checkpoint
  - [x] Phase 2 loads Phase 1 model weights (fresh optimizer)

### `train/sft.py` — Supervised Fine-Tuning ✅
- [x] `SFTConfig` — inherits TrainConfig (lr=5e-5, warmup=200, wd=0.01)
- [x] `SFTDataset` — loads JSONL from synth.py, tokenizes + pads/truncates
- [x] `run_sft()` — loads pretrained checkpoint, runs 1-2 epochs

### `train/grpo.py` — GRPO Reinforcement Learning ✅
- [x] `GRPOConfig` — lr=1e-6, G=4 candidates, KL coef=0.05
- [x] `GRPOTrainer`:
  - [x] Frozen reference model (SFT copy) for KL penalty
  - [x] Autoregressive generation with temperature sampling
  - [x] Binary rewards via test execution
  - [x] Group-relative advantages (reward - mean)
  - [x] Policy gradient on positive-advantage samples only
  - [x] KL divergence penalty against reference
  - [x] W&B logging + periodic checkpointing

---

## Phase 4: Evaluation (`eval/`) ✅ COMPLETE

### `eval/run_eval.py` — Evaluation Wrapper ✅
- [x] `EvalConfig` — benchmark selection, generation parameters
- [x] `ModelEvaluator`:
  - [x] Autoregressive `generate()` method (greedy or sampled)
  - [x] HumanEval evaluation (pass@1, pass@10)
  - [x] MBPP evaluation (pass@1)
  - [x] lm-eval-harness placeholder for ARC, HellaSwag
  - [x] Results saved to JSON + logged to W&B

### `eval/routing_analysis.py` — Routing Analysis
- [ ] Expert specialization heatmaps (token type × expert × layer)
- [ ] Per-layer entropy on eval set
- [ ] Compare routing on code vs NL vs math tokens
- [ ] Token classification: keyword, identifier, operator, whitespace, NL
- [ ] Export data for matplotlib/seaborn figure generation

---

## Phase 5: Training Execution 🔜 NEXT

### Phase 1: Foundation Run (1B tokens)
- [ ] Set up Modal secrets (`wandb-secret`, `hf-secret`)
- [ ] Create W&B `kt-gpt` project
- [ ] `modal run modal_train.py --phase 1`
- [ ] Monitor: loss curve, router entropy, grad norm
- [ ] Verify loss drops from ~10.4 → <4 within 500 steps
- [ ] Verify router entropy stays >2.0 per layer
- [ ] Run eval at final checkpoint

### Phase 2: Full Run (20B tokens)
- [ ] `modal run modal_train.py --phase 2`
- [ ] May need multiple 10hr runs (auto-resumes from checkpoint)
- [ ] Run eval at regular checkpoints
- [ ] Analyze routing patterns

### Post-Training Pipeline
- [ ] Generate synthetic data with `data/synth.py`
- [ ] Run SFT with `train/sft.py`
- [ ] Run GRPO with `train/grpo.py`
- [ ] Full evaluation suite

---

## Phase 6: Ablation Experiments (for Paper)

### Training All 4 Models
- [ ] **Model A — Dense Baseline (141M)**: `use_moe=False`, adjust FFN dim
- [ ] **Model B — MoE without Shared Expert**: 38 routed experts, no shared
- [ ] **Model C — MoE + Shared Expert (main model)**: 1 shared + 37 routed, bias LB ← **this is KT_GPT**
- [ ] **Model D — MoE with Aux Loss**: same as C but `load_balance_type="aux_loss"`

### Ablation Analysis
- [ ] Training loss curves (all 4 models)
- [ ] Loss per FLOP (sample efficiency)
- [ ] Final benchmark scores table
- [ ] Routing entropy comparison
- [ ] Expert load CV comparison (bias vs aux loss)
- [ ] Expert specialization heatmaps
- [ ] Inference speed: dense vs MoE

---

## Phase 7: Paper Writing

### Figures & Tables
- [ ] Architecture diagram
- [ ] Training loss curves (4 models)
- [ ] Loss per FLOP
- [ ] Router entropy per layer
- [ ] Expert specialization heatmap
- [ ] Expert load distribution
- [ ] Benchmark scores bar chart
- [ ] Architecture configs table
- [ ] Main results table

### Paper Sections
- [ ] Abstract, Introduction, Related Work, Architecture
- [ ] Experiments (setup, results, routing analysis, LB comparison, efficiency)
- [ ] Analysis & Discussion
- [ ] Conclusion, Appendix

---

## Key Hyperparameters Reference

| Hyperparameter       | Phase 1          | Phase 2          | SFT            | GRPO           |
| -------------------- | ---------------- | ---------------- | -------------- | -------------- |
| Learning rate        | 3e-4             | 1e-4             | 5e-5           | 1e-6           |
| LR schedule          | Cosine + warmup  | Cosine + warmup  | Cosine         | Constant       |
| Warmup steps         | 2000             | 2000             | 200            | 0              |
| Batch size (tokens)  | ~524K            | ~524K            | ~128K          | ~64K           |
| Weight decay         | 0.1              | 0.1              | 0.01           | 0.01           |
| Grad clip            | 1.0              | 1.0              | 1.0            | 1.0            |
| Precision            | bf16             | bf16             | bf16           | bf16           |
| Optimizer            | AdamW            | AdamW            | AdamW          | AdamW          |
| β1, β2              | 0.9, 0.95        | 0.9, 0.95        | 0.9, 0.95      | 0.9, 0.95      |
| Gamma (LB bias)      | 0.001            | 0.001            | 0.001          | 0.001          |

---

## Danger Zones (Things That Can Kill Your Run)

- [ ] **Routing collapse**: All tokens go to 1-2 experts. Monitor entropy. If < 0.5, stop and debug.
- [ ] **Loss spikes**: Gradient norm explodes. Always clip gradients. Log grad_norm.
- [ ] **NaN/Inf**: Usually from fp16 overflow. Use bf16 if A10G supports it.
- [ ] **OOM on A10G**: 24GB VRAM. Use gradient accumulation + mixed precision.
- [ ] **Data loading bottleneck**: Streaming from HuggingFace can be slow. Cache locally on Modal volume.
- [ ] **Checkpoint corruption**: Always verify checkpoint loads correctly after saving.
- [ ] **Unfair ablation**: All models MUST train on identical data, identical tokens. Use same seed.

---

## File Structure

```
mini-moe/                         (dir name, project = KT_GPT)
├── modal_train.py                Modal A10G deployment (Phase 1 + 2)
├── pyproject.toml                Dependencies
├── TODO.md                       ← you are here
├── CLAUDE.md                     Context doc for future AI sessions
│
├── model/
│   ├── config.py                 KTGPTConfig dataclass
│   └── model.py                  KTGPT, KTGPTBlock, MLAAttention, MoEFFN, etc.
│
├── data/
│   ├── mix.py                    Streaming data pipeline + phase configs
│   ├── filter.py                 Execution-based code filtering
│   └── synth.py                  R1-style synthetic data generation
│
├── train/
│   ├── pretrain.py               2-phase pretraining loop + Trainer class
│   ├── sft.py                    Supervised fine-tuning on synth data
│   └── grpo.py                   GRPO reinforcement learning
│
├── eval/
│   ├── run_eval.py               HumanEval, MBPP, lm-eval benchmarks
│   └── routing_analysis.py       Expert specialization analysis (scaffold)
│
├── scripts/
│   ├── smoke_test.py             End-to-end model verification
│   ├── count_params.py           Parameter count verification
│   └── sweep_config.py           Architecture config sweep tool
│
└── configs/                      YAML configs for ablation models
```
