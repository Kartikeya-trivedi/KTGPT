"""
Modal Deployment — KT_GPT Training
====================================

Launch 2-phase pretraining on Modal A10G GPUs.

Usage:
    # Phase 1: 1B tokens (stability + fluency)
    modal run modal_train.py --phase 1

    # Phase 2: 20B tokens (structured intelligence), resumes from Phase 1
    modal run modal_train.py --phase 2

    # Resume a crashed run (auto-loads latest checkpoint)
    modal run modal_train.py --phase 1

Secrets required (set up via Modal dashboard or CLI):
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    modal secret create hf-secret HF_TOKEN=<your-token>

Volume:
    Checkpoints persist across runs in the 'kt-gpt-checkpoints' volume.
    The volume is mounted at /checkpoints inside the container.
"""

import modal

# ── Modal App & Infrastructure ───────────────────────────────────────

app = modal.App("kt-gpt")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "zstandard>=0.22.0",
        "wandb>=0.16.0",
        "accelerate>=0.25.0",
        gpu="A100",                 # ensures CUDA-enabled torch is installed
    )
    # Mount the project source code into the container
    .add_local_dir(
        ".",
        remote_path="/root/kt-gpt",
        ignore=[
            "**/__pycache__", "**/.venv", "**/checkpoints", "**/.git", 
            "**/node_modules", "**/venv", "**/.gemini", 
            "**/.ipynb_checkpoints", "**/*.jsonl", "**/*.pt",
            "**/weights", "**/ktgpt_chat"
        ],
    )
)

# Persistent volume for checkpoints (survives across runs)
checkpoint_volume = modal.Volume.from_name(
    "kt-gpt-checkpoints", create_if_missing=True
)

CHECKPOINT_MOUNT = "/checkpoints"


# ═══════════════════════════════════════════════════════════════════════
#  Phase 0: Dry-Run Sanity Check
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100-80GB",
    timeout=300,              # 5 minutes max — should finish in ~30s
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
    cpu=8.0,
)
def test_setup() -> str:
    """Quick sanity check: GPU, secrets, imports, model init, fwd/bwd, volume I/O.

    Run with:  modal run modal_train.py --phase 0
    """
    import os
    import sys
    import time

    t0 = time.time()
    results = []

    # ── 1. GPU Check ──────────────────────────────────────────────
    import torch

    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    results.append(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    results.append(f"✅ PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    results.append(f"✅ bf16 supported: {torch.cuda.is_bf16_supported()}")

    # ── 2. Secrets Check ──────────────────────────────────────────
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    results.append(f"✅ WANDB_API_KEY: {'set (' + wandb_key[:4] + '...)' if wandb_key else '❌ MISSING'}")
    results.append(f"✅ HF_TOKEN: {'set (' + hf_token[:4] + '...)' if hf_token else '❌ MISSING'}")

    # ── 3. Import Check ───────────────────────────────────────────
    sys.path.insert(0, "/root/kt-gpt")
    from model.config import KTGPTConfig
    from model.model import KTGPT
    from train.pretrain import Trainer, TrainConfig
    from data.mix import create_dataloader
    results.append("✅ All project imports successful")

    # ── 4. Model Init ─────────────────────────────────────────────
    device = torch.device("cuda")
    config = KTGPTConfig()
    model = KTGPT(config).to(device=device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    vram_used = torch.cuda.memory_allocated() / 1e9
    results.append(f"✅ Model initialized: {total_params / 1e6:.1f}M params, {vram_used:.2f} GB VRAM used")

    # ── 5. Forward + Backward Pass ────────────────────────────────
    dummy_input = torch.randint(0, config.vocab_size, (2, 128), device=device)
    logits, _ = model(dummy_input)
    loss = logits[:, :-1].contiguous().view(-1, logits.size(-1)).float().mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    results.append(f"✅ Forward pass: logits shape {tuple(logits.shape)}")
    results.append(f"✅ Backward pass: grad_norm={grad_norm:.4f}")
    results.append(f"✅ Peak VRAM: {peak_vram:.2f} GB / {vram_gb:.1f} GB ({peak_vram/vram_gb*100:.0f}%)")

    # ── 6. Data Pipeline Check ────────────────────────────────────
    from data.mix import get_tokenizer

    tokenizer = get_tokenizer()
    results.append(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
                   f"model expects={config.vocab_size}")
    if tokenizer.vocab_size != config.vocab_size:
        results.append(f"⚠️  VOCAB MISMATCH: tokenizer={tokenizer.vocab_size} vs config={config.vocab_size}")

    # Stream a few batches from Phase 1 data to verify HF access + packing
    dataloader = create_dataloader(
        phase=1,
        batch_size=2,
        seq_len=128,          # short seq for speed
        total_tokens=1024,    # just a handful of tokens
        seed=42,
        num_workers=0,        # no multiprocessing for quick test
    )
    batch_count = 0
    for batch in dataloader:
        ids = batch["input_ids"]
        labels = batch["labels"]
        assert ids.shape == (2, 128), f"Unexpected input_ids shape: {ids.shape}"
        assert labels.shape == (2, 128), f"Unexpected labels shape: {labels.shape}"
        assert ids.dtype == torch.long, f"Unexpected dtype: {ids.dtype}"
        assert ids.max() < config.vocab_size, f"Token ID {ids.max()} >= vocab_size {config.vocab_size}"
        batch_count += 1
        if batch_count >= 2:
            break
    results.append(f"✅ Data pipeline: streamed {batch_count} batches, "
                   f"shape={tuple(ids.shape)}, token range=[{ids.min()}, {ids.max()}]")

    # Decode first sample for visual sanity check
    sample_text = tokenizer.decode(ids[0, :200].tolist(), skip_special_tokens=False)
    results.append(f"\n📄 Sample batch (first 200 tokens decoded):\n{'─'*60}\n{sample_text}\n{'─'*60}")

    # ── 7. W&B Init Check ─────────────────────────────────────────
    try:
        import wandb
        run = wandb.init(
            project="kt-gpt",
            name="phase0-dry-run",
            tags=["dry-run"],
            config={
                "phase": 0,
                "total_params": total_params,
                "gpu": gpu_name,
            },
        )
        assert run is not None, "wandb.init returned None"
        wandb.finish()
        results.append(f"✅ wandb.init: authenticated & run created successfully")
    except Exception as e:
        results.append(f"❌ wandb.init FAILED: {e}")

    # ── 8. Volume I/O Check ───────────────────────────────────────
    test_file = os.path.join(CHECKPOINT_MOUNT, "_dry_run_test.txt")
    with open(test_file, "w") as f:
        f.write("dry-run OK")
    with open(test_file, "r") as f:
        assert f.read() == "dry-run OK"
    os.remove(test_file)
    results.append(f"✅ Volume I/O: write/read/delete at {CHECKPOINT_MOUNT}")

    # ── 9. Cleanup ────────────────────────────────────────────────
    del model, dummy_input, logits, loss
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results.append(f"\n🎉 All checks passed in {elapsed:.1f}s — ready for training!")

    report = "\n".join(results)
    print(report)
    return report


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Foundation Training (1B tokens)
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100-80GB:4",
    timeout=36000,           # 10 hours max
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,            # 32GB system RAM for data loading
    cpu=8.0,
)
def train_phase1() -> str:
    """Launch Phase 1 pretraining: 1B tokens, lr=3e-4, FineWeb-Edu + DCLM using 4-GPU DDP.

    Automatically resumes from latest checkpoint if a previous run was
    interrupted. Checkpoints are saved to the persistent volume every
    500 optimizer steps.

    Returns a status message on completion.
    """
    import subprocess
    import os
    print("Launching Phase 1 DDP on 4 GPUs...")
    subprocess.run(
        ["torchrun", "--nproc_per_node=4", "-m", "train.pretrain", "--phase", "1", "--checkpoint-dir", CHECKPOINT_MOUNT],
        check=True,
        cwd="/root/kt-gpt",
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )
    return "Phase 1 Complete"


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: Full Training (20B tokens)
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100-80GB:4",
    timeout=36000,           # 10 hours — may need multiple runs for 20B
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
    cpu=8.0,
)
def train_phase2() -> str:
    """Launch Phase 2 pretraining: 20B tokens, lr=1e-4, full data mix using 4-GPU DDP.

    Loads the Phase 1 checkpoint as initialization, then trains with the
    full data mix. Returns a status message on completion.
    """
    import subprocess
    import os
    print("Launching Phase 2 DDP on 4 GPUs...")
    subprocess.run(
        ["torchrun", "--nproc_per_node=4", "-m", "train.pretrain", "--phase", "2", "--checkpoint-dir", CHECKPOINT_MOUNT],
        check=True,
        cwd="/root/kt-gpt",
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )
    return "Phase 2 Complete"


@app.function(
    gpu="A100-80GB",
    timeout=36000,
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
    cpu=8.0,
)
def train_sft_stage(
    stage_name: str,
    data_file: str,
    input_ckpt: str,
    output_dir: str,
    epochs: int = 3,
    checkpoint_every: int = 100,
    seq_len: int = 512,
    lr: float = 5e-6,
) -> str:
    """Generic SFT stage runner for the curriculum pipeline."""
    import subprocess
    import os
    print(f"Launching SFT {stage_name} on Single GPU...")
    print(f"  Input:   {input_ckpt}")
    print(f"  Data:    {data_file}")
    print(f"  Output:  {output_dir}")
    print(f"  Epochs:  {epochs}, seq_len={seq_len}, lr={lr:.2e}")
    subprocess.run(
        ["python", "-m", "train.sft",
         "--checkpoint", input_ckpt,
         "--data", data_file,
         "--output-dir", output_dir,
         "--epochs", str(epochs),
         "--seq-len", str(seq_len),
         "--lr", str(lr)],
        check=True,
        cwd="/root/kt-gpt",
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )
    return f"SFT {stage_name} Complete"


@app.function(
    gpu="A100-80GB",
    timeout=36000,
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
    cpu=8.0,
)
def train_grpo() -> str:
    """Launch Stage 3 GRPO using a single powerful GPU."""
    import subprocess
    import os
    print("Launching Stage 3 GRPO on Single GPU...")
    subprocess.run(
        ["python", "-m", "train.grpo",
         "--checkpoint", f"{CHECKPOINT_MOUNT}/sft_stage2/phase3/final.pt",
         "--output_dir", f"{CHECKPOINT_MOUNT}/grpo"],
        check=True,
        cwd="/root/kt-gpt",
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )
    return "Stage 3 GRPO Complete"


@app.function(
    gpu="A100-80GB",
    timeout=36000,
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
    cpu=8.0,
)
def train_sft_lora(
    input_ckpt: str,
    data_file: str,
    output_dir: str,
    epochs: int = 1,
    seq_len: int = 512,
    lr: float = 1e-4,
) -> str:
    """Launch LoRA SFT Training."""
    import subprocess
    import os
    print("Launching LoRA SFT on Single GPU...")
    print(f"  Input:   {input_ckpt}")
    print(f"  Data:    {data_file}")
    print(f"  Output:  {output_dir}")
    subprocess.run(
        ["python", "-m", "train.sft_lora",
         "--checkpoint", input_ckpt,
         "--data", data_file,
         "--output-dir", output_dir,
         "--epochs", str(epochs),
         "--seq-len", str(seq_len),
         "--lr", str(lr)],
        check=True,
        cwd="/root/kt-gpt",
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    )
    return "LoRA SFT Complete"


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
)
def eval_lora(checkpoint_path: str):
    """Run the eval script on Modal."""
    import subprocess
    import os
    print(f"Running LoRA Evaluation...")
    subprocess.run(
        ["python", "scripts/eval_lora.py", 
         "--checkpoint", checkpoint_path],
        check=True,
        cwd="/root/kt-gpt",
    )



# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Stage Definitions
# ═══════════════════════════════════════════════════════════════════════

# fmt: off
PIPELINE_STAGES = {
    # code:  (stage_name,  data_file,                                                                  input_checkpoint,                                          output_dir,                          epochs, ckpt_every, seq_len, lr)
    "30a":  ("Stage 0A",  f"{CHECKPOINT_MOUNT}/data/pipeline/stage0a_basic_math.jsonl",              f"{CHECKPOINT_MOUNT}/phase2/final.pt",                    f"{CHECKPOINT_MOUNT}/sft_stage0a",   5,  100, 128, 2e-5),
    "30b":  ("Stage 0B",  f"{CHECKPOINT_MOUNT}/data/pipeline/stage0b_expanded_math.jsonl",           f"{CHECKPOINT_MOUNT}/sft_stage0a/phase3/final.pt",       f"{CHECKPOINT_MOUNT}/sft_stage0b",   5,   50, 128, 2e-5),
    "30c":  ("Stage 0C",  f"{CHECKPOINT_MOUNT}/data/pipeline/stage0c_multistep_math.jsonl",          f"{CHECKPOINT_MOUNT}/sft_stage0b/phase3/final.pt",       f"{CHECKPOINT_MOUNT}/sft_stage0c",   5,  200, 128, 2e-5),
    "31":   ("Stage 1",   f"{CHECKPOINT_MOUNT}/data/pipeline/stage1_instruct.jsonl",                 f"{CHECKPOINT_MOUNT}/sft_stage0c/phase3/final.pt",       f"{CHECKPOINT_MOUNT}/sft_stage1",    2,  200, 512, 5e-6),
    "315":  ("Stage 1.5", f"{CHECKPOINT_MOUNT}/data/pipeline/stage1_5_context_grounding.jsonl",      f"{CHECKPOINT_MOUNT}/sft_stage1/phase3/final.pt",        f"{CHECKPOINT_MOUNT}/sft_stage1_5",  2,  200, 512, 5e-6),
    "32":   ("Stage 2",   f"{CHECKPOINT_MOUNT}/data/pipeline/stage2_function_calling.jsonl",         f"{CHECKPOINT_MOUNT}/sft_stage1_5/phase3/final.pt",      f"{CHECKPOINT_MOUNT}/sft_stage2",    2,  200, 512, 5e-6),
    "lora": ("LoRA SFT",  f"{CHECKPOINT_MOUNT}/data/lora_final.jsonl",                               f"{CHECKPOINT_MOUNT}/phase2/final.pt",                   f"{CHECKPOINT_MOUNT}/sft_lora",      1,  None, 512, 1e-4),
    "eval": ("Eval LoRA", None, None, None, None, None, None, None),
    "base_eval": ("Eval Base", None, None, None, None, None, None, None),
    "4":    ("Stage 3 GRPO", None, None, None, None, None, None, None),  # handled separately
}
# fmt: on


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(phase: str = "1") -> None:
    """Launch training from the command line.

    Usage:
        modal run modal_train.py --phase 0    (dry run)
        modal run modal_train.py --phase 1    (Phase 1 pretrain)
        modal run modal_train.py --phase 2    (Phase 2 pretrain)
        modal run modal_train.py --phase 30a  (SFT Stage 0A: basic math)
        modal run modal_train.py --phase 30b  (SFT Stage 0B: expanded math)
        modal run modal_train.py --phase 30c  (SFT Stage 0C: multi-step)
        modal run modal_train.py --phase 31   (SFT Stage 1: instruction)
        modal run modal_train.py --phase 315  (SFT Stage 1.5: context/RAG)
        modal run modal_train.py --phase 32   (SFT Stage 2: function calling)
        modal run modal_train.py --phase lora (LoRA SFT on Phase 2 Base)
        modal run modal_train.py --phase eval (Run LoRA Evaluation)
        modal run modal_train.py --phase base_eval (Run Base Model Evaluation)
        modal run modal_train.py --phase 4    (GRPO: tool decision RL)
    """
    print(f"\nLaunching KT_GPT Phase {phase} training on Modal...")
    print(f"  Checkpoint volume: kt-gpt-checkpoints\n")

    if phase == "0":
        result = test_setup.remote()
    elif phase == "1":
        result = train_phase1.remote()
    elif phase == "2":
        result = train_phase2.remote()
    elif phase == "eval":
        lora = f"{CHECKPOINT_MOUNT}/sft_lora/phase3/final.pt"
        result = eval_lora.remote(lora)
    elif phase == "base_eval":
        base = f"{CHECKPOINT_MOUNT}/phase2/final.pt"
        result = eval_lora.remote(base)
    elif phase in PIPELINE_STAGES:
        if phase == "4":
            result = train_grpo.remote()
        else:
            stage_name, data_file, input_ckpt, output_dir, epochs, ckpt_every, seq_len, lr = PIPELINE_STAGES[phase]
            print(f"  Stage:   {stage_name}")
            print(f"  Data:    {data_file}")
            print(f"  Input:   {input_ckpt}")
            print(f"  Output:  {output_dir}")
            print(f"  Epochs:  {epochs}, seq_len={seq_len}, lr={lr:.2e}")
            if phase == "lora":
                result = train_sft_lora.remote(input_ckpt, data_file, output_dir, epochs, seq_len, lr)
            else:
                result = train_sft_stage.remote(stage_name, data_file, input_ckpt, output_dir, epochs, ckpt_every, seq_len, lr)
    else:
        valid = ["0", "1", "2"] + list(PIPELINE_STAGES.keys())
        raise ValueError(f"Invalid phase: {phase}. Must be one of: {valid}")

    print(f"\nResult: {result}")