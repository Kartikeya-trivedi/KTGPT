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
        gpu="A10G",                 # ensures CUDA-enabled torch is installed
    )
    # Mount the project source code into the container
    .add_local_dir(
        ".",
        remote_path="/root/kt-gpt",
        ignore=["**/__pycache__", "**/.venv", "**/checkpoints", "**/.git"],
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
    gpu="A10G",
    timeout=300,              # 5 minutes max — should finish in ~30s
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
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

    # ── 7. Volume I/O Check ───────────────────────────────────────
    test_file = os.path.join(CHECKPOINT_MOUNT, "_dry_run_test.txt")
    with open(test_file, "w") as f:
        f.write("dry-run OK")
    with open(test_file, "r") as f:
        assert f.read() == "dry-run OK"
    os.remove(test_file)
    results.append(f"✅ Volume I/O: write/read/delete at {CHECKPOINT_MOUNT}")

    # ── 8. Cleanup ────────────────────────────────────────────────
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
    gpu="A10G",
    timeout=36000,           # 10 hours max
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,            # 32GB system RAM for data loading
)
def train_phase1() -> str:
    """Launch Phase 1 pretraining: 1B tokens, lr=3e-4, FineWeb-Edu + DCLM.

    Automatically resumes from latest checkpoint if a previous run was
    interrupted. Checkpoints are saved to the persistent volume every
    500 optimizer steps.

    Returns a status message on completion.
    """
    import sys
    sys.path.insert(0, "/root/kt-gpt")

    import torch
    from model.config import KTGPTConfig
    from model.model import KTGPT
    from train.pretrain import Trainer, TrainConfig
    from data.mix import create_dataloader

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Model ──────────────────────────────────────────────────────
    model_config = KTGPTConfig()
    model = KTGPT(model_config).to(device=device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")

    # ── Training config ────────────────────────────────────────────
    train_config = TrainConfig.phase1()
    train_config.checkpoint_dir = CHECKPOINT_MOUNT

    # ── Data ───────────────────────────────────────────────────────
    dataloader = create_dataloader(
        phase=1,
        batch_size=train_config.micro_batch_size,
        seq_len=train_config.seq_len,
        total_tokens=train_config.total_tokens,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )

    # ── Trainer ────────────────────────────────────────────────────
    trainer = Trainer(model=model, config=train_config, device=device)
    trainer.load_checkpoint()  # auto-resume if checkpoint exists

    trainer.train(dataloader)

    # Commit volume to persist checkpoints
    checkpoint_volume.commit()

    return f"Phase 1 complete. Steps: {trainer.global_step}, Tokens: {trainer.tokens_seen:,}"


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: Full Training (20B tokens)
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A10G",
    timeout=36000,           # 10 hours — may need multiple runs for 20B
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    memory=32768,
)
def train_phase2() -> str:
    """Launch Phase 2 pretraining: 20B tokens, lr=1e-4, full data mix.

    Loads the Phase 1 checkpoint as initialization, then trains with the
    full data mix (Stack v2 + FineWeb + DCLM + OpenWebMath).

    If a Phase 2 checkpoint exists, resumes from it instead.
    This allows multi-run training across Modal timeout boundaries.

    Returns a status message on completion.
    """
    import sys
    sys.path.insert(0, "/root/kt-gpt")

    import torch
    from model.config import KTGPTConfig
    from model.model import KTGPT
    from train.pretrain import Trainer, TrainConfig
    from data.mix import create_dataloader

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Model ──────────────────────────────────────────────────────
    model_config = KTGPTConfig()
    model = KTGPT(model_config).to(device=device, dtype=torch.bfloat16)

    # ── Training config ────────────────────────────────────────────
    train_config = TrainConfig.phase2()
    train_config.checkpoint_dir = CHECKPOINT_MOUNT

    # ── Data ───────────────────────────────────────────────────────
    dataloader = create_dataloader(
        phase=2,
        batch_size=train_config.micro_batch_size,
        seq_len=train_config.seq_len,
        total_tokens=train_config.total_tokens,
        seed=train_config.seed,
        num_workers=train_config.num_workers,
    )

    # ── Trainer ────────────────────────────────────────────────────
    trainer = Trainer(model=model, config=train_config, device=device)

    # Try Phase 2 resume first; if none, load Phase 1 final weights
    if not trainer.load_checkpoint():
        if not trainer.load_phase1_checkpoint():
            print("WARNING: No Phase 1 checkpoint found. Training from scratch.")

    trainer.train(dataloader)
    checkpoint_volume.commit()

    return f"Phase 2 complete. Steps: {trainer.global_step}, Tokens: {trainer.tokens_seen:,}"


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(phase: int = 1) -> None:
    """Launch training from the command line.

    Usage:
        modal run modal_train.py --phase 1
        modal run modal_train.py --phase 2
    """
    print(f"\nLaunching KT_GPT Phase {phase} training on Modal...")
    print(f"  GPU: A10G (24GB)")
    print(f"  Timeout: 10 hours")
    print(f"  Checkpoint volume: kt-gpt-checkpoints\n")

    if phase == 0:
        result = test_setup.remote()
    elif phase == 1:
        result = train_phase1.remote()
    elif phase == 2:
        result = train_phase2.remote()
    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 0, 1, or 2.")

    print(f"\nResult: {result}")