"""
Pre-tokenization Pipeline
=========================

Streams data from HuggingFace, tokenizes it, and saves it to a persistent
binary file (.bin) on the Modal volume using uint16.
This completely eliminates network/CPU bottlenecks during training.

Usage:
    modal run scripts/pretokenize.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import modal

app = modal.App("kt-gpt-pretokenize")

# Define project root to ensure we mount the entire project, not just the scripts folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/kt-gpt",
        ignore=["**/__pycache__", "**/.venv", "**/checkpoints", "**/.git"],
    )
)

checkpoint_volume = modal.Volume.from_name(
    "kt-gpt-checkpoints", create_if_missing=True
)

CHECKPOINT_MOUNT = "/checkpoints"

@app.function(
    image=image,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    timeout=86000,           # 24 hours max to allow 20B Phase 2 tokens to finish
    memory=32768,            # 32GB RAM for safety
    cpu=8.0,                 # Need fast CPU for tokenizing
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def run_pretokenize(phase: int = 1, total_tokens: int = 1_000_000_000, restart: bool = False) -> str:
    # Ensure sys.path is set inside the container
    sys.path.insert(0, "/root/kt-gpt")
    
    from data.mix import PHASE_1_SOURCES, PHASE_2_SOURCES, get_tokenizer, PackedDataset
    
    # Target directory
    data_dir = Path(CHECKPOINT_MOUNT) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    out_filename = f"phase{phase}_{total_tokens // 1_000_000}M.bin"
    out_path = data_dir / out_filename
    
    print(f"Starting pre-tokenization for Phase {phase}")
    print(f"Target tokens: {total_tokens:,}")
    print(f"Output file: {out_path}")
    
    if out_path.exists():
        print(f"WARNING: File {out_path} already exists. Overwriting.")
    
    sources = PHASE_1_SOURCES if phase == 1 else PHASE_2_SOURCES
    tokenizer = get_tokenizer()
    
    # Use a smaller block size so the tqdm progress bar updates much faster 
    # and doesn't appear "stuck" while waiting to accumulate large chunks
    block_size = 4096 
    
    start_time = time.time()
    
    # --- Checkpoint / Resume logic ---
    existing_tokens = 0
    file_mode = "wb"
    
    if restart and os.path.exists(out_path):
        os.remove(out_path)
        print("🗑️ Deleted existing file for a fresh start.")
        
    if os.path.exists(out_path):
        # We write uint16 (2 bytes per token), so tokens = file size // 2
        file_size = os.path.getsize(out_path)
        existing_tokens = file_size // 2
        
        if 0 < existing_tokens < total_tokens:
            print(f"\n[RESUME] Found existing file with {existing_tokens:,} tokens ({file_size / (1024**3):.2f} GB).")
            print("[RESUME] Natively fast-forwarding Hugging Face datasets (blazing fast!)...")
            file_mode = "ab"
        elif existing_tokens >= total_tokens:
            print(f"\n✅ File already completed ({existing_tokens:,} tokens). Exiting early.")
            return str(out_path)
            
    # We write tokens as uint16 (2 bytes each). Mistral vocab is 32k, so it fits perfectly.
    dtype = np.uint16
    
    dataset = PackedDataset(
        sources=sources,
        tokenizer=tokenizer,
        seq_len=block_size,
        total_tokens=total_tokens,
        seed=42,
        skip_tokens=existing_tokens,
    )
            
    tokens_written = existing_tokens
    
    with open(out_path, file_mode) as f:
        # Use a high-level progress bar
        pbar = tqdm(total=total_tokens, initial=existing_tokens, desc=f"Writing {out_filename}", unit="tok", unit_scale=True)
        
        for batch in dataset:
            # batch["input_ids"] is a tensor of shape (block_size,)
            token_ids = batch["input_ids"].numpy().astype(dtype)
            
            # Write bytes to disk
            f.write(token_ids.tobytes())
            
            tokens_written += len(token_ids)
            pbar.update(len(token_ids))
            
            # If we hit the limit early
            if tokens_written >= total_tokens:
                break
                
        pbar.close()
        
    # Flush and commit the volume
    checkpoint_volume.commit()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Pre-tokenization complete!")
    print(f"Total tokens written: {tokens_written:,}")
    print(f"File size: {os.path.getsize(out_path) / (1024**3):.2f} GB")
    print(f"Time taken: {elapsed:.2f}s ({tokens_written / elapsed:,.0f} tok/s)")
    
    return str(out_path)

@app.local_entrypoint()
def main(phase: int = 1, restart: bool = False):
    # Phase 1 is 1 Billion tokens
    # Phase 2 is 5 Billion tokens
    target_tokens = 1_000_000_000 if phase == 1 else 5_000_000_000
    
    print(f"Launching pre-tokenization for Phase {phase} on Modal...")
    if restart:
        print("⚠️ Restart flag set: Will overwrite any existing progress.")
        
    result = run_pretokenize.remote(phase=phase, total_tokens=target_tokens, restart=restart)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
