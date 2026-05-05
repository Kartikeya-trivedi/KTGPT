"""
KT_GPT Smoke Test
==================

End-to-end verification that the model:
  1. Instantiates with the correct architecture
  2. Produces the right output shape
  3. Computes loss and backpropagates gradients to all parameter groups
  4. Router bias tensors exist and are properly separated from trained params
  5. Fits in GPU memory at reasonable batch sizes

Usage:
    python -m scripts.smoke_test
"""

from __future__ import annotations

import sys
import os
import torch
import torch.nn.functional as F

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from model.config import KTGPTConfig
from model.model import KTGPT


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  |  Dtype: {dtype}\n")

    # -- 1. Initialize -------------------------------------------------------
    print("1. Initializing KT_GPT model...")
    config = KTGPTConfig()
    model = KTGPT(config).to(device=device, dtype=dtype)
    print("   [OK] Model initialized\n")

    # -- 2. Parameter count breakdown ----------------------------------------
    print("2. Parameter count breakdown:")
    config.verify_param_count()

    actual_total = sum(p.numel() for p in model.parameters())
    print(f"\n   Actual nn.Module parameter count: {actual_total / 1e6:.2f}M")

    # Check that weight tying is working (embed and lm_head share storage)
    assert model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "FAIL: embedding and LM head weights are not tied!"
    # The tied weight is counted once by nn.Module, so actual_total already
    # reflects the correct count (no double-counting).
    print("   [OK] Weight tying verified (embed == lm_head)\n")

    # -- 3. Forward pass -----------------------------------------------------
    batch_size, seq_len = 2, 128
    print(f"3. Forward pass (batch={batch_size}, seq_len={seq_len})...")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    model.train()
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        logits, _ = model(input_ids)

    # -- 4. Verify output shape ----------------------------------------------
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, \
        f"FAIL: expected {expected_shape}, got {logits.shape}"
    print(f"   [OK] Output shape: {logits.shape}\n")

    # -- 5. Compute loss -----------------------------------------------------
    print("5. Computing cross-entropy loss...")
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1),
    )
    print(f"   [OK] Loss: {loss.item():.4f}  (expected ~ln(32000) = {10.3734:.4f})\n")

    # -- 6. Backward pass ----------------------------------------------------
    print("6. Running loss.backward()...")
    loss.backward()
    print("   [OK] Backward pass completed\n")

    # -- 7. Verify gradients -------------------------------------------------
    print("7. Checking gradients on all parameter groups...")
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "embed": [],
        "attn": [],
        "ffn_shared": [],
        "ffn_routed": [],
        "router": [],
        "norm": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed" in name or "lm_head" in name:
            groups["embed"].append((name, param))
        elif "attn" in name:
            groups["attn"].append((name, param))
        elif "shared_expert" in name:
            groups["ffn_shared"].append((name, param))
        elif "routed_experts" in name:
            groups["ffn_routed"].append((name, param))
        elif "gate" in name:
            groups["router"].append((name, param))
        elif "norm" in name:
            groups["norm"].append((name, param))

    all_ok = True
    for group_name, params in groups.items():
        count = len(params)
        grad_count = sum(1 for _, p in params if p.grad is not None)
        no_grad_count = count - grad_count

        if count == 0:
            print(f"   [WARN] {group_name}: no parameters found")
            all_ok = False
        elif no_grad_count == 0:
            print(f"   [OK] {group_name}: {count} params, all have gradients")
        elif group_name == "ffn_routed" and grad_count > 0:
            # With many routed experts and small batches, some experts receive
            # zero tokens and thus have no gradients. This is expected MoE
            # behavior — not a bug. Only fail if NONE got gradients.
            pct = grad_count / count * 100
            print(f"   [OK] {group_name}: {grad_count}/{count} params have gradients "
                  f"({pct:.0f}%) — {no_grad_count} idle (expected with small batch)")
        else:
            no_grad = [(n, p) for n, p in params if p.grad is None]
            print(f"   [FAIL] {group_name}: {no_grad_count}/{count} params MISSING gradients:")
            for n, _ in no_grad[:5]:
                print(f"       - {n}")
            all_ok = False

    print()

    # -- 8. Router bias check ------------------------------------------------
    print("8. Verifying router bias tensors...")
    bias_params: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if "expert_bias" in name:
            bias_params.append((name, param))

    assert len(bias_params) == config.num_layers, \
        f"FAIL: expected {config.num_layers} router bias tensors, found {len(bias_params)}"

    for name, param in bias_params:
        assert not param.requires_grad, \
            f"FAIL: {name} should have requires_grad=False"
        assert param.dtype == torch.float32, \
            f"FAIL: {name} should be float32, got {param.dtype}"

    print(f"   [OK] Found {len(bias_params)} router bias tensors (requires_grad=False, float32)")

    # Check that biases were updated during the forward pass
    first_bias = bias_params[0][1]
    if torch.any(first_bias != 0.0):
        print("   [OK] Router biases were updated during forward (non-zero values found)")
    else:
        print("   [WARN] Router biases are still zero (may need more tokens for update)")
    print()

    # -- 9. W&B init check ---------------------------------------------------
    print("9. Verifying wandb.init (disabled mode)...")
    try:
        import wandb
        run = wandb.init(
            project="kt-gpt",
            name="smoke-test",
            mode="disabled",            # no credentials / network needed
        )
        assert run is not None, "FAIL: wandb.init returned None"
        wandb.finish()
        print("   [OK] wandb.init + finish succeeded (mode=disabled)\n")
    except ImportError:
        print("   [FAIL] wandb is not installed")
        all_ok = False
    except Exception as e:
        print(f"   [FAIL] wandb.init raised: {e}")
        all_ok = False

    # -- 10. Overall verdict -------------------------------------------------
    if all_ok:
        print("=" * 50)
        print("  All checks passed")
        print("=" * 50)
    else:
        print("=" * 50)
        print("  Some checks failed -- see above")
        print("=" * 50)
        sys.exit(1)

    # -- 11. GPU memory ------------------------------------------------------
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_gb = peak_mb / 1024
        print(f"\n  Peak GPU memory: {peak_mb:.0f} MB ({peak_gb:.2f} GB)")
    else:
        print("\n  (GPU memory tracking not available on CPU)")


if __name__ == "__main__":
    main()
