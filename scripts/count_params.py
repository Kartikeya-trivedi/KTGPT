"""
Parameter count verification script.

Computes total and active parameter counts for all 4 ablation models
using the config-level estimator (no model instantiation needed).

Also prints a design-space exploration table to help choose hyperparams
that hit the ~1B total / ~130M active target.

Usage:
    python -m scripts.count_params
"""

from __future__ import annotations

import sys
from dataclasses import replace
from model.config import ModelConfig


def fmt(n: int) -> str:
    """Format large numbers with M/B suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    return f"{n / 1e6:.1f}M"


def print_model_table() -> None:
    """Print a summary table for all 4 ablation model configs."""
    models = [
        ("A  Dense Baseline", ModelConfig.model_a_dense()),
        ("B  MoE (no shared)", ModelConfig.model_b_moe_no_shared()),
        ("C  MoE + Shared", ModelConfig.model_c_moe_shared()),
        ("D  MoE + Aux Loss", ModelConfig.model_d_moe_aux_loss()),
    ]

    header = f"{'Model':<22} {'Experts':>8} {'Top-k':>5} {'FFN':>5} {'Total':>10} {'Active':>10} {'Sparse%':>8}"
    print(header)
    print("-" * len(header))

    for name, cfg in models:
        if cfg.use_moe:
            experts_str = f"{cfg.n_shared_experts}s+{cfg.n_routed_experts}r"
        else:
            experts_str = "dense"
        total = cfg.total_params()
        active = cfg.active_params()
        sparsity = f"{1 - active / total:.0%}" if cfg.use_moe else "0%"
        print(f"{name:<22} {experts_str:>8} {cfg.top_k:>5} {cfg.ffn_dim:>5} "
              f"{fmt(total):>10} {fmt(active):>10} {sparsity:>8}")


def explore_design_space() -> None:
    """Sweep ffn_dim to find configs hitting the 1B/130M target."""
    print("\n\n=== Design Space Exploration ===")
    print("Fixed: hidden=768, layers=20, 12Q/4KV, vocab=32k, SwiGLU")
    print("Varying: n_experts (total), n_shared, ffn_dim\n")

    header = (f"{'Config':<28} {'FFN':>5} {'#Exp':>5} {'Shared':>6} "
              f"{'Total':>10} {'Active':>10} {'Ratio':>6}")
    print(header)
    print("-" * len(header))

    targets = []

    for n_total_experts in [8, 12, 16]:
        for n_shared in [0, 1]:
            n_routed = n_total_experts - n_shared
            for ffn_dim in [512, 768, 1024, 1280, 1536, 2048]:
                cfg = ModelConfig(
                    hidden_dim=768,
                    n_layers=20,
                    n_heads=12,
                    n_kv_heads=4,
                    ffn_dim=ffn_dim,
                    vocab_size=32_000,
                    max_seq_len=4096,
                    use_moe=True,
                    n_routed_experts=n_routed,
                    n_shared_experts=n_shared,
                    top_k=2,
                )
                total = cfg.total_params()
                active = cfg.active_params()

                # highlight configs near the target
                near_1b = 0.8e9 <= total <= 1.2e9
                near_130m = 100e6 <= active <= 160e6

                marker = ""
                if near_1b and near_130m:
                    marker = " <<<< HIT"
                elif near_1b:
                    marker = " << ~1B total"
                elif near_130m:
                    marker = " << ~130M active"

                tag = f"{n_total_experts}exp({n_shared}s+{n_routed}r) top2 ffn{ffn_dim}"
                print(f"{tag:<28} {ffn_dim:>5} {n_total_experts:>5} {n_shared:>6} "
                      f"{fmt(total):>10} {fmt(active):>10} {active/total:>5.1%}{marker}")

                if near_1b and near_130m:
                    targets.append((tag, cfg, total, active))

    if targets:
        print(f"\n{'=' * 60}")
        print(f"  CONFIGS HITTING ~1B TOTAL / ~130M ACTIVE:")
        print(f"{'=' * 60}")
        for tag, cfg, total, active in targets:
            print(f"  {tag}")
            print(f"    total={fmt(total)}, active={fmt(active)}, "
                  f"expert_params={fmt(cfg._expert_params())}")
            print()


def main() -> None:
    print("=" * 60)
    print("  Mini-MoE Parameter Count Verification")
    print("=" * 60)
    print()

    # Detailed breakdown for each model
    for name, factory in [
        ("Model A -- Dense Baseline", ModelConfig.model_a_dense),
        ("Model B -- MoE (no shared)", ModelConfig.model_b_moe_no_shared),
        ("Model C -- MoE + Shared (main)", ModelConfig.model_c_moe_shared),
        ("Model D -- MoE + Aux Loss", ModelConfig.model_d_moe_aux_loss),
    ]:
        cfg = factory()
        print(f"\n{name}")
        print(cfg.param_summary())

    # Summary table
    print("\n\n=== Summary Table ===\n")
    print_model_table()

    # Design space exploration
    explore_design_space()

    # Target check
    print("\n\n=== Target Verification ===")
    cfg_c = ModelConfig.model_c_moe_shared()
    total_c = cfg_c.total_params()
    active_c = cfg_c.active_params()
    print(f"  Model C total:  {fmt(total_c):>8}  (target: ~1B)    {'PASS' if 0.8e9 < total_c < 1.3e9 else 'MISS'}")
    print(f"  Model C active: {fmt(active_c):>8}  (target: ~130M)  {'PASS' if 100e6 < active_c < 160e6 else 'MISS'}")

    cfg_a = ModelConfig.model_a_dense()
    total_a = cfg_a.total_params()
    print(f"  Model A total:  {fmt(total_a):>8}  (target: ~130M)  {'PASS' if 100e6 < total_a < 160e6 else 'MISS'}")


if __name__ == "__main__":
    main()
