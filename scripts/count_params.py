"""
Parameter count verification script for KT_GPT.

Computes total and active parameter counts for all 4 ablation models
using the KTGPTConfig dataclass (no model instantiation needed).

Usage:
    python -m scripts.count_params
"""

from __future__ import annotations

import sys
from model.config import KTGPTConfig


def fmt(n: int) -> str:
    """Format large numbers with M/B suffix."""
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    return f"{n / 1e6:.2f}M"


def print_model_table() -> None:
    """Print a summary table for all 4 ablation model configs."""
    models = [
        ("Model A (Dense Baseline)", KTGPTConfig(num_routed_experts=0, num_shared_experts=1, expert_ffn_dim=960, top_k=0)),
        ("Model B (MoE - No Shared)", KTGPTConfig(num_routed_experts=38, num_shared_experts=0, expert_ffn_dim=320)),
        ("Model C (KT_GPT - Main MoE)", KTGPTConfig(num_routed_experts=37, num_shared_experts=1, expert_ffn_dim=320)),
        ("Model D (MoE - Aux Loss)", KTGPTConfig(num_routed_experts=37, num_shared_experts=1, expert_ffn_dim=320)),
    ]

    header = f"{'Model':<28} {'Experts':>8} {'Top-k':>5} {'FFN':>5} {'Total':>10} {'Active':>10} {'Sparse%':>8}"
    print(header)
    print("-" * len(header))

    for name, cfg in models:
        if cfg.num_routed_experts > 0 or cfg.num_shared_experts > 1:
            experts_str = f"{cfg.num_shared_experts}s+{cfg.num_routed_experts}r"
        else:
            experts_str = "dense"
        total = cfg.total_params()
        active = cfg.active_params()
        sparsity = f"{1 - active / total:.1%}" if cfg.num_routed_experts > 0 else "0.0%"
        print(f"{name:<28} {experts_str:>8} {cfg.top_k:>5} {cfg.expert_ffn_dim:>5} "
              f"{fmt(total):>10} {fmt(active):>10} {sparsity:>8}")


def main() -> None:
    print("=" * 60)
    print("  KT_GPT Parameter Count Verification")
    print("=" * 60)
    print()

    # Detailed breakdown for our main model (Model C)
    cfg_c = KTGPTConfig()
    cfg_c.verify_param_count()

    # Summary table
    print("\n\n=== Ablation Models Summary Table ===\n")
    print_model_table()
    print()


if __name__ == "__main__":
    main()
