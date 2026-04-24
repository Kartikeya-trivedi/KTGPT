"""
KT_GPT Configuration
====================

Dataclass holding every hyperparameter for the KT_GPT 1B-parameter
decoder-only Mixture-of-Experts language model.

The model uses Multi-head Latent Attention (MLA) from DeepSeek-V2/V3
and a SwiGLU-based MoE FFN with bias-based load balancing.

Key design targets:
  - Total parameters:  ~1B   (all expert weights counted)
  - Active parameters: ~138M (only top-k routed + shared expert + attention)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KTGPTConfig:
    """All hyperparameters for the KT_GPT model.

    Grouped into: dimensions, MLA attention, MoE FFN, and training knobs.
    Every numeric constant used anywhere in the model traces back here —
    no magic numbers in model code.
    """

    # ── Core dimensions ──────────────────────────────────────────────
    hidden_dim: int = 704
    num_layers: int = 36
    vocab_size: int = 32_000
    max_seq_len: int = 4096

    # ── MLA (Multi-head Latent Attention) ────────────────────────────
    q_lora_rank: int = 352       # low-rank query compression dimension (hidden_dim / 2)
    kv_lora_rank: int = 128      # low-rank KV compression dimension (MLA core)
    num_heads: int = 11          # hidden_dim / head_dim = 704 / 64
    head_dim: int = 64           # = qk_nope_dim + qk_rope_dim
    qk_nope_dim: int = 32       # portion of head WITHOUT positional encoding
    qk_rope_dim: int = 32       # portion of head WITH RoPE
    v_head_dim: int = 64

    # ── MoE FFN ──────────────────────────────────────────────────────
    expert_ffn_dim: int = 320
    num_routed_experts: int = 37
    num_shared_experts: int = 1
    top_k: int = 2

    # ── Training / normalization ─────────────────────────────────────
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    bias_update_speed: float = 0.001
    init_std: float = 0.02

    # ──────────────────────────────────────────────────────────────────

    @property
    def total_experts(self) -> int:
        """Total expert count (routed + shared)."""
        return self.num_routed_experts + self.num_shared_experts

    # ── Parameter-count helpers ──────────────────────────────────────

    def _embedding_params(self) -> int:
        """Token embedding (weight-tied with LM head, counted once)."""
        return self.vocab_size * self.hidden_dim

    def _attention_params_per_layer(self) -> int:
        """MLA attention parameters for one layer (no bias terms)."""
        q_down = self.hidden_dim * self.q_lora_rank
        q_up = self.q_lora_rank * (self.num_heads * self.head_dim)
        kv_down = self.hidden_dim * (self.kv_lora_rank + self.qk_rope_dim)
        kv_norm = self.kv_lora_rank  # RMSNorm weight
        kv_up = self.kv_lora_rank * self.num_heads * (self.qk_nope_dim + self.v_head_dim)
        out = self.num_heads * self.v_head_dim * self.hidden_dim
        return q_down + q_up + kv_down + kv_norm + kv_up + out

    def _expert_params(self) -> int:
        """Parameters for a single SwiGLU expert (gate + up + down)."""
        return 3 * self.hidden_dim * self.expert_ffn_dim

    def _moe_params_per_layer(self) -> int:
        """All MoE FFN parameters for one layer."""
        shared = self.num_shared_experts * self._expert_params()
        routed = self.num_routed_experts * self._expert_params()
        router = self.hidden_dim * self.num_routed_experts  # router linear
        # router bias scalars (15) are separate from main params
        return shared + routed + router

    def _norm_params_per_layer(self) -> int:
        """Two RMSNorms per block (pre-attention, pre-FFN)."""
        return 2 * self.hidden_dim

    def _per_layer_params(self) -> int:
        return (self._attention_params_per_layer()
                + self._moe_params_per_layer()
                + self._norm_params_per_layer())

    def total_params(self) -> int:
        """Total parameter count (all experts, weight-tied embedding)."""
        layers = self.num_layers * self._per_layer_params()
        final_norm = self.hidden_dim
        embeddings = self._embedding_params()
        return layers + final_norm + embeddings

    def active_params(self) -> int:
        """Parameters active per token (top-k routed + shared + attention)."""
        attn = self._attention_params_per_layer()
        shared = self.num_shared_experts * self._expert_params()
        routed_active = self.top_k * self._expert_params()
        router = self.hidden_dim * self.num_routed_experts
        norms = self._norm_params_per_layer()
        per_layer = attn + shared + routed_active + router + norms
        return (self.num_layers * per_layer
                + self.hidden_dim  # final norm
                + self._embedding_params())

    def verify_param_count(self) -> None:
        """Print a detailed breakdown of parameter counts."""
        def fmt(n: int) -> str:
            if n >= 1e9:
                return f"{n / 1e9:.3f}B"
            return f"{n / 1e6:.2f}M"

        print("=" * 60)
        print("  KT_GPT Parameter Count Breakdown")
        print("=" * 60)

        print(f"\n  Embeddings (weight-tied):    {fmt(self._embedding_params()):>10}")

        attn = self._attention_params_per_layer()
        print(f"\n  --- Per Layer ({self.num_layers} layers) ---")
        print(f"  MLA Attention:              {fmt(attn):>10}")
        print(f"  Single Expert (SwiGLU):     {fmt(self._expert_params()):>10}")
        print(f"  Shared Expert(s) [{self.num_shared_experts}]:       {fmt(self.num_shared_experts * self._expert_params()):>10}")
        print(f"  Routed Experts [{self.num_routed_experts}]:         {fmt(self.num_routed_experts * self._expert_params()):>10}")
        print(f"  Router Linear:              {fmt(self.hidden_dim * self.num_routed_experts):>10}")
        print(f"  Layer Norms (2x RMSNorm):   {fmt(self._norm_params_per_layer()):>10}")
        print(f"  Layer Total:                {fmt(self._per_layer_params()):>10}")

        print(f"\n  --- Totals ---")
        print(f"  All Layers:                 {fmt(self.num_layers * self._per_layer_params()):>10}")
        print(f"  Final RMSNorm:              {fmt(self.hidden_dim):>10}")
        print(f"  Embeddings:                 {fmt(self._embedding_params()):>10}")
        print(f"  ─────────────────────────────────────")
        print(f"  Total Parameters:           {fmt(self.total_params()):>10}")
        print(f"  Active Parameters / token:  {fmt(self.active_params()):>10}")
        print(f"  Sparsity:                   {1 - self.active_params() / self.total_params():.1%}")
        print(f"  KV Cache / token / layer:   {self.kv_lora_rank + self.qk_rope_dim} values"
              f"  (vs {2 * self.num_heads * self.head_dim} for standard MHA)")
        print("=" * 60)
