"""
KT_GPT Model
============

Complete implementation of a ~1B parameter decoder-only Transformer with:
  - Multi-head Latent Attention (MLA) from DeepSeek-V2/V3
  - Mixture-of-Experts FFN with SwiGLU activation
  - Bias-based load balancing (no auxiliary loss)

All modules are pure nn.Module — no HuggingFace dependencies.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import KTGPTConfig


# ═══════════════════════════════════════════════════════════════════════
#  RMSNorm
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler and faster than LayerNorm: normalizes by RMS of activations
    without centering.  No bias term.  Used everywhere in the model.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, cast back
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms).to(dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════════
#  Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin caches for Rotary Position Embeddings.

    Only applied to the *rope* portion of query/key heads in MLA.
    The rope dimension is split in half: the first half is rotated with
    cos and the second half with sin, following the standard RoPE formulation.
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        # dim = qk_rope_dim (e.g. 32).  We need dim/2 frequency bands.
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                   # (max_seq_len, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) slices for positions [offset, offset+seq_len)."""
        return (self.cos_cached[offset: offset + seq_len],
                self.sin_cached[offset: offset + seq_len])


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to tensor x along its last dimension.

    Uses the half-split convention: x is split into two halves along the
    last dim, then rotated using the standard 2D rotation formula.

    Args:
        x:   (..., rope_dim)  — any leading dimensions are fine
        cos: broadcastable to x's first-half shape
        sin: broadcastable to x's first-half shape
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  Multi-head Latent Attention (MLA)
# ═══════════════════════════════════════════════════════════════════════

class MLAAttention(nn.Module):
    """Multi-head Latent Attention from DeepSeek-V2/V3.

    Core idea: instead of caching per-head K and V tensors, we cache a
    single low-rank *compressed latent* (kv_lora_rank=128 dims) plus the
    RoPE key component (qk_rope_dim=32 dims).  At inference time, the
    full K and V are reconstructed from this compressed representation.
    This gives ~10x KV-cache compression vs standard MHA.

    Forward pass:
      1. Compress query → low-rank → expand to per-head q_nope, q_rope
      2. Compress KV → single latent + separate k_rope
      3. Expand latent → per-head k_nope, v
      4. Apply RoPE to q_rope and k_rope only
      5. Concat k = [k_nope | k_rope], run scaled dot-product attention
      6. Project output back to hidden_dim
    """

    def __init__(self, config: KTGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.qk_nope_dim = config.qk_nope_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.head_dim = config.head_dim          # nope + rope = 64
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # ── Query path: hidden → compressed → per-head ──────────────
        self.q_down_proj = nn.Linear(config.hidden_dim, config.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(config.q_lora_rank,
                                   config.num_heads * config.head_dim, bias=False)

        # ── KV path: hidden → (compressed_latent || k_rope) ─────────
        # The single projection produces both the latent and the shared
        # RoPE key component in one shot.
        self.kv_down_proj = nn.Linear(config.hidden_dim,
                                      config.kv_lora_rank + config.qk_rope_dim,
                                      bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)

        # Expand compressed latent → per-head k_nope and v
        self.kv_up_proj = nn.Linear(config.kv_lora_rank,
                                    config.num_heads * (config.qk_nope_dim + config.v_head_dim),
                                    bias=False)

        # ── Output projection ───────────────────────────────────────
        self.out_proj = nn.Linear(config.num_heads * config.v_head_dim,
                                  config.hidden_dim, bias=False)

        # ── Positional encoding ──────────────────────────────────────
        self.rotary_emb = RotaryEmbedding(config.qk_rope_dim,
                                          config.max_seq_len,
                                          config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x:              (batch, seq_len, hidden_dim)
            attention_mask:  not used directly — we rely on is_causal in SDPA
            use_cache:       whether to return the KV cache tuple
            past_kv:         (cached_kv_compressed, cached_k_rope) from prior steps

        Returns:
            output:  (batch, seq_len, hidden_dim)
            cache:   (kv_compressed, k_rope) if use_cache else None
        """
        B, S, _ = x.shape

        # ── 1. Query ────────────────────────────────────────────────
        q = self.q_up_proj(self.q_down_proj(x))               # (B, S, H*head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim)
        # Split into non-positional and RoPE portions
        q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # ── 2. KV compression ──────────────────────────────────────
        kv_full = self.kv_down_proj(x)                         # (B, S, kv_rank + rope)
        kv_compressed, k_rope = kv_full.split(
            [self.kv_lora_rank, self.qk_rope_dim], dim=-1
        )
        # kv_compressed: (B, S, kv_lora_rank)  — this is what we cache
        # k_rope:        (B, S, qk_rope_dim)   — this too

        kv_compressed = self.kv_norm(kv_compressed)

        # ── 5. Handle KV cache ──────────────────────────────────────
        # Cache the *compressed* representation, not expanded K/V
        if past_kv is not None:
            past_kv_comp, past_k_rope = past_kv
            kv_compressed = torch.cat([past_kv_comp, kv_compressed], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)

        new_cache = (kv_compressed, k_rope) if use_cache else None

        # Total KV sequence length (may include cached tokens)
        kv_len = kv_compressed.shape[1]

        # ── Expand compressed KV → per-head k_nope, v ──────────────
        kv_expanded = self.kv_up_proj(kv_compressed)           # (B, kv_len, H*(nope+v))
        kv_expanded = kv_expanded.view(
            B, kv_len, self.num_heads, self.qk_nope_dim + self.v_head_dim
        )
        k_nope, v = kv_expanded.split([self.qk_nope_dim, self.v_head_dim], dim=-1)

        # ── 3. Apply RoPE ──────────────────────────────────────────
        # Position offset for the query (always the new tokens)
        q_offset = kv_len - S
        cos_q, sin_q = self.rotary_emb(S, offset=q_offset)
        # Reshape for (B, S, H, rope_dim): need (1, S, 1, rope_dim//2)
        cos_q = cos_q[None, :, None, :]
        sin_q = sin_q[None, :, None, :]
        q_rope = apply_rotary_emb(q_rope, cos_q, sin_q)

        # k_rope: (B, kv_len, rope_dim) — shared across heads
        cos_k, sin_k = self.rotary_emb(kv_len)
        cos_k = cos_k[None, :, None, :]  # (1, kv_len, 1, rope_dim//2)
        sin_k = sin_k[None, :, None, :]
        # Expand k_rope to have a head dimension for concat
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        k_rope_expanded = apply_rotary_emb(k_rope_expanded, cos_k, sin_k)

        # ── 4. Reconstruct full keys ───────────────────────────────
        k = torch.cat([k_nope, k_rope_expanded], dim=-1)      # (B, kv_len, H, head_dim)

        # ── 6. Scaled dot-product attention ────────────────────────
        # Transpose to (B, H, S/kv_len, D) for F.scaled_dot_product_attention
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        q_full = q_full.transpose(1, 2).contiguous()                       # (B, H, S, head_dim)
        k = k.transpose(1, 2).contiguous()                                 # (B, H, kv_len, head_dim)
        v = v.transpose(1, 2).contiguous()                                 # (B, H, kv_len, v_head_dim)

        # Use causal mask only when not using cache (full prefill)
        # During generation with cache, S=1 and causal is implicit.
        is_causal = (past_kv is None) and (S > 1)

        attn_out = F.scaled_dot_product_attention(
            q_full, k, v,
            attn_mask=None,
            is_causal=is_causal,
            scale=self.scale,
        )                                                      # (B, H, S, v_head_dim)

        # ── 7. Output projection ───────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous()       # (B, S, H, v_head_dim)
        attn_out = attn_out.view(B, S, self.num_heads * self.v_head_dim)
        return self.out_proj(attn_out), new_cache


# ═══════════════════════════════════════════════════════════════════════
#  SwiGLU Expert
# ═══════════════════════════════════════════════════════════════════════

class SwiGLUExpert(nn.Module):
    """Single feed-forward expert using SwiGLU activation.

    SwiGLU(x) = (Swish(gate(x)) ⊙ up(x)) · down
    where Swish = SiLU = x·σ(x).

    Three projection matrices: gate, up (both hidden→ffn), down (ffn→hidden).
    No bias on any linear layer.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: elementwise product of gated and ungated paths
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════════
#  MoE Router
# ═══════════════════════════════════════════════════════════════════════

class MoERouter(nn.Module):
    """Token-to-expert router with bias-based load balancing (DeepSeek-V3).

    Routing:
      1. scores = softmax(Linear(hidden_dim, num_routed_experts)(x))
      2. selection = scores + bias  (bias only affects which experts are *chosen*)
      3. top2_idx = topk(selection, k=2)
      4. top2_scores = scores[top2_idx]  (weighting uses original scores, not biased)

    Load balancing is done via a separate online bias update — NOT through
    the main training loss.  This avoids the auxiliary-loss / gradient
    interference issues described in DeepSeek-V3.
    """

    def __init__(self, config: KTGPTConfig) -> None:
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.top_k
        self.bias_update_speed = config.bias_update_speed

        self.gate = nn.Linear(config.hidden_dim, config.num_routed_experts, bias=False)
        # Per-expert bias for load-balancing selection (NOT a normal param)
        # Kept in float32 even when the model runs in bf16.
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_routed_experts, dtype=torch.float32),
            requires_grad=False,    # updated manually, not by optimizer
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch * seq_len, hidden_dim) — flattened token representations

        Returns:
            top_indices: (num_tokens, top_k) — expert indices per token
            top_scores:  (num_tokens, top_k) — softmax weights (no bias)
            expert_counts: (num_routed_experts,) — tokens assigned per expert
        """
        # Compute routing probabilities
        logits = self.gate(x).float()                         # (T, E) in float32
        scores = F.softmax(logits, dim=-1)                     # (T, E)

        # Add bias for selection only (does not affect final weighting)
        selection = scores + self.expert_bias.unsqueeze(0)

        # Select top-k experts per token
        top_scores_biased, top_indices = torch.topk(selection, self.top_k, dim=-1)

        # Gather the *original* softmax scores for the selected experts
        top_scores = scores.gather(dim=-1, index=top_indices)

        # Normalize top-k scores so they sum to 1 per token
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Count tokens per expert for load-balancing update
        expert_counts = torch.zeros(
            self.num_routed_experts, device=x.device, dtype=torch.float32
        )
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                0, top_indices[:, k],
                torch.ones(top_indices.shape[0], device=x.device, dtype=torch.float32),
            )

        return top_indices, top_scores.to(x.dtype), expert_counts

    @torch.no_grad()
    def update_bias(self, expert_counts: torch.Tensor, total_tokens: int) -> None:
        """Online bias update for load balancing (runs after each forward).

        If an expert is assigned more tokens than the uniform threshold,
        decrease its bias (making it less likely to be selected).
        If underloaded, increase its bias.

        This is a direct scalar update — no gradient, no loss, no backprop.
        """
        # Multiply total_tokens by top_k because each token selects top_k experts
        threshold = (total_tokens * self.top_k) / self.num_routed_experts
        overloaded = expert_counts > threshold
        underloaded = expert_counts < threshold
        self.expert_bias[overloaded] -= self.bias_update_speed
        self.expert_bias[underloaded] += self.bias_update_speed


# ═══════════════════════════════════════════════════════════════════════
#  MoE FFN Layer
# ═══════════════════════════════════════════════════════════════════════

class MoEFFN(nn.Module):
    """Mixture-of-Experts Feed-Forward layer.

    Architecture:
      - 1 shared expert (always active, processes every token)
      - 15 routed experts (top-2 selected per token via router)

    Output = sum(score_i * expert_i(x) for i in top-2) + shared_expert(x)

    The shared expert provides a stable baseline representation that
    every token sees, while routed experts specialize on token subsets.
    """

    def __init__(self, config: KTGPTConfig) -> None:
        super().__init__()
        self.config = config

        # Shared expert — always active
        self.shared_expert = SwiGLUExpert(config.hidden_dim, config.expert_ffn_dim)

        # Routed experts (Batched for performance)
        # Instead of 37 separate Linear layers, we stack all weights into single tensors
        # shape: (num_experts, out_features, in_features)
        self.expert_gate_weight = nn.Parameter(
            torch.empty(config.num_routed_experts, config.expert_ffn_dim, config.hidden_dim)
        )
        self.expert_up_weight = nn.Parameter(
            torch.empty(config.num_routed_experts, config.expert_ffn_dim, config.hidden_dim)
        )
        self.expert_down_weight = nn.Parameter(
            torch.empty(config.num_routed_experts, config.hidden_dim, config.expert_ffn_dim)
        )

        self.router = MoERouter(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)                                # (T, D)  T = B*S
        T = x_flat.shape[0]

        # ── Route tokens to experts ────────────────────────────────
        top_indices, top_scores, expert_counts = self.router(x_flat)
        # top_indices: (T, top_k)    top_scores: (T, top_k)

        # ── Execute routed experts (Batched) ───────────────────────
        flat_indices = top_indices.flatten()
        flat_x = x_flat.repeat_interleave(self.config.top_k, dim=0)
        flat_scores = top_scores.flatten()

        # Sort tokens by expert assignment
        sorted_experts, sorted_idx = torch.sort(flat_indices)
        sorted_x = flat_x[sorted_idx]
        sorted_scores = flat_scores[sorted_idx]

        # Count tokens per expert and find boundaries
        true_counts = torch.bincount(sorted_experts, minlength=self.config.num_routed_experts)
        true_starts = torch.zeros(self.config.num_routed_experts, device=x.device, dtype=torch.long)
        true_starts[1:] = true_counts.cumsum(0)[:-1]

        # Position of each token within its assigned expert's group
        token_pos_in_group = torch.arange(T * self.config.top_k, device=x.device) - true_starts[sorted_experts]

        # Create padded tensor: (num_experts, max_tokens, D)
        max_tokens = true_counts.max().item()
        if max_tokens == 0:
            max_tokens = 1
            
        padded_x = torch.zeros((self.config.num_routed_experts, max_tokens, D), device=x.device, dtype=x.dtype)
        padded_x[sorted_experts, token_pos_in_group] = sorted_x

        # Batched matmuls: (E, max_tokens, D) @ (E, D, F) -> (E, max_tokens, F)
        # Optional LoRA adapters can inject low-rank deltas on routed experts.
        expert_gate_weight = self.expert_gate_weight
        expert_up_weight = self.expert_up_weight
        expert_down_weight = self.expert_down_weight
        if hasattr(self, "lora_expert_gate_weight"):
            expert_gate_weight = expert_gate_weight + self.lora_expert_gate_weight.delta(expert_gate_weight.dtype)
        if hasattr(self, "lora_expert_up_weight"):
            expert_up_weight = expert_up_weight + self.lora_expert_up_weight.delta(expert_up_weight.dtype)
        if hasattr(self, "lora_expert_down_weight"):
            expert_down_weight = expert_down_weight + self.lora_expert_down_weight.delta(expert_down_weight.dtype)

        gate = torch.bmm(padded_x, expert_gate_weight.transpose(1, 2))
        up = torch.bmm(padded_x, expert_up_weight.transpose(1, 2))
        h = F.silu(gate) * up
        exp_out_padded = torch.bmm(h, expert_down_weight.transpose(1, 2))

        # Extract valid tokens back to flat shape
        exp_out_flat = exp_out_padded[sorted_experts, token_pos_in_group]
        
        # Weight by router scores
        exp_out_flat = exp_out_flat * sorted_scores.unsqueeze(-1)

        # Scatter back to original token ordering
        token_indices_orig = torch.arange(T, device=x.device).repeat_interleave(self.config.top_k)
        sorted_token_indices = token_indices_orig[sorted_idx]
        
        routed_out = torch.zeros_like(x_flat)
        routed_out.scatter_add_(0, sorted_token_indices.unsqueeze(-1).expand(-1, D), exp_out_flat)

        # ── Execute shared expert on ALL tokens ────────────────────
        shared_out = self.shared_expert(x_flat)

        # ── Combine ────────────────────────────────────────────────
        output = routed_out + shared_out

        # ── Update router bias during training ─────────────────────
        if self.training:
            self.router.update_bias(expert_counts, T)

        return output.view(B, S, D)


# ═══════════════════════════════════════════════════════════════════════
#  Transformer Block
# ═══════════════════════════════════════════════════════════════════════

class KTGPTBlock(nn.Module):
    """Single transformer block: MLA attention + MoE FFN with residuals.

    Layout (standard sequential, NOT parallel):
      residual = x
      x = RMSNorm(x)
      x = MLA(x) + residual

      residual = x
      x = RMSNorm(x)
      x = MoEFFN(x) + residual
    """

    def __init__(self, config: KTGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.attn = MLAAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.ffn = MoEFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with pre-norm and residual
        residual = x
        x = self.attn_norm(x)
        attn_out, new_cache = self.attn(
            x, attention_mask=attention_mask, use_cache=use_cache, past_kv=past_kv
        )
        x = attn_out + residual

        # MoE FFN with pre-norm and residual
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x) + residual

        return x, new_cache


# ═══════════════════════════════════════════════════════════════════════
#  KT_GPT — Full Model
# ═══════════════════════════════════════════════════════════════════════

class KTGPT(nn.Module):
    """Complete KT_GPT decoder-only language model.

    Architecture:
      - Token embedding (weight-tied with LM head)
      - 36 x KTGPTBlock (MLA + MoE FFN)
      - Final RMSNorm
      - Linear LM head → vocab logits

    Forward returns logits and an optional list of KV caches for
    autoregressive generation.
    """

    def __init__(self, config: KTGPTConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            KTGPTBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # Final normalization before LM head
        self.final_norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # LM head — weight-tied with embedding
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Scale output projections by 1/sqrt(2*num_layers) for training stability
        output_scale = 1.0 / math.sqrt(2 * config.num_layers)
        for layer in self.layers:
            layer.attn.out_proj.weight.data.mul_(output_scale)
            layer.ffn.shared_expert.down_proj.weight.data.mul_(output_scale)
            layer.ffn.expert_down_weight.data.mul_(output_scale)

        # Tie embedding and LM head weights
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize all Linear and Embedding layers with N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Parameter) and len(module.shape) >= 2:
            # For batched expert weights
            nn.init.normal_(module, mean=0.0, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv_list: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids:    (batch, seq_len)  token IDs
            attention_mask: unused (causal masking handled by SDPA)
            use_cache:    whether to return KV caches for generation
            past_kv_list: list of (kv_compressed, k_rope) per layer

        Returns:
            logits:       (batch, seq_len, vocab_size)
            new_caches:   list of cache tuples if use_cache else None
        """
        x = self.embed(input_ids)                              # (B, S, D)

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, layer in enumerate(self.layers):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, cache = layer(
                x, attention_mask=attention_mask,
                use_cache=use_cache, past_kv=past_kv,
            )
            if cache is not None:
                new_caches.append(cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_caches if use_cache else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching and repetition penalty.
        
        Args:
            input_ids: (batch, seq_len)
            max_new_tokens: how many tokens to generate
            temperature: 1.0 = normal, <1.0 = focused, >1.0 = creative
            top_p: nucleus sampling threshold
            top_k: only sample from top K tokens
            repetition_penalty: >1.0 penalizes repeated tokens
            eos_token_id: stop generating if this token is produced
        """
        B, S = input_ids.shape
        curr_ids = input_ids
        past_kv_list = None
        
        for _ in range(max_new_tokens):
            model_input = curr_ids if past_kv_list is None else curr_ids[:, -1:]
            
            logits, past_kv_list = self.forward(
                model_input, 
                use_cache=True, 
                past_kv_list=past_kv_list
            )
            
            next_token_logits = logits[:, -1, :].clone()
            
            # 1. Repetition Penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    # For each token in the sequence, penalize its logit
                    for token_id in set(curr_ids[b].tolist()):
                        if next_token_logits[b, token_id] > 0:
                            next_token_logits[b, token_id] /= repetition_penalty
                        else:
                            next_token_logits[b, token_id] *= repetition_penalty

            # 2. Temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # 3. Top-K Sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # 4. Top-P (Nucleus) Sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # 5. Final Sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return curr_ids
