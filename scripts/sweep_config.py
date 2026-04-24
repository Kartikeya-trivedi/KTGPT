"""Sweep hidden_dim, layers, ffn_dim, num_experts to find configs hitting ~1B total / ~130-150M active."""

def compute(h, L, ffn, n_routed, n_shared=1, top_k=2, kv_lora=128, rope=32, nope=32, v_dim=64):
    num_heads = h // 64
    q_lora = h // 2
    head_dim = nope + rope  # 64

    # MLA attention per layer
    attn = (h * q_lora          # q_down
          + q_lora * h          # q_up (num_heads*head_dim = h)
          + h * (kv_lora + rope)  # kv_down
          + kv_lora               # kv_norm
          + kv_lora * num_heads * (nope + v_dim)  # kv_up
          + num_heads * v_dim * h)  # out_proj

    expert = 3 * h * ffn  # SwiGLU: gate + up + down
    router = h * n_routed
    norms = 2 * h
    embed = 32000 * h
    final_norm = h

    n_total = n_routed + n_shared
    n_active = top_k + n_shared

    per_layer_total = attn + n_total * expert + router + norms
    per_layer_active = attn + n_active * expert + router + norms

    total = embed + L * per_layer_total + final_norm
    active = embed + L * per_layer_active + final_norm

    return total, active, attn, expert

def fmt(n):
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.3f}B"

print(f"{'h':>4} {'L':>3} {'ffn':>5} {'routed':>6} {'total_exp':>9} {'expert_sz':>9} "
      f"{'attn/L':>8} {'Total':>10} {'Active':>10} {'ratio':>6}")
print("-" * 95)

results = []
for h in [640, 704, 768]:
    for L in [36, 38, 40, 42]:
        for ffn in range(256, 1025, 64):
            for n_routed in range(15, 65, 1):
                total, active, attn, expert = compute(h, L, ffn, n_routed)
                if 0.95e9 <= total <= 1.10e9 and 125e6 <= active <= 155e6:
                    ratio = ffn / h
                    results.append((h, L, ffn, n_routed, total, active, attn, expert, ratio))

# Sort by how close to 1B total and 140M active
results.sort(key=lambda r: abs(r[4] - 1e9) + abs(r[5] - 140e6))

seen = set()
for h, L, ffn, n_r, total, active, attn, expert, ratio in results[:30]:
    key = (h, L, ffn)
    if key in seen:
        continue
    seen.add(key)
    n_total = n_r + 1
    print(f"{h:>4} {L:>3} {ffn:>5} {n_r:>6} {n_total:>9} {fmt(expert):>9} "
          f"{fmt(attn):>8} {total/1e9:>9.3f}B {active/1e6:>9.1f}M {ratio:>5.2f}x")
