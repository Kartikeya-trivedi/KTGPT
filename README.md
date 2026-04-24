# Mini-MoE: Sparse Mixture-of-Experts for Code at Sub-200M Active Parameters

> **Research Question:** Does MoE help at the sub-200M active parameter regime for domain-specific (code) tasks?

## Architecture Summary

| Parameter          | Value                              |
| ------------------ | ---------------------------------- |
| Total params       | ~1B                                |
| Active params      | ~130M                              |
| Experts            | 8 total (1 shared + 7 routed), top-2 routing |
| Layers             | 20                                 |
| Hidden dim         | 768                                |
| Attention heads    | 12 (head_dim = 64)                 |
| KV heads           | 4 (GQA, 3:1 ratio)                |
| Expert FFN dim     | 1536 per expert                    |
| Shared experts     | 1 always-active                    |
| Routed experts     | 7, activate top-2                  |
| Context length     | 2048 (phase 1) → 4096 (phase 2)   |
| Vocab size         | 32000 (Llama 2 tokenizer)          |

## Ablation Models (for paper)

| Model   | Description                              | Purpose                      |
| ------- | ---------------------------------------- | ---------------------------- |
| Model A | Dense baseline, 125M params              | Control                      |
| Model B | MoE, 125M active / 1B total, top-2, 8E  | Main MoE comparison          |
| Model C | MoE + 1 shared expert (always-active)    | Shared expert ablation       |
| Model D | MoE, bias-based LB (no aux loss)         | Load balancing ablation      |

## Budget

| Phase                        | Hours | Cost   |
| ---------------------------- | ----- | ------ |
| Phase 1 pilot (1B tokens)    | ~15   | ~$17   |
| Ablation runs (5 × 5hr)     | ~25   | ~$28   |
| Phase 2 training (10B tokens)| ~80   | ~$88   |
| RL fine-tuning               | ~30   | ~$33   |
| Buffer / eval / reruns       | —     | ~$334  |
| **Total**                    |       | **$500** |

## Target Venues

- EMNLP 2026 (Apr/May deadline)
- COLM 2026 (Feb deadline)
- ACL SRW 2026 (student track)
