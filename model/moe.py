# TODO: Mixture of Experts layer
# - Router: Linear(hidden_dim, n_experts), top-k selection, softmax over selected
# - Bias-based load balancing (DeepSeek-V3 style)
# - Aux-loss load balancing (for ablation Model D)
# - Expert: FFN (Linear → SiLU → Linear, dim=1536)
# - MoELayer: 1 shared expert + 7 routed experts, top-2
# - Routing metrics: entropy, load CV, per-token expert assignment
