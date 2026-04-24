# TODO: Full transformer model
# - TransformerBlock: RMSNorm → GQA → residual → RMSNorm → MoE/FFN → residual
# - MiniMoEModel: embedding + 12 blocks + final norm + LM head (weight-tied)
# - MiniDenseModel: same but standard FFN (for Model A ablation)
# - Config dataclass with all hyperparameters
# - Parameter count utility
