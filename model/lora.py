from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for a standard nn.Linear layer."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.merged = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + lora * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.sub_(delta.to(self.base.weight.dtype))
        self.merged = False


class LoRABatchedExperts(nn.Module):
    """LoRA adapters for batched routed expert tensors: (E, O, I)."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(num_experts, r, in_features))
        self.lora_B = nn.Parameter(torch.empty(num_experts, out_features, r))
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def delta(self, dtype: torch.dtype) -> torch.Tensor:
        # (E, O, r) @ (E, r, I) -> (E, O, I)
        return torch.bmm(self.lora_B, self.lora_A).to(dtype=dtype) * self.scaling


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_attention: tuple[str, ...] = ("q_down_proj", "q_up_proj", "kv_down_proj", "kv_up_proj", "out_proj")
    target_experts: tuple[str, ...] = ("expert_gate_weight", "expert_up_weight", "expert_down_weight")
    target_shared_expert: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    setattr(parent, child_name, new_module)


def inject_lora(model: nn.Module, config: LoRAConfig | None = None) -> nn.Module:
    """Inject LoRA into MLA attention + MoE experts and freeze base weights."""
    cfg = config or LoRAConfig()

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # Replace attention and shared-expert linears.
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name in cfg.target_attention or child_name in cfg.target_shared_expert:
                wrapped = LoRALinear(child, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
                _set_child_module(module, child_name, wrapped)

    # Add LoRA adapters for batched routed expert tensors.
    for module in model.modules():
        has_all = all(hasattr(module, name) for name in cfg.target_experts)
        if not has_all:
            continue
        for name in cfg.target_experts:
            w: torch.Tensor = getattr(module, name)
            if not isinstance(w, nn.Parameter):
                continue
            adapter = LoRABatchedExperts(
                num_experts=w.shape[0],
                out_features=w.shape[1],
                in_features=w.shape[2],
                r=cfg.r,
                alpha=cfg.alpha,
            )
            setattr(module, f"lora_{name}", adapter)

    # Enable LoRA parameters only.
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], strict: bool = False) -> None:
    model.load_state_dict(state, strict=strict)


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def merge_lora_linears(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
        # Merge batched experts if they exist
        target_experts = ("expert_gate_weight", "expert_up_weight", "expert_down_weight")
        for name in target_experts:
            if hasattr(module, f"lora_{name}"):
                adapter = getattr(module, f"lora_{name}")
                if hasattr(adapter, "merged") and adapter.merged:
                    continue
                w = getattr(module, name)
                delta = adapter.delta(w.dtype)
                w.data.add_(delta)
                adapter.merged = True

def iter_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            yield param
