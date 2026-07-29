"""LoRA fine-tuning machinery (PLAN P4.T1).

`apply_finetune_mode(model, mode, rank, alpha)` configures a ModernDraftBot for:
  - "full": everything trainable (the default full fine-tune)
  - "lora": base frozen; low-rank adapters on attention qkv/o + SwiGLU FFN
            projections; per-set card table + pointer projections stay
            full-rank trainable (they are new-set-specific by design)
  - "head": base frozen; only card table + pointer projections train

LoRA is implemented with torch parametrizations (W → W + B@A·α/r), which works
uniformly for nn.Linear weights and nn.MultiheadAttention's fused in_proj
Parameter. Adapters serialize separately via adapter_state / load_adapter.
"""

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRAParam(nn.Module):
    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / rank

    def forward(self, W):
        return W + (self.B @ self.A) * self.scale


def _lora_targets(model) -> list[tuple[nn.Module, str]]:
    """(module, weight-attr) pairs for attention qkv/o + FFN projections."""
    targets = []
    for mod in model.modules():
        if isinstance(mod, nn.MultiheadAttention):
            targets.append((mod, "in_proj_weight"))
            targets.append((mod.out_proj, "weight"))
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (
                name.endswith(("in_proj", "out_proj")) and "ffn" in name
                or name == "fusion"):
            targets.append((mod, "weight"))
    return targets


ALWAYS_TRAINABLE = ("embedding.learned", "q_proj", "k_proj")


def apply_finetune_mode(model: nn.Module, mode: str, rank: int = 4,
                        alpha: float = 16.0) -> dict:
    """Configure trainability; returns {"trainable": n, "total": n, "frac": f}."""
    assert mode in ("full", "lora", "head")
    if mode != "full":
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if any(t in name for t in ALWAYS_TRAINABLE):
                p.requires_grad_(True)
    if mode == "lora":
        for mod, attr in _lora_targets(model):
            weight = getattr(mod, attr)
            parametrize.register_parametrization(
                mod, attr, LoRAParam(weight, rank, alpha))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total,
            "frac": trainable / max(total, 1)}


def adapter_state(model: nn.Module) -> dict:
    """State dict of ONLY the trainable tensors (LoRA A/B + table + pointer)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if any(t in k for t in ALWAYS_TRAINABLE) or "parametrizations" in k}


def load_adapter(model: nn.Module, state: dict) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"adapter has unknown tensors: {unexpected[:5]}"
