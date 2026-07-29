"""P4.T1 acceptance: LoRA touches <5% of params, gradients flow only to
adapters, adapters serialize separately and round-trip."""

import numpy as np
import torch

from draftbot.data.dataset import PAD
from draftbot.models.base import slot_nll
from draftbot.models.lora import (adapter_state, apply_finetune_mode,
                                  load_adapter)
from draftbot.models.modern import ModernDraftBot

N_CARDS, T, P, F = 60, 8, 5, 14


def build():
    torch.manual_seed(0)
    feats = torch.randn(N_CARDS + 1, F)
    return ModernDraftBot(N_CARDS, feats, t=T, emb_dim=64, layers=2, heads=4)


def batch(b=6):
    rng = np.random.default_rng(0)
    packs = torch.from_numpy(rng.integers(0, N_CARDS, (b, T, P)).astype(np.int64))
    picks = packs[:, :, 0].clone()
    prev = torch.full((b, T), PAD, dtype=torch.long)
    prev[:, 1:] = picks[:, :-1]
    return packs, prev, picks


def test_lora_param_fraction_under_5pct_at_production_dims():
    """DoD: <5% trainable at the real trunk scale (7.7M, emb 256 / 8 layers)."""
    torch.manual_seed(0)
    model = ModernDraftBot(339, torch.randn(340, 771), t=45, emb_dim=256,
                           layers=8, heads=8, feature_only=True,
                           use_set_token=True)
    stats = apply_finetune_mode(model, "lora")  # default rank
    assert stats["frac"] < 0.05, f"{stats['frac']:.2%} trainable"


def test_gradients_only_reach_adapters():
    model = build()
    apply_finetune_mode(model, "lora", rank=8)
    packs, prev, picks = batch()
    loss = slot_nll(model(packs, prev), packs, picks)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        assert p.grad is None, f"frozen param {name} received grad"
    lora_grads = [p.grad for n, p in model.named_parameters()
                  if "parametrizations" in n and p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in lora_grads)


def test_adapter_roundtrip_changes_output():
    model = build().eval()
    apply_finetune_mode(model, "lora", rank=8)
    packs, prev, picks = batch()
    with torch.no_grad():
        base_out = model(packs, prev)
    # nudge the adapters, capture state
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    loss = slot_nll(model(packs, prev), packs, picks)
    loss.backward()
    opt.step()
    state = adapter_state(model)
    with torch.no_grad():
        tuned_out = model(packs, prev)
    assert not torch.allclose(base_out, tuned_out)

    fresh = build().eval()
    apply_finetune_mode(fresh, "lora", rank=8)
    load_adapter(fresh, state)
    with torch.no_grad():
        restored = fresh(packs, prev)
    torch.testing.assert_close(restored, tuned_out, atol=1e-5, rtol=1e-5)


def test_head_mode_trains_only_table_and_pointer():
    model = build()
    stats = apply_finetune_mode(model, "head")
    for name, p in model.named_parameters():
        expected = any(t in name for t in
                       ("embedding.learned", "q_proj", "k_proj"))
        assert p.requires_grad == expected, name
    assert stats["frac"] < 0.5
