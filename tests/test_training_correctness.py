"""P1.T4 — model-agnostic training-correctness tests, parametrized over every
class implementing the DraftModel protocol. Synthetic data: fast + CI-safe."""

import numpy as np
import pytest
import torch

from draftbot.data.dataset import PAD
from draftbot.models.base import slot_nll
from draftbot.models.legacy import LegacyDraftBot

N_CARDS, T, P, F = 50, 6, 4, 12


def make_model(cls):
    torch.manual_seed(0)
    feats = torch.randn(N_CARDS + 1, F)
    if cls is LegacyDraftBot:
        return cls(N_CARDS, feats, t=T, emb_dim=32, enc_layers=1, dec_layers=1,
                   enc_heads=4, dec_heads=4)
    return cls(N_CARDS, feats, t=T, emb_dim=32, layers=1, heads=4)


def model_classes():
    classes = [LegacyDraftBot]
    try:
        from draftbot.models.modern import ModernDraftBot
        classes.append(ModernDraftBot)
    except ImportError:
        pass
    return classes


def synth_batch(b=8, seed=0):
    rng = np.random.default_rng(seed)
    packs = rng.integers(0, N_CARDS, size=(b, T, P)).astype(np.int64)
    packs[:, :, -1][: b // 2] = PAD  # some padding
    pick_slot = rng.integers(0, P - 1, size=(b, T))
    picks = np.take_along_axis(packs, pick_slot[..., None], -1)[..., 0]
    prev = np.full((b, T), PAD, dtype=np.int64)
    prev[:, 1:] = picks[:, :-1]
    return (torch.from_numpy(packs), torch.from_numpy(prev), torch.from_numpy(picks))


@pytest.mark.parametrize("cls", model_classes())
def test_causality(cls):
    """Changing future packs/picks must not alter logits at earlier steps."""
    model = make_model(cls).eval()
    packs, prev, _ = synth_batch()
    with torch.no_grad():
        base = model(packs, prev)
        k = 3
        packs2, prev2 = packs.clone(), prev.clone()
        packs2[:, k:] = torch.roll(packs2[:, k:], 1, dims=0)
        prev2[:, k + 1:] = torch.roll(prev2[:, k + 1:], 1, dims=0)
        perturbed = model(packs2, prev2)
    torch.testing.assert_close(base[:, :k], perturbed[:, :k], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cls", model_classes())
def test_pack_mask(cls):
    """PAD slots must carry zero probability."""
    model = make_model(cls).eval()
    packs, prev, _ = synth_batch()
    with torch.no_grad():
        logits = model(packs, prev)
    probs = torch.softmax(logits, -1)
    assert probs[packs == PAD].max() < 1e-6
    valid_sum = probs.masked_fill(packs == PAD, 0).sum(-1)
    torch.testing.assert_close(valid_sum, torch.ones_like(valid_sum))


@pytest.mark.parametrize("cls", model_classes())
def test_loss_finite_with_all_stats_missing(cls):
    """Feature rows of all zeros (the masked day-0 condition) must not NaN."""
    torch.manual_seed(0)
    feats = torch.zeros(N_CARDS + 1, F)
    if cls is LegacyDraftBot:
        model = cls(N_CARDS, feats, t=T, emb_dim=32, enc_layers=1, dec_layers=1,
                    enc_heads=4, dec_heads=4)
    else:
        model = cls(N_CARDS, feats, t=T, emb_dim=32, layers=1, heads=4)
    packs, prev, picks = synth_batch()
    logits = model(packs, prev)
    loss = slot_nll(logits, packs, picks)
    assert torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("cls", model_classes())
def test_overfit_canary(cls):
    """~100 tiny drafts must reach >90% top-1 quickly — proves the wiring."""
    torch.manual_seed(1)
    model = make_model(cls)
    packs, prev, picks = synth_batch(b=100, seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    top1 = 0.0
    for step in range(400):
        opt.zero_grad()
        logits = model(packs, prev)
        loss = slot_nll(logits, packs, picks)
        loss.backward()
        opt.step()
        if step % 50 == 49:
            with torch.no_grad():
                pred_slot = model(packs, prev).argmax(-1)
            pred_ids = torch.gather(packs, 2, pred_slot[..., None])[..., 0]
            top1 = float((pred_ids == picks).float().mean())
            if top1 > 0.9:
                break
    assert top1 > 0.9, f"overfit canary stuck at {top1:.3f}"
