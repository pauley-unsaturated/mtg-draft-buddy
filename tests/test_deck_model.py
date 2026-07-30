"""P5.T1d training-correctness tests for the stage-1 deck builder.

Data-free tests always run; the overfit canary needs the MSH deck data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from draftbot.data.dataset import PAD
from draftbot.models.builder import DeckBuilder, TorchDeckBuilder, build_deck

DECKS_PQ = Path("data/processed/MSH/decks.parquet")
SPLITS_JSON = Path("data/splits/MSH.json")
needs_data = pytest.mark.skipif(
    not (DECKS_PQ.exists() and SPLITS_JSON.exists()),
    reason="MSH deck data not built")


def tiny_model(n_cards=30, f=12, emb=32, heads=4, blocks=2, seed=0):
    torch.manual_seed(seed)
    return DeckBuilder(torch.randn(n_cards + 1, f), emb_dim=emb, heads=heads,
                       blocks=blocks, dropout=0.0)


def test_pool_permutation_equivariance():
    m = tiny_model().eval()
    ids = torch.tensor([[3, 7, 9, 12, 15, PAD, PAD]])
    cnt = torch.tensor([[1, 2, 1, 1, 3, 0, 0]])
    with torch.no_grad():
        mem1, land1, bas1 = m(ids, cnt, ids == PAD)
    perm = torch.tensor([4, 0, 2, 1, 3])
    ids2, cnt2 = ids.clone(), cnt.clone()
    ids2[0, :5], cnt2[0, :5] = ids[0, perm], cnt[0, perm]
    with torch.no_grad():
        mem2, land2, bas2 = m(ids2, cnt2, ids2 == PAD)
    assert torch.allclose(mem2[0, :5], mem1[0, perm], atol=1e-4)
    assert torch.allclose(land1, land2, atol=1e-4)
    assert torch.allclose(bas1, bas2, atol=1e-4)


def test_padding_inert():
    m = tiny_model().eval()
    ids1 = torch.tensor([[3, 7, 9]])
    cnt1 = torch.tensor([[1, 2, 1]])
    ids2 = torch.tensor([[3, 7, 9, PAD, PAD, PAD]])
    cnt2 = torch.tensor([[1, 2, 1, 0, 0, 0]])
    with torch.no_grad():
        mem1, land1, bas1 = m(ids1, cnt1, ids1 == PAD)
        mem2, land2, bas2 = m(ids2, cnt2, ids2 == PAD)
    assert torch.allclose(mem1[0], mem2[0, :3], atol=1e-4)
    assert torch.allclose(land1, land2, atol=1e-4)
    assert torch.allclose(bas1, bas2, atol=1e-4)
    assert (mem2[0, 3:] < -1e8).all()  # padded slots forced off


@pytest.mark.parametrize("decode", ["greedy", "expected"])
def test_decode_always_legal(decode):
    rng = np.random.default_rng(7)
    is_land = rng.random(60) < 0.15
    for trial in range(200):
        n = int(rng.integers(3, 45))
        ids = rng.choice(60, n, replace=False).astype(np.int16)
        counts = rng.integers(1, 4, n).astype(np.int16)
        d = build_deck(rng.random(n), rng.random(7), rng.random(5),
                       ids, counts, is_land, decode=decode)
        total = sum(d["deck"].values()) + int(np.sum(d["basics"]))
        assert total == 40, f"trial {trial}: {total} cards"
        pool = dict(zip(ids.tolist(), counts.tolist()))
        assert all(c <= pool[k] for k, c in d["deck"].items())
        assert (np.asarray(d["basics"]) >= 0).all()


def test_loss_finite_without_stats():
    """Zeroed feature table (the all-masked day-0 condition) must give a
    finite loss and finite gradients."""
    from draftbot.data.deck_dataset import DeckArrays
    from draftbot.train.decks import deck_loss, deck_targets

    torch.manual_seed(0)
    m = DeckBuilder(torch.zeros(31, 12), emb_dim=32, heads=4, blocks=2)
    rng = np.random.default_rng(0)
    n, P = 8, 20
    pool_ids = np.full((n, P), PAD, dtype=np.int16)
    pool_counts = np.zeros((n, P), dtype=np.int16)
    deck_counts = np.zeros((n, P), dtype=np.int16)
    for i in range(n):
        k = int(rng.integers(5, P))
        pool_ids[i, :k] = rng.choice(30, k, replace=False)
        pool_counts[i, :k] = rng.integers(1, 3, k)
        deck_counts[i, :k] = rng.integers(0, pool_counts[i, :k] + 1)
    arr = DeckArrays(pool_ids=pool_ids, pool_counts=pool_counts,
                     deck_counts=deck_counts,
                     basics=rng.integers(0, 8, (n, 5)).astype(np.int16),
                     meta=pd.DataFrame({"n_wins": rng.integers(0, 8, n)}))
    targets = deck_targets(arr, np.zeros(30, dtype=bool))
    loss = deck_loss(m, arr, targets, np.ones(n, np.float32),
                     np.arange(n), torch.device("cpu"), {})
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters()
               if p.grad is not None)


@needs_data
def test_overfit_canary():
    """200 real builds memorized to F1 ≥ 0.80 in a few hundred CPU steps
    proves the wiring end to end (loss → heads → decode)."""
    from draftbot.data.deck_dataset import land_flags, load_deck_arrays
    from draftbot.eval.decks import _f1, _true_build, _with_basics
    from draftbot.train.decks import deck_loss, deck_targets

    cards = pd.read_parquet("data/processed/MSH/cards.parquet")
    arr = load_deck_arrays("MSH", view="most_played")
    keep = np.arange(200)
    arr = type(arr)(pool_ids=arr.pool_ids[keep], pool_counts=arr.pool_counts[keep],
                    deck_counts=arr.deck_counts[keep], basics=arr.basics[keep],
                    meta=arr.meta.iloc[keep].reset_index(drop=True))
    is_land = land_flags(cards)
    torch.manual_seed(0)
    feats = torch.randn(len(cards) + 1, 16)  # random ids-as-features suffice
    model = DeckBuilder(feats, emb_dim=64, heads=4, blocks=2, dropout=0.0)
    targets = deck_targets(arr, is_land)
    w = np.ones(arr.n_builds, np.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    dev = torch.device("cpu")
    for step in range(300):
        opt.zero_grad(set_to_none=True)
        loss = deck_loss(model, arr, targets, w, np.arange(arr.n_builds),
                         dev, {})
        loss.backward()
        opt.step()
    builder = TorchDeckBuilder(model, "canary", dev, is_land)
    builds = builder.build_all(arr)
    f1 = np.mean([_f1(_with_basics(p["deck"], p["basics"]),
                      _with_basics(*_true_build(arr, i)))
                  for i, p in enumerate(builds)])
    assert f1 >= 0.80, f"canary F1 {f1:.3f}"
