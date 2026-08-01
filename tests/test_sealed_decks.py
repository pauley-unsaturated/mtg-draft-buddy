"""Sealed deck-dataset validation gate (sealed analogue of test_decks.py).

Same five checks per set with sealed data built, with one swap: there is no
draft log in sealed — the pool IS deck ∪ sideboard by construction — so the
pool-identity test becomes a pool-size sanity check (a 6-booster sealed pool
that isn't ~84 nonbasics means column mapping drifted).
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path("data")
SETS = sorted(p.parent.name
              for p in DATA.glob("processed/*/decks.sealed.parquet"))

pytestmark = pytest.mark.skipif(not SETS, reason="no sealed deck data built")


@pytest.fixture(scope="module", params=SETS or ["none"])
def sealed(request):
    code = request.param
    decks = pd.read_parquet(DATA / f"processed/{code}/decks.sealed.parquet")
    cards = pd.read_parquet(DATA / f"processed/{code}/cards.parquet")
    return code, decks, cards


def _pool(row) -> Counter:
    """Nonbasic pool of a build = multiset(deck ∪ sideboard)."""
    pool = Counter(dict(zip((int(i) for i in row["deck_ids"]),
                            (int(c) for c in row["deck_counts"]))))
    pool.update(dict(zip((int(i) for i in row["side_ids"]),
                         (int(c) for c in row["side_counts"]))))
    return pool


def test_1_pool_size_sanity(sealed):
    """Sealed pool ≈ 6 boosters of nonbasics: 55–110 for ≥99% of builds."""
    code, decks, _ = sealed
    totals = (decks["deck_counts"].apply(lambda c: int(np.sum(c)))
              + decks["side_counts"].apply(lambda c: int(np.sum(c))))
    ok = totals.between(55, 110)
    assert ok.mean() >= 0.99, (
        f"{code}: only {ok.mean():.2%} of sealed pools sized 55-110; "
        f"range seen {totals.min()}-{totals.max()}")


def test_2_deck_size_sanity(sealed):
    """nonbasics + basics ∈ [40, 43] for ≥99% of builds."""
    code, decks, _ = sealed
    totals = (decks["deck_counts"].apply(lambda c: int(np.sum(c)))
              + decks["basics"].apply(lambda b: int(np.sum(b))))
    ok = totals.between(40, 43)
    assert ok.mean() >= 0.99, (
        f"{code}: only {ok.mean():.2%} of builds sized 40-43; "
        f"range seen {totals.min()}-{totals.max()}")


def test_3_membership_labels_valid(sealed):
    """deck_counts ≤ pool counts per card; ids in-vocab, positive counts, no
    basics in the sparse nonbasic arrays. Deterministic ≤2000-row subsample
    per set (the sweep runs this across the whole corpus)."""
    code, decks, cards = sealed
    basics = set(cards.loc[cards["is_basic"] == 1, "id"])
    vocab = len(cards)
    sample = decks.iloc[:: max(1, len(decks) // 2000)]
    for _, row in sample.iterrows():
        pool = _pool(row)
        for i, c in zip(row["deck_ids"], row["deck_counts"]):
            assert 0 <= i < vocab and int(i) not in basics, f"{code}: id {i}"
            assert 1 <= c <= pool[int(i)], f"{code}: count {c} of id {i}"
        for i, c in zip(row["side_ids"], row["side_counts"]):
            assert 0 <= i < vocab and int(i) not in basics, f"{code}: id {i}"
            assert c >= 1


def test_4_rebuild_pool_integrity(sealed):
    """All builds within one sealed entry share the same nonbasic pool."""
    code, decks, _ = sealed
    multi = decks[decks.duplicated("draft_id", keep=False)]
    bad = []
    for did, g in multi.groupby("draft_id"):
        pools = [_pool(row) for _, row in g.iterrows()]
        if any(p != pools[0] for p in pools[1:]):
            bad.append(did)
    assert not bad, (f"{code}: {len(bad)} sealed entries whose rebuilds "
                     f"disagree on pool, e.g. {bad[:3]}")


def test_5_split_hygiene(sealed):
    """data/splits/<SET>.sealed.json: disjoint partitions, 100% coverage of
    builds (splits are derived from the sealed decks themselves), ~90/5/5."""
    code, decks, _ = sealed
    path = DATA / f"splits/{code}.sealed.json"
    assert path.exists(), f"{code}: sealed splits not persisted"
    splits = json.loads(path.read_text())["random"]
    part = {i: p for p, ids in splits.items() for i in ids}
    assert len(part) == sum(len(ids) for ids in splits.values()), "overlap"

    mapped = decks["draft_id"].map(part)
    assert mapped.notna().all(), (
        f"{code}: {mapped.isna().sum()} builds missing from the split file")
    frac = mapped.value_counts(normalize=True)
    assert 0.85 <= frac["train"] <= 0.95
    assert 0.02 <= frac["val"] <= 0.08 and 0.02 <= frac["test"] <= 0.08
