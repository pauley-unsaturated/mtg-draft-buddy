"""P5.T1 deck-dataset validation gate (docs/DECK_DATA_PLAN.md).

Five tests that must be green before any builder-model work. Skipped when MSH
data hasn't been built locally (CI has no data/); the gate requires them to
run and pass on the dev box.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path("data")
CARDS_PQ = DATA / "processed/MSH/cards.parquet"
DRAFT_PQ = DATA / "processed/MSH/draft.PremierDraft.parquet"
DECKS_PQ = DATA / "processed/MSH/decks.parquet"
SPLITS_JSON = DATA / "splits/MSH.json"

needs_data = pytest.mark.skipif(
    not (CARDS_PQ.exists() and DRAFT_PQ.exists() and DECKS_PQ.exists()
         and SPLITS_JSON.exists()),
    reason="MSH deck data not built",
)


@pytest.fixture(scope="module")
def cards():
    return pd.read_parquet(CARDS_PQ)


@pytest.fixture(scope="module")
def decks():
    return pd.read_parquet(DECKS_PQ)


def _pool(row) -> Counter:
    """Nonbasic pool of a build = multiset(deck ∪ sideboard)."""
    pool = Counter(dict(zip((int(i) for i in row["deck_ids"]),
                            (int(c) for c in row["deck_counts"]))))
    pool.update(dict(zip((int(i) for i in row["side_ids"]),
                         (int(c) for c in row["side_counts"]))))
    return pool


@needs_data
def test_1_pool_identity_vs_draft_log(decks, cards):
    """multiset(deck ∪ side nonbasics) == multiset(picks minus drafted basics)
    for ≥100 sampled drafts. Catches column-mapping and vocab drift."""
    draft = pd.read_parquet(DRAFT_PQ, columns=["draft_id", "pick"])
    basics = set(cards.loc[cards["is_basic"] == 1, "id"])
    first_builds = decks.drop_duplicates("draft_id").set_index("draft_id")
    shared = sorted(set(first_builds.index) & set(draft["draft_id"].unique()))
    assert len(shared) >= 100, "too few drafts shared between game and draft data"
    sample = shared[:: max(1, len(shared) // 150)][:150]  # deterministic spread

    picks = draft[draft["draft_id"].isin(set(sample))]
    picks_by_draft = picks.groupby("draft_id")["pick"].apply(list)
    mismatched = []
    for did in sample:
        want = Counter(int(p) for p in picks_by_draft[did] if int(p) not in basics)
        if _pool(first_builds.loc[did]) != want:
            mismatched.append(did)
    assert not mismatched, (
        f"{len(mismatched)}/{len(sample)} pools mismatch draft log, "
        f"e.g. {mismatched[:3]}")


@needs_data
def test_2_deck_size_sanity(decks):
    """nonbasics + basics ∈ [40, 43] for ≥99% of builds."""
    totals = (decks["deck_counts"].apply(lambda c: int(np.sum(c)))
              + decks["basics"].apply(lambda b: int(np.sum(b))))
    ok = totals.between(40, 43)
    assert ok.mean() >= 0.99, (
        f"only {ok.mean():.2%} of builds sized 40-43; "
        f"range seen {totals.min()}-{totals.max()}")


@needs_data
def test_3_membership_labels_valid(decks, cards):
    """deck_counts ≤ pool counts per card, everywhere; ids in-vocab, positive
    counts, and no basics hiding in the sparse nonbasic arrays."""
    basics = set(cards.loc[cards["is_basic"] == 1, "id"])
    vocab = len(cards)
    for _, row in decks.iterrows():
        pool = _pool(row)
        for i, c in zip(row["deck_ids"], row["deck_counts"]):
            assert 0 <= i < vocab and int(i) not in basics
            assert 1 <= c <= pool[int(i)]
        for i, c in zip(row["side_ids"], row["side_counts"]):
            assert 0 <= i < vocab and int(i) not in basics
            assert c >= 1


@needs_data
def test_4_rebuild_pool_integrity(decks):
    """All builds within a draft share the same nonbasic pool."""
    multi = decks[decks.duplicated("draft_id", keep=False)]
    bad = []
    for did, g in multi.groupby("draft_id"):
        pools = [_pool(row) for _, row in g.iterrows()]
        if any(p != pools[0] for p in pools[1:]):
            bad.append(did)
    assert not bad, f"{len(bad)} drafts whose rebuilds disagree on pool, e.g. {bad[:3]}"


@needs_data
def test_5_split_hygiene(decks):
    """Decks join data/splits/MSH.json by draft_id — the SAME splits as the
    draft model, so a test-split draft is never trained on by either model."""
    splits = json.loads(SPLITS_JSON.read_text())["random"]
    part = {i: p for p, ids in splits.items() for i in ids}
    assert len(part) == sum(len(ids) for ids in splits.values()), "splits overlap"

    draft_ids = set(pd.read_parquet(DRAFT_PQ, columns=["draft_id"])["draft_id"])
    in_draft = decks["draft_id"].isin(draft_ids)
    mapped = decks["draft_id"].map(part)
    # every deck whose draft the draft model knows must resolve to a split
    assert mapped[in_draft].notna().all()
    # the deck dataset must actually be covered by the shared splits
    assert mapped.notna().mean() >= 0.95, (
        f"only {mapped.notna().mean():.2%} of builds join the split file")
    # and the joined fractions must look like the 90/5/5 draft-id split
    frac = mapped.value_counts(normalize=True)
    assert 0.85 <= frac["train"] <= 0.95
    assert 0.03 <= frac["val"] <= 0.07 and 0.03 <= frac["test"] <= 0.07
