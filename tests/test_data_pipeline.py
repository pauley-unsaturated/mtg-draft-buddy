"""P0.T4/P0.T5 acceptance tests. Skipped when MSH data hasn't been built locally
(CI has no data/); the Phase-0 gate requires them to run and pass on the dev box."""

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path("data")
CARDS_PQ = DATA / "processed/MSH/cards.parquet"
DRAFT_PQ = DATA / "processed/MSH/draft.PremierDraft.parquet"
RAW_CSV = DATA / "raw/draft_data_public.MSH.PremierDraft.csv.gz"

needs_data = pytest.mark.skipif(
    not (CARDS_PQ.exists() and DRAFT_PQ.exists() and RAW_CSV.exists()),
    reason="MSH data not built",
)


@pytest.fixture(scope="module")
def cards():
    return pd.read_parquet(CARDS_PQ)


@pytest.fixture(scope="module")
def draft():
    return pd.read_parquet(DRAFT_PQ)


@needs_data
def test_vocab_exactly_339_stable_ids(cards):
    assert len(cards) == 339
    assert list(cards["id"]) == list(range(339))
    assert list(cards["name"]) == sorted(cards["name"])  # id order = sorted-name order
    assert cards["is_basic"].sum() == 5


@needs_data
def test_every_csv_pick_resolves(cards):
    """Zero KeyErrors over the FULL raw pick column (P0.T5 DoD)."""
    from draftbot.data.cards import name_to_id, norm_name

    nti = name_to_id(cards)
    with gzip.open(RAW_CSV, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        pick_col = header.index("pick")
        unresolved = set()
        for row in reader:
            if norm_name(row[pick_col]) not in nti:
                unresolved.add(row[pick_col])
    assert not unresolved, f"unresolved picks: {sorted(unresolved)[:10]}"


@needs_data
def test_draft_shape_and_dtypes(draft):
    assert draft["position"].max() == 41
    per_draft = draft.groupby("draft_id")["position"].count()
    assert (per_draft == 42).all()
    assert draft["user_n_games"].max() >= 500  # int8 would have overflowed here
    assert draft["rank"].str.islower().all()


@needs_data
def test_pool_reconstruction_roundtrip(draft, cards):
    """Pool is not stored; verify it's exactly reconstructable from prior picks
    by comparing against the raw CSV pool_ columns for a sampled draft."""
    rng = np.random.default_rng(0)
    target_ids = rng.choice(draft["draft_id"].unique(), size=3, replace=False)
    sub = draft[draft["draft_id"].isin(target_ids)]

    from draftbot.data.cards import name_to_id, norm_name

    nti = name_to_id(cards)
    raw_rows = {did: {} for did in target_ids}
    with gzip.open(RAW_CSV, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["draft_id"] in raw_rows:
                pos = int(row["pack_number"]) * 14 + int(row["pick_number"])
                pool = {}
                for k, v in row.items():
                    if k.startswith("pool_") and v not in ("", "0"):
                        pool[nti[norm_name(k[len("pool_"):])]] = int(v)
                raw_rows[row["draft_id"]][pos] = pool

    for did in target_ids:
        g = sub[sub["draft_id"] == did].sort_values("position")
        running: dict[int, int] = {}
        for _, r in g.iterrows():
            expected = raw_rows[did][r["position"]]
            assert running == expected, f"pool mismatch at {did} pos {r['position']}"
            running[r["pick"]] = running.get(r["pick"], 0) + 1


@needs_data
def test_pick_always_in_pack(draft):
    sample = draft.sample(50_000, random_state=0)
    ok = [p in set(pc) for p, pc in zip(sample["pick"], sample["pack_cards"])]
    assert all(ok)


@needs_data
def test_pack_sizes_follow_position(draft):
    """Play booster: 14 fresh cards at pick 0 of each pack, decreasing by 1."""
    sample = draft.sample(20_000, random_state=1)
    sizes = sample["pack_cards"].apply(len).to_numpy()
    picks_in_pack = (sample["position"] % 14).to_numpy()
    assert ((14 - picks_in_pack) == sizes).mean() > 0.99
