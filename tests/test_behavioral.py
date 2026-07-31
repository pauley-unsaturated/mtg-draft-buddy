"""P1.T3 — behavioral fixture suite. Scenario JSONs are frozen; never edit them
to make a test pass. Easy tier must pass for gih-greedy; medium/hard tiers are
xfail until a trained model clears them (Phase-2 gate re-runs these)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCEN_DIR = Path(__file__).parent / "fixtures" / "scenarios"
DATA_OK = Path("data/processed/MSH/cards.parquet").exists()

needs_data = pytest.mark.skipif(not DATA_OK, reason="MSH data not built")

T, P = 42, 14


def load_scenarios():
    return sorted(SCEN_DIR.glob("*.json"), key=lambda p: p.name)


def run_scenario(scorer, scen: dict, name_to_id: dict) -> tuple[bool, list[str]]:
    from draftbot.data.dataset import PAD

    def nid(n):
        return name_to_id[n.lower()]

    pos = scen["position"]
    packs = np.full((1, T, P), PAD, dtype=np.int16)
    filler = nid("Plains")
    packs[0, :, 0] = filler  # every step needs ≥1 card
    pack_ids = [nid(n) for n in scen["pack"]]
    packs[0, pos, :] = PAD
    packs[0, pos, : len(pack_ids)] = pack_ids
    prev = np.full((1, T), PAD, dtype=np.int16)
    pool_ids = [nid(n) for n in scen["pool"]]
    for i, cid in enumerate(pool_ids):
        prev[0, i + 1] = cid
    if pos + 1 < T:
        prev[0, pos + 1:] = filler
    scores = scorer.score_drafts(packs, prev)[0, pos]
    order = np.argsort(-scores)
    ranked_ids = []
    for slot in order:
        cid = packs[0, pos, slot]
        if cid != PAD and cid not in ranked_ids:
            ranked_ids.append(int(cid))
    k = scen["expect"]["k"]
    want = {nid(n) for n in scen["expect"]["any_of"]}
    id_to_name = {v: k2 for k2, v in name_to_id.items()}
    topk_names = [id_to_name[c] for c in ranked_ids[:k]]
    return bool(want & set(ranked_ids[:k])), topk_names


@pytest.fixture(scope="module")
def gih_scorer():
    from draftbot.data.features import load_snapshot
    from draftbot.eval.bots import build_bots

    cards = pd.read_parquet("data/processed/MSH/cards.parquet")
    static = pd.read_parquet("data/processed/MSH/features.static.parquet")
    snap = load_snapshot("MSH", "full")
    return {b.name: b for b in build_bots(cards, static, snap)}["gih-greedy"]


@pytest.fixture(scope="module")
def name_map():
    from draftbot.data.cards import name_to_id

    cards = pd.read_parquet("data/processed/MSH/cards.parquet")
    return name_to_id(cards)


@needs_data
@pytest.mark.parametrize("path", load_scenarios(), ids=lambda p: p.stem)
def test_easy_tier_gih_greedy(path, gih_scorer, name_map):
    scen = json.loads(path.read_text())
    if scen["tier"] != "easy":
        pytest.skip("gih-greedy only guarantees the easy tier")
    ok, topk = run_scenario(gih_scorer, scen, name_map)
    assert ok, f"{scen['name']}: got {topk}; {scen['rationale']}"


def check_model_fixtures(scorer, tiers=("easy", "medium"), verbose=False):
    """Reusable entry point for evaluating any model on the fixture suite
    (used by Phase-2 gate and the eval CLI, not just pytest)."""
    from draftbot.data.cards import name_to_id

    cards = pd.read_parquet("data/processed/MSH/cards.parquet")
    nm = name_to_id(cards)
    results = {}
    for path in load_scenarios():
        scen = json.loads(path.read_text())
        if scen["tier"] not in tiers:
            continue
        ok, topk = run_scenario(scorer, scen, nm)
        results[scen["name"]] = {"pass": ok, "topk": topk, "tier": scen["tier"]}
        if verbose and not ok:
            print(f"FAIL {scen['name']}: top-k {topk}\n  {scen['rationale']}")
    return results
