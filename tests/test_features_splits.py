"""P0.T6/P0.T7 acceptance tests. Scaler/mask tests are pure-unit (always run);
snapshot/split integration tests skip when MSH data isn't built."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draftbot.data.features import apply_scaler, fit_scaler, load_scaler, save_scaler

DATA = Path("data")
needs_data = pytest.mark.skipif(
    not (DATA / "processed/MSH/draft.PremierDraft.parquet").exists(),
    reason="MSH data not built",
)


def test_scaler_roundtrip(tmp_path):
    df = pd.DataFrame({"cmc": [1.0, 3.0, 5.0], "flag": [0.0, 1.0, 0.0]})
    scaler = fit_scaler(df)
    save_scaler(scaler, tmp_path / "s.json")
    reloaded = load_scaler(tmp_path / "s.json")
    a = apply_scaler(df, scaler, add_masks=False)
    b = apply_scaler(df, reloaded, add_masks=False)
    pd.testing.assert_frame_equal(a, b)
    assert abs(a["cmc"].mean()) < 1e-9  # z-scored
    assert (a["flag"] == df["flag"]).all()  # binary passthrough


def test_masks_flag_genuinely_missing():
    df = pd.DataFrame({"wr": [0.5, np.nan, 0.6], "seen": [1.0, 2.0, np.nan]})
    scaler = fit_scaler(df)
    out = apply_scaler(df, scaler, add_masks=True)
    assert out["wr__miss"].tolist() == [0.0, 1.0, 0.0]
    assert out["seen__miss"].tolist() == [0.0, 0.0, 1.0]
    # missing values are exactly 0 after scaling, NOT a fake "0% win rate"
    assert out.loc[1, "wr"] == 0.0
    # ...while a real value of 0 would z-score to nonzero
    assert out.loc[0, "wr"] != 0.0


@needs_data
def test_splits_stable_and_disjoint():
    from draftbot.data.splits import build_splits

    s1 = build_splits("MSH")
    s2 = build_splits("MSH")  # must load persisted file, not regenerate
    assert s1 == s2
    r = s1["random"]
    assert not (set(r["train"]) & set(r["val"]))
    assert not (set(r["train"]) & set(r["test"]))
    assert not (set(r["val"]) & set(r["test"]))
    total = len(r["train"]) + len(r["val"]) + len(r["test"])
    assert 0.88 < len(r["train"]) / total < 0.92
    t = s1["temporal"]
    assert not (set(t["train"]) & (set(t["val"]) | set(t["test"])))
    assert s1["expert_subset"]["min_games"] >= 50
    assert s1["low_skill_subset"]["min_games"] >= 50


@needs_data
def test_snapshots_exist_and_differ():
    reg = json.loads((DATA / "snapshots/MSH/registry.json").read_text())
    assert "full" in reg and "week1" in reg
    from draftbot.data.features import load_snapshot

    full = load_snapshot("MSH", "full")
    week1 = load_snapshot("MSH", "week1")
    assert full.shape == week1.shape
    # week1 has less data: strictly fewer games recorded
    gc = [c for c in full.columns if c == "game_count"]
    assert full[gc[0]].sum() > week1[gc[0]].sum()
