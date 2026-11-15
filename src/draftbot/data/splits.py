"""Split builder (PLAN P0.T7).

- Random split: deterministic sha1(draft_id) mod 100 → train/val/test = 90/5/5.
  No RNG, so regeneration is impossible to get wrong; still persisted once to
  data/splits/<SET>.json and loaded from there ever after (guardrail).
- Temporal split: train = first 80% of draft days, val/test = alternating halves
  of the remaining days (deployment-realistic variant).
- Expert subset (test-split picks): user_win_rate ≥ 0.62 AND user_n_games ≥ 100;
  thresholds relaxed automatically if the subset lands under 50k picks.
- Low-skill subset (owner requirement, 2026-07-28 journal): experienced-but-losing
  drafters — user_win_rate ≤ low_wr AND user_n_games ≥ 100 — used by the eval
  harness to verify models agree with good drafters MORE than with bad ones.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from draftbot.data.cards import PROCESSED_DIR

SPLITS_DIR = Path("data/splits")


def _bucket(draft_id: str) -> int:
    return int(hashlib.sha1(draft_id.encode()).hexdigest(), 16) % 100


def _pick_threshold(picks: pd.DataFrame, expert: bool) -> tuple[float, int]:
    """Headline thresholds, relaxed until the subset holds ≥ 50k picks."""
    ladder = ([(0.62, 100), (0.60, 100), (0.58, 50)] if expert
              else [(0.46, 100), (0.48, 100), (0.50, 50)])
    for wr, games in ladder:
        cond = (picks["user_win_rate"] >= wr) if expert else (picks["user_win_rate"] <= wr)
        n = int((cond & (picks["user_n_games"] >= games)).sum())
        if n >= 50_000:
            return wr, games
    return ladder[-1]


def build_splits(set_code: str, event: str = "PremierDraft") -> dict:
    out_path = SPLITS_DIR / f"{set_code}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    df = pd.read_parquet(PROCESSED_DIR / set_code / f"draft.{event}.parquet",
                         columns=["draft_id", "draft_time", "user_win_rate",
                                  "user_n_games"])
    ids = df["draft_id"].unique().tolist()
    buckets = {i: _bucket(i) for i in ids}
    random = {
        "train": [i for i in ids if buckets[i] < 90],
        "val": [i for i in ids if 90 <= buckets[i] < 95],
        "test": [i for i in ids if buckets[i] >= 95],
    }

    first = df.groupby("draft_id")["draft_time"].first()
    days = first.dt.normalize()
    cutoff = days.quantile(0.8)
    late_ids = sorted(days[days > cutoff].index.tolist())
    temporal = {
        "train": sorted(days[days <= cutoff].index.tolist()),
        "val": [i for i in late_ids if buckets[i] < 50],
        "test": [i for i in late_ids if buckets[i] >= 50],
    }

    test_picks = df[df["draft_id"].isin(set(random["test"]))]
    exp_wr, exp_games = _pick_threshold(test_picks, expert=True)
    low_wr, low_games = _pick_threshold(test_picks, expert=False)

    splits = {
        "set": set_code, "event": event, "method": "sha1(draft_id) mod 100",
        "random": random, "temporal": temporal,
        "expert_subset": {"min_win_rate": exp_wr, "min_games": exp_games},
        "low_skill_subset": {"max_win_rate": low_wr, "min_games": low_games},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits))
    return splits


def build_sealed_splits(set_code: str) -> dict:
    """Sealed draft_ids are a separate universe from draft draft_ids, so sealed
    decks get their own split file (same sha1 bucketing, persisted once)."""
    out_path = SPLITS_DIR / f"{set_code}.sealed.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    from draftbot.data.deck_dataset import deck_parquet_path
    df = pd.read_parquet(deck_parquet_path(set_code, "sealed"),
                         columns=["draft_id"])
    ids = df["draft_id"].unique().tolist()
    buckets = {i: _bucket(i) for i in ids}
    splits = {
        "set": set_code, "event": "sealed", "method": "sha1(draft_id) mod 100",
        "random": {
            "train": [i for i in ids if buckets[i] < 90],
            "val": [i for i in ids if 90 <= buckets[i] < 95],
            "test": [i for i in ids if buckets[i] >= 95],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits))
    return splits


def load_splits(set_code: str, source: str = "draft") -> dict:
    suffix = ".json" if source == "draft" else f".{source}.json"
    return json.loads((SPLITS_DIR / f"{set_code}{suffix}").read_text())


def split_picks(picks: pd.DataFrame, splits: dict, part: str,
                scheme: str = "random") -> pd.DataFrame:
    return picks[picks["draft_id"].isin(set(splits[scheme][part]))]
