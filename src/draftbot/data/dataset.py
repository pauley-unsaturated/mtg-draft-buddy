"""Tensorization of the per-pick parquet into fixed-shape draft arrays.

A set with t picks and pack size P (MSH: t=42, P=15... actually P=14 for play
boosters — computed from data, never hard-coded) becomes, per split:

  packs      (N, t, P) int16   card ids, padded with -1
  picks      (N, t)    int16   human pick (card id)
  prev_picks (N, t)    int16   pick at t-1, -1 at t=0 (the "bias" slot)
  positions  (t,)      int16   0..t-1 (same for every draft)
  meta       DataFrame (N,)    draft_id, rank, user_win_rate, user_n_games,
                               event_wins, event_losses, draft_time
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR

PAD = -1


@dataclass
class DraftArrays:
    packs: np.ndarray
    picks: np.ndarray
    prev_picks: np.ndarray
    meta: pd.DataFrame

    @property
    def n_drafts(self) -> int:
        return self.packs.shape[0]

    @property
    def t(self) -> int:
        return self.packs.shape[1]

    @property
    def max_pack(self) -> int:
        return self.packs.shape[2]


def load_draft_arrays(set_code: str, event: str = "PremierDraft",
                      draft_ids: set[str] | None = None,
                      parquet: Path | None = None) -> DraftArrays:
    path = parquet or PROCESSED_DIR / set_code / f"draft.{event}.parquet"
    df = pd.read_parquet(path)
    if draft_ids is not None:
        df = df[df["draft_id"].isin(draft_ids)]
    df = df.sort_values(["draft_id", "position"], kind="mergesort")
    t = int(df["position"].max()) + 1
    n = len(df) // t
    assert len(df) == n * t, "non-rectangular draft data"

    max_pack = int(df["pack_cards"].map(len).max())
    flat = df["pack_cards"].to_numpy()
    packs = np.full((n * t, max_pack), PAD, dtype=np.int16)
    for i, lst in enumerate(flat):
        packs[i, : len(lst)] = lst
    packs = packs.reshape(n, t, max_pack)
    picks = df["pick"].to_numpy(np.int16).reshape(n, t)

    # 17lands glitch rows exist (~1e-6) where the pick isn't among the pack
    # cards; one such row makes the masked NLL explode. Drop those drafts.
    ok = (packs == picks[..., None]).any(-1).all(axis=1)
    if not ok.all():
        keep_ids = df["draft_id"].unique()[ok]
        df = df[df["draft_id"].isin(set(keep_ids))]
        n = len(df) // t
        packs, picks = packs[ok], picks[ok]
        print(f"dropped {(~ok).sum()} drafts with pick-not-in-pack glitch rows")

    prev = np.full_like(picks, PAD)
    prev[:, 1:] = picks[:, :-1]

    meta = (df.groupby("draft_id", sort=True)
            .agg(rank=("rank", "first"), user_win_rate=("user_win_rate", "first"),
                 user_n_games=("user_n_games", "first"),
                 event_wins=("event_wins", "first"),
                 event_losses=("event_losses", "first"),
                 draft_time=("draft_time", "first"))
            .reset_index())
    assert (meta["draft_id"] == df["draft_id"].unique()).all()
    return DraftArrays(packs=packs, picks=picks, prev_picks=prev, meta=meta)


def pick_slot_indices(packs: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """(N, t) index of the human pick within the pack slots (first match)."""
    hits = packs == picks[..., None]
    assert hits.any(-1).all(), "pick not present in its pack"
    return hits.argmax(-1)


def pools_before(picks: np.ndarray, t_step: int) -> np.ndarray:
    """Card ids picked before step t_step: (N, t_step) slice of picks."""
    return picks[:, :t_step]
