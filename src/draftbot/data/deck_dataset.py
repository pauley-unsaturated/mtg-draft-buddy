"""Tensorization of decks.parquet into fixed-shape pool/deck arrays (P5.T1d).

Per build (nonbasics only — basics live in their own 5-vector):

  pool_ids    (N, P) int16  distinct card ids in pool (deck ∪ side), padded -1
  pool_counts (N, P) int16  copies in pool
  deck_counts (N, P) int16  copies in the maindeck (0..pool_count per slot)
  basics      (N, 5) int16  W U B R G counts in the maindeck
  meta        DataFrame     draft_id, build_index, n_games, n_wins, user_win_rate

Train mode keeps every build (rebuilds are extra signal); eval mode keeps the
most-played build per draft (ties → lowest build_index), per DECK_DATA_PLAN.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.dataset import PAD


def deck_parquet_path(set_code: str, source: str = "draft"):
    """source: 'draft' (PremierDraft decks) | 'sealed' (Sealed ∪ TradSealed)."""
    if source not in ("draft", "sealed"):
        raise ValueError(f"unknown deck source {source!r}")
    name = "decks.parquet" if source == "draft" else "decks.sealed.parquet"
    return PROCESSED_DIR / set_code / name


@dataclass
class DeckArrays:
    pool_ids: np.ndarray
    pool_counts: np.ndarray
    deck_counts: np.ndarray
    basics: np.ndarray
    meta: pd.DataFrame

    @property
    def n_builds(self) -> int:
        return self.pool_ids.shape[0]

    @property
    def max_pool(self) -> int:
        return self.pool_ids.shape[1]

    def take(self, idx) -> "DeckArrays":
        """Row subset by boolean mask or integer indices."""
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        return DeckArrays(
            pool_ids=self.pool_ids[idx], pool_counts=self.pool_counts[idx],
            deck_counts=self.deck_counts[idx], basics=self.basics[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True))


def most_played(df: pd.DataFrame) -> pd.DataFrame:
    """One build per draft: most games; ties broken by lowest build_index."""
    return (df.sort_values(["draft_id", "n_games", "build_index"],
                           ascending=[True, False, True], kind="mergesort")
            .drop_duplicates("draft_id"))


def winningest(df: pd.DataFrame) -> pd.DataFrame:
    """One build per draft: the deck that DID the winning (owner refinement
    2026-07-30: trophy eval targets the build that trophied, not the
    most-played one). Ties → most games, then lowest build_index."""
    return (df.sort_values(["draft_id", "n_wins", "n_games", "build_index"],
                           ascending=[True, False, False, True],
                           kind="mergesort")
            .drop_duplicates("draft_id"))


def load_deck_arrays(set_code: str, draft_ids: set[str] | None = None,
                     view: str = "all", source: str = "draft") -> DeckArrays:
    """view: 'all' builds (training) | 'most_played' (typical-play eval) |
    'winningest' (trophy eval)."""
    df = pd.read_parquet(deck_parquet_path(set_code, source))
    if draft_ids is not None:
        df = df[df["draft_id"].isin(draft_ids)]
    if view == "most_played":
        df = most_played(df)
    elif view == "winningest":
        df = winningest(df)
    elif view != "all":
        raise ValueError(f"unknown view {view!r}")
    return arrays_from_deck_df(df)


def arrays_from_deck_df(df: pd.DataFrame) -> DeckArrays:
    df = df.sort_values(["draft_id", "build_index"]).reset_index(drop=True)
    n = len(df)
    pools, decks = [], []
    for deck_ids, deck_cnt, side_ids, side_cnt in zip(
            df["deck_ids"], df["deck_counts"], df["side_ids"], df["side_counts"]):
        dmap = dict(zip(deck_ids, deck_cnt))
        ids = np.union1d(deck_ids, side_ids).astype(np.int16)
        smap = dict(zip(side_ids, side_cnt))
        pools.append((ids, np.array([dmap.get(i, 0) + smap.get(i, 0) for i in ids],
                                    dtype=np.int16)))
        decks.append(np.array([dmap.get(i, 0) for i in ids], dtype=np.int16))

    P = max((len(ids) for ids, _ in pools), default=0)
    pool_ids = np.full((n, P), PAD, dtype=np.int16)
    pool_counts = np.zeros((n, P), dtype=np.int16)
    deck_counts = np.zeros((n, P), dtype=np.int16)
    for i, ((ids, cnts), dcnts) in enumerate(zip(pools, decks)):
        pool_ids[i, : len(ids)] = ids
        pool_counts[i, : len(ids)] = cnts
        deck_counts[i, : len(ids)] = dcnts

    basics = np.stack(df["basics"].to_numpy()).astype(np.int16)
    meta = df[["draft_id", "build_index", "n_games", "n_wins",
               "user_win_rate"]].reset_index(drop=True)
    return DeckArrays(pool_ids=pool_ids, pool_counts=pool_counts,
                      deck_counts=deck_counts, basics=basics, meta=meta)


def soft_skill(user_win_rate: np.ndarray) -> np.ndarray:
    """Smooth skill ramp (no hard filter — EXP-013 lesson): sigmoid centered at
    0.50 WR, width 0.04 → 0.42→0.12, 0.54→0.73, 0.62→0.95. Missing WR → 0.5."""
    wr = np.nan_to_num(np.asarray(user_win_rate, dtype=np.float64), nan=0.5)
    return 1.0 / (1.0 + np.exp(-(wr - 0.50) / 0.04))


def example_weights(meta: pd.DataFrame) -> np.ndarray:
    """DECK_DATA_PLAN curation: (1 + n_wins) · soft_skill(user_win_rate)."""
    w = (1.0 + meta["n_wins"].to_numpy(np.float64)) \
        * soft_skill(meta["user_win_rate"].to_numpy(np.float64))
    return w.astype(np.float32)


def land_flags(cards: pd.DataFrame) -> np.ndarray:
    """(n_cards,) bool — front-face type line contains Land (incl. nonbasics)."""
    return cards["type_line"].fillna("").str.contains("Land").to_numpy()
