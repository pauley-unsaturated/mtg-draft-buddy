"""Deck-example extraction for the deckbuilder (PLAN P5.T1).

game_data → data/processed/<SET>/decks.parquet, one row per (draft_id,
build_index): sparse nonbasic deck/sideboard counts, per-color basic counts,
games/wins for win-weighting. Pool = deck ∪ sideboard (nonbasics); membership
label = deck counts. Nonbasic lands are ordinary cards here — the model learns
which fixing belongs in which deck (see journal: the Subterranean Cavern
incident).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from draftbot.data.cards import PROCESSED_DIR, build_cards, name_to_id, norm_name
from draftbot.data.fetch import dest_path

BASIC_ORDER = ["plains", "island", "swamp", "mountain", "forest"]  # → W U B R G

SCHEMA = pa.schema([
    ("draft_id", pa.string()),
    ("build_index", pa.int8()),
    ("deck_ids", pa.list_(pa.int16())),
    ("deck_counts", pa.list_(pa.int16())),
    ("side_ids", pa.list_(pa.int16())),
    ("side_counts", pa.list_(pa.int16())),
    ("basics", pa.list_(pa.int16())),      # W U B R G counts in the deck
    ("n_games", pa.int16()),
    ("n_wins", pa.int16()),
    ("user_win_rate", pa.float32()),
])


def _sparse(mat: np.ndarray, vids: np.ndarray) -> tuple[list, list]:
    """Per-row (ids, counts) of the nonzero entries of a builds×cards matrix."""
    ids, counts = [], []
    for row in mat:
        nz = np.nonzero(row)[0]
        ids.append(vids[nz].tolist())
        counts.append(row[nz].tolist())
    return ids, counts


def extract(set_code: str, event: str = "PremierDraft") -> Path:
    cards = build_cards(set_code, event)
    nti = name_to_id(cards)
    basic_vids = {nti[b] for b in BASIC_ORDER if b in nti}
    csv_path = dest_path(set_code, event, kind="game")
    out_path = PROCESSED_DIR / set_code / "decks.parquet"

    header = pd.read_csv(csv_path, nrows=0).columns
    deck_cols = [c for c in header if c.startswith("deck_")]
    side_cols = [c for c in header if c.startswith("sideboard_")]
    deck_vids = np.array([nti[norm_name(c[len("deck_"):])] for c in deck_cols],
                         dtype=np.int16)
    side_vids = np.array([nti[norm_name(c[len("sideboard_"):])] for c in side_cols],
                         dtype=np.int16)

    usecols = ["draft_id", "build_index", "won", "user_game_win_rate_bucket"] \
        + deck_cols + side_cols
    dtypes = {c: "float32" for c in deck_cols + side_cols}
    dtypes["won"] = "bool"  # fail loudly if the column ever isn't True/False
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)
    df["build_index"] = df["build_index"].fillna(0).astype(np.int8)

    # Deck vectors are identical within a (draft, build): aggregate games/wins,
    # then take one representative row per build for the card counts.
    gb = df.groupby(["draft_id", "build_index"], sort=True)
    stats = gb.agg(n_games=("won", "size"), n_wins=("won", "sum"),
                   user_win_rate=("user_game_win_rate_bucket", "first"))
    rep = df.drop_duplicates(["draft_id", "build_index"]) \
            .set_index(["draft_id", "build_index"]).reindex(stats.index)
    deck_mat = np.nan_to_num(rep[deck_cols].to_numpy(np.float32)).astype(np.int16)
    side_mat = np.nan_to_num(rep[side_cols].to_numpy(np.float32)).astype(np.int16)
    del df, rep

    basic_pos = [int(np.where(deck_vids == nti[b])[0][0]) for b in BASIC_ORDER]
    basics = deck_mat[:, basic_pos]  # W U B R G columns, in order
    deck_nb = np.array([v not in basic_vids for v in deck_vids])
    side_nb = np.array([v not in basic_vids for v in side_vids])
    deck_ids, deck_counts = _sparse(deck_mat[:, deck_nb], deck_vids[deck_nb])
    side_ids, side_counts = _sparse(side_mat[:, side_nb], side_vids[side_nb])

    table = pa.Table.from_pydict({
        "draft_id": [d for d, _ in stats.index],
        "build_index": [b for _, b in stats.index],
        "deck_ids": deck_ids, "deck_counts": deck_counts,
        "side_ids": side_ids, "side_counts": side_counts,
        "basics": basics.tolist(),
        "n_games": stats["n_games"].tolist(),
        "n_wins": stats["n_wins"].tolist(),
        "user_win_rate": stats["user_win_rate"].astype("float32").tolist(),
    }, schema=SCHEMA)
    pq.write_table(table, out_path, compression="zstd")
    print(f"{set_code}: {len(stats)} deck builds -> {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--set", dest="set_code", required=True)
    args = p.parse_args(argv)
    extract(args.set_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
