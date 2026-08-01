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
import tarfile
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


def _open_game_csv(path: Path):
    """Old 17lands game files (e.g. AFR) are tar-wrapped inside the gzip;
    newer ones are plain csv.gz. Return something pandas can read either way."""
    try:
        tf = tarfile.open(path, "r:gz")
        member = next(m for m in tf.getmembers() if m.isfile())
        return tf.extractfile(member)
    except tarfile.ReadError:
        return path


def _sparse(mat: np.ndarray, vids: np.ndarray) -> tuple[list, list]:
    """Per-row (ids, counts) of the nonzero entries of a builds×cards matrix."""
    ids, counts = [], []
    for row in mat:
        nz = np.nonzero(row)[0]
        ids.append(vids[nz].tolist())
        counts.append(row[nz].tolist())
    return ids, counts


def _col_vids(cols: list[str], prefix: str, nti: dict,
              label: str) -> np.ndarray:
    names = [norm_name(c[len(prefix):]) for c in cols]
    missing = sorted({n for n in names if n not in nti})
    if missing:
        raise KeyError(f"{label}: {len(missing)} {prefix}column names not in "
                       f"cards.parquet vocab, e.g. {missing[:10]}")
    return np.array([nti[n] for n in names], dtype=np.int16)


def _event_table(set_code: str, event: str, cards: pd.DataFrame) -> pa.Table:
    nti = name_to_id(cards)
    basic_vids = {nti[b] for b in BASIC_ORDER if b in nti}
    csv_path = dest_path(set_code, event, kind="game")

    header = pd.read_csv(_open_game_csv(csv_path), nrows=0).columns
    deck_cols = [c for c in header if c.startswith("deck_")]
    side_cols = [c for c in header if c.startswith("sideboard_")]
    label = f"{set_code} {event}"
    deck_vids = _col_vids(deck_cols, "deck_", nti, label)
    side_vids = _col_vids(side_cols, "sideboard_", nti, label)

    # older game files name the WR bucket differently (AFR era)
    wr_col = next((c for c in ("user_game_win_rate_bucket",
                               "user_win_rate_bucket") if c in header), None)
    usecols = ["draft_id", "build_index", "won"] \
        + ([wr_col] if wr_col else []) + deck_cols + side_cols
    dtypes = {c: "float32" for c in deck_cols + side_cols}
    dtypes["won"] = "bool"  # fail loudly if the column ever isn't True/False
    df = pd.read_csv(_open_game_csv(csv_path), usecols=usecols, dtype=dtypes)
    if wr_col is None:
        df["user_game_win_rate_bucket"] = np.nan
    elif wr_col != "user_game_win_rate_bucket":
        df = df.rename(columns={wr_col: "user_game_win_rate_bucket"})
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

    return pa.Table.from_pydict({
        "draft_id": [d for d, _ in stats.index],
        "build_index": [b for _, b in stats.index],
        "deck_ids": deck_ids, "deck_counts": deck_counts,
        "side_ids": side_ids, "side_counts": side_counts,
        "basics": basics.tolist(),
        "n_games": stats["n_games"].tolist(),
        "n_wins": stats["n_wins"].tolist(),
        "user_win_rate": stats["user_win_rate"].astype("float32").tolist(),
    }, schema=SCHEMA)


def extract(set_code: str, event: str = "PremierDraft") -> Path:
    """event='PremierDraft' → decks.parquet; event='sealed' → merge the Sealed
    and TradSealed game files (whichever are downloaded) into
    decks.sealed.parquet. The vocab is always the DRAFT card universe — sealed
    boosters are the same play boosters, and any name that fails to resolve is
    a loud error, not a dropped column."""
    cards = build_cards(set_code)
    if event.lower() == "sealed":
        events = [e for e in ("Sealed", "TradSealed")
                  if dest_path(set_code, e, kind="game").exists()]
        if not events:
            raise SystemExit(f"{set_code}: no Sealed/TradSealed game files in "
                             "data/raw — download them first")
        table = pa.concat_tables([_event_table(set_code, e, cards)
                                  for e in events])
        out_path = PROCESSED_DIR / set_code / "decks.sealed.parquet"
        label = "+".join(events)
    else:
        table = _event_table(set_code, event, cards)
        out_path = PROCESSED_DIR / set_code / "decks.parquet"
        label = event
    pq.write_table(table, out_path, compression="zstd")
    print(f"{set_code} {label}: {table.num_rows} deck builds -> {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--set", dest="set_code", required=True)
    p.add_argument("--event", default="PremierDraft",
                   help="PremierDraft (default) or 'sealed' (Sealed+TradSealed)")
    args = p.parse_args(argv)
    extract(args.set_code, args.event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
