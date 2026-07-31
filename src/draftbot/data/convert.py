"""Stream 17lands draft CSV.gz → compact per-pick Parquet (PLAN P0.T4).

Schema (one row per pick):
  draft_id      str
  position      int16   0..t-1  (pack_number * picks_per_pack + pick_number)
  pack_cards    list<int16>     card ids in the pack, multiplicity preserved
  pick          int16
  pick_2        int16   (-1 when absent — MSH has this column empty)
  rank          str     (lowercase)
  user_win_rate float32
  user_n_games  int16   (buckets up to 1000 — int8 would overflow)
  event_wins    int8
  event_losses  int8
  event_type    str
  draft_time    timestamp[s]

Pool is NOT stored: reconstructable as the multiset of picks before `position`
(verified by tests/test_convert.py against the CSV pool_ columns).
Incomplete drafts (< t picks) are dropped and counted.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from draftbot.data.cards import build_cards, name_to_id, norm_name
from draftbot.data.fetch import dest_path

PROCESSED_DIR = Path("data/processed")
CHUNK_ROWS = 100_000

SCHEMA = pa.schema([
    ("draft_id", pa.string()),
    ("position", pa.int16()),
    ("pack_cards", pa.list_(pa.int16())),
    ("pick", pa.int16()),
    ("pick_2", pa.int16()),
    ("rank", pa.string()),
    ("user_win_rate", pa.float32()),
    ("user_n_games", pa.int16()),
    ("event_wins", pa.int8()),
    ("event_losses", pa.int8()),
    ("event_type", pa.string()),
    ("draft_time", pa.timestamp("s")),
])


def _draft_frames(csv_path: Path, meta_cols: list[str], pack_cols: list[str],
                  dtypes: dict):
    """Yield one DataFrame per complete-in-file draft, preserving row order.

    Assumes rows of a draft are contiguous (asserted downstream); carries the
    trailing partial draft across chunk boundaries.
    """
    carry = None
    reader = pd.read_csv(csv_path, usecols=meta_cols + pack_cols, dtype=dtypes,
                         chunksize=CHUNK_ROWS)
    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        ids = chunk["draft_id"].values
        # boundaries where draft_id changes
        change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(ids)]])
        # hold back the last group — it may continue in the next chunk
        for s, e in zip(starts[:-1], ends[:-1]):
            yield chunk.iloc[s:e]
        carry = chunk.iloc[starts[-1]:].copy()
    if carry is not None and len(carry):
        yield carry


def convert(set_code: str, event: str = "PremierDraft") -> Path:
    cards = build_cards(set_code, event)
    nti = name_to_id(cards)
    csv_path = dest_path(set_code, event)
    out_path = PROCESSED_DIR / set_code / f"draft.{event}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0).columns
    pack_cols = [c for c in header if c.startswith("pack_card_")]
    # pack_cols sorted by card id so that column j <-> card id j
    pack_cols = sorted(pack_cols, key=lambda c: nti[norm_name(c[len("pack_card_"):])])
    assert [nti[norm_name(c[len("pack_card_"):])] for c in pack_cols] == list(range(len(pack_cols)))
    meta_cols = ["draft_id", "draft_time", "event_match_wins",
                 "event_match_losses", "pack_number", "pick_number", "pick"]
    # optional columns by export era: rank/skill buckets absent pre-2022,
    # pick_2 present only 2026+
    optional = [c for c in ("rank", "user_n_games_bucket",
                            "user_game_win_rate_bucket", "pick_2") if c in header]
    meta_cols += optional
    has_pick2 = "pick_2" in optional
    dtypes = {c: "float32" for c in pack_cols}  # 2021 exports: NAs in counts
    # wins/losses/buckets AND pack/pick numbers are float: pre-2023 exports
    # contain NAs (malformed rows are dropped per draft below)
    dtypes.update({"pack_number": "float32", "pick_number": "float32",
                   "event_match_wins": "float32", "event_match_losses": "float32",
                   "user_n_games_bucket": "float32",
                   "user_game_win_rate_bucket": "float32",
                   "draft_id": "str", "rank": "str", "pick": "str", "pick_2": "str"})

    tmp_path = out_path.with_suffix(".parquet.tmp")
    # prescan for the set's true t = MODE of per-draft row counts. (Max-based
    # inference broke on TLA: a handful of 15-pick outlier drafts inflated t to
    # 45 and every real draft got dropped; first-draft inference broke on ECL.)
    scan = pd.read_csv(csv_path, usecols=["draft_id"], dtype="str")
    t_expected = int(scan.groupby("draft_id").size().mode().iloc[0])
    del scan

    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")
    n_drafts = n_dropped = n_picks = 0
    seen_ids: set[str] = set()
    try:
        for g in _draft_frames(csv_path, meta_cols, pack_cols, dtypes):
            did = g["draft_id"].iloc[0]
            if did in seen_ids:  # non-contiguous rows (seen in WOE): drop draft
                n_dropped += 1
                continue
            seen_ids.add(did)
            g = g.dropna(subset=["pack_number", "pick_number"])
            if len(g) != t_expected:
                n_dropped += 1
                continue
            # position = dense rank of (pack, pick) — robust to sets whose
            # exports omit steps (TLA lacks every P1P1 row)
            g = g.sort_values(["pack_number", "pick_number"], kind="mergesort")
            g = g.assign(_position=np.arange(len(g), dtype=np.int16))

            pack_mat = np.nan_to_num(g[pack_cols].to_numpy(dtype=np.float32)).astype(np.int16)  # (t, n_cards)
            rows_idx, cols_idx = np.nonzero(pack_mat)
            counts = pack_mat[rows_idx, cols_idx]
            # expand multiplicity (a pack can hold 2 copies of one name)
            rep_rows = np.repeat(rows_idx, counts)
            rep_cols = np.repeat(cols_idx, counts).astype(np.int16)
            offsets = np.zeros(len(g) + 1, dtype=np.int64)
            np.add.at(offsets, rows_idx + 1, counts)
            offsets = np.cumsum(offsets)
            pack_lists = pa.ListArray.from_arrays(
                pa.array(offsets, type=pa.int32()), pa.array(rep_cols, type=pa.int16()))

            pick_ids = g["pick"].map(lambda x: nti[norm_name(x)]).astype(np.int16)
            if has_pick2:
                pick2_ids = g["pick_2"].map(
                    lambda x: nti[norm_name(x)] if isinstance(x, str) and x else -1
                ).astype(np.int16)
            else:
                pick2_ids = pd.Series(np.full(len(g), -1, dtype=np.int16))

            table = pa.Table.from_arrays([
                pa.array([did] * len(g)),
                pa.array(g["_position"].to_numpy(np.int16), type=pa.int16()),
                pack_lists,
                pa.array(pick_ids.to_numpy(), type=pa.int16()),
                pa.array(pick2_ids.to_numpy(), type=pa.int16()),
                pa.array(g["rank"].fillna("unknown").str.lower() if "rank" in g
                         else ["unknown"] * len(g)),
                pa.array(g["user_game_win_rate_bucket"].to_numpy(np.float32)
                         if "user_game_win_rate_bucket" in g
                         else np.full(len(g), np.nan, np.float32)),
                pa.array((g["user_n_games_bucket"].fillna(-1).to_numpy(np.int16)
                          if "user_n_games_bucket" in g
                          else np.full(len(g), -1, np.int16)), type=pa.int16()),
                pa.array(g["event_match_wins"].fillna(0).to_numpy(np.int8), type=pa.int8()),
                pa.array(g["event_match_losses"].fillna(0).to_numpy(np.int8), type=pa.int8()),
                pa.array([event] * len(g)),
                pa.array(pd.to_datetime(g["draft_time"]).astype("datetime64[s]")),
            ], schema=SCHEMA)
            writer.write_table(table)
            n_drafts += 1
            n_picks += len(g)
    finally:
        writer.close()
    tmp_path.rename(out_path)  # atomic: partial files never masquerade as done

    print(f"{set_code} {event}: {n_drafts} drafts / {n_picks} picks written, "
          f"{n_dropped} incomplete drafts dropped, t={t_expected}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot convert")
    p.add_argument("--set", dest="set_code", required=True)
    p.add_argument("--event", default="PremierDraft")
    args = p.parse_args(argv)
    convert(args.set_code, args.event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
