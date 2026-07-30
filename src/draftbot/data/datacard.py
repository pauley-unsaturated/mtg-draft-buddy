"""Generate docs/data_cards/<SET>.md from the processed data (PLAN P0.T8).
Numbers are computed, never typed."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import COMPANION_SHEETS, PROCESSED_DIR


def build_datacard(set_code: str) -> Path:
    events = sorted(p.name.split(".")[1] for p in
                    (PROCESSED_DIR / set_code).glob("draft.*.parquet"))
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    lines = [f"# Data card: {set_code}", ""]
    reg_path = Path("data/snapshots") / set_code / "registry.json"
    snapshots = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    for event in events:
        df = pd.read_parquet(PROCESSED_DIR / set_code / f"draft.{event}.parquet")
        t = int(df["position"].max()) + 1
        n_drafts = df["draft_id"].nunique()
        basics = set(cards[cards["is_basic"] == 1]["id"])
        basic_picks = int(df["pick"].isin(basics).sum())
        pick2 = int((df["pick_2"] >= 0).sum())
        lines += [
            f"## {event}",
            "",
            f"- picks: {len(df):,} | drafts: {n_drafts:,} | t={t}",
            f"- dates: {df['draft_time'].min().date()} → {df['draft_time'].max().date()}",
            f"- basics picked: {basic_picks:,} ({basic_picks / len(df):.1%}) — basics ARE draftable",
            f"- pick_2 present on {pick2:,} rows",
            f"- ranks: {sorted(df['rank'].unique().tolist())}",
            f"- user_n_games buckets: {sorted(df['user_n_games'].unique().tolist())}",
            "",
        ]
    decks_path = PROCESSED_DIR / set_code / "decks.parquet"
    if decks_path.exists():
        d = pd.read_parquet(decks_path)
        sizes = (d["deck_counts"].apply(lambda c: int(np.sum(c)))
                 + d["basics"].apply(lambda b: int(np.sum(b))))
        n_basics = d["basics"].apply(lambda b: int(np.sum(b)))
        per_draft = d.groupby("draft_id").size()
        splits_path = Path("data/splits") / f"{set_code}.json"
        cov = ""
        if splits_path.exists():
            r = json.loads(splits_path.read_text())["random"]
            part = {i: p for p, ids in r.items() for i in ids}
            cov = f"- split coverage: {d['draft_id'].map(part).notna().mean():.1%} of builds join `data/splits/{set_code}.json`"
        lines += [
            "## Decks (game_data, PremierDraft)",
            "",
            f"- builds: {len(d):,} | drafts: {d['draft_id'].nunique():,} | "
            f"games: {int(d['n_games'].sum()):,} ({d['n_games'].mean():.1f} per build)",
            f"- rebuilds: {(per_draft == 1).mean():.1%} of drafts have 1 build; max {per_draft.max()}",
            f"- deck size: {(sizes == 40).mean():.1%} exactly 40 (range {sizes.min()}-{sizes.max()})",
            f"- basics per deck: mean {n_basics.mean():.1f} σ {n_basics.std():.1f} "
            f"(range {n_basics.min()}-{n_basics.max()})",
            f"- wins per build: {(d['n_wins'] == 0).mean():.1%} zero-win, "
            f"{(d['n_wins'] >= 5).mean():.1%} ≥5 wins, "
            f"{(d['n_wins'] >= 7).mean():.1%} ≥7 wins (trophy pool)",
        ] + ([cov] if cov else []) + [""]

    lines += [
        "## Cards",
        "",
        f"- vocab: {len(cards)} cards "
        f"({(cards['is_basic'] == 1).sum()} basics, {(cards['is_flip'] == 1).sum()} flip)",
        f"- printing sets: {cards['set'].value_counts().to_dict()}",
        f"- companion sheets (registry): {COMPANION_SHEETS.get(set_code.upper(), [])}",
        f"- rarities: {cards['rarity'].value_counts().to_dict()}",
        "",
        "## Snapshots",
        "",
    ] + [f"- {tag}: `{fname}`" for tag, fname in sorted(snapshots.items())]
    out = Path("docs/data_cards") / f"{set_code}.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--set", dest="set_code", required=True)
    args = p.parse_args(argv)
    print(build_datacard(args.set_code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
