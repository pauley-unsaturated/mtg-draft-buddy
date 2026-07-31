"""CLI: python -m draftbot.eval --model <baseline-name|checkpoint.pt> --set MSH
--split val|test [--stats-snapshot none|week1|full] [--out docs/scorecards/]"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.features import load_snapshot
from draftbot.eval.bots import build_bots
from draftbot.eval.harness import evaluate, render_markdown

BASELINES = {"random", "rarity-first", "alsa-greedy", "gih-greedy", "gih-in-lane"}


def resolve_scorer(name: str, set_code: str, stats: str):
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    static = pd.read_parquet(PROCESSED_DIR / set_code / "features.static.parquet")
    if name in BASELINES:
        snapshot = None if stats == "none" else load_snapshot(set_code, stats)
        bots = {b.name: b for b in build_bots(cards, static, snapshot)}
        if name not in bots:
            raise SystemExit(f"{name} needs a stats snapshot (--stats-snapshot week1|full)")
        return bots[name]
    ckpt = Path(name)
    if ckpt.exists():
        from draftbot.models.loading import scorer_from_checkpoint
        return scorer_from_checkpoint(ckpt, set_code, stats)
    raise SystemExit(f"unknown model {name!r} (not a baseline, not a checkpoint path)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.eval")
    p.add_argument("--model", required=True)
    p.add_argument("--set", dest="set_code", required=True)
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--stats-snapshot", default="full", choices=["none", "week1", "full"])
    p.add_argument("--scheme", default="random", choices=["random", "temporal"])
    p.add_argument("--deck", action="store_true",
                   help="deck-builder eval (PLAN P5.T1a) instead of draft eval")
    p.add_argument("--decode", default=None,
                   choices=["greedy", "expected", "maskgit", "rescore"],
                   help="override the checkpoint's decode (deck eval only; "
                        "the override is recorded in the model name)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.deck:
        from draftbot.eval.decks import evaluate_deck, render_deck_markdown
        card = evaluate_deck(args.model, args.set_code, args.split,
                             args.stats_snapshot, scheme=args.scheme,
                             out_dir=args.out, decode=args.decode)
        print(render_deck_markdown(card))
        return 0

    scorer = resolve_scorer(args.model, args.set_code, args.stats_snapshot)
    card = evaluate(scorer, args.set_code, args.split, args.stats_snapshot,
                    scheme=args.scheme, out_dir=args.out)
    print(render_markdown(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
