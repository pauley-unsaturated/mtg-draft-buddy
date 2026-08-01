"""One-shot deck build from a pasted pool (owner tool, 2026-07-31).

`python -m draftbot.build --set TLA --pool pool.txt [--ckpt checkpoints/EXP-130]`

Pool file: Arena's sealed-pool export ("2 Card Name (TLA) 123"), or plain
"2 Card Name" / bare "Card Name" lines; '-' reads stdin. Basic lands in the
pool are ignored (the basics head owns the manabase); every other name must
resolve against the set's card vocabulary or the tool refuses loudly.

Output: the proposed 40 with per-card membership confidence, plus the bubble —
the least-confident cards in the deck and the best cards left out.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from draftbot.data.cards import PROCESSED_DIR, name_to_id, norm_name
from draftbot.data.dataset import PAD
from draftbot.data.deck_dataset import DeckArrays, land_flags

DEFAULT_CKPT = "checkpoints/EXP-130"  # TLA sealed fine-tune (fast path)

_LINE = re.compile(r"^(?:(\d+)\s+)?(.+?)(?:\s+\([A-Z0-9]{2,6}\)\s+\S+)?$")
_SECTIONS = {"deck", "sideboard", "commander", "companion", "about"}


def parse_pool(text: str) -> list[tuple[int, str]]:
    """(count, name) per line; Arena set/collector suffixes stripped."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower() in _SECTIONS:
            continue
        m = _LINE.match(line)
        out.append((int(m.group(1) or 1), m.group(2).strip()))
    return out


def pool_arrays(entries: list[tuple[int, str]],
                cards: pd.DataFrame) -> tuple[DeckArrays, int]:
    nti = name_to_id(cards)
    basic_ids = set(cards.loc[cards["is_basic"] == 1, "id"])
    counts: dict[int, int] = {}
    unknown, n_basics = [], 0
    for count, name in entries:
        vid = nti.get(norm_name(name))
        if vid is None:
            unknown.append(name)
        elif vid in basic_ids:
            n_basics += count
        else:
            counts[vid] = counts.get(vid, 0) + count
    if unknown:
        raise SystemExit(f"{len(unknown)} pool names not in this set's vocab: "
                         f"{unknown[:10]}")
    if not counts:
        raise SystemExit("empty pool after dropping basics")
    ids = np.array(sorted(counts), dtype=np.int16)
    cnt = np.array([counts[i] for i in ids], dtype=np.int16)
    arr = DeckArrays(
        pool_ids=ids[None, :], pool_counts=cnt[None, :],
        deck_counts=np.zeros_like(ids)[None, :],
        basics=np.zeros((1, 5), dtype=np.int16),
        meta=pd.DataFrame({"draft_id": ["cli"], "build_index": [0],
                           "n_games": [0], "n_wins": [0],
                           "user_win_rate": [np.nan]}))
    return arr, n_basics


def render(build: dict, conf: np.ndarray, arr: DeckArrays,
           cards: pd.DataFrame, header: str) -> str:
    names = dict(zip(cards["id"], cards["name"]))
    is_land = land_flags(cards)
    ids = arr.pool_ids[0]
    deck = {int(k): int(v) for k, v in build["deck"].items()}
    basics = np.asarray(build["basics"], dtype=int)
    slot_conf = {int(i): float(c) for i, c in zip(ids, conf) if i >= 0}

    def line(vid, n):
        tag = "  [land]" if is_land[vid] else ""
        return f"  {slot_conf[vid]:5.0%}  {n} {names[vid]}{tag}"

    spells = [(vid, n) for vid, n in deck.items() if not is_land[vid]]
    nb_lands = [(vid, n) for vid, n in deck.items() if is_land[vid]]
    order = lambda xs: sorted(xs, key=lambda x: -slot_conf[x[0]])
    n_lands = int(basics.sum()) + sum(n for _, n in nb_lands)
    basics_txt = ", ".join(
        f"{int(b)} {n}" for b, n in
        zip(basics, ["Plains", "Island", "Swamp", "Mountain", "Forest"]) if b)

    cut = [(vid, n) for vid, n in deck.items()]
    cut = sorted(cut, key=lambda x: slot_conf[x[0]])[:3]
    left = [(int(i), int(c - deck.get(int(i), 0)))
            for i, c in zip(ids, arr.pool_counts[0])
            if i >= 0 and c - deck.get(int(i), 0) > 0]
    add = sorted(left, key=lambda x: -slot_conf[x[0]])[:5]

    out = [header, "",
           f"PROPOSED 40 — {sum(deck.values()) + basics.sum()} cards, "
           f"{n_lands} lands ({basics_txt or 'no basics'})", ""]
    out += [line(v, n) for v, n in order(spells)]
    if nb_lands:
        out += [""] + [line(v, n) for v, n in order(nb_lands)]
    out += ["", "ON THE BUBBLE",
            "  weakest slots in the deck:"]
    out += ["  " + line(v, n) for v, n in cut]
    out += ["  best left in the sideboard:"]
    out += ["  " + line(v, n) for v, n in add]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.build")
    p.add_argument("--set", dest="set_code", required=True)
    p.add_argument("--pool", required=True,
                   help="pool text file, or '-' for stdin")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--stats", default="full",
                   choices=["none", "week1", "full"])
    p.add_argument("--decode", default=None,
                   choices=["greedy", "expected", "maskgit", "rescore"])
    args = p.parse_args(argv)

    text = sys.stdin.read() if args.pool == "-" else Path(args.pool).read_text()
    cards = pd.read_parquet(PROCESSED_DIR / args.set_code / "cards.parquet")
    arr, n_basics = pool_arrays(parse_pool(text), cards)

    from draftbot.models.loading import deck_builder_from_checkpoint
    builder = deck_builder_from_checkpoint(Path(args.ckpt), args.set_code,
                                           args.stats)
    if args.decode:
        builder.decode = args.decode
    build = builder.build_all(arr)[0]

    ids = torch.from_numpy(arr.pool_ids.astype(np.int64)).to(builder.device)
    cnt = torch.from_numpy(arr.pool_counts.astype(np.int64)).to(builder.device)
    member, _, _ = builder._heads(ids, cnt, ids == PAD)
    conf = member[0].cpu().numpy()

    pool_n = int(arr.pool_counts.sum())
    header = (f"{builder.name} · {args.set_code} · stats {args.stats} · "
              f"{pool_n}-card pool"
              + (f" ({n_basics} basics ignored)" if n_basics else ""))
    print(render(build, conf, arr, cards, header))
    return 0


if __name__ == "__main__":
    sys.exit(main())
