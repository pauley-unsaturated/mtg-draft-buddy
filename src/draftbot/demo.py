"""Try-out CLI: compare model picks on real (or hand-typed) packs.

Sampled mode (default):
  uv run python -m draftbot.demo --set MSH \
      --models checkpoints/EXP-010,checkpoints/EXP-001,gih-greedy --n 3

Interactive mode (type a pack, optionally a pool):
  uv run python -m draftbot.demo --set MSH --models checkpoints/EXP-010 --interactive
  > pack: Lightning Strike; HULK SMASH!; Plains; ...
  > pool: Cruel Alliance; Widow's Bite
"""

import argparse
import sys

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR, name_to_id
from draftbot.data.dataset import PAD, load_draft_arrays
from draftbot.data.features import load_snapshot
from draftbot.data.splits import load_splits
from draftbot.eval.__main__ import resolve_scorer
from draftbot.eval.harness import _softmax


def fuzzy_id(query: str, nti: dict, names: list[str]) -> int:
    q = query.strip().lower()
    if q in nti:
        return nti[q]
    matches = [n for n in nti if n.startswith(q)]
    if len(matches) == 1:
        return nti[matches[0]]
    matches = [n for n in nti if q in n]
    if len(matches) == 1:
        return nti[matches[0]]
    raise SystemExit(f"card {query!r}: {'ambiguous ' + str(matches[:5]) if matches else 'no match'}")


def show_pack(pack_ids, pool_ids, position, scorers, cards, gih, human_pick=None):
    id2name = dict(zip(cards["id"], cards["name"]))
    rarity = dict(zip(cards["id"], cards["rarity"]))
    t, P = 42, 14
    packs = np.full((1, t, P), PAD, dtype=np.int16)
    packs[0, :, 0] = pool_ids[0] if len(pool_ids) else pack_ids[0]
    packs[0, position, :] = PAD
    packs[0, position, : len(pack_ids)] = pack_ids
    prev = np.full((1, t), PAD, dtype=np.int16)
    for i, cid in enumerate(pool_ids):
        prev[0, i + 1] = cid
    if position + 1 < t:
        prev[0, position + 1:] = packs[0, 0, 0]

    pack_no, pick_no = position // 14 + 1, position % 14 + 1
    print(f"\n━━ P{pack_no}P{pick_no}", "─" * 60)
    if pool_ids:
        pool_names = [id2name[c] for c in pool_ids]
        print(f"pool ({len(pool_ids)}): {', '.join(pool_names[-8:])}"
              + (" …" if len(pool_ids) > 8 else ""))
    rankings = {}
    for s in scorers:
        scores = s.score_drafts(packs, prev)[0, position]
        probs = _softmax(scores[None, :])[0]
        order = np.argsort(-scores)[: len(pack_ids)]
        rankings[s.name] = [(int(packs[0, position, i]), float(probs[i])) for i in order]

    header = f"{'pack card':<38}{'rar':<5}{'GIH':<7}" + "".join(
        f"{s.name[:16]:<18}" for s in scorers)
    print(header)
    for cid in pack_ids:
        row = f"{id2name[cid][:36]:<38}{rarity[cid][:3]:<5}"
        g = gih.get(cid, float("nan"))
        row += f"{g:.3f}  " if g == g else "  —    "
        for s in scorers:
            rank = next((i for i, (c, _) in enumerate(rankings[s.name]) if c == cid), None)
            prob = next((p for c, p in rankings[s.name] if c == cid), 0.0)
            mark = f"#{rank + 1} {prob * 100:4.1f}%" if rank is not None else ""
            row += f"{mark:<18}"
        marker = " ← human" if human_pick is not None and cid == human_pick else ""
        print(row + marker)


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.demo")
    p.add_argument("--set", dest="set_code", default="MSH")
    p.add_argument("--models", required=True,
                   help="comma-separated: checkpoint dirs and/or baseline names")
    p.add_argument("--stats", default="full", choices=["none", "week1", "full"])
    p.add_argument("--n", type=int, default=3, help="number of sampled picks")
    p.add_argument("--positions", default="0,20,35",
                   help="comma-separated draft positions to sample")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args(argv)

    cards = pd.read_parquet(PROCESSED_DIR / args.set_code / "cards.parquet")
    nti = name_to_id(cards)
    snap = load_snapshot(args.set_code, "full")
    gih = dict(zip(cards["id"], snap["ever_drawn_win_rate"].values))
    scorers = [resolve_scorer(m.strip(), args.set_code, args.stats)
               for m in args.models.split(",")]

    if args.interactive:
        pack = input("pack: ").strip()
        pool = input("pool (optional): ").strip()
        pack_ids = [fuzzy_id(c, nti, list(cards["name"])) for c in pack.split(";") if c.strip()]
        pool_ids = [fuzzy_id(c, nti, list(cards["name"])) for c in pool.split(";") if c.strip()]
        show_pack(pack_ids, pool_ids, len(pool_ids), scorers, cards, gih)
        return 0

    splits = load_splits(args.set_code)
    arrays = load_draft_arrays(args.set_code,
                               draft_ids=set(splits["random"]["test"]))
    rng = np.random.default_rng(args.seed)
    positions = [int(x) for x in args.positions.split(",")]
    for _ in range(args.n):
        d = int(rng.integers(0, arrays.n_drafts))
        for pos in positions:
            pack_ids = [c for c in arrays.packs[d, pos] if c != PAD]
            pool_ids = list(arrays.picks[d, :pos])
            show_pack(pack_ids, pool_ids, pos, scorers, cards, gih,
                      human_pick=int(arrays.picks[d, pos]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
