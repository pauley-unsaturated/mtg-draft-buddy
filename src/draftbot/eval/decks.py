"""Deck-eval harness (PLAN P5.T1a–c; metrics defined in PLAN §Phase 5).

`python -m draftbot.eval --deck --model <baseline|ckpt> --set MSH --split val`
→ scorecard JSON + markdown. Every deck leaderboard number comes from here.

A deck "builder" is anything with `.name` and
`.build_all(arr: DeckArrays) -> list[{"deck": {card_id: count}, "basics": np.ndarray(5)}]`.
`rebuild-ceiling` is the pseudo-builder measuring human self-agreement: the
second-most-played build of a draft "predicts" the most-played one.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.deck_dataset import (DeckArrays, arrays_from_deck_df,
                                        example_weights, land_flags,
                                        load_deck_arrays, most_played)
from draftbot.data.splits import load_splits

DECK_SIZE = 40
DEFAULT_LANDS = 17
BASIC_KEYS = {c: -(c + 1) for c in range(5)}  # basics as pseudo-ids in full F1
DECK_BASELINES = {"random-legal", "gih-top23", "gih-in-lane-build",
                  "rebuild-ceiling"}


# ------------------------------------------------------------------ metrics ---
def _f1(pred: dict, true: dict) -> float:
    inter = sum(min(c, true.get(k, 0)) for k, c in pred.items())
    denom = sum(pred.values()) + sum(true.values())
    return 2 * inter / denom if denom else 1.0


def _true_build(arr: DeckArrays, i: int) -> tuple[dict, np.ndarray]:
    m = arr.deck_counts[i] > 0
    return (dict(zip(arr.pool_ids[i][m].tolist(),
                     arr.deck_counts[i][m].tolist())), arr.basics[i])


def _with_basics(deck: dict, basics: np.ndarray) -> dict:
    out = dict(deck)
    for c in range(5):
        if basics[c]:
            out[BASIC_KEYS[c]] = int(basics[c])
    return out


def compute_deck_scorecard(builder, arr: DeckArrays, cards: pd.DataFrame,
                           meta: dict) -> dict:
    lands = land_flags(cards)
    cmc = cards["cmc"].fillna(0).to_numpy()
    builds = builder.build_all(arr)
    assert len(builds) == arr.n_builds

    f1s = np.zeros(arr.n_builds)
    nb_f1s = np.zeros(arr.n_builds)
    lands_err = np.zeros(arr.n_builds)
    basics_l1 = np.zeros(arr.n_builds)
    curve_l1 = np.zeros(arr.n_builds)
    legal = np.zeros(arr.n_builds, dtype=bool)
    for i, pred in enumerate(builds):
        pdeck, pbasics = pred["deck"], np.asarray(pred["basics"])
        tdeck, tbasics = _true_build(arr, i)
        f1s[i] = _f1(_with_basics(pdeck, pbasics), _with_basics(tdeck, tbasics))
        nb_f1s[i] = _f1(pdeck, tdeck)
        pl = pbasics.sum() + sum(c for k, c in pdeck.items() if lands[k])
        tl = tbasics.sum() + sum(c for k, c in tdeck.items() if lands[k])
        lands_err[i] = abs(int(pl) - int(tl))
        basics_l1[i] = np.abs(pbasics - tbasics).sum()
        hp, ht = np.zeros(8), np.zeros(8)
        for k, c in pdeck.items():
            if not lands[k]:
                hp[min(int(cmc[k]), 7)] += c
        for k, c in tdeck.items():
            if not lands[k]:
                ht[min(int(cmc[k]), 7)] += c
        curve_l1[i] = np.abs(hp - ht).sum()
        pool = dict(zip(arr.pool_ids[i].tolist(), arr.pool_counts[i].tolist()))
        legal[i] = (sum(pdeck.values()) + pbasics.sum() == DECK_SIZE
                    and all(c <= pool.get(k, 0) for k, c in pdeck.items())
                    and (pbasics >= 0).all())

    wins = arr.meta["n_wins"].to_numpy()
    trophy = wins >= 5
    trophy7 = wins >= 7
    w = example_weights(arr.meta)
    by_wins = {str(b): float(f1s[np.clip(wins, 0, 7) == b].mean())
               for b in range(8) if (np.clip(wins, 0, 7) == b).any()}

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "unknown"
    return {
        **meta,
        "git_sha": sha,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_builds": int(arr.n_builds),
        "n_trophy": int(trophy.sum()),
        "metrics": {
            "deck_f1": float(f1s.mean()),
            "nonbasic_f1": float(nb_f1s.mean()),
            "trophy_f1": float(f1s[trophy].mean()) if trophy.any() else float("nan"),
            "trophy7_f1": float(f1s[trophy7].mean()) if trophy7.any() else float("nan"),
            "win_weighted_f1": float((f1s * w).sum() / w.sum()),
            "lands_mae": float(lands_err.mean()),
            "basics_l1": float(basics_l1.mean()),
            "curve_l1": float(curve_l1.mean()),
            "legal_rate": float(legal.mean()),
        },
        "slices": {"f1_by_wins": by_wins},
    }


def render_deck_markdown(card: dict) -> str:
    m = card["metrics"]
    return "\n".join([
        f"### {card['model']} — {card['set']} {card['split']} decks "
        f"(stats: {card['stats']})",
        "",
        "| F1 | NB-F1 | trophy-F1 | 7-win-F1 | win-wtd F1 | lands-MAE "
        "| basics-L1 | curve-L1 | legal |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {m['deck_f1']:.4f} | {m['nonbasic_f1']:.4f} | {m['trophy_f1']:.4f} "
        f"| {m['trophy7_f1']:.4f} | {m['win_weighted_f1']:.4f} "
        f"| {m['lands_mae']:.3f} | {m['basics_l1']:.3f} | {m['curve_l1']:.3f} "
        f"| {m['legal_rate']:.3f} |",
        "",
        f"{card['n_builds']} builds ({card['n_trophy']} trophy) · "
        f"sha {card['git_sha']}",
    ])


def deck_leaderboard_row(card: dict, exp: str, config: str, wall: str) -> str:
    m = card["metrics"]
    return (f"| {exp} | {card['model']} | {card['set']} | {card['split']} "
            f"| {card['stats']} | {m['deck_f1']:.4f} | {m['nonbasic_f1']:.4f} "
            f"| {m['trophy_f1']:.4f} | {m['win_weighted_f1']:.4f} "
            f"| {m['lands_mae']:.3f} | {m['basics_l1']:.3f} "
            f"| {config} | {card['git_sha']} | {wall} "
            f"| {card['generated_at'][:10]} |")


# ---------------------------------------------------------------- baselines ---
def _allocate(shares: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder allocation of `total` basics ∝ shares (5,)."""
    s = np.clip(np.asarray(shares, dtype=np.float64), 0, None)
    s = s / s.sum() if s.sum() > 0 else np.full(5, 0.2)
    base = np.floor(s * total).astype(np.int16)
    order = np.argsort(-(s * total - base))
    base[order[: total - base.sum()]] += 1
    return base


def _copies(pool_ids: np.ndarray, pool_counts: np.ndarray) -> np.ndarray:
    """Per-copy expansion of one pool row (ids repeated by count, PAD dropped)."""
    m = pool_ids >= 0
    return np.repeat(pool_ids[m], pool_counts[m]).astype(np.int64)


def _deck_from_copies(chosen: np.ndarray, pips: np.ndarray,
                      n_basics: int) -> dict:
    ids, counts = np.unique(chosen, return_counts=True)
    deck = dict(zip(ids.tolist(), counts.tolist()))
    basics = _allocate(pips[chosen].sum(0), n_basics)
    return {"deck": deck, "basics": basics}


class RandomLegalBuilder:
    """Floor: 23 uniformly random pool copies + 17 basics split by their pips.
    Deterministic per draft_id (sha1 seed) — eval stays seed-free."""

    name = "random-legal"

    def __init__(self, pips: np.ndarray):
        self.pips = pips

    def build_all(self, arr: DeckArrays) -> list[dict]:
        out = []
        for i in range(arr.n_builds):
            seed = int(hashlib.sha1(
                arr.meta["draft_id"].iloc[i].encode()).hexdigest()[:8], 16)
            copies = _copies(arr.pool_ids[i], arr.pool_counts[i])
            take = min(DECK_SIZE - DEFAULT_LANDS, len(copies))
            chosen = np.random.default_rng(seed).choice(copies, take,
                                                        replace=False)
            out.append(_deck_from_copies(chosen, self.pips, DECK_SIZE - take))
        return out


class GihTop23Builder:
    """Best-rated 23 nonbasics by GIH-WR, colors ignored; pip-split basics."""

    name = "gih-top23"

    def __init__(self, gih: np.ndarray, pips: np.ndarray):
        self.gih, self.pips = gih, pips

    def build_all(self, arr: DeckArrays) -> list[dict]:
        out = []
        for i in range(arr.n_builds):
            copies = _copies(arr.pool_ids[i], arr.pool_counts[i])
            order = np.argsort(-self.gih[copies], kind="stable")
            take = min(DECK_SIZE - DEFAULT_LANDS, len(copies))
            chosen = copies[order[:take]]
            out.append(_deck_from_copies(chosen, self.pips, DECK_SIZE - take))
        return out


class GihInLaneBuilder:
    """~The HUD heuristic that misfired: pick the 2-color lane with the most
    pool GIH mass, maindeck the best 23 castable in it (backfill off-lane if
    short), pip-split basics. Nonbasic lands have no pips → always "castable",
    so bad fixing sneaks in on raw GIH — the flaw this phase exists to kill."""

    name = "gih-in-lane-build"

    def __init__(self, gih: np.ndarray, pips: np.ndarray):
        self.gih, self.pips = gih, pips

    def build_all(self, arr: DeckArrays) -> list[dict]:
        pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
        out = []
        for i in range(arr.n_builds):
            copies = _copies(arr.pool_ids[i], arr.pool_counts[i])
            off_pips = np.stack([
                self.pips[copies][:, [c for c in range(5) if c not in pair]].sum(1)
                for pair in pairs])            # (10, n_copies)
            castable = off_pips == 0
            mass = (castable * np.clip(self.gih[copies] - 0.40, 0, None)).sum(1)
            lane = castable[int(np.argmax(mass))]
            order = np.argsort(-(self.gih[copies] + 1.0 * lane), kind="stable")
            take = min(DECK_SIZE - DEFAULT_LANDS, len(copies))
            chosen = copies[order[:take]]
            out.append(_deck_from_copies(chosen, self.pips, DECK_SIZE - take))
        return out


def build_deck_baselines(static: pd.DataFrame,
                         snapshot: pd.DataFrame | None) -> dict:
    pips = static[[f"pips_{c}" for c in "WUBRG"]].to_numpy(np.float64)
    bots = {"random-legal": RandomLegalBuilder(pips)}
    if snapshot is not None:
        gih = np.nan_to_num(
            snapshot["ever_drawn_win_rate"].to_numpy(np.float64), nan=0.0)
        bots["gih-top23"] = GihTop23Builder(gih, pips)
        bots["gih-in-lane-build"] = GihInLaneBuilder(gih, pips)
    return bots


# ------------------------------------------------------------------ ceiling ---
class RebuildCeiling:
    """Human self-agreement: build B (second-most-played) "predicts" build A
    (most-played) for every multi-build draft. Trophy slice uses DRAFT-level
    wins (a trophy run's wins are split across its builds)."""

    name = "rebuild-ceiling"

    def __init__(self, b_arr: DeckArrays):
        self.b_arr = b_arr

    def build_all(self, arr: DeckArrays) -> list[dict]:
        assert (arr.meta["draft_id"].to_numpy()
                == self.b_arr.meta["draft_id"].to_numpy()).all()
        return [{"deck": d, "basics": b} for d, b in
                (_true_build(self.b_arr, i) for i in range(self.b_arr.n_builds))]


def rebuild_ceiling_pair(set_code: str,
                         split_ids: set[str]) -> tuple[DeckArrays, RebuildCeiling]:
    df = pd.read_parquet(PROCESSED_DIR / set_code / "decks.parquet")
    df = df[df["draft_id"].isin(split_ids)]
    multi = df[df.duplicated("draft_id", keep=False)]
    a = most_played(multi)
    rest = multi.loc[multi.index.difference(a.index)]
    b = most_played(rest)
    keep = set(a["draft_id"]) & set(b["draft_id"])
    a, b = a[a["draft_id"].isin(keep)], b[b["draft_id"].isin(keep)]
    draft_wins = df.groupby("draft_id")["n_wins"].sum()
    a_arr = arrays_from_deck_df(a)
    a_arr.meta["n_wins"] = a_arr.meta["draft_id"].map(draft_wins).to_numpy()
    return a_arr, RebuildCeiling(arrays_from_deck_df(b))


# --------------------------------------------------------------------- CLI ---
def evaluate_deck(builder_name: str, set_code: str, split: str, stats: str,
                  scheme: str = "random", out_dir: Path | None = None) -> dict:
    splits = load_splits(set_code)
    split_ids = set(splits[scheme][split])
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    static = pd.read_parquet(PROCESSED_DIR / set_code / "features.static.parquet")

    if builder_name == "rebuild-ceiling":
        arr, builder = rebuild_ceiling_pair(set_code, split_ids)
    else:
        arr = load_deck_arrays(set_code, draft_ids=split_ids, eval_builds=True)
        if builder_name in DECK_BASELINES:
            from draftbot.data.features import load_snapshot
            snapshot = None if stats == "none" else load_snapshot(set_code, stats)
            bots = build_deck_baselines(static, snapshot)
            if builder_name not in bots:
                raise SystemExit(f"{builder_name} needs a stats snapshot")
            builder = bots[builder_name]
        else:
            ckpt = Path(builder_name)
            if not ckpt.exists():
                raise SystemExit(f"unknown deck model {builder_name!r}")
            from draftbot.models.loading import deck_builder_from_checkpoint
            builder = deck_builder_from_checkpoint(ckpt, set_code, stats)

    card = compute_deck_scorecard(
        builder, arr, cards,
        {"model": builder.name, "set": set_code, "split": split,
         "stats": stats, "scheme": scheme, "task": "deck"})
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"deck.{builder.name}.{set_code}.{split}.{stats}"
        (out_dir / f"{stem}.json").write_text(json.dumps(card, indent=1))
        (out_dir / f"{stem}.md").write_text(render_deck_markdown(card) + "\n")
    return card
