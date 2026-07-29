"""Eval harness (PLAN P1.T2, metrics per Appendix A + owner's skill-gap metric).

`python -m draftbot.eval --model <baseline|ckpt.pt> --set MSH --split val|test
 [--stats-snapshot none|week1|full]` → scorecard JSON + markdown summary.

Every leaderboard number comes from here (no-metric-laundering guardrail).
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.dataset import PAD, DraftArrays, load_draft_arrays
from draftbot.data.features import load_snapshot
from draftbot.data.splits import load_splits

RARE_LIKE = {"rare", "mythic", "special", "bonus"}
SELF_DRAFT_N = 500
COHERENCE_PIP_THRESHOLD = 8


def _softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max(-1, keepdims=True)
    e = np.exp(np.clip(z, -60, 0))
    return e / e.sum(-1, keepdims=True)


def topk_hits(packs: np.ndarray, picks: np.ndarray, scores: np.ndarray,
              kmax: int = 3) -> np.ndarray:
    """(kmax, N, t) bool — whether the human pick is within the model's top-k
    DISTINCT card ids (duplicate copies of one card collapse to one entry)."""
    order = np.argsort(-scores, axis=-1, kind="stable")
    ids_in_order = np.take_along_axis(packs, order, axis=-1)
    P = packs.shape[-1]
    distinct_count = np.zeros(picks.shape, dtype=np.int16)
    found_rank = np.full(picks.shape, 99, dtype=np.int16)
    for p in range(P):
        cur = ids_in_order[..., p]
        earlier_same = np.zeros(picks.shape, dtype=bool)
        for q in range(p):
            earlier_same |= ids_in_order[..., q] == cur
        is_new = (cur != PAD) & ~earlier_same
        distinct_count = distinct_count + is_new
        hit = is_new & (cur == picks) & (found_rank == 99)
        found_rank[hit] = distinct_count[hit]
    return np.stack([found_rank <= k for k in range(1, kmax + 1)])


def pick_probabilities(packs, picks, scores):
    """(prob_of_human_pick, top1_id, top1_prob) per (N, t)."""
    probs = _softmax(scores)
    probs = np.where(packs == PAD, 0.0, probs)
    p_pick = (probs * (packs == picks[..., None])).sum(-1)
    top_slot = scores.argmax(-1)
    top1_id = np.take_along_axis(packs, top_slot[..., None], -1)[..., 0]
    p_top1 = (probs * (packs == top1_id[..., None])).sum(-1)
    return p_pick, top1_id, p_top1


def ece_10bin(conf: np.ndarray, correct: np.ndarray) -> float:
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_scorecard(scorer, arrays: DraftArrays, cards: pd.DataFrame,
                      splits: dict, static_feats: pd.DataFrame,
                      meta: dict) -> dict:
    packs, picks = arrays.packs, arrays.picks
    n, t, _ = packs.shape
    scores = scorer.score_drafts(packs, arrays.prev_picks)
    assert scores.shape == packs.shape

    hits = topk_hits(packs, picks, scores)  # (3, N, t)
    p_pick, top1_id, p_top1 = pick_probabilities(packs, picks, scores)
    nll = float(-np.log(np.clip(p_pick, 1e-12, 1)).mean())
    ece = ece_10bin(p_top1.ravel(), hits[0].ravel())

    rarity = cards.set_index("id")["rarity"]
    is_rare_like = rarity.isin(RARE_LIKE).to_numpy()
    pick_rarity = rarity.to_numpy()[picks]

    # --- skill slices (owner requirement: agree with good drafters, not bad) ---
    es, ls = splits["expert_subset"], splits["low_skill_subset"]
    wr = arrays.meta["user_win_rate"].to_numpy()
    games = arrays.meta["user_n_games"].to_numpy()
    expert_mask = (wr >= es["min_win_rate"]) & (games >= es["min_games"])
    low_mask = (wr <= ls["max_win_rate"]) & (games >= ls["min_games"])
    top1 = hits[0]
    expert_top1 = float(top1[expert_mask].mean())
    low_top1 = float(top1[low_mask].mean())

    # --- rare-take watchdog ---
    pack_has_nonrare = (~is_rare_like[np.where(packs == PAD, 0, packs)]
                        | (packs == PAD)).any(-1)
    scope = pack_has_nonrare  # picks where a non-rare option existed
    model_rare = is_rare_like[top1_id] & scope
    human_rare = is_rare_like[picks] & scope
    expert_human_rare_rate = float(human_rare[expert_mask].sum()
                                   / scope[expert_mask].sum())
    model_rare_rate = float(model_rare.sum() / scope.sum())

    # --- pool coherence via greedy self-draft on held-out pack sequences ---
    sd = scorer.self_draft(packs[:SELF_DRAFT_N])
    pips = static_feats[[f"pips_{c}" for c in "WUBRG"]].to_numpy()
    pool_pips = pips[sd].sum(axis=1)  # (n, 5)
    model_colors = float((pool_pips >= COHERENCE_PIP_THRESHOLD).sum(-1).mean())
    human_pool_pips = pips[picks[:SELF_DRAFT_N]].sum(axis=1)
    human_colors = float((human_pool_pips >= COHERENCE_PIP_THRESHOLD).sum(-1).mean())

    by_position = [float(top1[:, i].mean()) for i in range(t)]
    ppp = t // 3
    by_pack = [float(top1[:, i * ppp:(i + 1) * ppp].mean()) for i in range(3)]
    by_rarity = {r: float(top1[pick_rarity == r].mean())
                 for r in pd.unique(pick_rarity.ravel())}

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "unknown"

    return {
        **meta,
        "git_sha": sha,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_drafts": int(n), "n_picks": int(n * t),
        "expert_subset": es, "low_skill_subset": ls,
        "n_expert_drafts": int(expert_mask.sum()),
        "n_low_skill_drafts": int(low_mask.sum()),
        "metrics": {
            "top1": float(top1.mean()),
            "top2": float(hits[1].mean()),
            "top3": float(hits[2].mean()),
            "nll": nll,
            "ece": ece,
            "expert_top1": expert_top1,
            "low_skill_top1": low_top1,
            "skill_gap": expert_top1 - low_top1,
            "rare_take_rate": model_rare_rate,
            "human_rare_take_rate": float(human_rare.sum() / scope.sum()),
            "expert_human_rare_take_rate": expert_human_rare_rate,
            "rare_take_ratio_vs_expert": (model_rare_rate / expert_human_rare_rate
                                          if expert_human_rare_rate else float("nan")),
            "pool_coherence_colors": model_colors,
            "human_pool_coherence_colors": human_colors,
        },
        "slices": {"top1_by_position": by_position, "top1_by_pack": by_pack,
                   "top1_by_pick_rarity": by_rarity},
    }


def render_markdown(card: dict) -> str:
    m = card["metrics"]
    return "\n".join([
        f"### {card['model']} — {card['set']} {card['split']} "
        f"(stats: {card['stats']})",
        "",
        f"| top-1 | top-2 | top-3 | NLL | ECE | expert top-1 | low-skill top-1 "
        f"| skill-gap | rare-take (vs expert ×) | colors |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| {m['top1']:.4f} | {m['top2']:.4f} | {m['top3']:.4f} | {m['nll']:.4f} "
        f"| {m['ece']:.4f} | {m['expert_top1']:.4f} | {m['low_skill_top1']:.4f} "
        f"| {m['skill_gap']:+.4f} | {m['rare_take_rate']:.4f} "
        f"({m['rare_take_ratio_vs_expert']:.2f}×) "
        f"| {m['pool_coherence_colors']:.2f} (human {m['human_pool_coherence_colors']:.2f}) |",
        "",
        f"pack top-1: {[f'{x:.3f}' for x in card['slices']['top1_by_pack']]} · "
        f"{card['n_drafts']} drafts · sha {card['git_sha']}",
    ])


def leaderboard_row(card: dict, exp: str, config: str, wall_clock: str) -> str:
    m = card["metrics"]
    return (f"| {exp} | {card['model']} | {card['set']} | {card['split']} "
            f"| {card['stats']} | {m['top1']:.4f} | {m['top3']:.4f} "
            f"| {m['expert_top1']:.4f} | {m['skill_gap']:+.4f} | {m['nll']:.4f} "
            f"| {config} | {card['git_sha']} | {wall_clock} "
            f"| {card['generated_at'][:10]} |")


def evaluate(scorer, set_code: str, split: str, stats: str,
             scheme: str = "random", out_dir: Path | None = None) -> dict:
    splits = load_splits(set_code)
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    static = pd.read_parquet(PROCESSED_DIR / set_code / "features.static.parquet")
    arrays = load_draft_arrays(set_code, draft_ids=set(splits[scheme][split]))
    card = compute_scorecard(
        scorer, arrays, cards, splits, static,
        {"model": scorer.name, "set": set_code, "split": split,
         "stats": stats, "scheme": scheme})
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{scorer.name}.{set_code}.{split}.{stats}"
        (out_dir / f"{stem}.json").write_text(json.dumps(card, indent=1))
        (out_dir / f"{stem}.md").write_text(render_markdown(card) + "\n")
    return card
