"""Heuristic baseline bots (PLAN P1.T1).

Every bot exposes the Scorer interface the eval harness consumes:
  score_drafts(packs, prev_picks) -> (N, t, P) float scores (PAD slots = -inf)
  self_draft(packs)               -> (N, t) card ids picked autoregressively

Scores are vectorized over whole splits; the only bot with pool state
(gih-in-lane) recomputes its lane from whichever pick sequence applies.
"""

import numpy as np
import pandas as pd

from draftbot.data.dataset import PAD

NEG = -1e9

RARITY_SCORE = {"mythic": 3.0, "rare": 2.0, "special": 2.0, "bonus": 2.0,
                "uncommon": 1.0, "common": 0.0}


def _slot_scores_from_card_values(packs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map per-card values (n_cards,) onto pack slots; PAD → -inf."""
    safe = np.where(packs == PAD, 0, packs)
    scores = values[safe].astype(np.float64)
    scores[packs == PAD] = NEG
    return scores


class CardValueBot:
    """A bot fully described by a static per-card value vector."""

    def __init__(self, name: str, values: np.ndarray):
        self.name = name
        self.values = values

    def score_drafts(self, packs, prev_picks=None):
        return _slot_scores_from_card_values(packs, self.values)

    def self_draft(self, packs):
        scores = self.score_drafts(packs)
        idx = scores.argmax(-1)
        return np.take_along_axis(packs, idx[..., None], axis=-1)[..., 0]


class RandomBot:
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score_drafts(self, packs, prev_picks=None):
        rng = np.random.default_rng(self.seed)  # fixed seed: eval reproducible
        scores = rng.random(packs.shape)
        scores[packs == PAD] = NEG
        return scores

    def self_draft(self, packs):
        idx = self.score_drafts(packs).argmax(-1)
        return np.take_along_axis(packs, idx[..., None], axis=-1)[..., 0]


class GihInLaneBot:
    """GIH-greedy, but from global position 8 on, restrict to the pool's top-2
    colors (colorless/lands always allowed); fall back to plain GIH when the
    pack has no in-lane option."""

    name = "gih-in-lane"

    def __init__(self, gih: np.ndarray, pips: np.ndarray, card_colors: np.ndarray):
        self.gih = gih              # (n_cards,)
        self.pips = pips            # (n_cards, 5)
        self.card_colors = card_colors  # (n_cards, 5) 0/1

    def _scores_step(self, packs_t, pool_pips):
        base = _slot_scores_from_card_values(packs_t, self.gih)
        top2 = np.argsort(-pool_pips, axis=-1)[:, :2]  # (N, 2)
        lane_mask = np.zeros((packs_t.shape[0], 5), dtype=bool)
        np.put_along_axis(lane_mask, top2, True, axis=-1)
        safe = np.where(packs_t == PAD, 0, packs_t)
        colors = self.card_colors[safe]  # (N, P, 5)
        in_lane = (colors * ~lane_mask[:, None, :]).sum(-1) == 0
        return base + 10.0 * in_lane

    def _run(self, packs, picks_source):
        """picks_source(t, own_picks) -> the pick ids to accrue into the pool."""
        n, t, _ = packs.shape
        scores = np.full(packs.shape, NEG)
        pool_pips = np.zeros((n, 5))
        own_picks = np.zeros((n, t), dtype=np.int16)
        for step in range(t):
            if step < 8:
                scores[:, step] = _slot_scores_from_card_values(packs[:, step], self.gih)
            else:
                scores[:, step] = self._scores_step(packs[:, step], pool_pips)
            idx = scores[:, step].argmax(-1)
            own_picks[:, step] = np.take_along_axis(packs[:, step], idx[:, None], -1)[:, 0]
            accrued = picks_source(step, own_picks)
            pool_pips += self.pips[accrued]
        return scores, own_picks

    def score_drafts(self, packs, prev_picks=None):
        # teacher-forced: the pool accrues the HUMAN's picks (prev_picks shifted)
        assert prev_picks is not None
        def human_pick(step, _own):
            nxt = step + 1
            if nxt < prev_picks.shape[1]:
                return prev_picks[:, nxt]  # prev_picks[t+1] == human pick at t
            return np.zeros(prev_picks.shape[0], dtype=np.int16)
        # NOTE: at the final step the accrued pick no longer matters
        scores, _ = self._run(packs, human_pick)
        return scores

    def self_draft(self, packs):
        _, own = self._run(packs, lambda step, own: own[:, step])
        return own


def build_bots(cards: pd.DataFrame, static_feats: pd.DataFrame,
               snapshot: pd.DataFrame | None) -> list:
    """Instantiate all five baseline bots for a set. `snapshot` may be None
    (day-0 condition) — then only the stat-free bots are returned."""
    n_cards = len(cards)
    rarity = cards["rarity"].map(RARITY_SCORE).fillna(0).to_numpy()
    # deterministic tie-break inside a rarity tier: card id (stable, documented)
    rarity_vals = rarity * 1000.0 - cards["id"].to_numpy() * 1e-3
    bots = [RandomBot(), CardValueBot("rarity-first", rarity_vals)]
    if snapshot is None:
        return bots
    alsa = snapshot["avg_seen"].to_numpy(np.float64)
    alsa_vals = np.where(np.isnan(alsa), NEG / 2, -alsa)  # missing → never take
    gih = snapshot["ever_drawn_win_rate"].to_numpy(np.float64)
    gih_vals = np.where(np.isnan(gih), -1.0, gih)
    pips = static_feats[[f"pips_{c}" for c in "WUBRG"]].to_numpy(np.float64)
    colors = static_feats[[f"color_{c}" for c in "WUBRG"]].to_numpy(np.float64)
    assert pips.shape[0] == n_cards
    bots += [
        CardValueBot("alsa-greedy", alsa_vals),
        CardValueBot("gih-greedy", gih_vals),
        GihInLaneBot(gih_vals, pips, colors),
    ]
    return bots
