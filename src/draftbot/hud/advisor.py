"""Model advisor for the HUD (H2): turn the current DraftState into ranked
pick suggestions with calibrated probabilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.dataset import PAD


@dataclass
class Suggestion:
    card_id: int
    name: str
    rarity: str
    prob: float          # calibrated probability (duplicate copies merged)
    rank: int
    gih: float | None
    alsa: float | None


class Advisor:
    def __init__(self, scorers: list, set_code: str):
        self.scorers = scorers
        cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
        self.names = dict(zip(cards["id"], cards["name"]))
        self.rarity = dict(zip(cards["id"], cards["rarity"]))
        try:
            from draftbot.data.features import load_snapshot
            snap = load_snapshot(set_code, "full")
            self.gih = dict(zip(cards["id"], snap["ever_drawn_win_rate"]))
            self.alsa = dict(zip(cards["id"], snap["avg_seen"]))
        except Exception:
            self.gih, self.alsa = {}, {}

    def rank_pack(self, state) -> dict[str, list[Suggestion]]:
        """Per-scorer ranked suggestions for the state's current pack."""
        packs, prev, pos = state.arrays()
        out: dict[str, list[Suggestion]] = {}
        for scorer in self.scorers:
            scores = scorer.score_drafts(packs, prev)[0, pos]
            slots = packs[0, pos]
            valid = slots != PAD
            z = scores[valid] - scores[valid].max()
            probs = np.exp(z) / np.exp(z).sum()
            by_card: dict[int, float] = {}
            for cid, p in zip(slots[valid], probs):
                by_card[int(cid)] = by_card.get(int(cid), 0.0) + float(p)
            ranked = sorted(by_card.items(), key=lambda kv: -kv[1])
            out[scorer.name] = [
                Suggestion(card_id=cid, name=self.names.get(cid, f"#{cid}"),
                           rarity=self.rarity.get(cid, "?"), prob=p, rank=i + 1,
                           gih=self.gih.get(cid), alsa=self.alsa.get(cid))
                for i, (cid, p) in enumerate(ranked)]
        return out
