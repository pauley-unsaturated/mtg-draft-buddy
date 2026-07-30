"""DeckBuilder stage 1 (PLAN P5.T1): one-shot per-card maindeck membership.

Pool (nonbasic multiset, incl. nonbasic lands) → set attention → three heads:
  membership: per-card P(this copy belongs in the deck)
  land_count: distribution over total lands 14..20
  basics:     color distribution for basic lands

Inference greedily fills 40 slots: rank cards by membership, honor the
predicted land count, split basics per the color head. Nonbasic lands compete
for slots like any other card — a BG tapland in a WB deck gets the low
membership it deserves.
"""

import torch
from torch import nn

from draftbot.models.modern import PMA, SetAttentionBlock

LAND_MIN, LAND_MAX = 14, 20


class DeckBuilder(nn.Module):
    def __init__(self, card_features: torch.Tensor, emb_dim: int = 128,
                 heads: int = 8, blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("card_features", card_features)  # (n_cards, F)
        f = card_features.shape[1]
        self.encoder = nn.Sequential(
            nn.Linear(f, emb_dim * 2), nn.SiLU(), nn.Linear(emb_dim * 2, emb_dim))
        self.count_emb = nn.Embedding(8, emb_dim)  # copies of this card in pool
        self.blocks = nn.ModuleList(
            [SetAttentionBlock(emb_dim, heads, dropout) for _ in range(blocks)])
        self.pma = PMA(emb_dim, heads)
        self.member_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, 1))
        self.land_head = nn.Linear(emb_dim, LAND_MAX - LAND_MIN + 1)
        self.basics_head = nn.Linear(emb_dim, 5)
        self.drop = nn.Dropout(dropout)

    def forward(self, pool_ids: torch.Tensor, pool_counts: torch.Tensor,
                pad_mask: torch.Tensor):
        """pool_ids (B,N) vocab ids, pool_counts (B,N) copies, pad_mask (B,N)
        True where padded. Returns (member_logits (B,N), land_logits (B,7),
        basics_logits (B,5))."""
        x = self.encoder(self.card_features[pool_ids.clamp(min=0)])
        x = x + self.count_emb(pool_counts.clamp(min=0, max=7))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        pooled = self.pma(x, pad_mask)
        member = self.member_head(x)[..., 0].masked_fill(pad_mask, -1e9)
        return member, self.land_head(pooled), self.basics_head(pooled)


def build_deck(member_probs, land_probs, basics_probs, pool_ids, pool_counts,
               is_land_flags) -> dict:
    """Greedy 40-card assembly from head outputs (single example, numpy)."""
    import numpy as np

    total_lands = LAND_MIN + int(np.argmax(land_probs))
    order = np.argsort(-member_probs)
    deck: dict[int, int] = {}
    n_spells = n_nb_lands = 0
    for i in order:
        cid, copies = int(pool_ids[i]), int(pool_counts[i])
        for _ in range(copies):
            if is_land_flags[cid]:
                if n_nb_lands < total_lands and member_probs[i] > 0.5:
                    deck[cid] = deck.get(cid, 0) + 1
                    n_nb_lands += 1
            elif n_spells < 40 - total_lands:
                deck[cid] = deck.get(cid, 0) + 1
                n_spells += 1
    # basics fill whatever is left — the deck is ALWAYS exactly 40 cards even
    # if the pool ran short of spells or the land head over/under-shot
    n_basics = 40 - n_spells - n_nb_lands
    share = basics_probs / max(basics_probs.sum(), 1e-9)
    basic_counts = np.floor(share * n_basics).astype(int)
    while basic_counts.sum() < n_basics:
        basic_counts[int(np.argmax(share - basic_counts / max(n_basics, 1)))] += 1
    return {"deck": deck, "total_lands": n_basics + n_nb_lands,
            "basics": basic_counts}
