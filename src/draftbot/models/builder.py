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

import numpy as np
import torch
from torch import nn

from draftbot.data.dataset import PAD
from draftbot.models.modern import PMA, SetAttentionBlock

LAND_MIN, LAND_MAX = 14, 20
MASK_STATE = 8  # deck_state: 0..7 = revealed maindeck count; 8 = masked


class DeckBuilder(nn.Module):
    """Stage 1 (diffusion=False): one-shot membership from the pool alone.

    Stage 2 (diffusion=True, PLAN P5 stage-2 proposal): adds a per-slot
    deck_state embedding (revealed count or MASK) — trained with random
    masking, decoded by iterative commit — plus a deck-quality head trained on
    within-pool winner/loser contrast pairs. state_emb is zero-initialized so
    a warm-started model with everything masked reproduces stage-1 exactly."""

    def __init__(self, card_features: torch.Tensor, emb_dim: int = 128,
                 heads: int = 8, blocks: int = 3, dropout: float = 0.1,
                 diffusion: bool = False, quality_basics: bool = False):
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
        self.diffusion = diffusion
        self.quality_basics = quality_basics
        if diffusion:
            self.state_emb = nn.Embedding(MASK_STATE + 1, emb_dim)
            nn.init.zeros_(self.state_emb.weight)
            q_in = emb_dim + (5 if quality_basics else 0)
            self.quality_head = nn.Sequential(
                nn.Linear(q_in, emb_dim), nn.SiLU(), nn.Linear(emb_dim, 1))

    def _trunk(self, pool_ids, pool_counts, pad_mask, deck_state=None):
        x = self.encoder(self.card_features[pool_ids.clamp(min=0)])
        x = x + self.count_emb(pool_counts.clamp(min=0, max=7))
        if self.diffusion:
            if deck_state is None:
                deck_state = torch.full_like(pool_ids, MASK_STATE)
            x = x + self.state_emb(deck_state.clamp(min=0, max=MASK_STATE))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return x, self.pma(x, pad_mask)

    def forward(self, pool_ids: torch.Tensor, pool_counts: torch.Tensor,
                pad_mask: torch.Tensor, deck_state: torch.Tensor | None = None):
        """pool_ids (B,N) vocab ids, pool_counts (B,N) copies, pad_mask (B,N)
        True where padded, deck_state (B,N) revealed counts / MASK_STATE
        (diffusion models only). Returns (member_logits (B,N),
        land_logits (B,7), basics_logits (B,5))."""
        x, pooled = self._trunk(pool_ids, pool_counts, pad_mask, deck_state)
        member = self.member_head(x)[..., 0].masked_fill(pad_mask, -1e9)
        return member, self.land_head(pooled), self.basics_head(pooled)

    def quality(self, pool_ids, pool_counts, pad_mask, deck_state,
                basics: torch.Tensor | None = None):
        """(B,) deck-quality score of a FULLY revealed build (win-aware aux).
        basics: (B,5) basic counts / 20 — required when quality_basics (the
        manabase is part of the deck being judged)."""
        _, pooled = self._trunk(pool_ids, pool_counts, pad_mask, deck_state)
        if self.quality_basics:
            pooled = torch.cat([pooled, basics], -1)
        return self.quality_head(pooled)[..., 0]


class TorchDeckBuilder:
    """Batched inference + decode; implements the deck-eval interface
    (`.name`, `.build_all(DeckArrays)`).

    decode: greedy | expected — one forward pass through build_deck.
            maskgit — iterative commit (diffusion models): cosine schedule
              commits the most-confident slots each step, re-runs conditioned
              on them; deterministic, no sampling (eval stays seed-free).
            rescore — assemble deterministic candidates (one-shot + maskgit at
              several step counts), score each with the quality head, keep the
              best per pool (the winner-pref play)."""

    def __init__(self, model: DeckBuilder, name: str, device,
                 is_land_flags: np.ndarray, batch: int = 512,
                 decode: str = "greedy", steps: int = 8):
        self.model = model.to(device).eval()
        self.name = name
        self.device = device
        self.is_land = is_land_flags
        self.batch = batch
        self.decode = decode
        self.steps = steps

    @torch.no_grad()
    def _heads(self, ids, cnt, pad, state=None):
        member, land, basics = self.model(ids, cnt, pad, state)
        return (torch.sigmoid(member), torch.softmax(land, -1),
                torch.softmax(basics, -1))

    @torch.no_grad()
    def _maskgit_probs(self, ids, cnt, pad, steps: int):
        import math
        state = torch.full_like(ids, MASK_STATE)
        committed = torch.zeros_like(pad)
        n_valid = (~pad).sum(1)
        member, land, basics = self._heads(ids, cnt, pad, state)
        for t in range(steps - 1):
            frac_masked = math.cos(math.pi / 2 * (t + 1) / steps)
            target = (n_valid.float() * (1 - frac_masked)).ceil().long()
            conf = (member - 0.5).abs().masked_fill(pad | committed, -1.0)
            ranks = conf.argsort(1, descending=True).argsort(1)
            quota = (target - committed.sum(1)).clamp(min=0)
            newly = (ranks < quota[:, None]) & ~pad & ~committed
            counts = torch.round(member * cnt).long().clamp(0, MASK_STATE - 1)
            state = torch.where(newly, counts, state)
            committed |= newly
            member, land, basics = self._heads(ids, cnt, pad, state)
        pinned = torch.where(committed,
                             state.float() / cnt.clamp(min=1).float(), member)
        return pinned, land, basics

    def _assemble(self, member, land, basics, arr, s, decode="greedy"):
        out = []
        m, ln, b = member.cpu().numpy(), land.cpu().numpy(), basics.cpu().numpy()
        for j in range(m.shape[0]):
            i = s + j
            d = build_deck(m[j], ln[j], b[j], arr.pool_ids[i],
                           arr.pool_counts[i], self.is_land, decode=decode)
            out.append({"deck": d["deck"], "basics": d["basics"]})
        return out

    @torch.no_grad()
    def build_all(self, arr) -> list[dict]:
        out = []
        for s in range(0, arr.n_builds, self.batch):
            ids = torch.from_numpy(arr.pool_ids[s:s + self.batch]
                                   .astype(np.int64)).to(self.device)
            cnt = torch.from_numpy(arr.pool_counts[s:s + self.batch]
                                   .astype(np.int64)).to(self.device)
            pad = ids == PAD
            if self.decode == "maskgit":
                probs = self._maskgit_probs(ids, cnt, pad, self.steps)
                out.extend(self._assemble(*probs, arr, s))
            elif self.decode == "rescore":
                cands = [self._assemble(*self._heads(ids, cnt, pad), arr, s)]
                for T in (2, 4, self.steps, self.steps + 4):
                    cands.append(self._assemble(
                        *self._maskgit_probs(ids, cnt, pad, T), arr, s))
                scores = []
                for cand in cands:
                    state = torch.zeros_like(ids)
                    bas = torch.zeros((ids.shape[0], 5), device=self.device)
                    for j, pred in enumerate(cand):
                        row = {int(k): int(v) for k, v in pred["deck"].items()}
                        st = [row.get(int(c), 0) for c in arr.pool_ids[s + j]]
                        state[j] = torch.tensor(st, device=self.device)
                        bas[j] = torch.tensor(np.asarray(pred["basics"]) / 20.0,
                                              device=self.device,
                                              dtype=torch.float32)
                    state = state.clamp(0, MASK_STATE - 1).masked_fill(
                        pad, MASK_STATE)
                    scores.append(self.model.quality(ids, cnt, pad, state, bas))
                best = torch.stack(scores).argmax(0).cpu().numpy()
                out.extend(cands[int(best[j])][j] for j in range(ids.shape[0]))
            else:
                member, land, basics = self._heads(ids, cnt, pad)
                out.extend(self._assemble(member, land, basics, arr, s,
                                          decode=self.decode))
        return out


def build_deck(member_probs, land_probs, basics_probs, pool_ids, pool_counts,
               is_land_flags, decode: str = "greedy") -> dict:
    """Greedy 40-card assembly from head outputs (single example, numpy).
    Padded slots are harmless: their pool_count is 0.

    decode="expected": the number of nonbasic spells comes from the membership
    head's expected count (Σ p·copies over non-lands, clamped to 20..26)
    instead of 40 − land-head argmax — the two heads stop fighting over the
    spell/land boundary."""
    total_lands = LAND_MIN + int(np.argmax(land_probs))
    if decode == "expected":
        nonland = ~is_land_flags[np.clip(pool_ids, 0, None)]
        exp_spells = float((member_probs * pool_counts * nonland).sum())
        n_spells_target = int(np.clip(round(exp_spells), 40 - LAND_MAX,
                                      40 - LAND_MIN))
    else:
        n_spells_target = 40 - total_lands
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
            elif n_spells < n_spells_target:
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
