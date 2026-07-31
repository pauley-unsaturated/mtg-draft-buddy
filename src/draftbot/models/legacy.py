"""LegacyDraftBot — faithful PyTorch port of the 2022 TF DraftBot (PLAN P1.T6).

Architecture (mirrors mtg/ml/models.py):
- ConcatEmbedding: learned half + card-feature-MLP half, with a "bias" token
  (index n_cards) used as the previous-pick at P1P1.
- Encoder: mean-pooled pack embedding × sqrt(emb) + positional embedding →
  post-LN transformer blocks (ReLU FFN), causal mask.
- Decoder: previous-pick embeddings → self-attention (causal) + cross-attention
  to encoder output, post-LN.
- Output MLP (reverse bottleneck) → per-card logits, masked to the pack, then
  gathered into slot space.
- Loss handled by the trainer: slot NLL + triplet embedding loss + rare/cmc
  behavioral priors + importance weighting.

Basics stay in the vocab (2026 data has them picked; PLAN P0.T5).
"""

import torch
import torch.nn.functional as F
from torch import nn

from draftbot.data.dataset import PAD
from draftbot.models.base import NEG_INF


class ConcatEmbedding(nn.Module):
    def __init__(self, n_tokens: int, emb_dim: int, card_features: torch.Tensor):
        super().__init__()
        half = emb_dim // 2
        self.learned = nn.Embedding(n_tokens, half)
        feat_dim = card_features.shape[1]
        self.register_buffer("card_features", card_features)  # (n_tokens, F)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
            nn.Linear(feat_dim // 2, feat_dim // 4), nn.ReLU(),
            nn.Linear(feat_dim // 4, half),
        )

    def all_embeddings(self) -> torch.Tensor:
        return torch.cat([self.learned.weight, self.mlp(self.card_features)], -1)


class PostLNBlock(nn.Module):
    def __init__(self, emb: int, heads: int, ffn: int, dropout: float,
                 cross: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb, heads, batch_first=True)
        self.ln_attn = nn.LayerNorm(emb)
        self.cross = cross
        if cross:
            self.cross_attn = nn.MultiheadAttention(emb, heads, batch_first=True)
            self.ln_cross = nn.LayerNorm(emb)
        self.ffn = nn.Sequential(nn.Linear(emb, ffn), nn.ReLU(), nn.Linear(ffn, emb))
        self.ln_out = nn.LayerNorm(emb)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask, enc=None):
        a, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln_attn(x + self.drop(a))
        if self.cross:
            a, _ = self.cross_attn(x, enc, enc, attn_mask=causal_mask,
                                   need_weights=False)
            x = self.ln_cross(x + self.drop(a))
        return self.ln_out(x + self.drop(self.ffn(x)))


class LegacyDraftBot(nn.Module):
    name = "legacy"

    def __init__(self, n_cards: int, card_features: torch.Tensor, t: int = 42,
                 emb_dim: int = 128, enc_layers: int = 2, dec_layers: int = 2,
                 enc_heads: int = 8, dec_heads: int = 8, ffn_width: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        self.n_cards, self.t, self.emb_dim = n_cards, t, emb_dim
        ffn_width = ffn_width or emb_dim * 4
        assert card_features.shape[0] == n_cards + 1  # + bias token row
        self.embedding = ConcatEmbedding(n_cards + 1, emb_dim, card_features)
        self.pos_embedding = nn.Embedding(t, emb_dim)
        self.encoder = nn.ModuleList(
            [PostLNBlock(emb_dim, enc_heads, ffn_width, dropout) for _ in range(enc_layers)])
        self.decoder = nn.ModuleList(
            [PostLNBlock(emb_dim, dec_heads, ffn_width, dropout, cross=True)
             for _ in range(dec_layers)])
        self.out_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(),
            nn.Linear(emb_dim * 4, n_cards),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, packs: torch.Tensor, prev_picks: torch.Tensor,
                return_states: bool = False):
        B, t, P = packs.shape
        device = packs.device
        all_emb = self.embedding.all_embeddings()  # (n_cards+1, E)

        pack_ids = packs.clamp(min=0)
        pack_emb = all_emb[pack_ids]                       # (B,t,P,E)
        pack_valid = (packs != PAD).unsqueeze(-1)
        n_options = pack_valid.sum(2).clamp(min=1)
        pack_mean = (pack_emb * pack_valid).sum(2) / n_options  # (B,t,E)

        positions = torch.arange(t, device=device)
        x = pack_mean * (self.emb_dim ** 0.5) + self.pos_embedding(positions)[None]
        x = self.drop(x)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=device), 1)
        for block in self.encoder:
            x = block(x, causal)

        prev_ids = torch.where(prev_picks == PAD,
                               torch.full_like(prev_picks, self.n_cards), prev_picks)
        y = self.drop(all_emb[prev_ids])
        for block in self.decoder:
            y = block(y, causal, enc=x)

        vocab_logits = self.out_mlp(y)                     # (B,t,n_cards)
        slot_logits = torch.gather(
            vocab_logits, 2, pack_ids.view(B, t, P))
        slot_logits = slot_logits.masked_fill(packs == PAD, NEG_INF)
        if return_states:
            return slot_logits, y, pack_emb
        return slot_logits

    def auxiliary_loss(self, packs, picks, slot_logits, y, pack_emb, weights,
                       rare_flag: torch.Tensor, cmc: torch.Tensor,
                       margin: float = 0.1, emb_lambda: float = 1.0,
                       rare_lambda: float = 10.0, cmc_lambda: float = 1.0,
                       cmc_margin: float = 1.0):
        """Triplet embedding loss + rare/cmc behavioral priors (original semantics)."""
        valid = packs != PAD
        target = packs == picks.unsqueeze(-1)
        wsum = weights.sum().clamp(min=1e-8)

        dists = torch.linalg.vector_norm(pack_emb - y.unsqueeze(2), dim=-1)
        d_correct = (dists * target).sum(-1, keepdim=True) / target.sum(-1, keepdim=True).clamp(min=1)
        hinge = (d_correct - dists + margin).clamp(min=0) * (valid & ~target)
        triplet = (hinge.sum(-1) * weights).sum() / wsum

        probs = torch.softmax(slot_logits, dim=-1) * valid
        safe_ids = packs.clamp(min=0)
        rare_p = (probs * rare_flag[safe_ids]).sum(-1)
        took_rare = (rare_flag[picks.clamp(min=0)] > 0).float()
        rare_loss = (((1 - took_rare) * rare_p) * weights).sum() / wsum

        pred_cmc = (probs * cmc[safe_ids]).sum(-1)
        true_cmc = cmc[picks.clamp(min=0)]
        cmc_loss = (((pred_cmc - true_cmc + cmc_margin).clamp(min=0)) * weights).sum() / wsum

        return emb_lambda * triplet + rare_lambda * rare_loss + cmc_lambda * cmc_loss
