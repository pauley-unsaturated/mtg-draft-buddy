"""Shared model interface, loss, and the numpy<->torch scorer adapter.

DraftModel contract (consumed by P1.T4 correctness tests and the eval harness):
  forward(packs, prev_picks) -> slot logits (B, t, P); PAD slots = -inf.
  packs: (B, t, P) long, card ids with PAD (-1) padding
  prev_picks: (B, t) long, previous human pick per step (-1 = start-of-draft bias)

All losses/metrics live in slot space; probability of a card = sum of its slot
probabilities (duplicates collapse in the harness).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from draftbot.data.dataset import PAD

NEG_INF = -1e9


def device_auto() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def slot_nll(logits: torch.Tensor, packs: torch.Tensor, picks: torch.Tensor,
             weights: torch.Tensor | None = None,
             label_smoothing: float = 0.0) -> torch.Tensor:
    """-log P(picked card) with softmax over valid slots.

    logits: (B,t,P) with PAD slots already at -inf; picks: (B,t) card ids.
    weights: (B,t) importance weights (normalized by their sum).
    """
    valid = packs != PAD
    target = packs == picks.unsqueeze(-1)
    logz = torch.logsumexp(logits, dim=-1)
    log_target = torch.logsumexp(
        logits.masked_fill(~target, NEG_INF), dim=-1)
    nll = logz - log_target
    if label_smoothing > 0:
        # smooth toward uniform over the valid slots
        n_valid = valid.sum(-1).clamp(min=1)
        mean_logp = (logits.masked_fill(~valid, 0).sum(-1) / n_valid) - logz
        nll = (1 - label_smoothing) * nll - label_smoothing * mean_logp
    if weights is None:
        return nll.mean()
    return (nll * weights).sum() / weights.sum().clamp(min=1e-8)


class TorchScorer:
    """Wrap a trained DraftModel for the numpy eval harness."""

    def __init__(self, model: nn.Module, name: str, device=None,
                 batch_size: int = 256):
        self.model = model.eval()
        self.name = name
        self.device = device or device_auto()
        self.model.to(self.device)
        self.batch_size = batch_size

    @torch.no_grad()
    def score_drafts(self, packs: np.ndarray, prev_picks: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(packs), self.batch_size):
            p = torch.from_numpy(packs[i:i + self.batch_size].astype(np.int64)).to(self.device)
            pp = torch.from_numpy(prev_picks[i:i + self.batch_size].astype(np.int64)).to(self.device)
            out.append(self.model(p, pp).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    @torch.no_grad()
    def self_draft(self, packs: np.ndarray) -> np.ndarray:
        n, t, P = packs.shape
        own = np.full((n, t), PAD, dtype=np.int64)
        prev = np.full((n, t), PAD, dtype=np.int64)
        for step in range(t):
            scores = self.score_drafts(packs[:, : step + 1], prev[:, : step + 1])
            idx = scores[:, step].argmax(-1)
            own[:, step] = np.take_along_axis(packs[:, step], idx[:, None], -1)[:, 0]
            if step + 1 < t:
                prev[:, step + 1] = own[:, step]
        return own


def importance_weights(meta_rank: np.ndarray, user_win_rate: np.ndarray,
                       event_wins: np.ndarray, event_losses: np.ndarray,
                       draft_time: np.ndarray, t: int,
                       minim: float = 0.1, maxim: float = 1.0) -> np.ndarray:
    """Faithful port of the original importance_weighting (per draft × position)."""
    rank_to_score = {"bronze": 0.01, "silver": 0.1, "gold": 0.25,
                     "platinum": 0.5, "diamond": 0.75, "mythic": 1.0}
    rank_add = np.array([rank_to_score.get(r, 0.5) for r in meta_rank])
    wr = np.nan_to_num(user_win_rate, nan=0.5)
    scaled_wr = np.clip(wr ** (2 - rank_add), minim, maxim)
    total = event_wins + event_losses
    won = np.divide(event_wins, np.maximum(total, 1),
                    out=np.zeros_like(event_wins, dtype=float), where=total > 0)
    won = np.clip(won, 0.5, 1.0)
    dt = draft_time.astype("datetime64[D]")
    n_weeks = ((dt.max() - dt).astype(int)) // 7
    recency = 0.9 ** n_weeks
    per_draft = scaled_wr * won * recency  # (N,)

    picks_per_pack = t // 3
    pick_nums = (np.arange(t) % picks_per_pack) + 1
    alpha = np.e / 5.0
    position_scale = (np.log(pick_nums) + 1) / np.power(pick_nums, alpha)  # (t,)
    return per_draft[:, None] * position_scale[None, :]
