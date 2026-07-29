"""Rebuild a trained model + feature pathway from a checkpoint directory."""

from pathlib import Path

import pandas as pd
import torch

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.features import load_scaler
from draftbot.models.base import TorchScorer, device_auto


def scorer_from_checkpoint(ckpt: Path, set_code: str, stats: str) -> TorchScorer:
    """ckpt may be .../best.pt, .../last.pt or the EXP directory itself.

    `stats` selects the snapshot used to build features at EVAL time (none|week1|
    full) — the day-0/day-7 evaluation lever. The scaler always comes from
    training (scaler.json next to the checkpoint)."""
    ckpt = Path(ckpt)
    ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent
    ckpt_file = ckpt if ckpt.is_file() else ckpt_dir / "best.pt"
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = state["cfg"]

    from draftbot.train.loop import build_model, feature_tensor
    scaler = load_scaler(ckpt_dir / "scaler.json")
    has_stat_cols = "avg_seen" in scaler  # draft-stage stat present ⇔ stats-trained
    if stats == "none" and has_stat_cols:
        # stat-trained model in day-0 mode: full-width table, stats zeroed, masks on
        from draftbot.data.corpus import _statless, add_bias_row
        from draftbot.data.features import assemble
        feats_df, _ = assemble(set_code, "full", scaler=scaler)
        feats = torch.tensor(add_bias_row(_statless(feats_df)))
    elif stats != "none" and not has_stat_cols:
        raise SystemExit("model was trained without stats; use --stats-snapshot none")
    else:
        feats, _ = feature_tensor(set_code, stats, ckpt_dir=None, scaler=scaler)
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    t = 42 if set_code == "MSH" else cfg.get("t", 42)
    model = build_model(cfg, len(cards), feats, t)
    model.load_state_dict(state["model"])
    name = f"{cfg['exp']}@{ckpt_file.stem}"
    return TorchScorer(model, name, device_auto())
