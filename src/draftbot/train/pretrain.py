"""Multi-set pretraining (PLAN P3.T4/T5).

`python -m draftbot.train.pretrain --config configs/EXP-030.yaml [--resume]`

The trunk is a feature-only ModernDraftBot with a set-summary token, built with
a dummy home vocab; every batch swaps in its set's card context. Stat-dropout
(p per batch) swaps the statless feature table so day-0 inference is a trained
condition. Zero-shot MSH-val top-1 is logged every eval_every steps — the
hill-climb signal. MSH data NEVER enters training; its snapshots are used at
eval time only (quarantine discipline; see tests/test_quarantine.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from draftbot.data.corpus import add_bias_row, batch_iterator, load_corpus
from draftbot.data.features import assemble, load_scaler, save_scaler
from draftbot.data.splits import load_splits
from draftbot.eval.harness import topk_hits
from draftbot.models.base import device_auto, slot_nll
from draftbot.models.modern import ModernDraftBot
from draftbot.train.loop import CKPT_ROOT, cosine_lr, set_seed

T_MAX = 45  # draft-booster era: 3×15; play boosters: 3×14


def build_trunk(cfg: dict, feat_dim: int) -> ModernDraftBot:
    dummy = torch.zeros(1, feat_dim)
    return ModernDraftBot(
        0, dummy, t=T_MAX, emb_dim=cfg.get("emb_dim", 256),
        layers=cfg.get("layers", 6), heads=cfg.get("heads", 8),
        dropout=cfg.get("dropout", 0.1), use_set_encoder=True,
        pointer_head=True, feature_only=True, use_set_token=True)


class ZeroShotEval:
    """Frozen MSH-val pack sequences + feature tables in all three stat modes."""

    def __init__(self, scaler: dict, device):
        splits = load_splits("MSH")
        from draftbot.data.dataset import load_draft_arrays
        self.arr = load_draft_arrays("MSH", draft_ids=set(splits["random"]["val"]))
        self.tables = {}
        for mode in ("none", "week1", "full"):
            feats, _ = assemble("MSH", None if mode == "none" else mode,
                                scaler={c: scaler[c] for c in scaler})
            if mode == "none":
                # static-only table lacks stat columns; rebuild at full width
                full, _ = assemble("MSH", "full", scaler=scaler)
                from draftbot.data.corpus import _statless
                feats = _statless(full)
            self.tables[mode] = torch.tensor(add_bias_row(feats)).to(device)

    @torch.no_grad()
    def top1(self, model: ModernDraftBot, mode: str, device,
             batch: int = 512) -> float:
        model.eval()
        model.embedding.set_context(self.tables[mode])
        hits_sum = n = 0
        for i in range(0, self.arr.n_drafts, batch):
            p = torch.from_numpy(self.arr.packs[i:i + batch].astype(np.int64)).to(device)
            pp = torch.from_numpy(self.arr.prev_picks[i:i + batch].astype(np.int64)).to(device)
            scores = model(p, pp).float().cpu().numpy()
            h = topk_hits(self.arr.packs[i:i + batch], self.arr.picks[i:i + batch],
                          scores, kmax=1)
            hits_sum += h[0].sum()
            n += h[0].size
        model.train()
        return float(hits_sum / n)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="draftbot.train.pretrain")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    exp = cfg["exp"]
    ckpt_dir = CKPT_ROOT / exp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("seed", 17))
    device = device_auto()

    scaler = None
    if args.resume and (ckpt_dir / "scaler.json").exists():
        scaler = load_scaler(ckpt_dir / "scaler.json")
    bundles, scaler = load_corpus(Path(cfg["corpus"]),
                                  skill_filter=cfg.get("skill_filter"), scaler=scaler)
    save_scaler(scaler, ckpt_dir / "scaler.json")
    feat_dim = bundles[0].features_full.shape[1]
    (ckpt_dir / "meta.json").write_text(json.dumps(
        {"feat_dim": feat_dim, "t_max": T_MAX, "corpus": cfg["corpus"]}))

    model = build_trunk(cfg, feat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{exp}: trunk {n_params / 1e6:.1f}M params, {len(bundles)} sets, "
          f"feat_dim {feat_dim}", flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98),
                              eps=1e-9, weight_decay=cfg.get("weight_decay", 0.01))
    total_steps = cfg.get("total_steps", 20000)
    warmup = cfg.get("lr_warmup", 1000)
    peak = cfg.get("peak_lr", 3e-4)
    stat_dropout = cfg.get("stat_dropout", 0.25)
    eval_every = cfg.get("eval_every", 1000)
    smoothing = cfg.get("label_smoothing", 0.05)
    step = 0
    best_zs = -1.0

    if args.resume and (ckpt_dir / "last.pt").exists():
        state = torch.load(ckpt_dir / "last.pt", map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        step, best_zs = state["step"], state["best_zs"]
        print(f"resumed at step {step} (best zs {best_zs:.4f})", flush=True)

    zse = ZeroShotEval(scaler, device)
    writer = SummaryWriter(log_dir=str(Path("runs") / exp))
    rng = np.random.default_rng(cfg.get("seed", 17) + step)
    tables = {b.code: (torch.tensor(b.features_full).to(device),
                       torch.tensor(b.features_statless).to(device))
              for b in bundles}
    it = batch_iterator(bundles, cfg.get("batch_drafts", 64), rng)
    model.train()
    start = time.time()

    def save(path):
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                    "step": step, "best_zs": best_zs, "cfg": cfg}, path)

    for bundle, idx in it:
        if step >= total_steps:
            break
        full_tab, statless_tab = tables[bundle.code]
        table = statless_tab if rng.random() < stat_dropout else full_tab
        model.embedding.set_context(table)
        packs = torch.from_numpy(bundle.arrays.packs[idx].astype(np.int64)).to(device)
        prev = torch.from_numpy(bundle.arrays.prev_picks[idx].astype(np.int64)).to(device)
        picks = torch.from_numpy(bundle.arrays.picks[idx].astype(np.int64)).to(device)
        w = torch.from_numpy(bundle.weights[idx]).to(device)
        lr = cosine_lr(step, total_steps, warmup, peak)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        logits = model(packs, prev)
        loss = slot_nll(logits, packs, picks, w, smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optim.step()
        step += 1
        if step % 100 == 0:
            writer.add_scalar("pretrain/loss", float(loss.detach()), step)
            writer.add_scalar("pretrain/lr", lr, step)
        if step % eval_every == 0 or step == total_steps:
            zs = {m: zse.top1(model, m, device) for m in ("none", "week1", "full")}
            mins = (time.time() - start) / 60
            print(f"step {step}/{total_steps} loss {float(loss.detach()):.4f} "
                  f"zs none {zs['none']:.4f} week1 {zs['week1']:.4f} "
                  f"full {zs['full']:.4f} ({mins:.1f} min)", flush=True)
            for m, v in zs.items():
                writer.add_scalar(f"zeroshot_msh_val/{m}", v, step)
            save(ckpt_dir / "last.pt")
            if zs["week1"] > best_zs:
                best_zs = zs["week1"]
                save(ckpt_dir / "best.pt")
    writer.close()
    print(f"done: best zero-shot(week1) MSH val top-1 = {best_zs:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
