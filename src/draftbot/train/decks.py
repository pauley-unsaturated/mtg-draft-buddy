"""Deck-builder training (PLAN P5.T1d).

`python -m draftbot.train.decks --config configs/EXP-101.yaml [--resume]`

Loss = per-copy membership BCE (weighted by pool copies × example weight)
     + land_lambda · CE over total-land count (14..20)
     + basics_lambda · soft-CE over the W/U/B/R/G basic split.
Example weight per DECK_DATA_PLAN curation: (1 + n_wins) · soft_skill(wr),
config-switchable (`weighting: win_skill|uniform`). Trains on ALL builds;
selection = val trophy-F1 (agreement with ≥5-win builds), computed through the
same greedy decode used at eval time.
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.deck_dataset import (example_weights, land_flags,
                                        load_deck_arrays)
from draftbot.data.splits import load_splits
from draftbot.models.base import device_auto
from draftbot.models.builder import (LAND_MAX, LAND_MIN, DeckBuilder,
                                     TorchDeckBuilder)
from draftbot.train.loop import CKPT_ROOT, cosine_lr, feature_tensor


def build_deck_model(cfg: dict, card_features: torch.Tensor) -> DeckBuilder:
    return DeckBuilder(card_features, emb_dim=cfg.get("emb_dim", 128),
                       heads=cfg.get("heads", 8), blocks=cfg.get("blocks", 3),
                       dropout=cfg.get("dropout", 0.1))


def deck_targets(arr, is_land: np.ndarray):
    """(member_frac (N,P), slot_weight (N,P), land_class (N,), basics_frac (N,5))"""
    pool = arr.pool_counts.astype(np.float32)
    frac = np.divide(arr.deck_counts, pool, out=np.zeros_like(pool),
                     where=pool > 0)
    ids = np.clip(arr.pool_ids, 0, None)
    nb_lands = (arr.deck_counts * is_land[ids] * (arr.pool_ids >= 0)).sum(1)
    total_lands = arr.basics.sum(1) + nb_lands
    land_class = np.clip(total_lands, LAND_MIN, LAND_MAX) - LAND_MIN
    bsum = arr.basics.sum(1, keepdims=True).astype(np.float32)
    basics_frac = np.divide(arr.basics, bsum,
                            out=np.full(arr.basics.shape, 0.2, np.float32),
                            where=bsum > 0)
    return frac.astype(np.float32), pool, land_class.astype(np.int64), basics_frac


def deck_loss(model, arr, targets, weights: np.ndarray, b: np.ndarray,
              device, cfg: dict) -> torch.Tensor:
    """The full training objective for a batch of build indices `b`."""
    frac, pool_w, land_cls, basics_frac = targets
    ids = torch.from_numpy(arr.pool_ids[b].astype(np.int64)).to(device)
    cnt = torch.from_numpy(arr.pool_counts[b].astype(np.int64)).to(device)
    ex_w = torch.from_numpy(weights[b]).to(device)

    member, land, basics = model(ids, cnt, ids == -1)
    slot_w = torch.from_numpy(pool_w[b]).to(device) * ex_w[:, None]
    target = torch.from_numpy(frac[b]).to(device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        member.clamp(min=-30), target, reduction="none")  # pads sit at -1e9
    member_loss = (bce * slot_w).sum() / slot_w.sum().clamp(min=1)

    land_t = torch.from_numpy(land_cls[b]).to(device)
    land_ce = torch.nn.functional.cross_entropy(land, land_t, reduction="none")
    land_loss = (land_ce * ex_w).sum() / ex_w.sum()

    basics_t = torch.from_numpy(basics_frac[b]).to(device)
    basics_ce = -(basics_t * torch.log_softmax(basics, -1)).sum(-1)
    basics_loss = (basics_ce * ex_w).sum() / ex_w.sum()

    return (member_loss + cfg.get("land_lambda", 0.2) * land_loss
            + cfg.get("basics_lambda", 0.2) * basics_loss)


class DeckTrainer:
    def __init__(self, cfg: dict, resume: bool = False):
        self.cfg = cfg
        self.exp = cfg["exp"]
        self.dir = CKPT_ROOT / self.exp
        self.dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.get("seed", 17))
        np.random.seed(cfg.get("seed", 17))
        torch.manual_seed(cfg.get("seed", 17))
        self.device = device_auto()
        set_code = cfg["set"]
        splits = load_splits(set_code)

        self.train_arr = load_deck_arrays(set_code,
                                          set(splits["random"]["train"]))
        self.val_arr = load_deck_arrays(set_code, set(splits["random"]["val"]),
                                        eval_builds=True)
        cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
        self.cards = cards
        self.is_land = land_flags(cards)

        if cfg.get("min_wins"):  # hard curation lever (hill-climb contrast)
            keep = self.train_arr.meta["n_wins"].to_numpy() >= cfg["min_wins"]
            self.train_arr = type(self.train_arr)(
                pool_ids=self.train_arr.pool_ids[keep],
                pool_counts=self.train_arr.pool_counts[keep],
                deck_counts=self.train_arr.deck_counts[keep],
                basics=self.train_arr.basics[keep],
                meta=self.train_arr.meta[keep].reset_index(drop=True))
            print(f"min_wins={cfg['min_wins']} kept {keep.sum()}/{len(keep)}")

        if cfg.get("weighting", "win_skill") == "win_skill":
            self.weights = example_weights(self.train_arr.meta)
        else:
            self.weights = np.ones(self.train_arr.n_builds, dtype=np.float32)

        feats, _ = feature_tensor(set_code, cfg.get("stats", "full"),
                                  ckpt_dir=self.dir)
        self.model = build_deck_model(cfg, feats).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        print(f"{self.exp}: {n_params / 1e6:.2f}M params on {self.device}, "
              f"{self.train_arr.n_builds} train builds")

        self.targets = deck_targets(self.train_arr, self.is_land)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9,
            weight_decay=cfg.get("weight_decay", 0.01))
        self.step = 0
        self.epoch = 0
        self.best_val = -1.0
        self.bad_epochs = 0
        self.writer = SummaryWriter(log_dir=str(Path("runs") / self.exp))
        if resume:
            self._load(self.dir / "last.pt")

    def _save(self, path: Path):
        torch.save({
            "model": self.model.state_dict(), "optim": self.optim.state_dict(),
            "step": self.step, "epoch": self.epoch, "best_val": self.best_val,
            "bad_epochs": self.bad_epochs, "cfg": self.cfg,
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
        }, path)

    def _load(self, path: Path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optim.load_state_dict(state["optim"])
        self.step, self.epoch = state["step"], state["epoch"]
        self.best_val, self.bad_epochs = state["best_val"], state["bad_epochs"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["np_rng"])
        print(f"resumed at epoch {self.epoch}, best {self.best_val:.4f}")

    def _loss(self, b: np.ndarray) -> torch.Tensor:
        return deck_loss(self.model, self.train_arr, self.targets,
                         self.weights, b, self.device, self.cfg)

    def val_f1(self) -> tuple[float, float]:
        """(overall deck-F1, trophy-F1) on val eval-builds via the real decode."""
        from draftbot.eval.decks import _f1, _true_build, _with_basics
        builder = TorchDeckBuilder(self.model, self.exp, self.device,
                                   self.is_land)
        builds = builder.build_all(self.val_arr)
        f1s = np.array([
            _f1(_with_basics(p["deck"], p["basics"]),
                _with_basics(*_true_build(self.val_arr, i)))
            for i, p in enumerate(builds)])
        trophy = self.val_arr.meta["n_wins"].to_numpy() >= 5
        self.model.train()
        return float(f1s.mean()), float(f1s[trophy].mean())

    def train(self) -> float:
        cfg = self.cfg
        bs = cfg.get("batch_builds", 256)
        epochs = cfg.get("epochs", 30)
        total_steps = int(np.ceil(self.train_arr.n_builds / bs)) * epochs
        clip = cfg.get("grad_clip", 1.0)
        self.model.train()
        start = time.time()
        while self.epoch < epochs:
            idx = np.random.permutation(self.train_arr.n_builds)
            epoch_loss, n_batches = 0.0, 0
            for s in range(0, len(idx), bs):
                lr = cosine_lr(self.step, total_steps,
                               cfg.get("lr_warmup", 300),
                               cfg.get("peak_lr", 3e-4))
                for g in self.optim.param_groups:
                    g["lr"] = lr
                self.optim.zero_grad(set_to_none=True)
                loss = self._loss(idx[s:s + bs])
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optim.step()
                self.step += 1
                epoch_loss += float(loss.detach())
                n_batches += 1
                if self.step % 100 == 0:
                    self.writer.add_scalar("train/loss", float(loss), self.step)
                    self.writer.add_scalar("train/lr", lr, self.step)
            self.epoch += 1
            val_f1, trophy_f1 = self.val_f1()  # trophy drives selection
            self.writer.add_scalar("val/deck_f1", val_f1, self.step)
            self.writer.add_scalar("val/trophy_f1", trophy_f1, self.step)
            mins = (time.time() - start) / 60
            print(f"epoch {self.epoch}/{epochs} "
                  f"loss {epoch_loss / max(n_batches, 1):.4f} "
                  f"val_f1 {val_f1:.4f} trophy {trophy_f1:.4f} "
                  f"({mins:.1f} min)", flush=True)
            improved = trophy_f1 > self.best_val
            if improved:
                self.best_val = trophy_f1
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            self._save(self.dir / "last.pt")
            if improved:
                self._save(self.dir / "best.pt")
            elif self.bad_epochs >= cfg.get("patience", 4):
                print(f"early stop at epoch {self.epoch} "
                      f"(best {self.best_val:.4f})")
                break
        self.writer.close()
        return self.best_val


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.train.decks")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    ckpt_dir = CKPT_ROOT / cfg["exp"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, ckpt_dir / "config.yaml")
    best = DeckTrainer(cfg, resume=args.resume).train()
    print(f"best val trophy-F1: {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
