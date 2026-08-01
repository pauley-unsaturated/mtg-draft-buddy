"""Deck-builder section of the HUD (HUD_PLAN H5, design states B1-B5).

Turns the drafted pool into a proposed 40-card build plus the review affordances
the panel needs: per-card membership confidence, the shared boundary zone (last
cards in / first cards out), the cut list, mana summary, and a lock-and-rebuild
path.

Two checkpoints, per docs/DECKBUILDER_HANDOFF.md:
  primary  EXP-121  one-shot builder — the proposed build
  rebuild  EXP-116  diffusion variant — conditions on locked slots natively

Locks are expressed as a `deck_state` tensor (count = locked in, 0 = locked out,
MASK_STATE = model's call) and driven through `_maskgit_probs(init_state=…)`.
Without the diffusion checkpoint we still honour locks by pinning the primary
model's membership probabilities before greedy assembly — same guarantee, less
context for the model.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from draftbot.data.cards import PROCESSED_DIR
from draftbot.data.dataset import PAD
from draftbot.models.builder import MASK_STATE, build_deck

BASIC_NAMES = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
PIP_COLS = [f"pips_{c}" for c in "WUBRG"]

# boundary zone: rows below this membership confidence stop reading as
# settled mainboard and join the violet band with the nearest cuts
BOUNDARY_HI = 0.70
BOUNDARY_HI_DAY0 = 0.80   # day-0 builds are less certain — widen the band
BOUNDARY_MAX = 3          # rows per side; day-0 gets one more

# a rough limited curve, used ONLY to name the weakest slot in the provisional
# view. Not model output — a talking point while the draft is still live.
TYPICAL_CURVE = {1: 2, 2: 5, 3: 5, 4: 4, 5: 2, 6: 1}


@dataclass
class DeckAdvisor:
    """Holds the loaded builders, the player's locks, and the last build."""

    set_code: str
    stats: str = "full"
    primary_ckpt: Path | None = None
    rebuild_ckpt: Path | None = None
    format_id: int = 0        # 0=draft, 1=sealed (formats checkpoints only)
    locks: dict[int, str] = field(default_factory=dict)  # card_id -> in|out
    _primary: object | None = None
    _rebuild: object | None = None
    _cards: pd.DataFrame | None = None
    _pips: np.ndarray | None = None
    _last_deck: dict[int, int] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---------------------------------------------------------------- loading

    def _ensure_loaded(self):
        if self._cards is not None:
            return
        from draftbot.models.loading import deck_builder_from_checkpoint

        cards = pd.read_parquet(PROCESSED_DIR / self.set_code / "cards.parquet")
        self._cards = cards.set_index("id", drop=False).sort_index()
        self._name = self._cards["name"].to_numpy()
        self._rarity = self._cards["rarity"].to_numpy()
        self._cmc = np.nan_to_num(self._cards["cmc"].to_numpy(float)).round().astype(int)
        self._is_land = self._cards["type_line"].fillna("") \
            .str.contains("Land").to_numpy()
        self._is_basic = self._cards["is_basic"].to_numpy().astype(bool)
        static = pd.read_parquet(
            PROCESSED_DIR / self.set_code / "features.static.parquet")
        self._pips = static[PIP_COLS].to_numpy().astype(int)
        if self.primary_ckpt:
            self._primary = deck_builder_from_checkpoint(
                self.primary_ckpt, self.set_code, self.stats)
            if getattr(self._primary.model, "formats", False):
                self._primary.format_id = self.format_id
        if self.rebuild_ckpt:
            try:
                self._rebuild = deck_builder_from_checkpoint(
                    self.rebuild_ckpt, self.set_code, self.stats)
                if getattr(self._rebuild.model, "formats", False):
                    self._rebuild.format_id = self.format_id
            except Exception:   # optional: the panel degrades to pinned locks
                self._rebuild = None

    def preload(self):
        """Warm the checkpoints off the draft loop's thread — loading costs a
        couple of seconds and must not stall a live pick."""
        with self._lock:
            self._ensure_loaded()

    @property
    def loaded(self) -> bool:
        return self._primary is not None

    # ------------------------------------------------------------------ pool

    def _pool(self, picks: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Drafted nonbasics as (distinct ids, copies) — drafted basics are not
        pool members, the basics head owns the manabase (DECK_DATA_PLAN)."""
        counts: dict[int, int] = {}
        for cid in picks:
            if not self._is_basic[cid]:
                counts[int(cid)] = counts.get(int(cid), 0) + 1
        ids = np.array(sorted(counts), dtype=np.int16)
        return ids, np.array([counts[int(i)] for i in ids], dtype=np.int16)

    def _arrays(self, ids: np.ndarray, cnts: np.ndarray):
        from draftbot.data.deck_dataset import DeckArrays

        n = len(ids)
        pool_ids = np.full((1, max(n, 1)), PAD, dtype=np.int16)
        pool_counts = np.zeros((1, max(n, 1)), dtype=np.int16)
        pool_ids[0, :n], pool_counts[0, :n] = ids, cnts
        return DeckArrays(pool_ids=pool_ids, pool_counts=pool_counts,
                          deck_counts=np.zeros_like(pool_ids),
                          basics=np.zeros((1, 5), np.int16),
                          meta=pd.DataFrame({"draft_id": ["hud"]}))

    def _lock_state(self, ids: np.ndarray, cnts: np.ndarray) -> np.ndarray | None:
        if not self.locks:
            return None
        state = np.full(len(ids), MASK_STATE, dtype=np.int64)
        for j, cid in enumerate(ids):
            mode = self.locks.get(int(cid))
            if mode == "in":
                state[j] = min(int(cnts[j]), MASK_STATE - 1)
            elif mode == "out":
                state[j] = 0
        return state

    # ---------------------------------------------------------------- solving

    def _solve(self, ids, cnts, state):
        """-> (deck {card_id: count}, basics (5,), conf per slot, model name)."""
        arr = self._arrays(ids, cnts)
        model = self._rebuild if (state is not None and self._rebuild) else self._primary
        dev = model.device
        t_ids = torch.from_numpy(arr.pool_ids.astype(np.int64)).to(dev)
        t_cnt = torch.from_numpy(arr.pool_counts.astype(np.int64)).to(dev)
        pad = t_ids == PAD
        t_state = None
        if state is not None and model is self._rebuild:
            t_state = torch.from_numpy(state[None, :]).to(dev)

        member, land, basics = model._heads(t_ids, t_cnt, pad, t_state)
        conf = member[0].detach().cpu().numpy()[: len(ids)].copy()

        if state is None:
            pred = model.build_all(arr)[0]
            return pred["deck"], np.asarray(pred["basics"]), conf, model.name

        if model is self._rebuild:
            probs = model._maskgit_probs(t_ids, t_cnt, pad, model.steps,
                                         init_state=t_state)
            m, ln, b = (p[0].detach().cpu().numpy() for p in probs)
        else:   # no diffusion checkpoint: pin the locks onto the one-shot probs
            m = member[0].detach().cpu().numpy().copy()
            ln = land[0].detach().cpu().numpy()
            b = basics[0].detach().cpu().numpy()
            m[: len(ids)] = np.where(state == MASK_STATE, m[: len(ids)],
                                     np.where(state > 0, 2.0, -1.0))
        pred = build_deck(m, ln, b, arr.pool_ids[0], arr.pool_counts[0],
                          model.is_land)
        return pred["deck"], np.asarray(pred["basics"]), conf, model.name

    # ------------------------------------------------------------- presenting

    def _card_row(self, cid: int, count: int, conf: float) -> dict:
        pips = self._pips[cid]
        cmc = int(self._cmc[cid])
        return {"card_id": int(cid), "name": str(self._name[cid]),
                "rarity": str(self._rarity[cid]), "cmc": cmc,
                "count": int(count), "conf": float(conf),
                "is_land": bool(self._is_land[cid]),
                "pips": pips.tolist(), "generic": max(0, cmc - int(pips.sum())),
                "lock": self.locks.get(int(cid))}

    def _arena_list(self, deck: dict[int, int], basics: np.ndarray) -> str:
        lines = ["Deck"]
        for cid, n in sorted(deck.items(), key=lambda kv: str(self._name[kv[0]])):
            lines.append(f"{n} {self._name[cid]}")
        lines += [f"{int(n)} {nm}" for nm, n in zip(BASIC_NAMES, basics) if n]
        return "\n".join(lines)

    def _diff(self, deck: dict[int, int], conf_of) -> dict | None:
        prev = self._last_deck
        if prev is None:
            return None
        added = [self._card_row(c, n - prev.get(c, 0), conf_of(c))
                 for c, n in deck.items() if n > prev.get(c, 0)]
        removed = [self._card_row(c, n - deck.get(c, 0), conf_of(c))
                   for c, n in prev.items() if n > deck.get(c, 0)]
        unchanged = sum(1 for c, n in deck.items() if prev.get(c, 0) == n)
        return {"added": sorted(added, key=lambda r: -r["conf"]),
                "removed": sorted(removed, key=lambda r: -r["conf"]),
                "unchanged": unchanged,
                "locks_honored": sum(
                    1 for c, m in self.locks.items()
                    if (m == "in") == (deck.get(int(c), 0) > 0))}

    # -------------------------------------------------------------------- API

    def build(self, picks: list[int], provisional: bool = False,
              keep_diff: bool = False) -> dict:
        """Propose (or rebuild) a deck from the drafted pool. Thread-safe: the
        panel's /rebuild POST lands on the HTTP thread, the draft loop on its own."""
        with self._lock:
            self._ensure_loaded()
            ids, cnts = self._pool(picks)
            if len(ids) == 0:
                return {"available": False, "reason": "no cards drafted yet"}
            state = self._lock_state(ids, cnts)
            deck, basics, conf, model_name = self._solve(ids, cnts, state)
            payload = self._present(ids, cnts, deck, basics, conf, model_name,
                                    provisional, keep_diff)
            self._last_deck = dict(deck)
            return payload

    def _present(self, ids, cnts, deck, basics, conf, model_name,
                 provisional, keep_diff) -> dict:
        conf_by_id = {int(c): float(conf[j]) for j, c in enumerate(ids)}
        conf_of = lambda c: conf_by_id.get(int(c), 0.0)  # noqa: E731

        played = [self._card_row(c, n, conf_of(c)) for c, n in deck.items()]
        cut_ids = {int(c): int(n) for c, n in zip(ids, cnts)}
        cuts = []
        for cid, n in cut_ids.items():
            left = n - deck.get(cid, 0)
            if left > 0:
                cuts.append(self._card_row(cid, left, conf_of(cid)))
        cuts.sort(key=lambda r: -r["conf"])

        spells = [r for r in played if not r["is_land"]]
        nb_lands = [r for r in played if r["is_land"]]
        n_lands = int(basics.sum()) + sum(r["count"] for r in nb_lands)

        hi = BOUNDARY_HI_DAY0 if self.stats == "none" else BOUNDARY_HI
        cap = BOUNDARY_MAX + (1 if self.stats == "none" else 0)
        near_in = sorted([r for r in spells if r["conf"] < hi and not r["lock"]],
                         key=lambda r: r["conf"])[:cap]
        near_ids = {r["card_id"] for r in near_in}
        # a split card (1 of 2 copies played) is a cut, but showing it on both
        # sides of the boundary reads as a contradiction — keep it in the build
        played_ids = {r["card_id"] for r in played}
        near_out = [r for r in cuts
                    if not r["lock"] and r["card_id"] not in played_ids][:cap]
        out_ids = {r["card_id"] for r in near_out}

        groups: dict[int, list] = {}
        for r in spells:
            if r["card_id"] in near_ids:
                continue
            groups.setdefault(min(5, r["cmc"]), []).append(r)  # 5+ is one bucket
        group_list = [
            {"cmc": k, "label": _drop_label(k),
             "n": sum(r["count"] for r in groups[k]),
             "cards": sorted(groups[k], key=lambda r: -r["conf"])}
            for k in sorted(groups)]

        curve = [0] * 7
        pips = [0] * 5
        for r in spells:
            curve[min(6, r["cmc"])] += r["count"]
            for i in range(5):
                pips[i] += r["pips"][i] * r["count"]

        diff = self._diff(deck, conf_of) if keep_diff else None
        return {
            "available": True, "provisional": provisional,
            "n_cards": int(sum(deck.values()) + basics.sum()),
            "n_spells": sum(r["count"] for r in spells),
            "n_lands": n_lands, "legal": int(sum(deck.values()) + basics.sum()) == 40,
            "basics": [int(x) for x in basics],
            "basic_names": BASIC_NAMES,
            "nonbasic_lands": sorted(nb_lands, key=lambda r: -r["conf"]),
            "groups": group_list,
            "boundary": {"in": near_in[::-1], "out": near_out},
            "cuts": [r for r in cuts if r["card_id"] not in out_ids],
            "near_cuts": len(near_out), "n_cuts": len(cuts),
            "curve": curve, "pips": pips,
            "locks": {str(k): v for k, v in self.locks.items()},
            "n_locks": len(self.locks),
            "diff": diff,
            "model_id": model_name.split("@")[0],
            "rebuild_model": (self._rebuild.name.split("@")[0]
                              if self._rebuild else None),
            "stats_mode": self.stats, "set_code": self.set_code,
            "arena_list": self._arena_list(deck, basics),
            "pool_size": int(cnts.sum()),
            # pool copies the model would play at all — the provisional view's
            # "playables so far", which is about the POOL, not the trimmed 40
            "playables": int(sum(int(n) for c, n in zip(ids, cnts)
                                 if conf_by_id.get(int(c), 0.0) >= 0.5)),
            "weak_slot": _weak_slot(curve),
        }

    # ------------------------------------------------------------------ locks

    def set_lock(self, card_id: int, mode: str | None):
        with self._lock:
            if mode in (None, "clear"):
                self.locks.pop(int(card_id), None)
            else:
                self.locks[int(card_id)] = mode

    def clear_locks(self):
        with self._lock:
            self.locks.clear()


def _drop_label(cmc: int) -> str:
    if cmc >= 5:
        return "5+ drops"
    return "1 drop" if cmc == 1 else f"{cmc} drops"


def _weak_slot(curve: list[int]) -> dict | None:
    """Largest deficit against TYPICAL_CURVE — a heuristic talking point for the
    provisional view, not a model prediction."""
    worst = None
    for cmc, want in TYPICAL_CURVE.items():
        have = curve[cmc]
        if have < want and (worst is None or want - have > worst["short"]):
            worst = {"cmc": cmc, "have": have, "want": want, "short": want - have}
    return worst
