"""Draft-state reconstruction from follower events (HUD H1/H2).

Maps Arena grpIds (== Scryfall arena_id) into our per-set vocab, accumulates
pack/pick history, and exposes the (packs, prev_picks, position) arrays the
models consume. Position comes from COUNTING packs seen per draft — Arena's
reported pack/pick numbers vary in base across event kinds, so they are used
for display only.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from draftbot.data.cards import (COMPANION_SHEETS, PROCESSED_DIR,
                                 SCRYFALL_CACHE_DIR, name_to_id, norm_name)
from draftbot.data.dataset import PAD
from draftbot.hud.follower import (DraftCompleted, DraftJoined, PackSeen,
                                   PickMade, set_code_from_event)


_GLOBAL_GRP_INDEX: dict[int, str] | None = None


def global_grp_index() -> dict[int, str]:
    """arena grpId → set code, across every onboarded set. Lets the HUD infer
    the set from pack contents alone — real Arena logs don't reliably emit a
    parseable join message (seen live 2026-07-29: bare-JSON Course lines)."""
    global _GLOBAL_GRP_INDEX
    if _GLOBAL_GRP_INDEX is None:
        idx: dict[int, str] = {}
        for pq in PROCESSED_DIR.glob("*/cards.parquet"):
            set_code = pq.parent.name
            for grp in grp_to_vocab(set_code):
                idx.setdefault(grp, set_code)
        _GLOBAL_GRP_INDEX = idx
    return _GLOBAL_GRP_INDEX


def infer_set(card_ids: list[int]) -> str | None:
    """Majority-vote the set from a pack's grpIds."""
    idx = global_grp_index()
    votes: dict[str, int] = {}
    for g in card_ids:
        code = idx.get(int(g))
        if code:
            votes[code] = votes.get(code, 0) + 1
    if not votes:
        return None
    best = max(votes, key=votes.get)  # type: ignore[arg-type]
    return best if votes[best] >= max(2, len(card_ids) // 2) else None


def grp_to_vocab(set_code: str) -> dict[int, int]:
    """arena grpId → our card id, via the cached Scryfall set JSONs."""
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    nti = name_to_id(cards)
    mapping: dict[int, int] = {}
    sources = [set_code] + COMPANION_SHEETS.get(set_code.upper(), [])
    for code in sources:
        cache = SCRYFALL_CACHE_DIR / f"{code.lower()}.json"
        if not cache.exists():
            continue
        for card in json.loads(cache.read_text()):
            arena = card.get("arena_id")
            vid = nti.get(norm_name(card["name"]))
            if arena and vid is not None:
                mapping[int(arena)] = vid
    for f in (SCRYFALL_CACHE_DIR / "named").glob("*.json"):
        card = json.loads(f.read_text())
        arena = card.get("arena_id")
        vid = nti.get(norm_name(card.get("name", "")))
        if arena and vid is not None:
            mapping.setdefault(int(arena), vid)
    return mapping


@dataclass
class DraftState:
    t_max: int = 45
    pack_max: int = 15
    set_code: str | None = None
    draft_id: str | None = None
    event_name: str | None = None
    completed: bool = False
    packs: list[list[int]] = field(default_factory=list)   # vocab ids per step
    picks: list[int] = field(default_factory=list)
    unknown_grps: set[int] = field(default_factory=set)
    _grp_map: dict[int, int] | None = None

    @property
    def position(self) -> int:
        return len(self.packs) - 1  # current (last-seen) pack's step

    def _ensure_set(self, event_name: str | None):
        code = set_code_from_event(event_name)
        if code and code != self.set_code:
            self.set_code = code
            self._grp_map = grp_to_vocab(code)

    def _map(self, grp_ids: list[int]) -> list[int]:
        out = []
        for g in grp_ids:
            vid = (self._grp_map or {}).get(int(g))
            if vid is None:
                self.unknown_grps.add(int(g))
            else:
                out.append(vid)
        return out

    def apply(self, ev) -> bool:
        """Consume one event; returns True if the pick-context changed."""
        if isinstance(ev, DraftJoined):
            self._ensure_set(ev.event_name)
            return False
        if isinstance(ev, DraftCompleted):
            self.completed = ev.draft_id == self.draft_id or self.completed
            return False
        if isinstance(ev, PackSeen):
            if ev.event_name:
                self._ensure_set(ev.event_name)
            if self._grp_map is None:  # no join message parsed — infer from pack
                code = infer_set(ev.card_ids)
                if code:
                    self.set_code = code
                    self._grp_map = grp_to_vocab(code)
            if self._grp_map is None:
                return False
            if ev.draft_id != self.draft_id:  # new draft begins
                self.draft_id = ev.draft_id
                self.packs, self.picks = [], []
                self.completed = False
            # replays/dedup: identical pack at same step arrives via multiple
            # message kinds — only append if it's genuinely the next step
            ids = self._map(ev.card_ids)
            if self.packs and len(self.picks) < len(self.packs):
                self.packs[-1] = ids  # refresh current step (dedup)
            else:
                self.packs.append(ids)
            return True
        if isinstance(ev, PickMade):
            if ev.draft_id != self.draft_id or not self.packs:
                return False
            if len(self.picks) < len(self.packs):
                vids = self._map(ev.grp_ids)
                if vids:
                    self.picks.append(vids[0])
            return True
        return False

    def arrays(self) -> tuple[np.ndarray, np.ndarray, int]:
        """(packs (1,k,P), prev_picks (1,k), current position index)."""
        k = len(self.packs)
        assert k > 0, "no pack seen yet"
        packs = np.full((1, k, self.pack_max), PAD, dtype=np.int64)
        for i, ids in enumerate(self.packs):
            packs[0, i, : len(ids)] = ids[: self.pack_max]
        prev = np.full((1, k), PAD, dtype=np.int64)
        for i, p in enumerate(self.picks[: k - 1]):
            prev[0, i + 1] = p
        return packs, prev, k - 1

    def pool_ids(self) -> list[int]:
        return list(self.picks)
