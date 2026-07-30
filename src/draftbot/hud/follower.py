"""Arena Player.log follower (HUD milestone H1, see docs/HUD_PLAN.md).

Event schemas verified against 17lands' seventeenlands/mtga_follower.py:
- `Draft.Notify `            {draftId, SelfPack, SelfPick, PackCards: "id,id"}
- LogBusinessEvents + PickGrpId  {DraftId, EventId, PackNumber, PickNumber,
                                  CardsInPack: [...], PickGrpId, AutoPick,
                                  TimeRemainingOnPick}
- EventPlayerDraftMakePick   {DraftId, Pack, Pick, GrpIds: [...]}
- DraftStatus == "PickNext"  {EventName, DraftPack: [...], PackNumber, PickNumber}
- BotDraft_DraftPick         {PickInfo: {...}}
- Event_Join + Course        event metadata; Draft_CompleteDraft → done

Card ids are Arena grpIds (== Scryfall arena_id). Log statements start with a
`[UnityCrossThreadLogger]`/`Client GRE` prefix; their JSON payload may span
multiple lines, so we buffer until the next statement prefix and then decode
the first JSON object in the buffer. Pure stdlib — no torch/pandas here.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_LOG = Path.home() / "Library/Logs/Wizards Of The Coast/MTGA/Player.log"

LOG_START = re.compile(r"^(\[UnityCrossThreadLogger\]|\[Client GRE\])")
EVENT_NAME_SET = re.compile(r"(?:^|_)([A-Z][A-Z0-9]{2})_")  # PremierDraft_MSH_...

RELEVANT_TOKENS = ("Draft.Notify ", "EventPlayerDraftMakePick",
                   "LogBusinessEvents", "DraftStatus", "BotDraft_DraftPick",
                   "Draft_CompleteDraft", "Event_Join")


@dataclass
class PackSeen:
    draft_id: str
    pack_number: int          # as reported by Arena (base varies by event kind;
    pick_number: int          # DraftState derives position by COUNTING packs)
    card_ids: list[int]       # Arena grpIds
    source: str               # which log message produced this
    event_name: str | None = None


@dataclass
class PickMade:
    draft_id: str
    pack_number: int
    pick_number: int
    grp_ids: list[int]        # usually one; lists for pick-two mechanics
    auto_pick: bool = False
    source: str = ""


@dataclass
class DraftJoined:
    event_name: str
    draft_id: str | None = None


@dataclass
class DraftCompleted:
    draft_id: str


def _first_json(buffer: str) -> dict | None:
    start = buffer.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(buffer[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_statement(buffer: str) -> list:
    """Turn one buffered log statement into 0..n events."""
    if not any(tok in buffer for tok in RELEVANT_TOKENS):
        return []
    obj = _first_json(buffer)
    if obj is None:
        return []
    # Arena wraps request payloads: {"id": ..., "request": "<json string>"}
    # (seen live 2026-07-29 on EventPlayerDraftMakePick)
    if isinstance(obj.get("request"), str):
        try:
            inner = json.loads(obj["request"])
            if isinstance(inner, dict):
                obj = inner
        except json.JSONDecodeError:
            pass
    # order mirrors the 17lands client's dispatch
    if "DraftStatus" in obj:
        if obj.get("DraftStatus") == "PickNext" and "DraftPack" in obj:
            return [PackSeen(draft_id=obj.get("DraftId", obj.get("EventName", "bot")),
                             pack_number=int(obj.get("PackNumber", 0)),
                             pick_number=int(obj.get("PickNumber", 0)),
                             card_ids=[int(x) for x in obj["DraftPack"]],
                             source="DraftStatus",
                             event_name=obj.get("EventName"))]
        return []
    if "LogBusinessEvents" in buffer and "PickGrpId" in obj:
        # combined message: the pack AND the pick. On Arena builds where
        # Draft.Notify skips P1P1, this is the ONLY source for that pack —
        # emit PackSeen first so DraftState sees pack-then-pick.
        out = []
        if obj.get("CardsInPack"):
            out.append(PackSeen(draft_id=obj["DraftId"],
                                pack_number=int(obj["PackNumber"]),
                                pick_number=int(obj["PickNumber"]),
                                card_ids=[int(x) for x in obj["CardsInPack"]],
                                source="LogBusiness",
                                event_name=obj.get("EventId")))
        out.append(PickMade(draft_id=obj["DraftId"],
                            pack_number=int(obj["PackNumber"]),
                            pick_number=int(obj["PickNumber"]),
                            grp_ids=[int(obj["PickGrpId"])],
                            auto_pick=bool(obj.get("AutoPick", False)),
                            source="LogBusiness"))
        return out
    if "Draft.Notify " in buffer and "method" not in obj and "PackCards" in obj:
        return [PackSeen(draft_id=obj["draftId"],
                         pack_number=int(obj["SelfPack"]),
                         pick_number=int(obj["SelfPick"]),
                         card_ids=[int(x) for x in str(obj["PackCards"]).split(",") if x],
                         source="Draft.Notify")]
    if "EventPlayerDraftMakePick" in buffer and "GrpIds" in obj:
        return [PickMade(draft_id=obj["DraftId"],
                         pack_number=int(obj["Pack"]),
                         pick_number=int(obj["Pick"]),
                         grp_ids=[int(x) for x in obj["GrpIds"]],
                         source="EventPlayerDraftMakePick")]
    if "Draft_CompleteDraft" in buffer and "DraftId" in obj:
        return [DraftCompleted(draft_id=obj["DraftId"])]
    if "Event_Join" in buffer and "EventName" in obj:
        return [DraftJoined(event_name=obj["EventName"],
                            draft_id=obj.get("DraftId"))]
    return []


def set_code_from_event(event_name: str | None) -> str | None:
    if not event_name:
        return None
    m = EVENT_NAME_SET.search(event_name)
    return m.group(1) if m else None


class LogFollower:
    """Tails Player.log, yielding typed events. replay=True stops at EOF."""

    def __init__(self, path: Path = DEFAULT_LOG, replay: bool = False,
                 poll_seconds: float = 0.25, from_start: bool = True):
        self.path = Path(path)
        self.replay = replay
        self.poll_seconds = poll_seconds
        self.from_start = from_start  # scan history → mid-draft attach works
        self._buffer: list[str] = []

    def _flush(self) -> list:
        if not self._buffer:
            return []
        events = parse_statement("\n".join(self._buffer))
        self._buffer = []
        return events

    def events(self) -> Iterator[object]:
        f = open(self.path, "r", errors="replace")
        if not self.from_start and not self.replay:
            f.seek(0, 2)
        inode = self.path.stat().st_ino
        while True:
            line = f.readline()
            if line:
                if LOG_START.match(line):
                    yield from self._flush()
                self._buffer.append(line.rstrip("\n"))
                continue
            # EOF: a multi-line JSON may still be mid-write — only flush if the
            # buffered statement already parses; otherwise keep accumulating.
            if self._buffer:
                events = parse_statement("\n".join(self._buffer))
                if events:
                    self._buffer = []
                    yield from events
            if self.replay:
                return
            time.sleep(self.poll_seconds)
            try:
                if self.path.stat().st_ino != inode:  # rotation
                    f.close()
                    f = open(self.path, "r", errors="replace")
                    inode = self.path.stat().st_ino
            except FileNotFoundError:
                pass
