"""HUD H1 acceptance: a synthesized Player.log for a REAL test-split MSH draft
replays into exactly t PackSeen + t PickMade events whose ids resolve through
cards.parquet, and DraftState reconstructs the parquet arrays bit-for-bit.
Message shapes mirror seventeenlands/mtga_follower.py."""

import json
from pathlib import Path

import numpy as np
import pytest

from draftbot.hud.follower import (DraftCompleted, DraftJoined, LogFollower,
                                   PackSeen, PickMade)

DATA_OK = Path("data/processed/MSH/cards.parquet").exists()
needs_data = pytest.mark.skipif(not DATA_OK, reason="MSH data not built")


def synthesize_log(out_path: Path, n_drafts: int = 1) -> dict:
    """Write a Player.log covering real MSH test drafts; returns ground truth."""
    import pandas as pd

    from draftbot.data.dataset import PAD, load_draft_arrays
    from draftbot.data.splits import load_splits
    from draftbot.hud.state import grp_to_vocab

    splits = load_splits("MSH")
    arr = load_draft_arrays("MSH", draft_ids=set(splits["random"]["test"][:n_drafts]))
    vocab_to_grp = {v: g for g, v in grp_to_vocab("MSH").items()}

    lines = []
    truth = {"packs": arr.packs, "picks": arr.picks, "t": arr.t}
    for d in range(arr.n_drafts):
        draft_id = f"synthetic-{d}"
        lines.append('[UnityCrossThreadLogger]==> Event_Join '
                     + json.dumps({"EventName": f"PremierDraft_MSH_20260626",
                                   "DraftId": draft_id, "Course": {}}))
        for t in range(arr.t):
            pack_grps = [vocab_to_grp[int(c)] for c in arr.packs[d, t] if c != PAD]
            pick_grp = vocab_to_grp[int(arr.picks[d, t])]
            pack_no, pick_no = t // 14 + 1, t % 14 + 1
            # pack via Draft.Notify (multi-line JSON to exercise buffering)
            notify = json.dumps({"draftId": draft_id, "SelfPack": pack_no,
                                 "SelfPick": pick_no,
                                 "PackCards": ",".join(map(str, pack_grps))},
                                indent=1)
            lines.append("[UnityCrossThreadLogger]Draft.Notify " + notify.splitlines()[0])
            lines.extend(notify.splitlines()[1:])
            # pick via EventPlayerDraftMakePick
            lines.append('[UnityCrossThreadLogger]==> EventPlayerDraftMakePick '
                         + json.dumps({"DraftId": draft_id, "Pack": pack_no,
                                       "Pick": pick_no, "GrpIds": [pick_grp]}))
        lines.append('[UnityCrossThreadLogger]<== Draft_CompleteDraft '
                     + json.dumps({"DraftId": draft_id}))
    out_path.write_text("\n".join(lines) + "\n")
    return truth


@needs_data
def test_replay_reconstructs_real_draft(tmp_path):
    from draftbot.hud.state import DraftState

    log = tmp_path / "Player.log"
    truth = synthesize_log(log)
    events = list(LogFollower(log, replay=True).events())

    packs_seen = [e for e in events if isinstance(e, PackSeen)]
    picks_made = [e for e in events if isinstance(e, PickMade)]
    assert len(packs_seen) == truth["t"]
    assert len(picks_made) == truth["t"]
    assert any(isinstance(e, DraftJoined) for e in events)
    assert any(isinstance(e, DraftCompleted) for e in events)

    state = DraftState()
    for ev in events:
        state.apply(ev)
    assert state.set_code == "MSH"
    assert not state.unknown_grps, f"unresolved grpIds: {list(state.unknown_grps)[:5]}"
    assert state.completed

    packs, prev, pos = state.arrays()
    assert pos == truth["t"] - 1
    for t in range(truth["t"]):
        want = sorted(int(c) for c in truth["packs"][0, t] if c != -1)
        got = sorted(int(c) for c in packs[0, t] if c != -1)
        assert got == want, f"pack mismatch at step {t}"
    assert state.pool_ids() == [int(x) for x in truth["picks"][0]]
    # prev_picks: shifted human picks
    assert prev[0, 0] == -1
    assert list(prev[0, 1:]) == [int(x) for x in truth["picks"][0, :-1]]


@needs_data
def test_mid_draft_attach(tmp_path):
    """Scanning a log that already contains half a draft catches up correctly."""
    from draftbot.hud.state import DraftState

    log = tmp_path / "Player.log"
    truth = synthesize_log(log)
    events = list(LogFollower(log, replay=True).events())
    # feed only the first half (as if we attached mid-draft after a full scan)
    half = [e for e in events][: 2 + 2 * 20]  # join + 20 pack/pick pairs
    state = DraftState()
    for ev in half:
        state.apply(ev)
    packs, prev, pos = state.arrays()
    assert pos == len(state.packs) - 1
    assert len(state.picks) in (pos, pos + 1)


def test_combined_logbusiness_emits_pack_then_pick():
    """P1P1 on current Arena arrives ONLY via the combined message — the parser
    must emit PackSeen before PickMade from a single statement."""
    import json as _json

    from draftbot.hud.follower import parse_statement

    stmt = ('[UnityCrossThreadLogger]LogBusinessEvents '
            + _json.dumps({"DraftId": "d1", "EventId": "PremierDraft_MSH_20260626",
                           "PackNumber": 1, "PickNumber": 1,
                           "CardsInPack": [105093, 104936, 105094],
                           "PickGrpId": 105094, "AutoPick": False,
                           "TimeRemainingOnPick": 52.3}))
    events = parse_statement(stmt)
    assert len(events) == 2
    assert isinstance(events[0], PackSeen)
    assert events[0].card_ids == [105093, 104936, 105094]
    assert isinstance(events[1], PickMade)
    assert events[1].grp_ids == [105094]


REAL_FIXTURE = Path(__file__).parent / "fixtures" / "real_draft_msh.log"


@needs_data
@pytest.mark.skipif(not REAL_FIXTURE.exists(), reason="real fixture absent")
def test_real_captured_draft_parses_completely():
    """A REAL Arena draft log (captured 2026-07-29, sanitized to draft events)
    must parse into a full 42-step reconstruction — the authoritative fixture."""
    from draftbot.hud.state import DraftState

    events = list(LogFollower(REAL_FIXTURE, replay=True).events())
    packs_seen = [e for e in events if isinstance(e, PackSeen)]
    picks_made = [e for e in events if isinstance(e, PickMade)]
    assert len(packs_seen) == 42
    assert len(picks_made) == 42
    state = DraftState()
    for ev in events:
        state.apply(ev)
    assert state.set_code == "MSH"          # inferred from pack contents
    assert not state.unknown_grps
    assert len(state.picks) == 42
    packs, prev, pos = state.arrays()
    assert pos == 41
    # play-booster shape: 14 fresh cards at each pack's first pick
    assert (packs[0, 0] != -1).sum() == 14
    assert (packs[0, 14] != -1).sum() == 14
    assert (packs[0, 28] != -1).sum() == 14
