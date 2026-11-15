"""HUD sealed mode (P5.S4): parser → state → build, against the sanitized
real capture (tests/fixtures/real_sealed_tla.log, ArenaDirect TLA sealed,
2026-08-01). Artifact-dependent stages skip on machines without data."""

from pathlib import Path

import pytest

from draftbot.hud.follower import LogFollower, SealedPool

FIXTURE = Path("tests/fixtures/real_sealed_tla.log")
TLA_CARDS = Path("data/processed/TLA/cards.parquet")
CKPT = Path("checkpoints/EXP-135/best.pt")


def _pool_events():
    return [ev for ev in LogFollower(FIXTURE, replay=True).events()
            if isinstance(ev, SealedPool)]


def test_parser_extracts_sealed_pool():
    events = _pool_events()
    assert len(events) == 1
    ev = events[0]
    assert "Sealed" in ev.event_name and "TLA" in ev.event_name
    assert len(ev.card_ids) == 84  # six 14-card boosters
    assert all(isinstance(g, int) for g in ev.card_ids)


@pytest.mark.skipif(not TLA_CARDS.exists(), reason="TLA artifacts not built")
def test_sealed_state_maps_pool_and_drops_basics():
    from draftbot.hud.state import SealedState

    state = SealedState()
    assert state.apply(_pool_events()[0])
    assert state.set_code == "TLA"
    assert not state.unknown_grps
    # the captured pool contains 4 basic-land grpIds — not pool members
    assert len(state.pool) == 80


@pytest.mark.skipif(not (TLA_CARDS.exists() and CKPT.exists()),
                    reason="TLA artifacts or EXP-135 not built")
def test_sealed_advisor_builds_legal_40_and_honors_locks():
    from draftbot.hud.deck import DeckAdvisor
    from draftbot.hud.state import SealedState

    state = SealedState()
    state.apply(_pool_events()[0])
    adv = DeckAdvisor(set_code="TLA", primary_ckpt=CKPT.parent, format_id=1)
    d = adv.build(state.pool)
    assert d["available"] and d["legal"] and d["n_cards"] == 40

    out_id = d["boundary"]["out"][0]["card_id"]  # best card left out
    adv.set_lock(out_id, "in")
    d2 = adv.build(state.pool, keep_diff=True)
    played = {c["card_id"] for g in d2["groups"] for c in g["cards"]}
    played |= {c["card_id"] for c in d2["nonbasic_lands"]}
    played |= {c["card_id"] for c in d2["boundary"]["in"]}
    assert out_id in played and d2["legal"]
