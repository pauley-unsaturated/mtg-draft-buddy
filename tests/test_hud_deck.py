"""HUD H5 acceptance: the captured MSH draft replays into a legal proposed
build, locks survive a rebuild, and the /state contract the panel reads holds.

Model-backed tests need the (gitignored) deck checkpoints; the server and
end-of-draft tests are data-light and always run.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from draftbot.hud.follower import LogFollower
from draftbot.hud.state import DraftState

FIXTURE = Path(__file__).parent / "fixtures" / "real_draft_msh.log"
CARDS = Path("data/processed/MSH/cards.parquet")
PRIMARY = Path("checkpoints/EXP-121")
REBUILD = Path("checkpoints/EXP-116")

needs_data = pytest.mark.skipif(
    not (CARDS.exists() and FIXTURE.exists()), reason="MSH data/fixture not built")
needs_models = pytest.mark.skipif(
    not (PRIMARY.exists() and CARDS.exists() and FIXTURE.exists()),
    reason="deck checkpoints not present")

# every key panel.html reads off `state.deck` — drift here breaks the window
PANEL_KEYS = {
    "available", "provisional", "n_cards", "n_spells", "n_lands", "legal",
    "basics", "basic_names", "nonbasic_lands", "groups", "boundary", "cuts",
    "near_cuts", "n_cuts", "curve", "pips", "locks", "n_locks", "diff",
    "model_id", "rebuild_model", "stats_mode", "set_code", "arena_list",
    "pool_size", "playables", "weak_slot",
}
ROW_KEYS = {"card_id", "name", "rarity", "cmc", "count", "conf", "is_land",
            "pips", "generic", "lock"}


def replayed_state() -> DraftState:
    state = DraftState()
    for ev in LogFollower(FIXTURE, replay=True).events():
        state.apply(ev)
    return state


def advisor(rebuild=True):
    from draftbot.hud.deck import DeckAdvisor
    return DeckAdvisor(set_code="MSH", stats="full", primary_ckpt=PRIMARY,
                       rebuild_ckpt=REBUILD if (rebuild and REBUILD.exists()) else None)


@needs_data
def test_end_of_draft_detected_from_pack_shape():
    """Arena did not emit Draft_CompleteDraft in the captured session, so the
    HUD must recognise the end from the final one-card pack."""
    from draftbot.hud.__main__ import draft_finished

    state = replayed_state()
    assert len(state.picks) == 42
    assert not state.completed          # no Draft_CompleteDraft in this log
    assert draft_finished(state)

    # every pack ends on a one-card pack — only the THIRD one ends the draft
    events = list(LogFollower(FIXTURE, replay=True).events())
    partial = DraftState()
    seen_ends = 0
    for ev in events:
        partial.apply(ev)
        n = len(partial.picks)
        if n in (14, 28) and len(partial.packs) == n:
            seen_ends += 1
            assert not draft_finished(partial), f"finished early at pick {n}"
        elif n < 42:
            assert not draft_finished(partial)
    assert seen_ends == 2, "fixture did not cover both intermediate pack ends"
    assert draft_finished(partial)


@needs_models
def test_replay_ends_in_a_legal_forty_card_build():
    state = replayed_state()
    adv = advisor()
    d = adv.build(state.pool_ids())

    assert d["available"] and d["legal"]
    assert d["n_cards"] == 40
    assert d["n_spells"] + d["n_lands"] == 40
    assert 14 <= d["n_lands"] <= 20
    assert set(d) == PANEL_KEYS, set(d) ^ PANEL_KEYS

    rows = [c for g in d["groups"] for c in g["cards"]]
    rows += d["boundary"]["in"] + d["boundary"]["out"] + d["cuts"] + d["nonbasic_lands"]
    assert rows
    for r in rows:
        assert set(r) == ROW_KEYS, set(r) ^ ROW_KEYS
        assert 0.0 <= r["conf"] <= 1.0 and r["count"] >= 1
        assert len(r["pips"]) == 5

    # the build is drawn from the pool and never plays a drafted basic
    import pandas as pd
    cards = pd.read_parquet(CARDS).set_index("id", drop=False).sort_index()
    is_basic = cards["is_basic"].to_numpy().astype(bool)
    pool: dict[int, int] = {}
    for cid in state.pool_ids():
        if not is_basic[cid]:
            pool[int(cid)] = pool.get(int(cid), 0) + 1
    played: dict[int, int] = {}
    for g in d["groups"]:
        for c in g["cards"]:
            played[c["card_id"]] = played.get(c["card_id"], 0) + c["count"]
    for c in d["boundary"]["in"] + d["nonbasic_lands"]:
        played[c["card_id"]] = played.get(c["card_id"], 0) + c["count"]
    assert played, "no cards in the proposed build"
    for cid, n in played.items():
        assert not is_basic[cid], f"{cards.loc[cid, 'name']} is a basic land"
        assert n <= pool.get(cid, 0), f"played {n} of {cid}, pool has {pool.get(cid, 0)}"


@needs_models
def test_lock_two_cards_and_rebuild_honors_both():
    """DoD: lock 2 cards, rebuild honours both — one forced in, one forced out."""
    state = replayed_state()
    adv = advisor()
    first = adv.build(state.pool_ids())

    lock_in = first["cuts"][0]                      # a card the model cut
    lock_out = first["groups"][1]["cards"][0]       # one it played
    adv.set_lock(lock_in["card_id"], "in")
    adv.set_lock(lock_out["card_id"], "out")
    second = adv.build(state.pool_ids(), keep_diff=True)

    played = {c["card_id"] for g in second["groups"] for c in g["cards"]}
    played |= {c["card_id"] for c in second["boundary"]["in"] + second["nonbasic_lands"]}
    assert lock_in["card_id"] in played, "locked-in card was cut"
    assert lock_out["card_id"] not in played, "locked-out card was played"
    assert second["legal"] and second["n_cards"] == 40
    assert second["diff"]["locks_honored"] == 2
    assert second["n_locks"] == 2

    adv.clear_locks()
    third = adv.build(state.pool_ids())
    assert third["n_locks"] == 0 and third["legal"]


@needs_models
def test_provisional_build_from_a_partial_pool():
    state = replayed_state()
    d = advisor(rebuild=False).build(state.pool_ids()[:30], provisional=True)
    assert d["provisional"] and d["legal"] and d["n_cards"] == 40
    assert d["playables"] >= 1
    assert d["rebuild_model"] is None       # degrades without the diffusion ckpt


def test_server_merges_deck_section_and_serves_posts():
    """The deck section rides on the existing /state poll and survives a
    pick-view update; POSTs reach the action handler."""
    from draftbot.hud.server import HudServer

    seen = []
    srv = HudServer(port=8791)
    srv.set_action_handler(lambda a, p: (seen.append((a, p)), {"available": True,
                                                               "n_locks": len(seen)})[1])
    url = srv.start().rstrip("/")
    try:
        srv.update({"status": "live", "pos_label": "P1P1"})
        assert "deck" not in json.loads(urllib.request.urlopen(url + "/state").read())

        srv.update_deck({"available": True, "n_locks": 0})
        body = json.loads(urllib.request.urlopen(url + "/state").read())
        assert body["deck"]["available"] and body["pos_label"] == "P1P1"

        srv.update({"status": "live", "pos_label": "P1P2"})   # pick view moves on
        body = json.loads(urllib.request.urlopen(url + "/state").read())
        assert body["deck"]["available"], "deck section clobbered by a pick update"

        req = urllib.request.Request(
            url + "/lock", data=json.dumps({"card_id": 7, "mode": "in"}).encode(),
            method="POST")
        body = json.loads(urllib.request.urlopen(req).read())
        assert seen == [("lock", {"card_id": 7, "mode": "in"})]
        assert body["deck"]["n_locks"] == 1
    finally:
        srv.stop()


def test_server_rejects_unknown_posts():
    from draftbot.hud.server import HudServer

    srv = HudServer(port=8792)
    url = srv.start().rstrip("/")
    try:
        for path, code in (("/nope", 404), ("/rebuild", 503)):  # 503: no handler
            req = urllib.request.Request(url + path, data=b"{}", method="POST")
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req)
            assert exc.value.code == code
    finally:
        srv.stop()
