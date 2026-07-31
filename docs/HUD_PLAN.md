# Draft HUD — real-time pick advisor alongside Arena

**Goal:** a small always-on-top window on the Mac that, during a live Arena
draft, shows the model's ranking of the current pack with confidences, updating
within ~1s of each pack appearing. Owner request 2026-07-29 (promotes PLAN
P5.T4 into a concrete milestone; journal entry same date).

## How we get real-time data (verified against 17lands' client source)

The 17lands uploader (`rconroy293/mtga-log-client`, installed by the
`homebrew-seventeenlands` formula) tails Arena's own log file — no memory
reading, no injection, ToS-safe. Verified specifics from
`seventeenlands/mtga_follower.py`:

- **File:** `~/Library/Logs/Wizards Of The Coast/MTGA/Player.log`
  (plus `Player-prev.log` after restarts). Requires **Detailed Logs (Plugin
  Support)** enabled in Arena's settings — must be documented in our README.
- **Follow strategy:** poll the file, re-open on rotation/truncation
  (client uses a 60s force-refresh; we can poll at 250–500ms for HUD latency).
- **Line format:** events start with `[UnityCrossThreadLogger]`/`Client GRE`
  prefixes, often with a timestamp, followed by a JSON blob (sometimes
  multi-line — the parser must accumulate until the JSON closes).
- **Human (premier/trad) draft events:**
  - `Draft.Notify`: `{draftId, SelfPack, SelfPick, PackCards: "id,id,…"}` —
    the pack we're being offered.
  - Combined `LogBusiness` messages: `{DraftId, EventId, PackNumber,
    PickNumber, CardsInPack: […], PickGrpId, AutoPick, TimeRemainingOnPick}` —
    pack AND the pick that was made.
  - `EventPlayerDraftMakePick`: our pick (grpId).
- **Bot (quick) draft events:** `DraftStatus == "PickNext"` with `DraftPack`
  card list; `BotDraft_DraftPick` with `PickInfo`. (v2 scope.)
- **Event linking:** join messages carry `EventName` like
  `PremierDraft_MSH_20260626` → set code extraction.
- **Card ids:** Arena **grpIds** — which equal Scryfall `arena_id`. Our cached
  Scryfall data has arena_id for 375/375 MSH+MAR cards → grpId → name → our
  vocab id is a pure lookup (`data/processed/<SET>/cards.parquet` join).
  Fallback for unknown ids: Scryfall `/cards/arena/<id>` (cached).

## Architecture (three small pieces, one process)

```
Player.log ─▶ LogFollower ─▶ DraftState ─▶ Advisor(model) ─▶ HUD UI
              (tail+parse)    (pack/pool/     (TorchScorer      (rendered
                               position)       probs)            ranking)
```

1. **LogFollower** (`draftbot/hud/follower.py`): generator yielding typed
   events (`PackSeen`, `PickMade`, `DraftJoined`, `DraftCompleted`). Pure
   parsing, no model deps → unit-testable against captured log fixtures.
   Includes a `--replay <logfile>` mode (timed or instant) for development
   without Arena running.
2. **DraftState** (`draftbot/hud/state.py`): reconstructs `(packs, prev_picks,
   position)` arrays incrementally from events; maps grpIds → vocab ids;
   detects set code from the event name and lazy-loads that set's cards +
   feature table. Handles mid-draft attach (scan Player.log from the top to
   catch up — the log retains the whole session).
3. **Advisor** (`draftbot/hud/advisor.py`): wraps 1–2 checkpoints via the
   existing `scorer_from_checkpoint`; on each `PackSeen`, teacher-forces the
   picks so far and softmaxes the current step → per-card probabilities.
   Latency budget trivial (single 1×t×P forward on MPS, <50ms). Stats mode
   configurable (`full` normally; `week1`/`none` in a new set's first days —
   exactly the day-0 pathway EXP-030 was trained for).
4. **HUD UI**: two stages —
   - **v0 (terminal):** `rich` live table — card name, rarity, model prob bar,
     model rank vs pack order, pool color pips summary. Zero packaging risk.
   - **v1 (window):** `pywebview` (or plain browser tab served by FastAPI on
     localhost) rendering an HTML panel: ranked rows with Scryfall art
     thumbnails (`image_uris.small`, disk-cached at
     `data/raw/scryfall/images/`), confidence bars, top-3 highlighted, second
     model's opinion as a small delta chip (e.g. EXP-013 vs EXP-031), pool
     curve/pips strip. Always-on-top toggle.

## What the player sees (v1 target)

- Ranked pack list: `#1 Lightning Strike ▓▓▓▓▓▓ 46% · GIH .553 · ALSA 3.6`
  (model prob = calibrated — ECE ≈ 0.01 — so the % is honest).
- Divergence flag when the two models disagree on top-1.
- Pool summary: pip counts by color, curve histogram, picks so far.
- Stat-mode indicator (full/week1/none) + model/version footer.
- No draft detected → idle state with last-seen event and log-path health.

## Milestones

- **H1 — follower + fixtures:** parse a captured MSH premier draft log
  (capture one manually next draft; also synthesize from our parquet for CI).
  DoD: replay of fixture log emits exactly 42 PackSeen + 42 PickMade with ids
  resolving through cards.parquet.
- **H2 — advisor loop (terminal v0):** `python -m draftbot.hud --models
  checkpoints/EXP-013,checkpoints/EXP-031 [--replay fixture.log]` renders live
  rankings. DoD: end-to-end replay session < 1s/pick; works mid-draft attach.
- **H3 — window UI + art:** pywebview/FastAPI panel, Scryfall thumbnail cache,
  always-on-top. DoD: draft-along session on a real Arena draft.
- **H4 — polish:** quick-draft (bot) events; auto set detection → auto
  onboard-set if the set is unknown (ties into P4.T3); config file.
- **H5 — deck-builder view (DONE 2026-07-30):** second view of the same window
  (design states B1–B5). `hud/deck.py` runs the production builder for the
  proposed 40 (EXP-121 at ship; **EXP-126** since the 2026-07-31 model lock —
  see DECKBUILDER_HANDOFF.md) and EXP-116 for lock-conditioned rebuilds;
  `/state` carries a `deck` section and
  `POST /lock|/rebuild|/clear_locks` drive it. Full card list always visible
  (scrolling, sticky cmc headers, per-row mana cost), shared boundary zone,
  cuts view, rebuild diff, provisional peek from pick 30, day-0 badge, Arena
  list copy. DoD met: fixture replay → legal 40, two locks honoured on
  rebuild. See docs/hud/deck-*.png and the 2026-07-30 journal entry.

## Risks / notes

- Log schema drifts with Arena patches (the 17lands client handles several
  format generations — we mirror only current ones and keep the parser
  fixture-tested so drift is a test failure, not a mystery).
- Detailed Logs off → empty events; HUD must surface this loudly.
- New-set day 0: grpIds appear before our vocab exists → auto-run onboard-set
  (probe may 404 until 17lands publishes; static-only mode still works via
  Scryfall + `none` stats — that's the EXP-030 zero-shot path with masks on).
- Never upload anything; read-only on the log. Coexists fine with the 17lands
  uploader tailing the same file.
