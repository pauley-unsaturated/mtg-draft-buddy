# Design brief: "Draft Buddy" deck-builder view (HUD H5)

(Handoff prompt for a Claude Design session. Companion to
docs/HUD_DESIGN_BRIEF.md — the draft-pick HUD this view lives inside. The
engineering counterpart is docs/DECKBUILDER_HANDOFF.md.)

## What this is

When the 42nd pick lands, the draft is over and the player has ~60 seconds of
downtime before deck submission. The HUD's window transitions from "pick
advisor" to **deck builder**: an ML model (win-weighted imitation of trophy
builds; ~90% agreement with what 5-7-win players submit) proposes a complete
40-card deck from the drafted pool — which ~23 nonbasics to play, how many
lands, and the basic-land split. The player reviews it, optionally locks a few
cards they insist on, taps rebuild, and then assembles the result in Arena by
hand.

This view exists to answer three glances: **what do I play, what do I cut,
what's my mana**. It is not a collection browser or a stats screen.

## Design-system continuity (hard constraint)

This is a SECOND VIEW of the existing HUD window, not a new app. Reuse the
implemented system from `hud/panel.html`: dark instrument theme (#08090A bg,
#101216 surfaces), oklch confidence cyan for model conviction, split violet
for uncertainty/disagreement, rarity-tinted card names (gold/orange idiom),
the fill-bar-as-probability row treatment, 220ms state-change motion only.
Same window: ~340–420px wide, degrades to ~240px text-only, macOS/retina,
pywebview WebKit, no external assets except the local /art cache.

## The data available to display

- **The build**: ~23 nonbasic cards, each with: name, mana pips, rarity,
  membership confidence 0–1 (how sure the model is this card belongs — most
  mainboard cards sit >0.9; the interesting ones are 0.4–0.7), is-land flag,
  copies (occasionally 2×). Plus basics as five counts (e.g. 8 Plains /
  7 Swamp) and total lands (14–20, almost always 16–18).
- **The cuts**: the ~19 pool cards NOT in the build, same fields, sorted by
  how close they came (confidence descending). The 2–3 nearest misses are
  genuinely arguable — the model knows it (0.35–0.65 confidence) and the
  design should honor that honesty: near-miss cuts and near-miss inclusions
  are a *shared boundary zone*, not two cleanly separated lists.
- **Curve + colors**: cmc histogram (0–7+) of the build's spells; color pip
  counts. The pick-phase HUD already renders tiny versions of both.
- **Lock state**: any card can be locked-in or locked-out by the player;
  rebuild re-runs the model honoring locks (a diffusion variant conditions on
  them natively). Locks must be visible at a glance and cheap to undo.
- **Meta strip**: model id (e.g. EXP-121), stat mode ("full stats" /
  "week-1" / "day-0 — no stats yet" badge), set code, deck legality readout
  ("40 cards · 17 lands").

## States to design

1. **Build proposed** (hero): the moment the draft ends. The 40 shown as
   scannable in <5s: spells grouped by cmc or confidence, lands as a compact
   mana summary (17 lands: 8W 7B + 2 nonbasic named). The boundary zone
   (last cards in / first cards out) should be visually continuous.
2. **Reviewing cuts**: expand/scroll into the sideboard-to-be, confidence
   bars fading with distance from the boundary.
3. **Locked + rebuilding**: player locks 1–3 cards, taps rebuild; brief
   recompute (<1s), diff-highlight what changed (2–4 cards typically swap).
4. **Draft still live**: this view is reachable early (pick 30+) as a
   "provisional build" peek — badge it clearly as provisional; it must not
   distract from the pick timer. (Cheap version: only offer it between picks.)
5. **Day-0**: no stats exist for this set; build confidence is slightly
   lower. Same subtle badge idiom as the pick view.

## Feel

Same instrument calm as the pick view. The player is about to click 40 cards
in Arena by hand — the deck list IS the artifact, so density and legibility
beat decoration. Confidence reads as weight, not numerals. One deliberate
moment of satisfaction on build-proposed (the draft's payoff) is welcome —
a single 220ms settle, nothing looping.

## Deliverable

Same contract as the pick-view brief: design tokens + component hierarchy +
the 5 states, specified concretely enough that an engineer can implement it
in `hud/panel.html`'s existing idiom without inventing visual language.
