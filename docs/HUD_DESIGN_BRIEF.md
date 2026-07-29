# Design brief: "Draft Buddy" — a real-time pick-advisor HUD for MTG Arena

(Handoff prompt for a design-focused session. Context: docs/HUD_PLAN.md.)

## What this is
A small, always-on-top macOS window that sits beside (or overlapping the edge of)
the MTG Arena game window during a booster draft. Every ~30–75 seconds the game
offers the player a pack of up to 14 cards and a countdown timer; our ML model
ranks those cards. The HUD's one job: let the player absorb the model's opinion
in under one second of glancing, then get their eyes back to the game.

It is NOT a dashboard, stats site, or deck tracker. It's closer to a heads-up
rangefinder: one primary readout, minimal chrome, legible from a glance at
arm's length while attention is elsewhere.

## Technical frame (hard constraints)
- Rendered as HTML/CSS in a pywebview window (WebKit). No external CDNs; all
  assets local. Target window ~340–420px wide, height flexible up to ~700px;
  must also work collapsed to ~240px wide (text-only, no art).
- macOS, retina. User may resize; layout should degrade gracefully.
- Arena itself is a dark UI (near-black backgrounds, gold/amber accents,
  saturated card art). The HUD lives next to it — dark theme is the default;
  it should feel adjacent to Arena without imitating its trade dress.
- Card art thumbnails come from Scryfall (aspect ratio 488×680 full card, or
  art-crop 626×457). Art is optional per-row (nice-to-have tier).

## The data available to display (per pick)
- Up to 14 rows, each: card name (up to ~30 chars, e.g. "Black Panther,
  Wakandan King"), rarity (common/uncommon/rare/mythic), mana cost pips,
  model confidence as a calibrated probability (these are honest percentages,
  0–100, typically one card at 30–55%, two or three at 8–20%, the rest <5%),
  model rank, and two reference stats (GIH win-rate like ".603", ALSA like
  "3.2") that experienced drafters recognize.
- A second model's opinion exists; usually identical top-1. The interesting
  moment is DISAGREEMENT — when the two models' #1 picks differ, that should
  be visible without adding a whole second column of noise.
- Pool context (running state of the draft): count of picks so far (e.g.
  "P2P7 · 20 picks"), color commitment as five pip counts (W/U/B/R/G), and a
  tiny mana-curve histogram (7 buckets). Secondary information — visible but
  subordinate.
- Status strip: which set + stat mode ("MSH · full stats" vs "week-1 stats"
  vs "day-0 — no stats yet", the latter deserves a subtle badge since
  confidence quality differs), model version, log-connection health.

## States to design
1. **Live pick** (the hero state): ranked list, top choice unmistakable in
   peripheral vision — the #1 row should read from 2 feet away. Rows 2–3
   clearly above the rest. Sub-5% rows can compress to near-nothing.
2. **Between picks / waiting**: last pick's result ("took Lightning Strike —
   model agreed, 46%"), dimmed.
3. **Disagreement**: models split on #1. Make it noticeable, not alarming.
4. **Idle / no draft detected**: quiet placeholder + log health.
5. **Error/degraded**: Detailed Logs disabled in Arena (needs a clear call to
   action), unknown set (day-0 fallback active).

## Feel
Calm, precise, instrument-like. The player is under a timer; nothing should
animate for its own sake — motion only to mark state changes (new pack
arriving). Confidence should read as horizontal weight (bars/emphasis), not
as numbers alone. Rarity can tint names the way drafters expect (gold/orange
for rare/mythic) but the confidence signal must dominate the hierarchy.

## Deliverables
- Layout for the live-pick state at 380px and at 240px (no-art) widths.
- The five states above as variants.
- A visual hierarchy spec: exactly what makes #1 pop, how 2–3 differ from
  the tail, where art sits if enabled (left thumbnail? hover-reveal?).
- Color/typography tokens (dark theme; system font stack or one bundled face).
- HTML/CSS mockup preferred (it can be dropped straight into pywebview).
