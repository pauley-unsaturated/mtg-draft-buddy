# Journal (append-only)

Format: `## YYYY-MM-DD — <entry title>` followed by prose. Never edit past entries.

## 2026-07-28 — Session start: enacting docs/PLAN.md from P0.T1

Owner directive for this session (Mark): deliver BOTH tryable models — (1) the modernized
single-set model trained on MSH only (Phase 2), and (2) the pretrained/general model
fine-tuned on MSH (Phase 3). Target set is MSH (confirmed; it is the currently
available Arena set). Additional owner requirement: **validate that models agree with
good drafters, not bad drafters** — implemented as a first-class eval metric: the
scorecard reports top-1 agreement on the expert subset (WR ≥ 0.62, ≥ 100 games) AND on
a low-skill subset (bottom tail by user_win_rate), plus the gap. A model that mimics
skilled play must show expert-agreement > low-skill-agreement; the gap must not be
worse than GIH-greedy's. Added to P1.T2 scope (journal-sanctioned scope addition, no
plan tasks reordered).

Environment: uv 0.11.19 present, system python 3.14 (project pins 3.12 via uv).
Starting P0.T1 scaffolding.

## 2026-07-28 — 17lands ratings-API drift; backlog item promoted into P0.T6

The anonymous `card_ratings/data` API no longer serves win rates (all null) and
ignores `start_date`/`end_date` (verified on MSH, FIN, EOE, TDM; counts also ~100×
too small). This breaks the original stats pathway recorded in PLAN Appendix B.
Remedy: promoted the backlog item **"rating-free stat features derived from raw
game_data"** into P0.T6. Snapshots are now computed from the public
`game_data_public.<SET>.<EVENT>.csv.gz` + our converted draft parquet:
ALSA/ATA/seen/pick counts from draft data; GP/OH/GD/GIH/GNS win rates + IWD from
game data, overall + all 31 exact-deck-color filters; win rates NaN (masked) under
5 games. Date windows are exact (game_time date ∈ [start, end]), which makes the
week-1 "day-7 knowledge" snapshots *more* honest than the API ever was. The two
API-fetched snapshot files (null win rates, useless) were deleted and rebuilt via
`features.build_snapshot`; on-disk contract unchanged. Note for future sets:
game_data must be fetched alongside draft_data (fetch.py now takes kind="game").

## 2026-07-28 — Phase 0 complete

All P0 tasks done, 12 tests green. MSH: 68,609 complete drafts / 2,881,578 picks,
t=42, vocab 339 (281 msh + 58 mar), Premier parquet 231MB. Snapshots computed from
raw data: full (2026-06-26→2026-07-28) and week1 (→2026-07-03); 576-col API shape
replaced by 356 computed columns (draft-stage stats unsuffixed once, game-stage
×32 filters). Expert subset auto-relaxed to WR≥0.58/≥50 games; low-skill
WR≤0.50/≥50 games (both ≥50k test picks). `make data SET=<code>` is the
one-command pipeline. 24 incomplete drafts dropped (Premier), 1 (Trad).

## 2026-07-28 — Phase 1 complete: eval suite + EXP-001 baseline

Heuristic floor (MSH test, full stats): random .232 / rarity-first .327 /
gih-greedy .379 / gih-in-lane .424 / alsa-greedy .467 top-1. Skill-gap metric
behaves as designed: gih bots agree with experts >> low-skill (+.06), rarity-first
gap NEGATIVE (-.007) — the metric catches bad-drafter mimicry.

EXP-001 (legacy port, 1.68M params, 6 epochs / 2.7 min on MPS): val .576, test
.574 top-1 (beats every baseline; +19.5 over gih-greedy, +10.6 over alsa-greedy).
Expert top-1 .598, skill-gap +.039 (well above gih-greedy bar). ECE .034.
Weak spots for Phase 2: rare-take 0.38× expert (the λ=10 anti-rare prior
overshoots — v2 drops priors for skill curation + watchdog); pool coherence 1.81
colors vs human 2.16 (slightly over-committed); pack-1 positions 0-3 weakest
early-pick accuracy. Behavioral fixtures: 21 scenarios (10 easy all pass for
gih-greedy; medium/hard reserved for trained models). Analysis per position shows
P1P1 .57 rising to ~.60+ late-pack (wheel picks easier).

## 2026-07-28 — EXP-001 pathology: priors + basics-in-vocab = P1P1 basic-taking

The demo CLI surfaced it; quantified on val: EXP-001 top-1 is a basic land in
24.9% of P1P1s (humans: 0.0%), 18.2% vs human 6.9% when a pack contains a basic.
Cause: the legacy cmc prior (hinge toward cheap picks) + rare prior both point at
basics (cmc 0, never rare) under uncertainty; the 2022 original excluded basics
from the vocab so this couldn't happen. Kept as-is for baseline fidelity —
EXP-001 is the reproduction, priors on. Phase-2 v2 (priors off, skill curation)
must clear the easy-no-basic-over-playable fixture EXP-001 fails. Watchdog for
this now exists in the demo + this probe.

## 2026-07-29 — EXP-010: v2 architecture validates decisively

val top-1 .6725 vs EXP-001 .5758 (+9.7). Expert .6977, skill-gap +.041, ECE .007,
rare-take 1.12× expert (priors OFF — watchdog healthy), pool coherence 2.07
(human 2.14), P1P1 basic pathology eliminated (0.0%; overall 3.3% ≈ human).
All 21 behavioral fixtures pass (easy+medium+hard) → P2.T5 done early.
1.27M params, 11 epochs / 10.4 min (early stop @8). Skill filter kept 28,618/61,573
drafts (46%). Ablation chain EXP-011..014 + EXP-020 scale-up running.
Corpus: all 29 non-MSH sets downloaded (6.7GB); onboarding in progress —
MKM needed PLST (The List) companion sheet, registry extended, politeness sleeps
raised after Scryfall 429s.

## 2026-07-29 — Owner directive: expert agreement is THE signal

Mark (citing Ryan Saxe's original intent): the target signal is "what good
drafters draft", not the average 17lands user. Changes:
1. Leaderboard ranking + Phase-2/3 winner declarations use expert-subset top-1
   (evaluation-side; uniform across all runs, past and future).
2. Early stopping / best-checkpoint selection now tracks expert-subset val
   top-1 (train/loop.py) — applies from EXP-014 onward; EXP-010..013 selected on
   overall val top-1 (metrics move together; scorecard comparison unaffected).
3. Pretrain zero-shot tracker selects best.pt on expert-subset week1 MSH-val.
EXP-013 (no skill filter) expected to look competitive on overall top-1 but
worse on expert agreement — that's the point of running it.
