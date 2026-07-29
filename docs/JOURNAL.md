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

## 2026-07-29 — Design note for P5.T1 (deckbuilder), owner discussion

Mark floated diffusion; agreed direction: **masked discrete diffusion /
MaskGIT-style iterative infilling** over the deck set — the principled version of
the 2022 DAE (its sample-N-out corruption IS absorbing-state diffusion; its
iterative argmax IS single-path denoising). Plan: (1) one-shot baseline first:
set-transformer pool encoder (warm-start from draft trunk card encoder) +
per-card maindeck head + land-count head, trained on skill-curated winning
builds from game_data; (2) masked-infilling model with confidence-based commit
(5-10 steps), which gives partial-deck conditioning (lock cards, complete rest)
for free. Avoid continuous diffusion (discrete-rounding pain, no benefit at
40-slot scale). Eval: win-rate-weighted agreement (many builds co-optimal) +
curve/land sanity; expert builds are the reference, per the expert-signal
directive. AR pointer decoder rejected for decks: imposes order on a set.

## 2026-07-29 — Ablation ladder results (P2.T3): the skill-filter surprise

Ranked by expert-subset val top-1 (owner's headline metric):
EXP-013 no-hard-filter .7020 > EXP-014 ls0 .6994 > EXP-011 dense .6987 >
EXP-010 recipe .6977 > EXP-012 meanpool .6976.
Conclusions, one line each:
- (c) HARD skill filter is a net LOSS (-0.4pt expert): soft importance weighting
  already downweights weak drafters; the filter just burns 53% of data. Both
  owner and agent predicted the opposite — data > purity at this scale. KEEP
  soft weighting, DROP hard filter for the scale-up winner combo.
- (a) pointer vs dense head: tie (dense +0.1pt, noise). Pointer retained —
  required for Phase-3 vocab-agnostic transfer.
- (b) set-encoder vs mean pool: tie. Set encoder retained (cheap; Phase-3
  pack-conditioning rationale) but not a proven win at 1.3M single-set scale.
- (d) label smoothing 0 vs 0.05: tie on accuracy; 0.05 clearly better ECE
  (.007 vs .030). Keep 0.05.
- (e) Premier+Trad merge: NOT RUN this session (loader change); backlog.
EXP-020 (big, filtered) in flight; EXP-023 (big, no filter) queued after — the
two candidates for Phase-2 best.

## 2026-07-29 — PHASE 2 GATE PASSED; EXP-013 declared Phase-2 best

Test gate (run once, frozen in docs/scorecards/phase2/): EXP-013 test top-1
.6756 / expert .6985 — beats EXP-001 (.5736) by +10.2 (gate: ≥2) ✓; fixtures
21/21 ✓; rare-take 1.08× expert (≤1.5×) ✓; coherence 2.06 (human 2.16).
EXP-020 (8.5M filtered) .6902 expert and EXP-023 (8.5M unfiltered) .6962 expert
both UNDER the 1.3M EXP-013 → single-set MSH is data-limited; growth must come
from pretraining (the plan's predicted signal — P2.T4 conclusion).
Deliverable #1 for Mark = checkpoints/EXP-013 (demo: --models checkpoints/EXP-013).
Note: ceiling analysis (calibrated-confidence proxy) puts expert-agreement
ceiling in low-to-mid .70s; EXP-013 at .6985 is within ~2-4pts.

## 2026-07-29 — P3.T3 warm-start pilot: transfer VERDICT POSITIVE

EXP-021 (v2 on EOE, 42.3k skill-filtered drafts): EOE expert val .7040. EOE
quirk found: t=39 (13-pick packs); loading.py now infers t from checkpoint.
EXP-022 (EOE trunk → MSH fine-tune, 11 epochs, EOE scaler): MSH expert val
.6995 vs EXP-010 from-scratch .6977 at equal epochs — modest final gain, much
faster convergence (epoch 1: .6895 vs .6423 overall). Feature pathway confirmed
working cross-set → proceed to EXP-030 corpus pretrain (queued).
Also fixed this session: 4/4.07M EOE picks were 17lands glitch rows (pick not
in pack) inflating printed loss ×600 via the -1e9 mask; loader now drops such
drafts globally. Val metrics were never affected.

## 2026-07-29 — EXP-030 pretrain complete: zero-shot MSH is REAL

7.7M feature-only trunk + set token, 29 sets (2021 quartet kept via
no-skill-metadata fallback), stat-dropout 0.25, 20k steps / 98 min.
Zero-shot MSH val (never trained on MSH): none .5304/.5257exp,
week1 .6514/.6719exp, full .6587/.6845exp.
P3.T4 sanity bars demolished: zs(week1) vs ALSA-greedy +19pts; zs(none) vs
rarity-first +20pts. Phase-3 gate conditions 1-2 comfortably met on val
(GIH-greedy(week1) val = .3804/.4198exp; margin +27pts; gap to Phase-2 best
2.7pts < 6). Day-0 mode .53 = better than any stats-armed heuristic bot.
EXP-031 fine-tune launching via chain.

## 2026-07-29 — Phase-3 gate: 3 of 4 conditions pass; condition 3 = statistical tie

MSH TEST (declared gate run): (1) zs(week1) .6486 vs gih-greedy(week1) .3793
✓ +27pts; (2) gap to Phase-2 best .6756−.6486 = 2.7pts < 6 ✓; (4) zero-shot
easy fixtures 10/10 ✓. (3) EXP-031 FT .6752 vs EXP-013 .6756 — 0.04pt under
(≈0.3 SE on 154k picks; VAL had EXP-031 ahead +0.27pt expert). Reading: tie
within noise, not "pretraining hurts", but the gate is strict — one remedy
sanctioned: EXP-032 (gentler FT: peak_lr 7e-5, warmup 200, patience 5).
If it also lands under, escalate per protocol.
Zero-shot day-0 test .5239 with skill-gap −.011 (without stats the trunk can't
fully separate expert taste — noted for P3.T5 oracle-text lever).

## 2026-07-29 — Phase-3 gate condition 3: second attempt also ties → ESCALATED

EXP-032 (gentler FT) test: .6752/.6972exp — identical to EXP-031 (.6752/.6973).
Robust result: the pretrained+fine-tuned model TIES the MSH-only best
(Δ −0.0004 overall, −0.0013 expert; both under 1 SE). Val ordering favored the
fine-tunes (.7047/.7043 vs .7020) — test flip is textbook split noise.
"Pretraining must not hurt the data-rich case": demonstrated (no hurt). Strict
"≥": not met. Two remedies attempted → escalating the pass/hold decision to
Mark per §0. MSH stays quarantined until he rules. Both deliverables exist and
are demo-ready: EXP-013 (single-set best) and EXP-031 (pretrained→fine-tuned;
EXP-032 equivalent). Zero-shot results unaffected — those passed with +27pt
margins.

## 2026-07-29 — OWNER RULING: Phase-3 gate PASSED (tie accepted)

Mark ruled the condition-3 statistical tie acceptable ("good enough to move past
Phase 3"). Per the gate's on-pass protocol: MSH exits quarantine and joins the
corpus (configs/corpus/pretrain_v2.yaml — v1 frozen for EXP-030 provenance);
the NEXT Arena release becomes the new sandbag (code TBD at announcement —
update pretrain_v2 quarantine + PLAN §0 when known). P3.T5 (zero-shot
hill-climb) and P3.T6 (few-shot curves) deferred by owner in favor of Phase 4;
they remain unchecked in the plan and can be resumed any time. Committing and
pushing all session work before starting Phase 4.

## 2026-07-29 — P5.T4 promoted: real-time draft HUD plan (owner request)

Mark wants a HUD app running alongside Arena. Mechanism verified from
rconroy293/mtga-log-client source: tail ~/Library/Logs/Wizards Of The
Coast/MTGA/Player.log (Detailed Logs setting required), parse
Draft.Notify/LogBusiness/EventPlayerDraftMakePick JSON events; card ids are
Arena grpIds == Scryfall arena_id (375/375 coverage in our MSH cache → pure
lookup into cards.parquet). Full plan in docs/HUD_PLAN.md: LogFollower →
DraftState → Advisor (existing TorchScorer, calibrated probs) → rich-TUI v0
then pywebview panel with Scryfall art. Milestones H1-H5; H1 needs a captured
real draft log (Mark: enable Detailed Logs + save Player.log after next draft).

## 2026-07-29 — P4.T2 equivalence study: LoRA ≈ full-FT, decision LORA-BY-DEFAULT

Expert val top-1 across modes × MSH draft budgets (all init_from EXP-030):
            1k      5k      all(61.6k)   trainable
full-FT    .6935   .6966   .7043        100% (8.0M)
LoRA r4    .6919   .6946   .7021        4.2% (329k)
head-only  .6899   .6924   .6987        ~2% (174k)
LoRA sits 0.16-0.22pt under full-FT at every budget — equivalent within split
noise, at 1/24th the trainable params. DECISION: LoRA is the default for
onboard-set adapters (cheap, reusable base, separate adapter.pt); full-FT
reserved for flagship-set models where the last 0.2pt matters. Head-only is a
surprisingly strong floor (.6987 ≈ Phase-2 EXP-013's .6985 with 2% params) —
fine for day-3 quick adapters. Few-shot: ~1k drafts (≈ day 1-2 of a set)
reaches ~.69 expert in every mode — this table doubles as the P3.T6 evidence
at 3 budgets. LoRA device-placement + parametrized-checkpoint loading bugs
fixed along the way (params must be created on the base weight's device;
loader re-applies LoRA structure before load_state_dict).

## 2026-07-29 — PHASE 4 GATE PASSED

P4.T3 dry-run: `python -m draftbot.onboard_set --set KTK --budget-drafts 3000
--epochs 3` ran unattended: pipeline skip-detection → zero-shot scorecards
(KTK val .6475/.6719exp from the EXP-030 base) → LoRA adapter (0.32M trainable,
42s) → .6503/.6729exp scorecard. P4.T4 executed refresh: EXP-033 on corpus v2
(30 sets incl. MSH): MSH week1 .6852exp / full .6962exp — beats the v1 trunk on
every mode (recipe gate ✓); EXP-033 is now the recommended --base for future
onboard-set runs. Phase-4 gate: one-command onboarding ✓, adapter decision
documented with data ✓. Remaining open plan work: P3.T5/T6 (owner-deferred),
Phase 5 (deckbuilder implementation, HUD H1-H5 per docs/HUD_PLAN.md).

## 2026-07-29 — HUD H1+H2 landed (terminal advisor working in replay)

follower.py parses the exact message shapes from seventeenlands/mtga_follower
(Draft.Notify multi-line JSON buffering included); state.py maps grpId==
arena_id → vocab via the Scryfall cache and rebuilds (packs, prev, position) by
counting packs (Arena's reported numbers are display-only — base varies).
H1 DoD: synthesized Player.log of a REAL test draft replays to exact array
reconstruction (42+42 events, zero unresolved grpIds, mid-draft attach test).
H2: `python -m draftbot.hud --models checkpoints/EXP-033 [--replay log]`
renders live rich table: rank/name(rarity-tinted)/calibrated-prob bar/GIH/ALSA,
pool pips, model-disagreement flag in the title. Verified over a full 42-pick
replay with EXP-033. Awaiting Mark's captured real Player.log (Detailed Logs
on) to harden fixtures; H3 window UI awaits the design-brief round-trip.
