# Sealed deck builder (P5.S)

Owner directive 2026-07-31: help building decks for the live Avatar sealed
event. The set is **TLA** (there is no "AVA" — Scryfall has one Avatar
expansion). Everything here is an ADDITION: `source="sealed"` threads through
extractor / splits / dataset / eval / trainer with `draft` defaults, new
artifacts only, HUD untouched (P5.S4 is a future proposal).

## Why sealed is the easy case

- 17lands publishes Sealed + TradSealed game data for 27 of our 30 onboarded
  sets (AFR/MID/VOW predate it), schema-identical to draft game files.
- **No draft-log join.** Pool = deck ∪ sideboard by construction — the AFR
  failure class does not exist. Extraction is self-contained per game file.
- The whole model/eval stack is pool-shape-agnostic (set transformer): the
  only new thing sealed brings is ~84-card unfiltered pools, i.e. the model
  must CHOOSE colors rather than have the draft pre-filter them.

## Data

- `python -m draftbot.data.decks --set <CODE> --event sealed` →
  `data/processed/<CODE>/decks.sealed.parquet` (same schema as decks.parquet).
- Curation, logged at extract (gate findings): drop decks outside 40–43 cards
  (~1–2%/set; up to 91-card unbuilt piles) and entries whose rebuilds disagree
  on their own pool (17lands quirk, ~1% of LCI/EOE multi-builds). Same-pool
  rebuilds are load-bearing for the ceiling and winner-pref.
- Splits: `data/splits/<CODE>.sealed.json` — separate draft_id universe,
  sha1 mod 100 → 90/5/5, persisted once (§0).
- Gate: `tests/test_sealed_decks.py` — draft gate with the draft-log identity
  test replaced by pool-size sanity (6 boosters ≈ 84 nonbasics).
- Corpus: `configs/corpus/sealed_v1.yaml` — decks_v2 weights, EOE holdout,
  ~394k builds / 26 sets.

## Metrics & ceiling (TLA sealed val, full stats)

Same harness (`--deck --event sealed`). Human rebuild ceiling **0.9066 F1 /
0.9224 trophy-F1** — lower than draft's 0.95/0.965, as expected: bigger
unfiltered pools admit more legitimate builds. Winner-pref exists but has ~14
val pairs — ignore it at sealed scale. Controls: random-legal 0.4187,
gih-top23 0.5205, gih-in-lane 0.6024.

## Findings so far

- **The draft→sealed shift is color choice.** EXP-126 (draft trunk) zero-shot:
  0.6923/0.7195 — lands-MAE already excellent (0.29) but basics-L1 9.2.
- **EXP-130** (TLA sealed fine-tune of EXP-126, 0.7min): 0.7269/0.7570.
  9.6k builds close a third of the gap; the corpus pretrain owns the rest.
- **Corpus verdict (2026-07-31): EXP-134 is the sealed trunk.** Warm start ×
  scratch LR (3e-4, patience 4) → 0.7123 proxies; EXP-131 (warm, FT-lr)
  stalls at its starting point, EXP-132 (scratch) plateaus lower (0.7051).
  All three land near ~0.71 — sealed is DATA-bound (335k builds = 1/11 of
  decks_v2), not init- or scale-bound.
- **Holdout (EOE sealed, ceiling 0.8971 trophy):** EXP-134 0.6421 full /
  0.5432 none vs draft trunk 0.5708/0.4795 — the sealed corpus transfers.
- **In-corpus-needs-no-FT replicates:** TLA trophy band 0.7516–0.7570 across
  EXP-130/131/133/134. Production = EXP-134 (CLI default, in bundle);
  EXP-130/133 kept as TLA-max variants.
- **Joint program (2026-08-01): EXP-135 supersedes EXP-134 as production.**
  Format token (zero-init, additive) + joint corpora (sealed boost 4.0):
  draft decks preserved at EXP-126 level (≤0.2pt), TLA sealed 0.7576 with no
  per-set FT. EXP-134 keeps a 1-2pt edge on the unseen-set sealed holdout;
  next-set recipe = refresh the joint corpus. Annealing probes: uniform flat;
  win_skill (bombs hypothesis) +1pt proxies / +0.9pt TLA 7-win-F1 but flat
  trophy-F1 — the first lever to revisit as sealed data grows.
- Remaining future lever: sealed data volume (17lands accrues it weekly).

## Day-0 workflow for the NEXT sealed event

1. Set already onboarded for draft? Then when 17lands sealed data appears:
   `--event sealed` extract (seconds) + `build_sealed_splits` + gate.
2. Before data exists: the sealed corpus trunk zero-shots with `--stats none`
   (stat-dropout is a trained condition, same as the draft story).
3. Owner interface: `python -m draftbot.build --set <CODE> --pool pool.txt
   [--ckpt checkpoints/<BEST-SEALED>]` — Arena pool export in, proposed 40 +
   bubble out.
