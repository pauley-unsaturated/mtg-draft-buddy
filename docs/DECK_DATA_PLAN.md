# Deck dataset plan (P5.T1 data layer)

Owner directive 2026-07-29: nail the dataset before the builder model.

## Sources — inventory

| source | status | contents | use |
|---|---|---|---|
| `game_data_public.<SET>.<EVENT>.csv.gz` | **already on disk, all 30 sets** (~1.6GB; fetched in Phase 0 for card stats) | one row per game: full 40-card maindeck (`deck_<name>` counts — every card in the submitted build, played or not), `sideboard_<name>`, `build_index`, `won`, ranks, user WR bucket, `main_colors`, timestamps | the deck dataset |
| `draft_data_public...` | on disk (Phase 0) | per-pick draft logs | cross-validation of pools + shared splits |
| replay data | **not used** — per-action play-by-play; not at the public S3 path (audit 2026-07-28) | — | out of scope |

## Empirical facts (MSH full population, 377,514 games — supersedes the 30k probe)

- Deck sizes: 92.6% exactly 40; ≥99% within 40–43; true tail reaches 60
  (rate ≤1% — keep cap 43 for size-conditioned heads; oversize rows remain
  valid membership-training examples).
- Basics per deck: mean 14.8 (σ 1.6, range 4–23) → total lands ≈ 17 with ~2.3
  nonbasics; land-count head range 14–20 is right.
- Nonbasic pool sizes 38–42 — consistent with 42 picks minus drafted basics.
- No games with a missing deck block (`won` parses strictly as bool; zero NaNs).
- 83,457 (draft, build) pairs / 67,808 drafts on MSH (probe extrapolated ~95k);
  1 build for 81.0% of drafts, rebuilds up to 7; 4.5 games per build.
- game_data draft_ids are a subset of the draft parquet's → 100% of builds
  join `data/splits/MSH.json`.

## Extraction → `data/processed/<SET>/decks.parquet`

One row per (draft_id, build_index):
`deck_ids/deck_counts` (sparse nonbasics — nonbasic LANDS included as ordinary
cards), `side_ids/side_counts`, `basics` (W/U/B/R/G counts), `n_games`,
`n_wins`, `user_win_rate`. Extractor: `draftbot/data/decks.py` — vectorized
(whole-file read fits RAM easily; groupby size/sum for games/wins + one
representative row per build for the card vectors). MSH extracts in ~2 min;
ready for the 30-set scale-out as-is.

## Validation suite (tests/test_decks.py — gate for the model work)

1. **Pool identity**: for ≥100 sampled draft_ids, multiset(deck ∪ sideboard
   nonbasics) == multiset(picks from draft parquet, minus drafted basics).
   Catches column-mapping and vocab drift. (Drafted basics can't be separated
   from added basics in game_data — excluded from membership labels by design;
   the basics head owns them.)
2. Deck size sanity: nonbasics + basics ∈ [40, 43] for ≥99% of rows.
3. Membership label validity: deck_counts ≤ pool counts per card, everywhere.
4. Rebuild integrity: builds within a draft share the same pool.
5. Split hygiene: decks join `data/splits/<SET>.json` by draft_id — the SAME
   splits as the draft model, so a test-split draft is never seen by either
   model in training (his HUD draft's build eval stays honest).

## Curation & weighting (mirrors the draft-model lessons)

- No hard skill filter (EXP-013 lesson). Example weight =
  `(1 + n_wins) · soft_skill(user_win_rate)` — winning builds dominate,
  0-win builds still contribute negative signal.
- Trophy view for eval: the headline eval subset = builds with ≥5 wins
  (7-win trophies where available), expert users — "agree with winning
  builds by winning players".
- Rebuilds: all builds kept for training (more signal); eval uses the
  most-played build per draft.

## Scale-out

MSH first (model dev + his-draft demo). Then the same extractor over the 30-set
corpus (~1.5–2M builds) for a pretrained feature-based builder — same
set-agnostic card-feature pathway as the draft trunk, same quarantine rules if
a new sandbag exists by then.

## DoD

decks.parquet for MSH; all 5 validation tests green; data card section added
to docs/data_cards/MSH.md (deck counts, sizes, wins distribution, computed not
typed); journal entry with any surprises.

**Status 2026-07-29: MET.** decks.parquet (83,457 builds), tests/test_decks.py
5/5 green, data card §Decks regenerated, journal entry written. Model work
(membership + basics + land-count heads) may begin.
