# Engineering hand-off: HUD deck-builder section (HUD H5 / PLAN P5)

Audience: an agent implementing the deck-builder view in the HUD app.
Design spec: docs/DECKBUILDER_DESIGN_BRIEF.md (Claude Design round-trip).
Read CLAUDE.md + PLAN.md §0 first. Scope: `src/draftbot/hud/` ONLY — the
model/data layers are done and tested; do not modify `draftbot/{data,models,
train,eval}` or any test outside `tests/test_hud_*`.

## What exists (do not rebuild)

- **Models on disk** (gitignored; Mark's machine has them):
  - `checkpoints/EXP-127/` — production one-shot builder (val trophy-F1
    0.9011; corpus-pretrained trunk EXP-126 + MSH fine-tune). Use for the
    initial proposed build. (EXP-121 is the retired previous best.)
  - `checkpoints/EXP-116/` — diffusion variant: same quality class (0.8951)
    but supports **partial-deck conditioning** — use it for lock-and-rebuild.
- **Loading**: `draftbot.models.loading.deck_builder_from_checkpoint(ckpt_dir,
  set_code, stats) -> TorchDeckBuilder`. `stats` ∈ full|week1|none must match
  what the HUD's advisor already resolved for the set (day-0 → "none").
- **Inference contract**: `TorchDeckBuilder.build_all(arr: DeckArrays) ->
  [{"deck": {card_id: count}, "basics": ndarray(5)}]` — W/U/B/R/G order,
  always exactly 40 cards, deck ⊆ pool (tests/test_deck_model.py guarantees
  legality). Build a 1-row `DeckArrays` from the drafted pool:
  `draftbot.data.deck_dataset.DeckArrays` (pool_ids/pool_counts int16, PAD=-1,
  deck_counts/basics zeros, meta needs only a draft_id column).
- **Pool source**: `hud/state.py` tracks all 42 picks. The builder pool =
  the NONBASIC picks (drop ids where `cards.is_basic == 1` — drafted basics
  are not pool members; the basics head owns the manabase. See
  docs/DECK_DATA_PLAN.md "pool identity" note).
- **Membership confidences** for the boundary-zone UI: run the model forward
  directly (see `TorchDeckBuilder._heads`) and sigmoid the member logits;
  slot order matches your pool_ids row.
- **Server**: `hud/server.py` — stdlib HTTP, panel polls `/state` every
  500ms; `/art/<card_id>` serves cached Scryfall art. Extend `/state` with a
  `deck` section rather than adding a second poll loop.

## Lock-and-rebuild (the one genuinely new mechanic)

EXP-116 is `DeckBuilder(diffusion=True)`. Conditioning: build a `deck_state`
tensor (see `draftbot.models.builder.MASK_STATE`) — locked-IN card slots get
their count, locked-OUT get 0, everything else MASK_STATE — and drive
`TorchDeckBuilder._maskgit_probs`-style decoding from that initial state
instead of all-MASK. The current `_maskgit_probs` always starts fully masked:
add an optional `init_state=` parameter to it (this is the ONE permitted
core-code touch; keep it backward-compatible, default None = current
behavior, and extend `tests/test_deck_model.py` with a lock-respect test:
locked slots must appear/not-appear in the decoded 40 exactly as locked).

## Suggested flow

1. `state.py` detects pick 42 (or user opens the provisional view) → emits
   pool to a new `hud/deck.py` module.
2. `deck.py` holds both loaded builders, produces
   `{build, confidences, cuts, locks, model_id, stats_mode}` → merged into
   the server's `/state` payload.
3. Panel renders per the design spec; lock toggles POST to a new `/lock`
   endpoint; rebuild POSTs to `/rebuild` → EXP-116 path with the lock state.
4. Provisional mode (draft still live) reuses the same path with the partial
   pool — clearly badged, never stealing focus from the pick view.

## DoD

- Full replay of `tests/fixtures/real_draft_msh.log` through
  `python -m draftbot.hud --replay` ends in a proposed 40-card legal build
  rendered in the panel; lock 2 cards → rebuild honors both; `uv run pytest`
  green including the new lock-respect test; no regressions to the pick view
  (existing HUD tests untouched and passing).
- Journal entry (append-only) describing what shipped + a screenshot path.
- Do NOT add leaderboard rows (that is eval-harness territory) and do not
  touch `docs/PLAN.md` checkboxes — this is HUD_PLAN H5 work.
