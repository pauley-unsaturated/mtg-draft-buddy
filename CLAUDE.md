# mtg-draft-buddy

Fork of RyanSaxe/mtg (2022 transformer draft AI), being modernized into a
multi-set-pretrained drafter. Target set: MSH.

- **Master plan:** `docs/PLAN.md` — read its §0 ground rules at session start; work
  tasks in order; log to `docs/JOURNAL.md` (append-only) and `docs/LEADERBOARD.md`.
- **New code:** `src/draftbot/` (PyTorch + MPS). **Reference only:** `mtg/` (original TF, do not modify).
- **Env:** `uv sync --extra dev`; run things via `uv run …` (Python 3.12, torch on MPS).
- **Tests:** `uv run pytest`. Data lives under `data/` (never committed, raw/ immutable).
- **Hard guardrails** (PLAN.md §0): MSH is quarantined from all pretraining; splits by
  draft_id persisted once; leaderboard rows only from `python -m draftbot.eval`;
  seeds in configs; runs >20min must be resumable.
