# Leaderboard

All rows produced by `python -m draftbot.eval` (no hand-computed numbers).
Primary metric: top-1 on MSH test; headline: expert-subset top-1.
`skill-gap` = expert-subset top-1 − low-skill-subset top-1 (validates we mimic good drafters).

| EXP | Model | Set(s) | Split | Stats | Top-1 | Top-3 | Expert Top-1 | Skill-gap | NLL | Config | SHA | Wall-clock | Date |
|-----|-------|--------|-------|-------|-------|-------|--------------|-----------|-----|--------|-----|------------|------|
| — | random | MSH | test | full | 0.2322 | 0.5174 | 0.2347 | +0.0041 | 1.8302 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | rarity-first | MSH | test | full | 0.3270 | 0.6334 | 0.3213 | -0.0072 | 13.5253 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | alsa-greedy | MSH | test | full | 0.4672 | 0.8009 | 0.4837 | +0.0249 | 1.4449 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | gih-greedy | MSH | test | full | 0.3785 | 0.7171 | 0.4168 | +0.0626 | 1.7862 | baseline | 7c25359 | <1min | 2026-07-29 |
| — | gih-in-lane | MSH | test | full | 0.4236 | 0.7780 | 0.4608 | +0.0617 | 2.7696 | baseline | 7c25359 | <1min | 2026-07-29 |
| EXP-001 | EXP-001@best | MSH | val | full | 0.5758 | 0.8672 | 0.5994 | +0.0383 | 1.1683 | configs/EXP-001.yaml | ef5a849 | 2.7min | 2026-07-29 |
| EXP-001 | EXP-001@best | MSH | test | full | 0.5736 | 0.8648 | 0.5980 | +0.0390 | 1.1742 | configs/EXP-001.yaml | ef5a849 | 2.7min | 2026-07-29 |
| EXP-011 | EXP-011@best | MSH | val | full | 0.6731 | 0.9406 | 0.6987 | +0.0422 | 0.8602 | configs/EXP-011.yaml | 687af40 | 14min | 2026-07-29 |
| EXP-012 | EXP-012@best | MSH | val | full | 0.6719 | 0.9400 | 0.6976 | +0.0410 | 0.8639 | configs/EXP-012.yaml | 687af40 | 11min | 2026-07-29 |
| EXP-013 | EXP-013@best | MSH | val | full | 0.6778 | 0.9427 | 0.7020 | +0.0376 | 0.8474 | configs/EXP-013.yaml | 687af40 | 28min | 2026-07-29 |
| EXP-014 | EXP-014@best | MSH | val | full | 0.6718 | 0.9398 | 0.6994 | +0.0438 | 0.8569 | configs/EXP-014.yaml | 687af40 | 12min | 2026-07-29 |
| EXP-023 | EXP-023@best | MSH | val | full | 0.6765 | 0.9424 | 0.7011 | +0.0379 | 0.8519 | configs/EXP-023.yaml | 0f5159f | 38min | 2026-07-29 |
| EXP-013 | EXP-013@best | MSH | test | full | 0.6756 | 0.9410 | 0.6985 | +0.0386 | 0.8560 | configs/EXP-013.yaml | 0f5159f | 28min | 2026-07-29 |
| EXP-020 | EXP-020@best | MSH | test | full | 0.6671 | 0.9364 | 0.6902 | +0.0400 | 0.8789 | configs/EXP-020.yaml | 0f5159f | 17min | 2026-07-29 |
| EXP-023 | EXP-023@best | MSH | test | full | 0.6742 | 0.9409 | 0.6962 | +0.0366 | 0.8595 | configs/EXP-023.yaml | 0f5159f | 38min | 2026-07-29 |
| EXP-022 | EXP-022@best | MSH | val | full | 0.6741 | 0.9411 | 0.6995 | +0.0402 | 0.8554 | configs/EXP-022.yaml | 8abd93f | 10min | 2026-07-29 |
| — | gih-greedy | MSH | test | week1 | 0.3793 | 0.7177 | 0.4174 | +0.0637 | 1.7844 | baseline | 634c503 | <1min | 2026-07-29 |
| EXP-030 | EXP-030-zeroshot@best | MSH | test | week1 | 0.6486 | 0.9265 | 0.6696 | +0.0358 | 0.9324 | configs/EXP-030.yaml | 634c503 | 98min | 2026-07-29 |
| EXP-030 | EXP-030-zeroshot@best | MSH | test | none | 0.5239 | 0.8545 | 0.5167 | -0.0112 | 1.2040 | configs/EXP-030.yaml | 634c503 | 98min | 2026-07-29 |
| EXP-031 | EXP-031@best | MSH | test | full | 0.6752 | 0.9421 | 0.6973 | +0.0383 | 0.8533 | configs/EXP-031.yaml | 634c503 | 33min | 2026-07-29 |
| EXP-032 | EXP-032@best | MSH | test | full | 0.6752 | 0.9420 | 0.6972 | +0.0379 | 0.8531 | configs/EXP-032.yaml | ed4c23a | 42min | 2026-07-29 |
| EXP-040-1k | EXP-040-1k@best | MSH | val | full | 0.6666 | 0.9377 | 0.6935 | +0.0416 | 0.8757 | configs/EXP-040-1k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-040-5k | EXP-040-5k@best | MSH | val | full | 0.6716 | 0.9404 | 0.6966 | +0.0391 | 0.8651 | configs/EXP-040-5k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-041-1k | EXP-041-1k@best | MSH | val | full | 0.6654 | 0.9370 | 0.6919 | +0.0416 | 0.8802 | configs/EXP-041-1k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-041-5k | EXP-041-5k@best | MSH | val | full | 0.6698 | 0.9392 | 0.6946 | +0.0390 | 0.8672 | configs/EXP-041-5k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-041-all | EXP-041-all@best | MSH | val | full | 0.6770 | 0.9424 | 0.7021 | +0.0391 | 0.8498 | configs/EXP-041-all.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-042-1k | EXP-042-1k@best | MSH | val | full | 0.6634 | 0.9360 | 0.6899 | +0.0414 | 0.8851 | configs/EXP-042-1k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-042-5k | EXP-042-5k@best | MSH | val | full | 0.6672 | 0.9386 | 0.6924 | +0.0395 | 0.8732 | configs/EXP-042-5k.yaml | 6959f47 | grid | 2026-07-29 |
| EXP-042-all | EXP-042-all@best | MSH | val | full | 0.6733 | 0.9405 | 0.6987 | +0.0399 | 0.8611 | configs/EXP-042-all.yaml | 6959f47 | grid | 2026-07-29 |

## Deck builder (Phase 5, EXP-1NN)

Deck rows from `python -m draftbot.eval --deck`. Primary: deck-F1 (full 40 incl.
basics) on MSH val; headline: trophy-F1 (eval builds with ≥5 wins — agree with
winning builds). Target = rebuild-ceiling (human self-agreement).

| EXP | Model | Set | Split | Stats | F1 | NB-F1 | Trophy-F1 | Win-wtd F1 | Lands-MAE | Basics-L1 | Config | SHA | Wall-clock | Date |
|-----|-------|-----|-------|-------|----|-------|-----------|------------|-----------|-----------|--------|-----|------------|------|
| — | random-legal | MSH | val | full | 0.6821 | 0.6076 | 0.6880 | 0.6829 | 1.826 | 6.539 | baseline | a980284 | <1min | 2026-07-30 |
| — | gih-top23 | MSH | val | full | 0.7601 | 0.7362 | 0.7736 | 0.7699 | 2.221 | 6.499 | baseline | a980284 | <1min | 2026-07-30 |
| — | gih-in-lane-build | MSH | val | full | 0.7844 | 0.7765 | 0.7915 | 0.7933 | 2.658 | 6.474 | baseline | a980284 | <1min | 2026-07-30 |
| — | rebuild-ceiling | MSH | val | full | 0.9545 | 0.9469 | 0.9627 | 0.9608 | 0.162 | 0.973 | baseline | a980284 | <1min | 2026-07-30 |
| EXP-101 | EXP-101@best | MSH | val | full | 0.8829 | 0.8625 | 0.8920 | 0.8894 | 0.247 | 2.492 | configs/EXP-101.yaml | 8433d49 | 2.3min | 2026-07-30 |
