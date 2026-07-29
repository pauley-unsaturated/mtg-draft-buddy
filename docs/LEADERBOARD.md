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
