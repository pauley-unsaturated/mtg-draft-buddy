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
| EXP-102 | EXP-102@best | MSH | val | full | 0.8879 | 0.8672 | 0.8960 | 0.8942 | 0.248 | 2.320 | configs/EXP-102.yaml | 2fedfd8 | 2.5min | 2026-07-30 |
| EXP-103 | EXP-103@best | MSH | val | none | 0.8802 | 0.8567 | 0.8885 | 0.8864 | 0.244 | 2.412 | configs/EXP-103.yaml | 2fedfd8 | 2.4min | 2026-07-30 |
| EXP-104 | EXP-104@best | MSH | val | full | 0.8852 | 0.8659 | 0.8946 | 0.8920 | 0.273 | 2.467 | configs/EXP-104.yaml | 2fedfd8 | 3.6min | 2026-07-30 |
| EXP-105 | EXP-105@best | MSH | val | full | 0.8814 | 0.8621 | 0.8906 | 0.8881 | 0.248 | 2.589 | configs/EXP-105.yaml | 2fedfd8 | 1.4min | 2026-07-30 |

### Deck builder — trophy-F1 v2 + winner-pref (owner refinement 2026-07-30; supersedes rows above)

Trophy-F1 v2: pools that DID trophy, target = the build that won. Winner-pref:
model's deck strictly closer to the winning than the losing build of the same
pool (91 val pairs — underpowered at val scale; treat ±0.05 as noise).

| EXP | Model | Set | Split | Stats | F1 | NB-F1 | Trophy-F1v2 | Win-pref | Win-wtd F1 | Lands-MAE | Basics-L1 | Config | SHA | Wall-clock | Date |
|-----|-------|-----|-------|-------|----|-------|-------------|----------|------------|-----------|-----------|--------|-----|------------|------|
| — | random-legal | MSH | val | full | 0.6821 | 0.6076 | 0.6880 | 0.527 | 0.6829 | 1.826 | 6.539 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| — | gih-top23 | MSH | val | full | 0.7601 | 0.7362 | 0.7736 | 0.495 | 0.7699 | 2.221 | 6.499 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| — | gih-in-lane-build | MSH | val | full | 0.7844 | 0.7765 | 0.7915 | 0.473 | 0.7933 | 2.658 | 6.474 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| — | rebuild-ceiling | MSH | val | full | 0.9545 | 0.9469 | 0.9625 | nan | 0.9608 | 0.162 | 0.973 | baseline | 66dd1e6 | <1min | 2026-07-30 |
| EXP-101 | EXP-101@best | MSH | val | full | 0.8829 | 0.8625 | 0.8920 | 0.467 | 0.8894 | 0.247 | 2.492 | configs/EXP-101.yaml | 66dd1e6 | 2.3min | 2026-07-30 |
| EXP-102 | EXP-102@best | MSH | val | full | 0.8879 | 0.8672 | 0.8960 | 0.478 | 0.8942 | 0.248 | 2.320 | configs/EXP-102.yaml | 66dd1e6 | 2.5min | 2026-07-30 |
| EXP-103 | EXP-103@best | MSH | val | none | 0.8802 | 0.8567 | 0.8885 | 0.522 | 0.8864 | 0.244 | 2.412 | configs/EXP-103.yaml | 66dd1e6 | 2.4min | 2026-07-30 |
| EXP-104 | EXP-104@best | MSH | val | full | 0.8852 | 0.8659 | 0.8946 | 0.462 | 0.8920 | 0.273 | 2.467 | configs/EXP-104.yaml | 66dd1e6 | 3.6min | 2026-07-30 |
| EXP-105 | EXP-105@best | MSH | val | full | 0.8814 | 0.8621 | 0.8906 | 0.484 | 0.8881 | 0.248 | 2.589 | configs/EXP-105.yaml | 66dd1e6 | 1.4min | 2026-07-30 |
| EXP-106 | EXP-106@best | MSH | val | full | 0.8856 | 0.8673 | 0.8942 | 0.462 | 0.8919 | 0.256 | 2.504 | configs/EXP-106.yaml | 49ff68e | 5.5min | 2026-07-30 |
| EXP-107 | EXP-107@best | MSH | val | full | 0.8893 | 0.8675 | 0.8977 | 0.473 | 0.8956 | 0.248 | 2.217 | configs/EXP-107.yaml | 49ff68e | 2.6min | 2026-07-30 |
| EXP-108 | EXP-108@best | MSH | val | full | 0.8870 | 0.8676 | 0.8951 | 0.505 | 0.8933 | 0.678 | 2.404 | configs/EXP-108.yaml | 4c81490 | 2.7min | 2026-07-30 |
| EXP-109 | EXP-109@best | MSH | val | full | 0.8884 | 0.8676 | 0.8967 | 0.451 | 0.8945 | 0.252 | 2.299 | configs/EXP-109.yaml | 4c81490 | 2.9min | 2026-07-30 |
| EXP-110 | EXP-110@best | MSH | val | full | 0.8873 | 0.8668 | 0.8959 | 0.478 | 0.8937 | 0.252 | 2.345 | configs/EXP-110.yaml | 3a8c483 | 7.1min | 2026-07-30 |
| EXP-107 | EXP-107@best | MSH | test | full | 0.8888 | 0.8669 | 0.8968 | 0.386 | 0.8941 | 0.254 | 2.228 | configs/EXP-107.yaml | 3a8c483 | 2.6min | 2026-07-30 |
| — | rebuild-ceiling | MSH | test | full | 0.9535 | 0.9465 | 0.9654 | nan | 0.9583 | 0.180 | 1.044 | baseline | 3a8c483 | <1min | 2026-07-30 |
| EXP-111 | EXP-111@best | MSH | val | full | 0.8856 | 0.8635 | 0.8936 | 0.473 | 0.8917 | 0.243 | 2.324 | configs/EXP-111.yaml | 97d376b | 3.9min | 2026-07-30 |
| EXP-112 | EXP-112@best | MSH | val | full | 0.8861 | 0.8637 | 0.8946 | 0.467 | 0.8923 | 0.247 | 2.294 | configs/EXP-112.yaml | 97d376b | 4.1min | 2026-07-30 |
| EXP-113 | EXP-113@best | MSH | val | full | 0.8862 | 0.8642 | 0.8948 | 0.462 | 0.8924 | 0.246 | 2.307 | configs/EXP-113.yaml | 97d376b | 16min | 2026-07-30 |
