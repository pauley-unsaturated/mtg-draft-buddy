# Data card: MSH

## PremierDraft

- picks: 2,881,578 | drafts: 68,609 | t=42
- dates: 2026-06-23 → 2026-07-24
- basics picked: 99,160 (3.4%) — basics ARE draftable
- pick_2 present on 5 rows
- ranks: ['bronze', 'diamond', 'gold', 'mythic', 'platinum', 'silver', 'unknown']
- user_n_games buckets: [1, 5, 10, 50, 100, 500, 1000]

## TradDraft

- picks: 285,054 | drafts: 6,787 | t=42
- dates: 2026-06-23 → 2026-07-24
- basics picked: 9,295 (3.3%) — basics ARE draftable
- pick_2 present on 1 rows
- ranks: ['unknown']
- user_n_games buckets: [1, 5, 10, 50, 100, 500, 1000]

## Decks (game_data, PremierDraft)

- builds: 83,457 | drafts: 67,808 | games: 377,514 (4.5 per build)
- rebuilds: 81.0% of drafts have 1 build; max 7
- deck size: 92.6% exactly 40 (range 40-60)
- basics per deck: mean 14.8 σ 1.6 (range 4-23)
- wins per build: 18.4% zero-win, 20.1% ≥5 wins, 6.1% ≥7 wins (trophy pool)
- split coverage: 100.0% of builds join `data/splits/MSH.json`

## Cards

- vocab: 339 cards (5 basics, 5 flip)
- printing sets: {'msh': 281, 'mar': 58}
- companion sheets (registry): ['MAR']
- rarities: {'uncommon': 100, 'common': 96, 'mythic': 83, 'rare': 60}

## Snapshots

- full: `2026-06-26_2026-07-28.parquet`
- week1: `2026-06-26_2026-07-03.parquet`
