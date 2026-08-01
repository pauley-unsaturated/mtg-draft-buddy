"""Pool-paste build CLI (draftbot.build): parsing is pure-unit; the end-to-end
run needs local artifacts (TLA cards + EXP-130) and skips without them."""

from pathlib import Path

import pytest

from draftbot.build import parse_pool

CARDS = Path("data/processed/TLA/cards.parquet")
CKPT = Path("checkpoints/EXP-130/best.pt")


def test_parse_pool_formats():
    text = """Deck
2 Sokka, Master Strategist (TLA) 123
1 Appa, Steadfast Guardian
Cabbage Merchant
4 Island (TLA) 291

Sideboard
1 A-Rebalanced Thing (TLA) 12b
"""
    assert parse_pool(text) == [
        (2, "Sokka, Master Strategist"),
        (1, "Appa, Steadfast Guardian"),
        (1, "Cabbage Merchant"),
        (4, "Island"),
        (1, "A-Rebalanced Thing"),
    ]


@pytest.mark.skipif(not (CARDS.exists() and CKPT.exists()),
                    reason="TLA artifacts not built")
def test_end_to_end_legal_build(capsys):
    import pandas as pd

    from draftbot.build import main

    decks = pd.read_parquet("data/processed/TLA/decks.sealed.parquet")
    cards = pd.read_parquet(CARDS)
    names = dict(zip(cards["id"], cards["name"]))
    row = decks.iloc[0]
    pool: dict[int, int] = {}
    for ids, cnts in ((row["deck_ids"], row["deck_counts"]),
                      (row["side_ids"], row["side_counts"])):
        for i, c in zip(ids, cnts):
            pool[int(i)] = pool.get(int(i), 0) + int(c)
    text = "\n".join(f"{c} {names[i]}" for i, c in pool.items())
    text += "\n3 Island\n"  # basics in the export must be ignored

    pool_file = Path("data/processed/TLA") / "_cli_smoke_pool.txt"
    pool_file.write_text(text)
    try:
        main(["--set", "TLA", "--pool", str(pool_file)])
    finally:
        pool_file.unlink()
    out = capsys.readouterr().out
    assert "PROPOSED 40 — 40 cards" in out
    assert "3 basics ignored" in out
