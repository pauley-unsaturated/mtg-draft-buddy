"""Terminal HUD (H2): live model rankings alongside an Arena draft.

Live:    uv run python -m draftbot.hud --models checkpoints/EXP-033
Replay:  uv run python -m draftbot.hud --models ... --replay path/to/Player.log
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from draftbot.hud.advisor import Advisor
from draftbot.hud.follower import DEFAULT_LOG, LogFollower, PackSeen
from draftbot.hud.state import DraftState

RARITY_STYLE = {"mythic": "bold orange1", "rare": "gold3",
                "uncommon": "grey70", "common": "white"}
PIPS = "WUBRG"


def render(state: DraftState, ranked: dict, pips: list[int]) -> Table:
    pos = state.position
    pack_no, pick_no = pos // 14 + 1, pos % 14 + 1
    primary_name = next(iter(ranked))
    primary = ranked[primary_name]
    others = {n: r for n, r in ranked.items() if n != primary_name}

    title = f"P{pack_no}P{pick_no} · {state.set_code} · {len(state.picks)} picks"
    if others:
        for name, r in others.items():
            if r[0].card_id != primary[0].card_id:
                title += f"  ⚠ {name.split('@')[0]} prefers {r[0].name}"
    table = Table(title=title, expand=True, header_style="dim")
    table.add_column("#", width=3)
    table.add_column("card")
    table.add_column("conf", width=26)
    table.add_column("GIH", width=5)
    table.add_column("ALSA", width=5)
    for s in primary:
        bar = "▓" * max(1, int(s.prob * 24)) if s.prob >= 0.005 else "·"
        style = RARITY_STYLE.get(s.rarity, "white")
        emphasis = "bold " if s.rank == 1 else ""
        table.add_row(
            str(s.rank), Text(s.name, style=emphasis + style),
            f"{bar} {s.prob * 100:4.1f}%",
            f"{s.gih:.3f}" if s.gih == s.gih and s.gih is not None else "—",
            f"{s.alsa:.1f}" if s.alsa == s.alsa and s.alsa is not None else "—",
        )
    pip_str = "  ".join(f"{c}:{n}" for c, n in zip(PIPS, pips) if n)
    table.caption = f"pool pips  {pip_str or '—'}"
    return table


def pool_pips(state: DraftState, set_code: str) -> list[int]:
    import pandas as pd
    static = pd.read_parquet(f"data/processed/{set_code}/features.static.parquet")
    pip_cols = static[[f"pips_{c}" for c in PIPS]].to_numpy()
    total = [0] * 5
    for cid in state.pool_ids():
        for i in range(5):
            total[i] += int(pip_cols[cid, i])
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(prog="draftbot.hud")
    ap.add_argument("--models", required=True)
    ap.add_argument("--stats", default="full", choices=["none", "week1", "full"])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--replay", type=Path, default=None)
    args = ap.parse_args(argv)

    console = Console()
    state = DraftState()
    advisor = None
    scorers = None
    follower = LogFollower(args.replay or args.log, replay=args.replay is not None)
    console.print(f"[dim]following {follower.path}…[/dim]")

    with Live(console=console, auto_refresh=False) as live:
        for ev in follower.events():
            changed = state.apply(ev)
            if not changed or not isinstance(ev, PackSeen):
                continue
            if scorers is None and state.set_code:
                from draftbot.eval.__main__ import resolve_scorer
                scorers = [resolve_scorer(m.strip(), state.set_code, args.stats)
                           for m in args.models.split(",")]
                advisor = Advisor(scorers, state.set_code)
            if advisor is None:
                continue
            ranked = advisor.rank_pack(state)
            live.update(render(state, ranked, pool_pips(state, state.set_code)),
                        refresh=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
