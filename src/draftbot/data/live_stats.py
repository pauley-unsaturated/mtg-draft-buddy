"""Provisional stats for a day-0 set from 17lands' live card-ratings endpoint.

    uv run python -m draftbot.data.live_stats --set TRK

Between release and the public dataset (~3 weeks) the only statistics that
exist are the ones behind 17lands' card-ratings page. The anonymous endpoint
is limited (verified 2026-11-21): it ignores date and deck-colour filters,
serves one recent window, and withholds a value until its sample is large
enough (ALSA / ATA need 200 picks-or-sightings, every win rate 500 games). So
this is a PARTIAL snapshot: overall columns only, rares and mythics mostly
without win rates. Missing values stay NaN -> masked, as everywhere else.

Simulated on HOB / FRA with the trunk that never saw them (EXP-063; week-1
snapshot cut to a 10% sample with the same thresholds): expert top-1
.582 -> .646 (HOB) and .544 -> .620 (FRA), about 60% of the way from no stats
to real week-1 stats (journal 2026-11-21).

Only for a set still carrying the day-0 marker: the snapshot is registered as
`week1`, `full` and `live`, replacing the all-NaN placeholder, and is thrown
away with the other provisional artifacts when the public dataset is
onboarded. Inference only — the quarantined set's stats never reach a
pretraining corpus or scaler (those read manifests, not this).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from draftbot.data.cards import DAY0_MARKER, HEADERS, PROCESSED_DIR, norm_name
from draftbot.data.features import SNAPSHOT_DIR, _registry_path, set_release_date

URL = "https://www.17lands.com/card_ratings/data"
RAW_DIR = Path("data/raw/17lands_api")
LIVE_SUFFIX = ".live.parquet"

# snapshot column <- endpoint field (overall, no deck-colour filter)
FIELDS = {
    "seen_count": "seen_count", "avg_seen": "avg_seen",
    "pick_count": "pick_count", "avg_pick": "avg_pick",
    "game_count": "game_count", "game_win_rate": "win_rate",
    "opening_hand_count": "opening_hand_game_count",
    "opening_hand_win_rate": "opening_hand_win_rate",
    "drawn_count": "drawn_game_count", "drawn_win_rate": "drawn_win_rate",
    "ever_drawn_count": "ever_drawn_game_count",
    "ever_drawn_win_rate": "ever_drawn_win_rate",
    "never_drawn_count": "never_drawn_game_count",
    "never_drawn_win_rate": "never_drawn_win_rate",
    "drawn_improvement_win_rate": "drawn_improvement_win_rate",
}


def fetch(set_code: str, event: str = "PremierDraft") -> list[dict]:
    """One request; the raw response is kept (timestamped, never rewritten)."""
    resp = requests.get(URL, params={"expansion": set_code, "format": event},
                        headers=HEADERS, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RAW_DIR / set_code / f"{event}.{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    return rows


def to_snapshot(rows: list[dict], card_names: list[str], columns: list[str]) -> pd.DataFrame:
    """Endpoint rows -> a snapshot frame in the standard column layout
    (deck-colour columns all NaN), indexed by normalised card name."""
    by = {norm_name(r["name"]): r for r in rows}
    snap = pd.DataFrame(np.nan, index=[norm_name(n) for n in card_names], columns=columns)
    for col, field in FIELDS.items():
        snap[col] = [(by.get(k) or {}).get(field) for k in snap.index]
    return snap.astype(float)


def build_live_snapshot(set_code: str, rows: list[dict] | None = None) -> pd.DataFrame:
    from draftbot.data.day0 import _template_columns
    code = set_code.upper()
    proc = PROCESSED_DIR / code
    if not (proc / DAY0_MARKER).exists():
        raise SystemExit(f"{code} is not a day-0 set: its snapshots come from the public "
                         "dataset (onboard_set) and are not replaced by live ratings")
    cards = pd.read_parquet(proc / "cards.parquet")
    snap = to_snapshot(rows if rows is not None else fetch(code),
                       list(cards["name"]), _template_columns(code))
    n = int(snap["avg_seen"].notna().sum())
    if n < 20:
        raise SystemExit(f"{code}: only {n} cards have pick-order stats — too early, nothing written")
    rel = set_release_date(code)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    out_dir = SNAPSHOT_DIR / code
    path = out_dir / f"{rel}_{stamp}{LIVE_SUFFIX}"
    snap.to_parquet(path)
    reg_path = _registry_path(code)
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    placeholder = out_dir / f"{rel}_{rel}.parquet"  # the all-NaN day-0 snapshot stays reachable
    if placeholder.exists():
        reg.setdefault("day0", placeholder.name)
    reg.update({"week1": path.name, "full": path.name, "live": path.name})
    reg_path.write_text(json.dumps(reg, indent=2))
    real = cards["is_basic"].to_numpy() == 0
    have = lambda c: int(snap[c].notna().to_numpy()[real].sum())
    print(f"{code}: live stats for {int(real.sum())} cards — ALSA {have('avg_seen')}, "
          f"ATA {have('avg_pick')}, GP WR {have('game_win_rate')}, "
          f"GIH WR {have('ever_drawn_win_rate')} -> {path}")
    return snap


def main(argv=None):
    p = argparse.ArgumentParser(prog="draftbot.data.live_stats")
    p.add_argument("--set", dest="set_code", required=True)
    build_live_snapshot(p.parse_args(argv).set_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
