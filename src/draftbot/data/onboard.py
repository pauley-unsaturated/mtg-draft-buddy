"""Batch-onboard corpus sets: vocab → parquet → static features → full-season
snapshot → splits → data card. Skips whatever already exists; logs failures and
continues (a set that fails vocab resolution is reported, not fatal).

Usage: uv run python -m draftbot.data.onboard --sets MKM,OTJ,... [--corpus configs/corpus/pretrain_v1.yaml]
"""

import argparse
import sys
import traceback
from pathlib import Path

import yaml

from draftbot.data.cards import PROCESSED_DIR, build_cards
from draftbot.data.convert import convert
from draftbot.data.datacard import build_datacard
from draftbot.data.features import (build_snapshot, build_static_features,
                                    set_release_date)
from draftbot.data.fetch import dest_path
from draftbot.data.splits import build_splits


def onboard(set_code: str, end_date: str = "2026-07-28") -> None:
    if not dest_path(set_code, "PremierDraft").exists():
        raise FileNotFoundError("draft csv not downloaded")
    pq = PROCESSED_DIR / set_code / "draft.PremierDraft.parquet"
    if not pq.exists():
        build_cards(set_code)
        convert(set_code, "PremierDraft")
    if not (PROCESSED_DIR / set_code / "features.static.parquet").exists():
        build_static_features(set_code)
    rel = set_release_date(set_code)
    build_snapshot(set_code, rel, end_date, tag="full")
    build_splits(set_code)
    build_datacard(set_code)
    print(f"=== {set_code} onboarded ===", flush=True)


def _onboard_worker(code: str) -> str | None:
    """Runs in a subprocess. Returns an error string or None."""
    try:
        onboard(code)
        return None
    except Exception as e:
        return f"{e!r}\n{traceback.format_exc()}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default=None)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--parallel", type=int, default=1,
                   help="worker processes for the CPU phase (vocab stays serial "
                        "— Scryfall rate limits)")
    args = p.parse_args(argv)
    if args.corpus:
        manifest = yaml.safe_load(args.corpus.read_text())
        sets = list(manifest["sets"])
        assert not (set(sets) & set(manifest["quarantine"]))
    else:
        sets = [s.strip().upper() for s in args.sets.split(",")]

    failures = {}
    # Phase A (serial): vocab building — the only network-heavy step.
    ready = []
    for code in sets:
        try:
            build_cards(code)
            ready.append(code)
            print(f"vocab ok: {code}", flush=True)
        except Exception as e:
            failures[code] = repr(e)
            traceback.print_exc()

    # Phase B: everything else is per-set CPU work — parallelize.
    if args.parallel > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            for code, err in zip(ready, pool.map(_onboard_worker, ready)):
                if err:
                    failures[code] = err
                    print(f"FAILED {code}: {err}", flush=True)
    else:
        for code in ready:
            err = _onboard_worker(code)
            if err:
                failures[code] = err

    print(f"done: {len(sets) - len(failures)}/{len(sets)} onboarded")
    for code, err in failures.items():
        print(f"FAILED {code}: {err.splitlines()[0]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
