"""Laptop-bundle builder smoke test (skips without local data/checkpoints)."""

import tarfile
from pathlib import Path

import pytest

needs_artifacts = pytest.mark.skipif(
    not (Path("checkpoints/EXP-126/best.pt").exists()
         and Path("data/processed/MSH/cards.parquet").exists()),
    reason="production checkpoints/data not on this machine")


@needs_artifacts
def test_bundle_builds_and_contains_essentials(tmp_path):
    from draftbot.bundle import build

    out = build(["MSH"], ["EXP-126"], out=tmp_path / "b.tar.gz")
    names = set(tarfile.open(out).getnames())
    assert "checkpoints/EXP-126/best.pt" in names
    assert "checkpoints/EXP-126/scaler.json" in names
    assert "data/processed/MSH/cards.parquet" in names
    assert "data/processed/MSH/features.static.parquet" in names
    assert not any(n.endswith("last.pt") for n in names)
    assert not any("draft.PremierDraft.parquet" in n for n in names)
