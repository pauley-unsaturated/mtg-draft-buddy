"""Stop-the-world guardrail (PLAN §0): the sandbagged set must never enter any
pretraining corpus manifest, warm-start config, or scaler-fit source."""

from pathlib import Path

import yaml

CORPUS_DIR = Path("configs/corpus")
CONFIG_DIR = Path("configs")

QUARANTINED = {"MSH"}


def test_quarantined_sets_absent_from_all_corpus_manifests():
    manifests = list(CORPUS_DIR.glob("*.yaml"))
    assert manifests, "no corpus manifests found"
    for path in manifests:
        m = yaml.safe_load(path.read_text())
        assert QUARANTINED <= set(m.get("quarantine", [])), (
            f"{path}: quarantine list must contain {QUARANTINED}")
        overlap = QUARANTINED & set(m.get("sets", {}))
        assert not overlap, f"{path}: quarantined sets {overlap} in corpus!"


def test_no_pretrain_config_trains_on_quarantined_set():
    """Any config whose model pretrains multi-set must not list MSH as a source.
    Single-set configs targeting MSH (set: MSH) are fine — that's fine-tuning/
    baseline territory, not the pretrained trunk."""
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict):
            continue
        corpus = cfg.get("corpus")
        if corpus is None:
            continue  # single-set config
        manifest = yaml.safe_load((Path(corpus)).read_text()) if isinstance(corpus, str) \
            else corpus
        sets = manifest.get("sets", manifest if isinstance(manifest, dict) else {})
        overlap = QUARANTINED & set(sets)
        assert not overlap, f"{path}: pretrains on quarantined {overlap}"
