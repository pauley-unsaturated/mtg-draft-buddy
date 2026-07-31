# One-command per-set data pipeline (Phase-0 gate). Usage: make data SET=MSH
SET ?= MSH

data:
	uv run python -m draftbot.data.fetch --set $(SET) --event PremierDraft
	uv run python -c "from draftbot.data.fetch import download; download('$(SET)', 'PremierDraft', kind='game')"
	uv run python -m draftbot.data.convert --set $(SET) --event PremierDraft
	uv run python -c "\
from datetime import datetime, timedelta; \
from draftbot.data.features import build_snapshot, set_release_date, build_static_features; \
rel = set_release_date('$(SET)'); \
w1 = (datetime.strptime(rel, '%Y-%m-%d') + timedelta(days=7)).date(); \
build_snapshot('$(SET)', rel, '2026-07-28', tag='full'); \
build_snapshot('$(SET)', rel, str(w1), tag='week1'); \
build_static_features('$(SET)')"
	uv run python -c "from draftbot.data.splits import build_splits; build_splits('$(SET)')"
	uv run python -m draftbot.data.datacard --set $(SET)

data-msh:
	$(MAKE) data SET=MSH

test:
	uv run pytest

.PHONY: data data-msh test
