PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: help corpus check test lint evidence validate clean

help:
	@echo "  make corpus     fetch PriMock57 (CC BY 4.0, text only -- no audio needed)"
	@echo "  make check      lint + tests + evidence + validation  (the full gate)"
	@echo "  make test       unit tests"
	@echo "  make lint       ruff"
	@echo "  make evidence   do existing metrics predict clinical safety?"
	@echo "  make validate   does Witness flag what clinicians flagged?"

corpus:
	@test -d primock57 || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
		https://github.com/babylonhealth/primock57.git
	@echo "  PriMock57 ready (audio is git-lfs and optional: git lfs pull)"

lint:
	@.venv/bin/ruff check src tests && echo "  ruff clean"

test:
	@.venv/bin/pytest -q

evidence:
	@$(PY) -m witness.cli evidence

validate:
	@$(PY) -m witness.cli validate

check: lint test evidence validate
	@echo "  all checks passed"

clean:
	@rm -rf data/out .pytest_cache .ruff_cache
