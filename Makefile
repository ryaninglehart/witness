PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: help corpus check test lint evidence validate calibrate sweep prove crossval semantic review clean

help:
	@echo "  make corpus     fetch PriMock57 (CC BY 4.0, text only -- no audio needed)"
	@echo "  make check      lint + tests + evidence + validation  (the full gate)"
	@echo "  make test       unit tests"
	@echo "  make lint       ruff"
	@echo "  make evidence   do existing metrics predict clinical safety?"
	@echo "  make validate   does Witness flag what clinicians flagged?"
	@echo "  make calibrate  derive thresholds from clinician-written notes"
	@echo "  make sweep      trace the sensitivity/specificity trade"
	@echo "  make prove      break it on purpose; see what is caught"
	@echo "  make crossval   held-out evaluation; nothing scored in-sample"
	@echo "  make semantic   measure the LLM grader (needs Ollama; not in make check)"
	@echo "  make review     export a one-file review page you can click through"

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

calibrate:
	@$(PY) -m witness.cli calibrate

sweep:
	@$(PY) -m witness.cli sweep

prove:
	@$(PY) -m witness.cli prove

crossval:
	@$(PY) -m witness.cli crossval

# Not part of `make check`: it needs a model server and takes tens of
# minutes on a local 7B. Cassettes make the second run free.
semantic:
	@$(PY) -m witness.cli semantic $(ARGS)

review:
	@$(PY) -m witness.cli review

check: lint test evidence validate calibrate prove crossval
	@echo "  all checks passed"

clean:
	@rm -rf data/out .pytest_cache .ruff_cache
