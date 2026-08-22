"""The claims the README makes about itself.

Two of them are load-bearing and neither was enforced. "Reproduce every number
on a laptop in under a minute" is the reason a reviewer would try this at all,
and "the same input gives the same verdict" is a precondition for using any of
it as a release gate -- a gate that decides differently on reruns is not a
gate, it is a coin with a log file.

Both are cheap to check and silent when they break, which is the combination
that gets things written down instead of tested.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from witness.corpus import PriMock57
from witness.crossval import folds
from witness.gate import Gate, Thresholds
from witness.graders.lexical import LexicalGrader
from witness.graders.statistical import fit
from witness.pipeline import Pipeline
from witness.segment import segment

ROOT = Path(__file__).resolve().parents[1] / "primock57"


@pytest.fixture(scope="module")
def corpus() -> PriMock57:
    if not ROOT.exists():
        pytest.skip("PriMock57 not fetched; run `make corpus`")
    return PriMock57(ROOT)


class TestDeterminism:
    def test_the_same_note_grades_identically_twice(self, corpus: PriMock57):
        cons = corpus.load_consultation(corpus.consultation_ids()[0])
        g = LexicalGrader()
        a = g.ground_note("c", "m", segment(cons.clinician_note), cons.utterances)
        b = g.ground_note("c", "m", segment(cons.clinician_note), cons.utterances)
        assert [(x.verdict, x.rationale) for x in a.assertions] == [
            (x.verdict, x.rationale) for x in b.assertions
        ]

    def test_the_gate_reaches_the_same_verdict_twice(self, corpus: PriMock57):
        cons = corpus.load_consultation(corpus.consultation_ids()[0])
        pipe = Pipeline(gate=Gate(Thresholds()))
        a = pipe.run(cons, cons.clinician_note, "m", publish=False)
        b = pipe.run(cons, cons.clinician_note, "m", publish=False)
        assert a.gate.published == b.gate.published
        assert a.gate.ruleset_hash == b.gate.ruleset_hash

    def test_the_ruleset_fingerprint_is_stable_across_processes(self):
        # A hash seeded by PYTHONHASHSEED would change per process and every
        # published artifact would claim a different ruleset than the last.
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from witness.gate import Thresholds; print(Thresholds().fingerprint())"
        )
        env_a = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=ROOT.parent, check=True,
        ).stdout
        env_b = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=ROOT.parent, check=True,
        ).stdout
        assert env_a == env_b and env_a.strip()

    def test_the_learned_model_is_deterministic(self, corpus: PriMock57):
        # An SVD is deterministic, but nothing said so and a switch to a
        # randomised solver would silently make every published figure a
        # sample rather than a measurement.
        cons = [corpus.load_consultation(c) for c in corpus.consultation_ids()[:20]]
        a, b = fit(cons, dims=16), fit(cons, dims=16)
        assert a.vocab == b.vocab
        assert a.nearest("cough", 5) == b.nearest("cough", 5)

    def test_folds_do_not_depend_on_input_order(self, corpus: PriMock57):
        ids = corpus.consultation_ids()
        assert folds(ids, 5) == folds(list(reversed(ids)), 5)


class TestRuntime:
    """The README says about a minute. Nothing enforced it until now."""

    def test_grading_the_whole_corpus_stays_interactive(self, corpus: PriMock57):
        # The budget is deliberately loose -- this asserts the harness has not
        # become a nightly job, not that it hits a particular speed on a
        # particular machine. CI runners vary by more than a factor of two.
        grader = LexicalGrader()
        started = time.monotonic()
        for cid in corpus.consultation_ids():
            c = corpus.load_consultation(cid)
            grader.ground_note(cid, "clinician", segment(c.clinician_note), c.utterances)
        elapsed = time.monotonic() - started
        assert elapsed < 30.0, f"grading 57 consultations took {elapsed:.1f}s"

    def test_loading_the_corpus_is_not_the_bottleneck(self, corpus: PriMock57):
        started = time.monotonic()
        for cid in corpus.consultation_ids():
            corpus.load_consultation(cid)
        assert time.monotonic() - started < 15.0
