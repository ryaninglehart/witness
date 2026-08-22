"""Tests for the sabotage harness.

WIT-013 is the reason this file exists. Two defects hid each other for the life
of the project and both produced published numbers: a false-positive control
whose injection matched nothing on 56 of 57 consultations, and a "clean"
baseline that failed the gate on its own for 31 of them. Neither raised an
error. Both produced a plausible figure.

So the tests here are mostly about the harness's own validity rather than about
detection rates: that the manipulation happens, that a case that tested nothing
says so, and that credit is subtracted for what the baseline was already doing.
"""

from __future__ import annotations

from pathlib import Path

from witness.corpus import Consultation, Utterance
from witness.prove import (
    SABOTAGES,
    TAXONOMY,
    clean_baseline,
    run_proof,
    taxonomy_coverage,
)

NOTE = (
    "3/7 hx of diarrhoea, mainly watery. No blood in stool. Fever on first day, "
    "nil since. PMH: Asthma. DH: Inhalers. Imp: gastroenteritis."
)


def _corpus_or_skip():
    import pytest

    from witness.corpus import PriMock57

    root = Path(__file__).resolve().parents[1] / "primock57"
    if not root.exists():
        pytest.skip("PriMock57 not fetched; run `make corpus`")
    return PriMock57(root)


def _consultation() -> Consultation:
    lines = [
        ("doctor", "What brings you in today?"),
        ("patient", "I have had watery diarrhoea for about three days"),
        ("doctor", "Any blood in it?"),
        ("patient", "No, no blood at all"),
        ("patient", "I had a fever the first day but nothing since"),
        ("patient", "I have asthma and I use an inhaler"),
        ("doctor", "Sounds like gastroenteritis"),
    ]
    return Consultation(
        consultation_id="day1_consultation01",
        utterances=[
            Utterance(speaker=s, text=t, start_s=float(i * 3), end_s=float(i * 3 + 2))
            for i, (s, t) in enumerate(lines)
        ],
        presenting_complaint="",
        clinician_note=NOTE,
        highlights=[],
    )


class TestTheManipulationActuallyHappens:
    """The check that would have caught WIT-013 on the day it was written."""

    def test_no_injection_is_vacuous_across_the_corpus(self):
        """The invariant that would have caught WIT-013 the day it was written.

        Some injections are content-dependent by nature -- flipping a negation
        needs a negation to flip -- so being vacuous on a particular note is
        expected and is now reported. Being vacuous on nearly *every* note is
        the WIT-013 signature: the old paraphrase control fired on 1 of 57.
        """
        corpus = _corpus_or_skip()
        ids = corpus.consultation_ids()
        pairs = [
            (c := corpus.load_consultation(cid), clean_baseline(c)) for cid in ids
        ]
        weak = {}
        for s in SABOTAGES:
            if s.key == "weakened_threshold":
                continue
            live = sum(1 for cons, b in pairs if s.inject(b, cons) != b)
            if live < 0.2 * len(ids):
                weak[s.key] = f"{live}/{len(ids)}"
        assert not weak, f"injections that match almost nothing: {weak}"

    def test_the_control_is_live_on_almost_every_consultation(self):
        # A control is the one case that must work everywhere, because it is
        # the only thing standing between a noisy harness and a clean report.
        corpus = _corpus_or_skip()
        ids = corpus.consultation_ids()
        control = next(s for s in SABOTAGES if not s.should_block)
        live = 0
        for cid in ids:
            cons = corpus.load_consultation(cid)
            b = clean_baseline(cons)
            live += int(control.inject(b, cons) != b)
        assert live >= 0.8 * len(ids), f"control only rewords {live}/{len(ids)} notes"

    def test_the_paraphrase_control_rewords_without_changing_a_claim(self):
        control = next(s for s in SABOTAGES if not s.should_block)
        out = control.inject(NOTE, _consultation())
        assert out != NOTE
        # Content preserved: the negation survives as a negation, and the
        # clinical terms are all still present.
        for term in ("diarrhoea", "blood", "stool", "asthma", "gastroenteritis"):
            assert term in out.lower()
        assert "denies" in out.lower()

    def test_a_vacuous_injection_is_reported_not_passed(self):
        cons = _consultation()
        results = run_proof(cons, "Nothing here resembles the patterns.")
        vacuous = [r for r in results if r.vacuous]
        assert vacuous, "expected some injections to match nothing on this note"
        for r in vacuous:
            assert not r.as_expected, "a case that tested nothing must not read as a pass"
            assert "tested nothing" in r.evidence


class TestDeltaScoring:
    def test_results_record_what_the_baseline_already_failed(self):
        cons = _consultation()
        results = run_proof(cons, clean_baseline(cons))
        assert all(isinstance(r.baseline_findings, list) for r in results)

    def test_a_rule_the_baseline_already_fires_gives_no_credit(self):
        # The WIT-013 failure mode: a baseline that fails on its own hands
        # every case a catch it never made.
        cons = _consultation()
        noisy = clean_baseline(cons) + " Allergic to penicillin. Prescribed amoxicillin 500mg."
        results = run_proof(cons, noisy)
        for r in results:
            if r.blocked and r.sabotage.expect_rule:
                assert r.sabotage.expect_rule not in r.baseline_findings


class TestTaxonomy:
    def test_every_case_is_tagged(self):
        untagged = [s.key for s in SABOTAGES if not s.taxonomy]
        assert not untagged, f"untagged sabotage cases: {untagged}"

    def test_every_mapped_category_points_at_a_real_case(self):
        keys = {s.key for s in SABOTAGES}
        missing = {c: k for c, k in TAXONOMY.items() if k is not None and k not in keys}
        assert not missing, f"taxonomy references cases that do not exist: {missing}"

    def test_out_of_scope_categories_stay_in_the_denominator(self):
        # Trimming them out would flatter the coverage figure.
        cons = _consultation()
        text = taxonomy_coverage(run_proof(cons, clean_baseline(cons)))
        assert f"of {len(TAXONOMY)} categories" in text
        assert "out of scope" in text

    def test_the_control_is_not_counted_as_a_detected_defect(self):
        control = next(s for s in SABOTAGES if not s.should_block)
        assert TAXONOMY["13. Good behaviour"] == control.key
