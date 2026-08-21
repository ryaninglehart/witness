"""Gate tests: the rules that decide whether a note may be published.

The most important test in this file is
``test_weakening_a_threshold_changes_the_fingerprint``. Without it the gate can
be quietly relaxed between runs while every artifact still claims to have been
produced under the same ruleset, and no reader can tell the difference.
"""

from __future__ import annotations

import dataclasses

import pytest

from witness.gate import Acknowledgement, Gate, Thresholds
from witness.grounding import (
    Assertion,
    GroundedAssertion,
    NoteGrounding,
    Slot,
    Span,
    Verdict,
)


def _span(text: str = "said it") -> Span:
    return Span(utterance_index=0, speaker="patient", start_s=1.0, end_s=2.0, text=text)


def _note(*graded: GroundedAssertion) -> NoteGrounding:
    return NoteGrounding("c1", "m1", "test", list(graded))


def _ok(i: int, slot: Slot = Slot.SYMPTOM) -> GroundedAssertion:
    return GroundedAssertion(
        Assertion(i, f"finding {i}", slot), Verdict.SUPPORTED, [_span()], grader="test"
    )


def _bad(i: int, verdict: Verdict, slot: Slot = Slot.SYMPTOM) -> GroundedAssertion:
    spans = [_span()] if verdict is Verdict.CONTRADICTED else []
    return GroundedAssertion(
        Assertion(i, f"finding {i}", slot), verdict, spans, grader="test"
    )


class TestProvenanceInvariant:
    def test_supported_without_a_span_is_rejected(self) -> None:
        """A citation you cannot follow is not provenance.

        Making this unrepresentable matters more than catching it later: the
        whole product claim is that every supported statement names the moment
        it came from.
        """
        with pytest.raises(ValueError, match="requires at least one source span"):
            GroundedAssertion(Assertion(0, "x"), Verdict.SUPPORTED, [])

    def test_inferred_without_a_span_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires at least one source span"):
            GroundedAssertion(Assertion(0, "ECOG 1"), Verdict.INFERRED, [])

    def test_unsupported_needs_no_span(self) -> None:
        GroundedAssertion(Assertion(0, "x"), Verdict.UNSUPPORTED, [])


class TestBlockingRules:
    def test_a_single_contradiction_blocks(self) -> None:
        r = Gate().evaluate(_note(_ok(0), _ok(1), _bad(2, Verdict.CONTRADICTED)))
        assert not r.published
        assert any(f.rule == "contradicted_claims" for f in r.blocking)

    def test_ungrounded_allergy_blocks(self) -> None:
        r = Gate().evaluate(
            _note(_ok(0), _ok(1), _bad(2, Verdict.UNSUPPORTED, Slot.ALLERGY))
        )
        assert not r.published
        assert any(f.rule == "unsupported_safety_claims" for f in r.blocking)

    def test_one_fabricated_allergy_blocks_in_an_otherwise_noisy_note(self) -> None:
        """Why per-slot ceilings exist.

        Calibrated against real clinician notes the aggregate safety ceiling
        lands around five, and at five a single fabricated allergy publishes.
        Allergy claims are rare, so their own bar can be zero while the
        aggregate stays loose enough to accept human writing. See WIT-008.
        """
        loose = Thresholds(
            max_safety_unsupported=5, max_unsupported_rate=0.9,
            slot_ceilings="ALLERGY:0",
        )
        note = _note(*[_ok(i) for i in range(19)],
                     _bad(19, Verdict.UNSUPPORTED, Slot.ALLERGY))
        r = Gate(loose).evaluate(note)
        assert not r.published
        assert any(f.rule == "slot_ceiling_exceeded" for f in r.blocking)

    def test_slot_ceilings_are_inside_the_fingerprint(self) -> None:
        a = Thresholds(slot_ceilings="ALLERGY:0")
        b = Thresholds(slot_ceilings="ALLERGY:9")
        assert a.fingerprint() != b.fingerprint()

    def test_ungrounded_social_detail_does_not_block_on_the_safety_rule(self) -> None:
        """Asymmetry is the point.

        An ungrounded claim about someone's occupation is a quality problem. An
        ungrounded claim about their allergies is a safety problem. Collapsing
        them into one hallucination rate is how a harness stays green while
        shipping something harmful.
        """
        r = Gate().evaluate(
            _note(*[_ok(i) for i in range(9)], _bad(9, Verdict.UNSUPPORTED, Slot.SOCIAL))
        )
        assert not any(f.rule == "unsupported_safety_claims" for f in r.findings)

    def test_a_near_empty_note_blocks_rather_than_passing_trivially(self) -> None:
        """An empty note has a 0% unsupported rate and is not therefore safe."""
        r = Gate().evaluate(_note(_ok(0)))
        assert not r.published
        assert any(f.rule == "note_too_thin" for f in r.blocking)

    def test_a_clean_note_publishes(self) -> None:
        r = Gate().evaluate(_note(*[_ok(i) for i in range(6)]))
        assert r.published
        assert not r.blocking


class TestAcknowledgement:
    def test_acknowledged_block_publishes_and_is_recorded(self) -> None:
        g = Gate()
        # Padded so the unsupported-rate rule does not also fire: this test is
        # about acknowledgement, and a note tripping two rules would not isolate it.
        note = _note(*[_ok(i) for i in range(19)], _bad(19, Verdict.CONTRADICTED))
        r = g.evaluate(
            note,
            [
                Acknowledgement(
                    "contradicted_claims",
                    "Patient corrected themselves at 04:12; the note records the correction.",
                    "r.inglehart",
                )
            ],
        )
        assert r.published
        assert r.blocking, "the finding must survive the override, not be erased by it"
        assert r.acknowledgements[0].by == "r.inglehart"

    def test_override_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="written reason"):
            Acknowledgement("contradicted_claims", "   ", "someone")

    def test_override_without_an_approver_is_refused(self) -> None:
        with pytest.raises(ValueError, match="identified approver"):
            Acknowledgement("contradicted_claims", "looks fine", "")

    def test_overriding_one_rule_does_not_override_another(self) -> None:
        note = _note(
            *[_ok(i) for i in range(18)],
            _bad(18, Verdict.CONTRADICTED),
            _bad(19, Verdict.UNSUPPORTED, Slot.DOSE),
        )
        r = Gate().evaluate(
            note, [Acknowledgement("contradicted_claims", "reviewed", "r.inglehart")]
        )
        assert not r.published
        # The dose failure legitimately trips two rules: the per-slot ceiling
        # for DOSE and the aggregate safety count. Both must survive an
        # acknowledgement aimed at a third rule.
        assert {f.rule for f in r.unacknowledged_blocking} == {
            "unsupported_safety_claims",
            "slot_ceiling_exceeded",
        }


class TestRulesetFingerprint:
    def test_weakening_a_threshold_changes_the_fingerprint(self) -> None:
        """The rule that makes every other rule auditable.

        A gate whose fingerprint does not cover its thresholds can be relaxed
        between runs while both builds claim the same ruleset. Two artifacts
        look identical; one was checked against half the bar.
        """
        strict = Gate(Thresholds())
        loose = Gate(Thresholds(max_unsupported_rate=0.90))
        assert strict.ruleset_hash != loose.ruleset_hash

    def test_every_threshold_field_is_covered(self) -> None:
        base = Thresholds()
        for field_name, value in (
            ("max_contradicted", 5),
            ("max_safety_unsupported", 5),
            ("max_unsupported_rate", 0.99),
            ("warn_inferred_rate", 0.99),
            ("min_assertions", 99),
        ):
            changed = Thresholds(**{**base.__dict__, field_name: value})
            assert changed.fingerprint() != base.fingerprint(), (
                f"{field_name} is not inside the ruleset fingerprint; it could be "
                "changed without any artifact recording that the rules moved"
            )

    def test_fingerprint_is_stable_across_identical_thresholds(self) -> None:
        assert Thresholds().fingerprint() == Thresholds().fingerprint()

    def test_thresholds_cannot_be_mutated_after_hashing(self) -> None:
        """Frozen, so a build cannot be stamped with a ruleset it was not checked against."""
        t = Thresholds()
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.max_contradicted = 99  # type: ignore[misc]
