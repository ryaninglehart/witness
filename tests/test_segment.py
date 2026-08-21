"""Segmentation and negation tests.

These carry more weight than their size suggests. Every verdict downstream is
computed on the fragments this module produces, so a segmentation bug does not
announce itself -- it silently caps the accuracy of the whole harness and looks
like a grader problem.
"""

from __future__ import annotations

import pytest

from witness.grounding import Slot
from witness.segment import classify_slot, is_negated, segment


class TestNegation:
    @pytest.mark.parametrize(
        "text",
        [
            "no blood in stool",
            "nil radiation",
            "denies chest pain",
            "patient has no fever",
            "negative for weight loss",
            "without nausea",
            "NKDA",
            "doesn't smoke",
        ],
    )
    def test_detects_clinical_negation(self, text: str) -> None:
        assert is_negated(text) or classify_slot(text) is Slot.ALLERGY

    @pytest.mark.parametrize(
        "text",
        [
            "no longer smoking",
            "not only diarrhoea but vomiting",
        ],
    )
    def test_pseudo_negation_does_not_flip_polarity(self, text: str) -> None:
        """"No longer smoking" asserts a change, not an absence.

        Treating it as a negative finding would make the grader contradict an
        encounter in which the patient described quitting -- a false alarm on a
        true claim, which is how a gate gets switched off.
        """
        assert not is_negated(text)

    def test_affirmative_is_not_negated(self) -> None:
        assert not is_negated("crampy intermittent lower abdominal pain")


class TestSegmentation:
    def test_splits_on_clinical_delimiters_not_just_sentences(self) -> None:
        """The motivating case from the module docstring.

        One "sentence" carrying a positive finding and a negative one. A
        sentence splitter returns a single fragment and the negation is
        invisible to the grader.
        """
        got = segment("Associated LLQ pain - crampy, intermittent, nil radiation.")
        assert len(got) > 1
        assert any(a.negated for a in got), "the negative finding must survive segmentation"

    def test_bare_headers_are_not_claims(self) -> None:
        assert segment("PMH:") == []
        assert segment("Plan:") == []

    def test_non_clinical_fragments_are_dropped(self) -> None:
        """Regression: WIT-002.

        These were once graded UNSUPPORTED, which put at least one failure in
        every note and drove the note-level flag rate to 100% -- strict-looking
        and useless. They are not clinical claims and must not reach the grader.
        """
        assert segment("Hello. Thanks very much. Okay then.") == []

    def test_real_note_segments_into_multiple_claims(self) -> None:
        note = (
            "3/7 hx of diarrhea, mainly watery. No blood in stool. "
            "PMH: Asthma. DH: Inhalers."
        )
        got = segment(note)
        assert len(got) >= 3
        assert any(a.negated for a in got)


class TestSlots:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("NKDA", Slot.ALLERGY),
            ("penicillin allergy", Slot.ALLERGY),
            ("salbutamol 100mcg bd", Slot.DOSE),
            ("takes inhalers", Slot.MEDICATION),
            ("Imp: gastroenteritis", Slot.DIAGNOSIS),
            ("Plan: review in 3-5d", Slot.PLAN),
            ("works as an accountant", Slot.SOCIAL),
        ],
    )
    def test_slot_classification(self, text: str, expected: Slot) -> None:
        assert classify_slot(text) is expected

    def test_allergy_outranks_medication(self) -> None:
        """A drug allergy names a drug. The safety reading must win.

        If this returned MEDICATION, an ungrounded allergy claim would miss the
        allergy-specific gate rule and a fabricated allergy could publish.
        """
        assert classify_slot("allergic to penicillin") is Slot.ALLERGY

    def test_dose_outranks_medication(self) -> None:
        assert classify_slot("paracetamol 500mg qds") is Slot.DOSE

    def test_stool_frequency_is_not_a_dose(self) -> None:
        """Regression: bare numbers must not read as prescriptions.

        "6 times a day" is a symptom. Classifying it as DOSE would put it in a
        safety slot and let a symptom description block a release.
        """
        assert classify_slot("opening bowels 6 times a day") is not Slot.DOSE
