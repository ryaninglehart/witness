"""Omission and calibration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from witness.corpus import Consultation, PriMock57, Utterance
from witness.omission import Tier, extract_salient, find_omissions

CORPUS = Path(__file__).resolve().parents[1] / "primock57"
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "human_eval_data").is_dir(), reason="run: make corpus"
)


def _consult(pairs, note="", highlights=()):
    utts, t = [], 0.0
    for speaker, text in pairs:
        utts.append(Utterance(speaker, t, t + 4, text))
        t += 5
    return Consultation("test", utts, "", note, list(highlights))


class TestSalience:
    def test_a_doctors_question_is_not_a_fact(self) -> None:
        """The elicitation is not the finding; the answer is.

        Counting questions doubles every fact and makes the doctor's agenda
        look like the patient's history.
        """
        c = _consult(
            [("doctor", "Any blood in the stool at all?"),
             ("patient", "No, no blood, just watery stool.")],
            note="No blood in stool. Watery stool.",
        )
        facts = extract_salient(c)
        assert all(f.speaker == "patient" for f in facts)

    def test_safety_categories_are_critical_without_a_clinician_signal(self) -> None:
        c = _consult(
            [("patient", "I am allergic to penicillin, it gives me a rash.")], note=""
        )
        facts = extract_salient(c)
        assert facts and facts[0].tier == Tier.CRITICAL

    def test_clinician_highlights_are_major_not_critical(self) -> None:
        """Regression: WIT-007.

        Tiering highlights as CRITICAL made all 57 clinician-written notes fail
        the omission rule. A severity scale where most things are top severity
        is not a scale.
        """
        c = _consult(
            [("patient", "I have had diarrhoea for three days now.")],
            note="3/7 hx of diarrhoea", highlights=["diarrhoea"],
        )
        facts = extract_salient(c)
        assert facts and all(f.tier != Tier.CRITICAL for f in facts)

    def test_filler_is_not_a_fact(self) -> None:
        c = _consult([("patient", "Yeah."), ("patient", "Okay, thanks.")], note="x")
        assert extract_salient(c) == []


class TestCoverage:
    def test_a_carried_fact_is_not_reported_as_omitted(self) -> None:
        """Regression: WIT-005.

        Coverage once scored over every token including filler, so no fact was
        ever judged covered and the module reported 24 omissions per note.
        """
        c = _consult(
            [("patient", "Um, I have been allergic to penicillin since I was a child.")],
            note="Allergic to penicillin.",
        )
        r = find_omissions(c, "Allergic to penicillin.")
        assert r.omitted == [], f"should be covered, got {[f.text for f in r.omitted]}"

    def test_a_dropped_allergy_is_reported(self) -> None:
        c = _consult(
            [("patient", "I am allergic to penicillin.")],
            note="Sore throat for three days. Plan: rest and fluids.",
        )
        r = find_omissions(c, "Sore throat for three days. Plan: rest and fluids.")
        assert any(f.tier == Tier.CRITICAL for f in r.omitted)

    def test_an_inverted_negative_is_not_counted_as_covered(self) -> None:
        """The note mentions the finding but with the wrong polarity.

        Term overlap alone would call this covered. It is not: the encounter
        denied it and the note asserts it.
        """
        c = _consult(
            [("patient", "No chest pain at all, none.")], note="No chest pain."
        )
        r = find_omissions(c, "Patient reports chest pain.")
        assert all(f.text != "" for f in r.omitted)


@requires_corpus
class TestCalibration:
    def test_thresholds_are_derived_and_admit_most_human_notes(self) -> None:
        """The bar is set from what clinicians write, not from intuition.

        Regression: WIT-008. The first hand-picked thresholds rejected 57 of 57
        clinician-written notes.
        """
        from witness.calibration import calibrate

        cal = calibrate(PriMock57(CORPUS), percentile=0.90)
        assert 0.0 < cal.human_flag_rate < 0.5, (
            "a gate flagging no human notes measures nothing; one flagging most "
            "of them gets switched off"
        )
        assert cal.thresholds.max_unsupported_rate > 0.15
