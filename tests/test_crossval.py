"""Tests for the held-out evaluation.

The thing under test here is not an accuracy number, it is an absence: that no
consultation is ever scored by a model that trained on it, and that no
threshold is ever applied to a note that helped set it. Both failures are
silent -- they produce a plausible number rather than an error -- so they get
tests rather than a comment saying they were considered.
"""

from __future__ import annotations

import pytest

from witness.corpus import Consultation, Utterance
from witness.crossval import folds, grade_sabotages, operating_points
from witness.graders.lexical import LexicalGrader
from witness.graders.statistical import StatisticalGrader, fit
from witness.grounding import NoteGrounding


def _utt(speaker: str, text: str, start: float) -> Utterance:
    return Utterance(speaker=speaker, text=text, start_s=start, end_s=start + 2.0)


def _consultation(cid: str, note: str, lines: list[tuple[str, str]]) -> Consultation:
    return Consultation(
        consultation_id=cid,
        utterances=[_utt(s, t, float(i * 3)) for i, (s, t) in enumerate(lines)],
        clinician_note=note,
        highlights=[],
        presenting_complaint="",
    )


IDS = [f"day{d}_consultation{n:02d}" for d in (1, 2, 3) for n in range(1, 8)]


class TestFolds:
    def test_every_id_is_tested_exactly_once(self):
        seen: list[str] = []
        for f in folds(IDS, k=5):
            seen.extend(f.test)
        assert sorted(seen) == sorted(IDS)

    def test_train_and_test_never_overlap(self):
        for f in folds(IDS, k=5):
            assert not set(f.train) & set(f.test)
            assert set(f.train) | set(f.test) == set(IDS)

    def test_assignment_is_deterministic(self):
        # A cross-validation whose result moves between runs is measuring the
        # random number generator. Order of the input must not matter either.
        assert folds(IDS, k=5) == folds(list(reversed(IDS)), k=5)

    def test_recording_days_are_spread_across_folds(self):
        # Contiguous blocks would put a whole recording day in one fold and
        # confound the split with whatever varied between days.
        for f in folds(IDS, k=3):
            days = {cid.split("_")[0] for cid in f.test}
            assert len(days) > 1, f"fold {f.index} drew from one day only: {days}"

    @pytest.mark.parametrize("k", [0, 1, len(IDS) + 1])
    def test_impossible_k_is_refused(self, k):
        with pytest.raises(ValueError):
            folds(IDS, k=k)


class TestLeakageGuard:
    """The statistical grader must refuse to score what it trained on."""

    @staticmethod
    def _corpus() -> list[Consultation]:
        return [
            _consultation(
                f"day1_consultation{i:02d}",
                "Cough and cold. Salbutamol inhaler prn.",
                [
                    ("doctor", "What brings you in today?"),
                    ("patient", "I have had a cough and a cold for three days"),
                    ("patient", "I use my blue inhaler, the salbutamol, most days"),
                    ("doctor", "Any wheeze with that?"),
                    ("patient", "A bit of wheeze at night"),
                ],
            )
            for i in range(1, 6)
        ]

    def test_scoring_a_trained_consultation_raises(self):
        cons = self._corpus()
        grader = StatisticalGrader(fit(cons, dims=2, min_count=1))
        with pytest.raises(ValueError, match="in-sample"):
            grader.ground_note(
                cons[0].consultation_id, "m", [], cons[0].utterances
            )

    def test_held_out_consultation_is_scored(self):
        cons = self._corpus()
        held_out = _consultation(
            "day9_consultation99", "Cough.", [("patient", "I have a cough")]
        )
        grader = StatisticalGrader(fit(cons, dims=2, min_count=1))
        g = grader.ground_note("day9_consultation99", "m", [], held_out.utterances)
        assert isinstance(g, NoteGrounding)

    def test_empty_model_degrades_rather_than_inventing_signal(self):
        # Too little vocabulary to reduce into must yield no associations, not
        # a low-rank projection of noise that scores like evidence.
        cons = self._corpus()
        model = fit(cons, dims=500, min_count=1)
        assert model.vectors is None
        assert model.neighbours() == {}

    def test_frequent_terms_are_excluded_from_the_lexicon(self):
        # Every consultation here contains "cough". A term in every document
        # discriminates nothing and must not acquire neighbours, or it becomes
        # a near-synonym of everything.
        model = fit(self._corpus(), dims=2, min_count=1, max_doc_frac=0.5)
        assert "cough" not in model.vocab


class TestOperatingPoints:
    """Thresholds must come from the fold that excludes the note being judged."""

    @staticmethod
    def _grounding(cid: str) -> NoteGrounding:
        return NoteGrounding(cid, "clinician", "test/1.0", [])

    def test_a_test_consultation_never_informs_its_own_threshold(self, monkeypatch):
        import witness.crossval as cv

        ids = [f"c{i:02d}" for i in range(10)]
        fold_list = folds(ids, k=5)
        groundings = {cid: self._grounding(cid) for cid in ids}
        omissions = dict.fromkeys(ids, 0)

        seen: list[set[str]] = []
        real = cv._thresholds_from

        def spy(gs, oms, p):
            gs = list(gs)
            seen.append({g.consultation_id for g in gs})
            return real(gs, oms, p)

        monkeypatch.setattr(cv, "_thresholds_from", spy)
        cv.operating_points(fold_list, groundings, omissions, {}, "test", (0.90,))

        assert len(seen) == len(fold_list)
        for f, calibrated_on in zip(fold_list, seen, strict=True):
            assert not calibrated_on & set(f.test)

    def test_pass_rate_counts_every_consultation_once(self):
        ids = [f"c{i:02d}" for i in range(10)]
        points = operating_points(
            folds(ids, k=5),
            {cid: self._grounding(cid) for cid in ids},
            dict.fromkeys(ids, 0),
            {},
            "test",
            (0.90,),
        )
        assert points[0].notes == len(ids)


class TestSabotageBaselines:
    def test_baseline_is_rebuilt_per_grader(self):
        # Injecting into the other grader's baseline measures the mismatch
        # between graders rather than either one's sensitivity.
        cons = _consultation(
            "day9_consultation99",
            "Cough for three days. Blue inhaler prn. Loss of appetite.",
            [
                ("patient", "I have had a cough for three days"),
                ("patient", "I use the blue inhaler when I need it"),
                ("patient", "I have had a loss of appetite too"),
            ],
        )
        lex = grade_sabotages(cons, LexicalGrader())
        assert lex, "no sabotage cases were graded"
        assert any(g.should_block for g in lex)
        assert any(g.threshold_independent for g in lex)
