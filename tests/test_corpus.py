"""Corpus tests, including the join that silently lost 19% of the data."""

from __future__ import annotations

from pathlib import Path

import pytest

from witness.corpus import CRITICAL, NONCRITICAL, PriMock57, canonical_id, parse_findings

CORPUS = Path(__file__).resolve().parents[1] / "primock57"
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "human_eval_data").is_dir(), reason="run: make corpus"
)


class TestFindingParsing:
    def test_severity_prefixes(self) -> None:
        got = parse_findings("! normal food colour.\n- minor wording\n")
        assert [f.severity for f in got] == [CRITICAL, NONCRITICAL]
        assert got[0].text == "normal food colour."

    def test_bare_marker_carries_no_information_and_is_dropped(self) -> None:
        assert parse_findings("!\n-\n") == []

    def test_empty_cell(self) -> None:
        assert parse_findings(None) == [] and parse_findings("") == []


class TestConsultationIdPadding:
    def test_three_digit_ids_map_onto_two_digit_filenames(self) -> None:
        """Regression: WIT-001.

        The human-eval CSV writes day1_consultation010; the files are named
        day1_consultation10. Ids 1-9 agree, so the mismatch survives a
        spot-check of the first rows and then drops 220 of 1,425 annotations
        with no error raised.
        """
        assert canonical_id("day1_consultation010") == "day1_consultation10"
        assert canonical_id("day2_consultation015") == "day2_consultation15"

    def test_two_digit_ids_are_untouched(self) -> None:
        assert canonical_id("day1_consultation01") == "day1_consultation01"
        assert canonical_id("day1_consultation09") == "day1_consultation09"


@pytest.fixture(scope="module")
def corpus() -> PriMock57:
    return PriMock57(CORPUS)


@requires_corpus
class TestCorpus:
    def test_every_annotation_resolves_to_a_consultation(self, corpus: PriMock57) -> None:
        """The guard that would have caught WIT-001 on the first run."""
        anns = corpus.annotations()
        corpus.require_consultations({a.consultation_id for a in anns})
        assert len(anns) == 1140

    def test_unresolvable_ids_raise_rather_than_being_skipped(self, corpus: PriMock57) -> None:
        with pytest.raises(KeyError, match="do not resolve"):
            corpus.require_consultations({"day9_consultation99"})

    def test_utterances_are_ordered_and_timed(self, corpus: PriMock57) -> None:
        c = corpus.load_consultation("day1_consultation01")
        assert len(c.utterances) > 50
        starts = [u.start_s for u in c.utterances]
        assert starts == sorted(starts)
        assert all(u.end_s >= u.start_s for u in c.utterances)
        assert {u.speaker for u in c.utterances} == {"doctor", "patient"}

    def test_lfs_pointer_is_not_mistaken_for_audio(self, corpus: PriMock57) -> None:
        """A 133-byte pointer file exists but is not a wav.

        Treating it as audio produces a baffling decoder error far from the
        cause, so it is rejected where the reason is obvious.
        """
        assert corpus.audio_path("day1_consultation01", "doctor") is None
