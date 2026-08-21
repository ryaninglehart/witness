"""Alignment between clinician findings and Witness verdicts."""

from __future__ import annotations

from witness.graders.lexical import content_terms
from witness.matching import _overlap


def test_containment_not_jaccard() -> None:
    """An annotator quoting part of a claim is still referring to that claim.

    Jaccard would score this pairing low because the assertion is longer, which
    is the wrong reason to miss a match.
    """
    finding = content_terms("normal food colour")
    assertion = content_terms("vomit was normal food colour, no blood")
    assert _overlap(finding, assertion) >= 0.9


def test_unrelated_texts_do_not_match() -> None:
    assert _overlap(content_terms("penicillin allergy"), content_terms("no chest pain")) == 0.0


def test_empty_is_not_a_match() -> None:
    assert _overlap(set(), content_terms("anything")) == 0.0
