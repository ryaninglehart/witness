"""Did Witness flag the claim the clinician flagged?

Note-level ranking turned out to be the wrong evaluation for this corpus. When
98.4% of notes contain at least one critical error, ordering notes by risk is
close to meaningless -- every decile is contaminated, every detector "detects",
and the resulting numbers say more about the base rate than about the detector.
The first version of this validation reported exactly such a number and it was
uninterpretable.

The corpus supports a much sharper question, because the annotators did not
just count errors: they wrote out the text of each one. "! normal food colour."
is a specific claim a specific clinician judged critically wrong.

So the evaluation is assertion-level. For each labelled finding, did Witness
raise a failure on the assertion that finding refers to? That yields ordinary
precision and recall against 3,471 clinician-identified critical statements,
and it is answerable rather than saturated.

Matching finding text to assertion text is itself approximate -- annotators
paraphrase, abbreviate and sometimes quote loosely. The matcher is therefore
deliberately generous about *finding* the referent and strict about counting
it, and the residual ambiguity is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from witness.corpus import CRITICAL, Annotation, Finding
from witness.graders.lexical import content_terms
from witness.grounding import NoteGrounding


def _overlap(a: set[str], b: set[str]) -> float:
    """Fraction of the smaller term set present in the larger.

    Containment rather than Jaccard: an annotator writing "normal food colour"
    about the assertion "vomit was normal food colour, no blood" is referring
    to it, and Jaccard would score that pairing low for the wrong reason.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class MatchResult:
    findings: int
    matched: int
    flags: int
    flags_matching_a_finding: int
    unmatchable_findings: int

    @property
    def recall(self) -> float:
        """Of the clinician's critical findings, how many did Witness flag?"""
        denom = self.findings - self.unmatchable_findings
        return self.matched / denom if denom else 0.0

    @property
    def precision(self) -> float:
        """Of Witness's flags, how many land on a clinician-identified error?"""
        return self.flags_matching_a_finding / self.flags if self.flags else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match_note(
    annotation: Annotation,
    grounding: NoteGrounding,
    threshold: float = 0.5,
) -> tuple[int, int, int, int, int]:
    """Align one note's clinician findings with Witness's verdicts.

    Returns (findings, matched, flags, flags_on_a_finding, unmatchable).
    """
    critical = [f for f in annotation.incorrect if f.severity == CRITICAL]
    finding_terms = [content_terms(f.text) for f in critical]
    assertion_terms = [content_terms(ga.assertion.text) for ga in grounding.assertions]
    flagged = [i for i, ga in enumerate(grounding.assertions) if ga.is_failure]

    # A finding whose text shares no clinical terms with *any* assertion cannot
    # be attributed to one. Usually the annotator described a structural
    # problem ("disjointed note") rather than quoting a claim. Counting these
    # as misses would penalise the detector for not finding something that is
    # not in its input, so they are excluded from recall and reported.
    unmatchable = 0
    matched = 0
    matched_flag_indices: set[int] = set()

    for ft in finding_terms:
        best_i, best_score = -1, 0.0
        for i, at in enumerate(assertion_terms):
            s = _overlap(ft, at)
            if s > best_score:
                best_i, best_score = i, s
        if best_score < threshold:
            unmatchable += 1
            continue
        if best_i in flagged:
            matched += 1
            matched_flag_indices.add(best_i)

    return (
        len(critical),
        matched,
        len(flagged),
        len(matched_flag_indices),
        unmatchable,
    )


def match_note_union(
    annotations: list[Annotation],
    grounding: NoteGrounding,
    threshold: float = 0.5,
) -> tuple[int, int, int, int, int]:
    """Score one note against what *any* of the five clinicians flagged.

    Precision against a single annotator is not a measure of the detector
    alone. The five agree exactly 27.5% of the time, so a flag that lands on a
    real error one clinician did not happen to write down is counted as a false
    positive, and the detector is charged for the annotator's omission.

    Pooling the findings across all five evaluators of the same note gives the
    other bound. Neither figure is the truth: the single-annotator number
    understates precision by charging for disagreement, and the union
    overstates it by accepting anything anyone flagged. Reporting both, with
    the inter-rater ceiling beside them, is the honest form -- picking whichever
    one flatters the detector is the thing this project objects to elsewhere.
    """
    pooled: list[Finding] = []
    for a in annotations:
        pooled.extend(f for f in a.incorrect if f.severity == CRITICAL)

    finding_terms = [content_terms(f.text) for f in pooled]
    assertion_terms = [content_terms(ga.assertion.text) for ga in grounding.assertions]
    flagged = [i for i, ga in enumerate(grounding.assertions) if ga.is_failure]

    unmatchable = 0
    matched = 0
    matched_flag_indices: set[int] = set()
    for ft in finding_terms:
        best_i, best_score = -1, 0.0
        for i, at in enumerate(assertion_terms):
            s = _overlap(ft, at)
            if s > best_score:
                best_i, best_score = i, s
        if best_score < threshold:
            unmatchable += 1
            continue
        if best_i in flagged:
            matched += 1
            matched_flag_indices.add(best_i)

    return (
        len(pooled),
        matched,
        len(flagged),
        len(matched_flag_indices),
        unmatchable,
    )


def match_omissions(
    annotations: list[Annotation],
    salient_texts: list[str],
    omitted_texts: list[str],
    threshold: float = 0.5,
) -> tuple[int, int, int, int, int]:
    """Did Witness report the omissions the clinicians reported?

    The harder half of the problem and the one most work skips: 62.5% of the
    critical errors in this corpus are omissions.

    The two arguments are not interchangeable and conflating them is how the
    first version of this function returned 100% recall by construction.
    ``salient_texts`` is everything the extractor judged the encounter
    established -- the universe of facts Witness could possibly have reported
    missing. ``omitted_texts`` is the subset it actually reported as dropped.

    So an annotator's omission is *attributable* if it corresponds to some
    salient fact, and *detected* only if it corresponds to one Witness
    reported. Scoring attributability as detection makes every attributable
    finding a hit and the recall meaningless -- the denominator becomes the
    numerator.
    """
    pooled: list[Finding] = []
    for a in annotations:
        pooled.extend(f for f in a.omissions if f.severity == CRITICAL)

    salient = [content_terms(t) for t in salient_texts]
    reported = [content_terms(t) for t in omitted_texts]

    def best(ft: set[str], candidates: list[set[str]]) -> tuple[int, float]:
        bi, bs = -1, 0.0
        for i, ct in enumerate(candidates):
            sc = _overlap(ft, ct)
            if sc > bs:
                bi, bs = i, sc
        return bi, bs

    unmatchable = 0
    matched = 0
    matched_reports: set[int] = set()
    for f in pooled:
        ft = content_terms(f.text)
        # Could the extractor have seen this at all?
        _, salient_score = best(ft, salient)
        if salient_score < threshold:
            unmatchable += 1
            continue
        # It could. Did Witness report it as dropped?
        ri, report_score = best(ft, reported)
        if report_score >= threshold:
            matched += 1
            matched_reports.add(ri)

    return (
        len(pooled),
        matched,
        len(reported),
        len(matched_reports),
        unmatchable,
    )
