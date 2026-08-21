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

from witness.corpus import CRITICAL, Annotation
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
