"""Measuring the grader against 1,140 clinician-graded notes.

The question this answers is not "does Witness find errors" -- any strict
checker finds errors. It is whether Witness's failures are *the same failures a
clinician would care about*, and whether ranking notes by its score beats
ranking them by the metrics the industry already has.

The comparison is scoped honestly. Grounding reads the note against the
transcript, so it can see fabrication and contradiction. It structurally cannot
see omission: nothing in a note tells you what is missing from it. So grounding
is scored against ``incorrect_critical`` only, and omission is a separate
module measured separately. Reporting one combined number would let the easier
half carry the harder one, which is the kind of flattering aggregate this
project was built to distrust.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from witness.corpus import Annotation, PriMock57
from witness.graders.lexical import LexicalGrader
from witness.grounding import NoteGrounding, Verdict
from witness.segment import segment

#: Weights turning a set of verdicts into one comparable risk score.
#: Contradiction outranks fabrication because writing "no known allergies" over
#: a stated allergy is both dangerous and disguised as diligence. Safety-slot
#: failures are weighted again on top -- see ``score_note``.
_VERDICT_WEIGHT = {
    Verdict.CONTRADICTED: 3.0,
    Verdict.UNSUPPORTED: 1.0,
    Verdict.INFERRED: 0.15,
    Verdict.SUPPORTED: 0.0,
}
_SAFETY_MULTIPLIER = 2.0


def score_note(g: NoteGrounding) -> float:
    """A single risk score for one note. Higher is worse.

    Deliberately a transparent weighted sum rather than anything learned. A
    fitted scorer would score better on this corpus and would also be
    unauditable and overfitted to five annotators' habits, which is the wrong
    trade for something whose output is meant to justify blocking a release.
    """
    total = 0.0
    for ga in g.assertions:
        w = _VERDICT_WEIGHT[ga.verdict]
        if w and ga.assertion.slot.is_safety_critical:
            w *= _SAFETY_MULTIPLIER
        total += w
    return total


@dataclass
class ValidationResult:
    n: int
    spearman_rho: float
    p_value: float
    #: Mean clinician-labelled critical incorrect statements across all notes.
    base_mean: float
    #: Same, among the decile Witness scored safest.
    safest_decile_mean: float
    detection_rate: float
    false_alarm_rate: float

    @property
    def error_reduction(self) -> float:
        if self.base_mean == 0:
            return 0.0
        return 1.0 - (self.safest_decile_mean / self.base_mean)


def validate(corpus: PriMock57, grader: LexicalGrader | None = None) -> ValidationResult:
    grader = grader or LexicalGrader()
    annotations = corpus.annotations()
    # Fail loudly on an unresolvable join rather than quietly grading a subset.
    corpus.require_consultations({a.consultation_id for a in annotations})
    consultations = {cid: corpus.load_consultation(cid) for cid in corpus.consultation_ids()}

    scores: list[float] = []
    truth: list[int] = []
    flagged_and_wrong = flagged = actually_wrong = 0

    for a in annotations:
        cons = consultations[a.consultation_id]
        g = grader.ground_note(
            a.consultation_id, a.model_id, segment(a.generated_note), cons.utterances
        )
        s = score_note(g)
        scores.append(s)
        truth.append(a.critical_incorrect)

        # Note-level agreement: did Witness raise any failure, and did the
        # clinician record any critical incorrect statement?
        w_flag = any(ga.is_failure for ga in g.assertions)
        h_flag = a.critical_incorrect > 0
        flagged += w_flag
        actually_wrong += h_flag
        flagged_and_wrong += w_flag and h_flag

    rho, p = stats.spearmanr(scores, truth)
    n = max(1, int(len(scores) * 0.10))
    safest = sorted(zip(scores, truth, strict=True), key=lambda t: t[0])[:n]

    return ValidationResult(
        n=len(scores),
        spearman_rho=float(rho),
        p_value=float(p),
        base_mean=sum(truth) / len(truth),
        safest_decile_mean=sum(t for _, t in safest) / len(safest),
        detection_rate=flagged_and_wrong / actually_wrong if actually_wrong else 0.0,
        false_alarm_rate=(flagged - flagged_and_wrong) / max(1, len(scores) - actually_wrong),
    )


def agreement_on(
    corpus: PriMock57, consultation_id: str | None = None
) -> dict[str, float]:
    """Inter-annotator agreement among the five clinicians.

    Pass a consultation id for one encounter; pass nothing for the whole
    corpus, which is what any statement about "the clinicians" needs. The
    published figure used to be one consultation's -- 27.5% exact on 40 pairs
    -- quoted as a property of the corpus. Across all 2,280 pairs it is 19.8%.
    Reporting the friendlier of the two as though it were the general case is
    the error this project spends its time finding elsewhere. See WIT-015.

    This is the ceiling. Witness cannot meaningfully be asked to agree with
    "the clinicians" more than the clinicians agree with each other, and the
    scoping review (medRxiv 2025) names unreported inter-rater reliability as
    one of the field's four systematic failures.

    One caveat travels with it: this is agreement on the *count* of critical
    incorrect statements per note, not on which statements those are. Two
    clinicians who each find three different errors agree perfectly here. So
    it bounds nothing formally -- it is context for an item-level precision
    figure, not a derivation, and treating it as a hard ceiling would be one
    more unit mismatch.
    """
    rows = (
        [a for a in corpus.annotations() if a.consultation_id == consultation_id]
        if consultation_id is not None
        else list(corpus.annotations())
    )
    # Keyed by note, not by model: the same model id appears across every
    # consultation, and pooling those would pair clinicians who reviewed
    # different encounters.
    by_model: dict[tuple[str, str], list[Annotation]] = {}
    for a in rows:
        by_model.setdefault((a.consultation_id, a.model_id), []).append(a)

    pairs: list[tuple[int, int]] = []
    for anns in by_model.values():
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                pairs.append((anns[i].critical_incorrect, anns[j].critical_incorrect))
    if len(pairs) < 3:
        return {"pairs": len(pairs)}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho, _ = stats.spearmanr(xs, ys)
    exact = sum(1 for x, y in pairs if x == y) / len(pairs)
    within_one = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
    return {
        "pairs": len(pairs),
        "spearman_rho": float(rho),
        "exact_agreement": exact,
        "within_one": within_one,
    }
