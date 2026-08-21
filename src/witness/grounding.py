"""Verdicts, and what it means for a clinical claim to be supported.

Every assertion in a generated note gets exactly one verdict and, unless it is
fabricated, a span pointing at the utterance that justifies it. The span is the
product. A score with no span is an opinion; a span is something a clinician can
check in four seconds by playing the audio.

The four verdicts are not a severity scale. They are four different
relationships between a claim and an encounter, and they fail in different ways:

``SUPPORTED``
    Someone said it. The cheapest and least interesting case.

``INFERRED``
    Nobody said it, but a clinician would write it anyway. "She's up and about
    most of the day but tires easily" becomes "ECOG 1". That is correct clinical
    reasoning, and a naive entailment checker calls it a hallucination and is
    wrong. This category is the reason the project is not a thin wrapper around
    an NLI model. An inference must still cite the span it reasoned *from* --
    if it can point at nothing, it is not an inference, it is a guess, and it
    is graded ``UNSUPPORTED``.

``UNSUPPORTED``
    Nothing in the encounter supports it and no clinical rule derives it.

``CONTRADICTED``
    The encounter says otherwise. Worse than ``UNSUPPORTED``: fabricating an
    allergy is dangerous, but recording "no known allergies" over a stated
    penicillin allergy is dangerous *and* looks like diligence.

The PriMock57 clinicians who labelled 1,425 notes used categories that map onto
these: "Hallucination", "Incorrect statement", "Contradiction", and separately
"Symptoms mentioned is reported as fact" -- which is precisely the
inference/fabrication boundary, found independently by working clinicians
before it was designed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    INFERRED = "INFERRED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"

    @property
    def is_failure(self) -> bool:
        return self in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)


class Slot(StrEnum):
    """What kind of clinical claim this is.

    The gate treats slots asymmetrically. An unsupported statement about a
    patient's occupation is a quality problem; an unsupported statement about a
    medication dose is a safety problem. Flattening them into one "hallucination
    rate" is how a harness ends up green while shipping something harmful.
    """

    MEDICATION = "MEDICATION"
    ALLERGY = "ALLERGY"
    DOSE = "DOSE"
    DIAGNOSIS = "DIAGNOSIS"
    SYMPTOM = "SYMPTOM"
    NEGATIVE_FINDING = "NEGATIVE_FINDING"
    PLAN = "PLAN"
    SOCIAL = "SOCIAL"
    OTHER = "OTHER"

    @property
    def is_safety_critical(self) -> bool:
        return self in (Slot.MEDICATION, Slot.ALLERGY, Slot.DOSE, Slot.DIAGNOSIS)


@dataclass(frozen=True)
class Span:
    """A pointer into the encounter, precise enough to play.

    ``start_s``/``end_s`` are what turn a citation into evidence: the reviewer
    does not have to trust the classifier, they can listen. This is the same
    idea as DeepScribe's "Clinical Moments", and the reason it is on the Span
    rather than bolted on later is that a verdict without a timestamp is not
    reviewable, so it should not be representable.
    """

    utterance_index: int
    speaker: str
    start_s: float
    end_s: float
    text: str


@dataclass
class Assertion:
    """One clinical claim extracted from a generated note."""

    index: int
    text: str
    slot: Slot = Slot.OTHER
    #: True when the claim asserts the *absence* of something ("no chest pain").
    #: Tracked separately because negation is where grounding most often gets
    #: it backwards -- and where getting it backwards is most dangerous.
    negated: bool = False


@dataclass
class GroundedAssertion:
    assertion: Assertion
    verdict: Verdict
    spans: list[Span] = field(default_factory=list)
    #: Free-text reason. Required for INFERRED (name the clinical rule) and for
    #: CONTRADICTED (say what the encounter actually said), because a verdict a
    #: reviewer cannot audit is a verdict they have to take on faith.
    rationale: str = ""
    confidence: float = 0.0
    grader: str = ""

    def __post_init__(self) -> None:
        # A grader that returns SUPPORTED or INFERRED without a span has not
        # done the job -- it has produced the appearance of provenance. Failing
        # loudly here beats emitting an uncheckable citation, which is the exact
        # failure this project exists to catch in other systems.
        if self.verdict in (Verdict.SUPPORTED, Verdict.INFERRED) and not self.spans:
            raise ValueError(
                f"{self.verdict.value} requires at least one source span "
                f"(assertion {self.assertion.index}: {self.assertion.text[:60]!r})"
            )

    @property
    def is_failure(self) -> bool:
        return self.verdict.is_failure

    @property
    def is_safety_failure(self) -> bool:
        return self.is_failure and self.assertion.slot.is_safety_critical


@dataclass
class NoteGrounding:
    """The full result for one generated note."""

    consultation_id: str
    model_id: str
    grader: str
    assertions: list[GroundedAssertion] = field(default_factory=list)

    def count(self, verdict: Verdict) -> int:
        return sum(1 for a in self.assertions if a.verdict is verdict)

    @property
    def n(self) -> int:
        return len(self.assertions)

    @property
    def unsupported_rate(self) -> float:
        if not self.n:
            return 0.0
        return sum(1 for a in self.assertions if a.is_failure) / self.n

    @property
    def safety_failures(self) -> list[GroundedAssertion]:
        return [a for a in self.assertions if a.is_safety_failure]

    def summary(self) -> dict[str, float | int]:
        return {
            "assertions": self.n,
            "supported": self.count(Verdict.SUPPORTED),
            "inferred": self.count(Verdict.INFERRED),
            "unsupported": self.count(Verdict.UNSUPPORTED),
            "contradicted": self.count(Verdict.CONTRADICTED),
            "safety_failures": len(self.safety_failures),
            "unsupported_rate": round(self.unsupported_rate, 4),
        }
