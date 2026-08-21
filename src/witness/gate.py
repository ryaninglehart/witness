"""The release gate: turning verdicts into a decision, and refusing to publish.

A score is advice. A gate is a decision, and a decision needs three properties
a score does not: a threshold someone chose on purpose, a record of who
overrode it and why, and a fingerprint that changes when the rules change.

That third one is the easiest to get wrong and the most damaging. A gate whose
thresholds are not inside its own fingerprint can be quietly weakened between
runs and every artifact will still claim to have been produced under the same
rules. Two builds look identical, one of them was checked against a bar half as
high, and nothing in the record distinguishes them. So ``ruleset_hash`` covers
the thresholds themselves, not merely the rule names.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from witness.grounding import NoteGrounding, Slot, Verdict

RULESET_VERSION = "1.0.0"


class Severity(StrEnum):
    BLOCK = "BLOCK"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class Thresholds:
    """Every number the gate can fail on.

    Frozen so a caller cannot mutate the thresholds after the hash is taken --
    which would produce a build stamped with a ruleset it was not actually
    checked against.
    """

    #: Any contradiction at all fails. Recording the opposite of what was said
    #: is not a quality issue on a spectrum; it is a different kind of wrong.
    max_contradicted: int = 0
    #: Unsupported claims in medication, dose, allergy or diagnosis slots.
    max_safety_unsupported: int = 0
    #: Unsupported claims anywhere, as a fraction of all assertions.
    max_unsupported_rate: float = 0.15
    #: Inferred claims are legitimate but should not dominate a note; a note
    #: that is mostly inference is mostly the model's reasoning, not the visit.
    warn_inferred_rate: float = 0.30
    #: A note with almost nothing in it passes every ratio test trivially.
    min_assertions: int = 3

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Finding:
    rule: str
    severity: Severity
    detail: str
    count: int = 0


@dataclass
class Acknowledgement:
    """An override, with the reason attached. Anonymous overrides are refused."""

    rule: str
    reason: str
    by: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(f"override of {self.rule!r} requires a written reason")
        if not self.by.strip():
            raise ValueError(f"override of {self.rule!r} requires an identified approver")


@dataclass
class GateResult:
    consultation_id: str
    model_id: str
    ruleset_version: str
    ruleset_hash: str
    findings: list[Finding] = field(default_factory=list)
    acknowledgements: list[Acknowledgement] = field(default_factory=list)
    published: bool = False
    decided_at: str = ""

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCK]

    @property
    def unacknowledged_blocking(self) -> list[Finding]:
        ack = {a.rule for a in self.acknowledgements}
        return [f for f in self.blocking if f.rule not in ack]

    def to_dict(self) -> dict:
        return {
            "consultation_id": self.consultation_id,
            "model_id": self.model_id,
            "ruleset_version": self.ruleset_version,
            "ruleset_hash": self.ruleset_hash,
            "decided_at": self.decided_at,
            "published": self.published,
            "findings": [asdict(f) | {"severity": f.severity.value} for f in self.findings],
            "acknowledgements": [asdict(a) for a in self.acknowledgements],
        }


class Gate:
    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or Thresholds()

    @property
    def ruleset_hash(self) -> str:
        return self.thresholds.fingerprint()

    def evaluate(
        self,
        grounding: NoteGrounding,
        acknowledgements: list[Acknowledgement] | None = None,
        now: datetime | None = None,
    ) -> GateResult:
        t = self.thresholds
        findings: list[Finding] = []

        contradicted = grounding.count(Verdict.CONTRADICTED)
        if contradicted > t.max_contradicted:
            worst = next(
                a for a in grounding.assertions if a.verdict is Verdict.CONTRADICTED
            )
            findings.append(
                Finding(
                    "contradicted_claims",
                    Severity.BLOCK,
                    f"{contradicted} claim(s) contradict the encounter, "
                    f"e.g. {worst.assertion.text[:70]!r}",
                    contradicted,
                )
            )

        safety = grounding.safety_failures
        if len(safety) > t.max_safety_unsupported:
            slots = sorted({a.assertion.slot.value for a in safety})
            findings.append(
                Finding(
                    "unsupported_safety_claims",
                    Severity.BLOCK,
                    f"{len(safety)} ungrounded claim(s) in safety slots: {', '.join(slots)}",
                    len(safety),
                )
            )

        if grounding.n < t.min_assertions:
            findings.append(
                Finding(
                    "note_too_thin",
                    Severity.BLOCK,
                    f"only {grounding.n} clinical assertion(s); ratio checks are "
                    "meaningless below the floor and would pass trivially",
                    grounding.n,
                )
            )
        elif grounding.unsupported_rate > t.max_unsupported_rate:
            findings.append(
                Finding(
                    "unsupported_rate",
                    Severity.BLOCK,
                    f"{grounding.unsupported_rate:.1%} of claims ungrounded, "
                    f"above {t.max_unsupported_rate:.0%}",
                    round(grounding.unsupported_rate * 100),
                )
            )

        inferred_rate = grounding.count(Verdict.INFERRED) / grounding.n if grounding.n else 0.0
        if inferred_rate > t.warn_inferred_rate:
            findings.append(
                Finding(
                    "high_inference",
                    Severity.WARN,
                    f"{inferred_rate:.1%} of claims are inferred rather than stated",
                    round(inferred_rate * 100),
                )
            )

        for slot in (Slot.ALLERGY, Slot.DOSE):
            n = sum(1 for a in grounding.assertions if a.assertion.slot is slot)
            if n:
                findings.append(
                    Finding(f"{slot.value.lower()}_present", Severity.INFO, f"{n} claim(s)", n)
                )

        result = GateResult(
            consultation_id=grounding.consultation_id,
            model_id=grounding.model_id,
            ruleset_version=RULESET_VERSION,
            ruleset_hash=self.ruleset_hash,
            findings=findings,
            acknowledgements=list(acknowledgements or []),
            decided_at=(now or datetime.now(UTC)).isoformat(),
        )
        result.published = not result.unacknowledged_blocking
        return result
