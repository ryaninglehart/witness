"""Where the thresholds come from.

The first set of thresholds in this project were chosen by judgement: no
contradictions, no ungrounded safety claims, under 15% ungrounded overall. They
seemed defensible. They rejected **57 of 57 notes written by practising
clinicians.**

That is not a strict gate, it is a broken one. A gate that fails everything is
switched off within a week, and a switched-off gate is worse than no gate,
because it still appears in the architecture diagram and the compliance
narrative while doing nothing.

The failure was not in the numbers, it was in where they came from. A threshold
is a claim about acceptable risk, and it can only be set against a reference
population. Here the natural reference is what a competent human produces: the
consulting clinician's own note, written during or shortly after the visit. If
the harness would reject that, the harness is wrong.

So thresholds are derived, not declared. Each is set at a percentile of the
human-written distribution, and the resulting operating point is published:
this is the fraction of clinician notes the gate would flag, and this is what
it catches at that setting. Suki's CMO asked the field for "statistical
procedures for release decisions" (Fierce Healthcare, July 2026). This is one.

**The reference population is not a gold standard.** Human notes contain errors
too -- PriMock57's own evaluators found them. Calibrating to human performance
sets the bar at "no worse than a hurried clinician", which is the honest bar
for a documentation tool, not a claim of correctness.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from witness.corpus import PriMock57
from witness.gate import Thresholds
from witness.graders.lexical import LexicalGrader
from witness.grounding import Verdict
from witness.omission import Tier, find_omissions
from witness.segment import segment


@dataclass
class Distribution:
    name: str
    values: list[float]

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        k = (len(s) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    def summary(self) -> dict[str, float]:
        return {
            "n": len(self.values),
            "median": statistics.median(self.values) if self.values else 0.0,
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "max": max(self.values) if self.values else 0.0,
        }


@dataclass
class CalibrationResult:
    unsupported_rate: Distribution
    safety_unsupported: Distribution
    contradicted: Distribution
    critical_omissions: Distribution
    thresholds: Thresholds
    #: Omission has no home in Thresholds because the gate never sees the
    #: transcript. It is calibrated here and carried alongside.
    max_critical_omissions: int
    #: Fraction of the human reference population the derived gate would flag.
    human_flag_rate: float

    def report(self) -> str:
        lines = [
            "  Calibrated against notes written by the consulting clinician",
            "  " + "-" * 66,
            f"  reference population   {self.unsupported_rate.summary()['n']} human-written notes",
            "",
            f"  {'measure':<26}{'median':>9}{'p90':>9}{'p95':>9}{'max':>9}",
        ]
        for d in (
            self.unsupported_rate,
            self.safety_unsupported,
            self.contradicted,
            self.critical_omissions,
        ):
            s = d.summary()
            lines.append(
                f"  {d.name:<26}{s['median']:>9.2f}{s['p90']:>9.2f}"
                f"{s['p95']:>9.2f}{s['max']:>9.2f}"
            )
        t = self.thresholds
        lines += [
            "",
            "  derived thresholds",
            f"    max_unsupported_rate      {t.max_unsupported_rate:.2f}",
            f"    max_safety_unsupported    {t.max_safety_unsupported}",
            f"    max_contradicted          {t.max_contradicted}",
            f"    max_critical_omissions    {self.max_critical_omissions}",
            f"    slot ceilings             {t.slot_ceilings}",
            f"    ruleset fingerprint       {t.fingerprint()}",
            "",
            f"  operating point: this gate flags {self.human_flag_rate:.0%} of the "
            "clinician-written notes",
            "  it was calibrated on. A gate that flagged none of them would be",
            "  measuring nothing; one that flagged all of them would be switched off.",
        ]
        return "\n".join(lines)


def calibrate(
    corpus: PriMock57,
    percentile: float = 0.90,
    grader: LexicalGrader | None = None,
) -> CalibrationResult:
    """Derive thresholds from what practising clinicians actually write."""
    grader = grader or LexicalGrader()
    unsupported: list[float] = []
    safety: list[float] = []
    contradicted: list[float] = []
    omissions: list[float] = []

    from collections import defaultdict

    per_slot: dict[str, list[int]] = defaultdict(list)
    groundings = []
    for cid in corpus.consultation_ids():
        cons = corpus.load_consultation(cid)
        g = grader.ground_note(
            cid, "clinician", segment(cons.clinician_note), cons.utterances
        )
        o = find_omissions(cons, cons.clinician_note, "clinician")
        groundings.append((g, o))
        unsupported.append(g.unsupported_rate)
        safety.append(len(g.safety_failures))
        contradicted.append(g.count(Verdict.CONTRADICTED))
        omissions.append(len(o.omitted_at(Tier.CRITICAL)))
        counts: dict[str, int] = defaultdict(int)
        for a in g.safety_failures:
            counts[a.assertion.slot.value] += 1
        # Slots absent from a note contribute a zero. Averaging only over notes
        # that happen to contain an allergy claim would set the allergy ceiling
        # from the rare noisy cases and miss that most notes have none.
        for slot in ("ALLERGY", "DOSE", "MEDICATION", "DIAGNOSIS"):
            per_slot[slot].append(counts.get(slot, 0))

    d_unsup = Distribution("unsupported rate", unsupported)
    d_safety = Distribution("ungrounded safety claims", safety)
    d_contra = Distribution("contradicted claims", contradicted)
    d_omit = Distribution("critical omissions", omissions)

    # Per-slot ceilings, each from its own distribution. Sparse slots earn a
    # tight bar; dense ones do not. This is why a fabricated allergy can block
    # in a note whose overall ungrounded count is well within tolerance.
    ceilings = {
        slot: int(Distribution(slot, vals).percentile(percentile))
        for slot, vals in per_slot.items()
    }
    thresholds = Thresholds(
        max_unsupported_rate=round(d_unsup.percentile(percentile), 3),
        max_safety_unsupported=int(d_safety.percentile(percentile)),
        max_contradicted=int(d_contra.percentile(percentile)),
        slot_ceilings=",".join(f"{k}:{v}" for k, v in sorted(ceilings.items())),
    )

    # Measure the operating point rather than asserting it. A calibration that
    # does not report how often it fires on its own reference population is
    # just a differently-shaped guess.
    from witness.gate import Gate

    gate = Gate(thresholds)
    flagged = 0
    for g, o in groundings:
        r = gate.evaluate(g)
        crit = len(o.omitted_at(Tier.CRITICAL))
        if r.unacknowledged_blocking or crit > int(d_omit.percentile(percentile)):
            flagged += 1

    return CalibrationResult(
        max_critical_omissions=int(d_omit.percentile(percentile)),
        unsupported_rate=d_unsup,
        safety_unsupported=d_safety,
        contradicted=d_contra,
        critical_omissions=d_omit,
        thresholds=thresholds,
        human_flag_rate=flagged / len(groundings) if groundings else 0.0,
    )


@dataclass
class OperatingPoint:
    percentile: float
    human_pass_rate: float
    sabotage_detection: float
    thresholds: Thresholds


def sweep(corpus: PriMock57, consultation_id: str = "day1_consultation01") -> list[OperatingPoint]:
    """Trace the sensitivity/specificity trade across calibration percentiles.

    A single operating point invites the reader to assume a better one exists
    just off-camera. Sweeping shows the whole curve, and in this case the curve
    is the result: with lexical grounding there is no percentile at which the
    gate both accepts notes written by clinicians and catches a single injected
    defect. Tightening to catch the defects rejects the humans; loosening to
    accept the humans lets the defects through.

    That is a property of the grader, not of the thresholds, and it is the
    measured argument for a semantic grader rather than an asserted one.
    """
    from witness.gate import Gate
    from witness.pipeline import Pipeline
    from witness.prove import run_proof

    cons = corpus.load_consultation(consultation_id)
    consultations = [corpus.load_consultation(c) for c in corpus.consultation_ids()]
    points: list[OperatingPoint] = []

    for p in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        cal = calibrate(corpus, percentile=p)
        pipe = Pipeline(
            gate=Gate(cal.thresholds), max_critical_omissions=cal.max_critical_omissions
        )
        passed = sum(
            1
            for c in consultations
            if pipe.run(c, c.clinician_note, "clinician", publish=False).gate.published
        )
        proof = run_proof(cons, cons.clinician_note, cal.thresholds, cal.max_critical_omissions)
        detectable = [r for r in proof if r.sabotage.should_block]
        points.append(
            OperatingPoint(
                percentile=p,
                human_pass_rate=passed / len(consultations),
                sabotage_detection=sum(1 for r in detectable if r.blocked) / len(detectable),
                thresholds=cal.thresholds,
            )
        )
    return points
