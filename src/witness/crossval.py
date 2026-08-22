"""Held-out evaluation. The module that stops the other numbers being flattering.

Two forms of leakage are possible in this project and both are invisible in the
output.

**The grader can have seen the answers.** :mod:`witness.graders.statistical`
learns its lexicon from transcripts paired with the notes clinicians wrote from
them. Fit it on all 57 consultations and then score those same 57, and it will
report that it grounds far more of a note than it really can, because the
associations were derived from the notes being graded. The number would look
like a result and be an echo.

**The thresholds can have seen the answers.** This one was already true and
already documented: the README says outright that the gate is calibrated on the
same 57 consultations it is evaluated on. With 57 encounters the choice was
between a calibration set and an evaluation set, neither large enough. Folds
resolve that -- every consultation is evaluated exactly once, by a model and a
threshold set derived only from the other consultations.

The design is two nested layers, because the two leaks are independent:

1. *Outer folds* partition the 57 consultations. For each fold the association
   model is fitted on the training consultations and every held-out note is
   scored by it. Repeated over all folds this yields one out-of-fold grounding
   per consultation, none of them in-sample.
2. *Thresholds* are then calibrated on the training consultations of the same
   fold, using out-of-fold groundings, and applied to the held-out ones.

Assignment is by stride over sorted consultation ids -- deterministic, no seed,
no RNG. A cross-validation whose result moves when you re-run it is a
measurement of the random number generator.

The comparison is always against the lexical grader **on the same folds**. A
new grader evaluated on a different split than the baseline it claims to beat
is not evidence of anything.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

from witness.calibration import Distribution
from witness.corpus import Consultation, PriMock57
from witness.gate import Gate, Thresholds
from witness.graders.lexical import LexicalGrader
from witness.graders.statistical import AssociationModel, StatisticalGrader, fit
from witness.grounding import Grader, NoteGrounding, Verdict
from witness.interval import Difference, Interval, bootstrap, median_of, paired_difference, ratio_of
from witness.matching import MatchResult, match_note
from witness.omission import Tier, find_omissions
from witness.segment import segment


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[str, ...]
    test: tuple[str, ...]


def folds(ids: Iterable[str], k: int = 5) -> list[Fold]:
    """Partition consultation ids into k deterministic folds.

    Stride assignment over the sorted ids. PriMock57 ids encode the recording
    day (``day1_consultation03``), so contiguous blocks would put a whole day
    in one fold and confound the split with whatever varied between days --
    clinicians, actors, room. Striding interleaves the days across folds.
    """
    ordered = sorted(ids)
    if k < 2 or k > len(ordered):
        raise ValueError(f"k must be between 2 and {len(ordered)}, got {k}")
    out: list[Fold] = []
    for f in range(k):
        test = tuple(cid for i, cid in enumerate(ordered) if i % k == f)
        train = tuple(cid for cid in ordered if cid not in set(test))
        out.append(Fold(f, train, test))
    return out


@dataclass
class GroundingProfile:
    """How much of a note a grader can actually witness, per note."""

    grader: str
    grounded_fraction: list[float] = field(default_factory=list)
    unsupported_rate: list[float] = field(default_factory=list)
    contradicted: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        g = self.grounded_fraction
        return {
            "notes": len(g),
            "median_grounded": statistics.median(g) if g else 0.0,
            "mean_grounded": statistics.fmean(g) if g else 0.0,
            "median_unsupported": (
                statistics.median(self.unsupported_rate) if self.unsupported_rate else 0.0
            ),
            "notes_with_contradiction": sum(1 for c in self.contradicted if c > 0),
        }


def _profile(grader_name: str, groundings: Iterable[NoteGrounding]) -> GroundingProfile:
    p = GroundingProfile(grader_name)
    for g in groundings:
        if g.n == 0:
            continue
        grounded = g.count(Verdict.SUPPORTED) + g.count(Verdict.INFERRED)
        p.grounded_fraction.append(grounded / g.n)
        p.unsupported_rate.append(g.unsupported_rate)
        p.contradicted.append(g.count(Verdict.CONTRADICTED))
    return p


# -- out-of-fold grounding ------------------------------------------------


@dataclass
class FoldModels:
    """One fold's fitted association model, kept so callers can inspect it."""

    fold: Fold
    model: AssociationModel


def fit_folds(
    corpus: PriMock57,
    fold_list: list[Fold],
    consultations: dict[str, Consultation] | None = None,
    **fit_kwargs: float | int,
) -> list[FoldModels]:
    cons = consultations or {c: corpus.load_consultation(c) for c in corpus.consultation_ids()}
    return [
        FoldModels(f, fit([cons[c] for c in f.train], **fit_kwargs))  # type: ignore[arg-type]
        for f in fold_list
    ]


def out_of_fold_clinician_notes(
    corpus: PriMock57,
    fold_list: list[Fold],
    models: list[FoldModels],
    consultations: dict[str, Consultation],
) -> tuple[dict[str, NoteGrounding], dict[str, NoteGrounding]]:
    """Ground every clinician-written note twice: lexical, and out-of-fold statistical.

    The lexical grader has nothing to learn and so nothing to leak; it is run
    over the same partition anyway, because comparing a held-out number to an
    in-sample one is the error this module exists to prevent.
    """
    lexical = LexicalGrader()
    lex: dict[str, NoteGrounding] = {}
    stat: dict[str, NoteGrounding] = {}
    for fm in models:
        grader = StatisticalGrader(fm.model)
        for cid in fm.fold.test:
            c = consultations[cid]
            assertions = segment(c.clinician_note)
            lex[cid] = lexical.ground_note(cid, "clinician", assertions, c.utterances)
            stat[cid] = grader.ground_note(cid, "clinician", assertions, c.utterances)
    return lex, stat


def out_of_fold_detection(
    corpus: PriMock57,
    models: list[FoldModels],
    consultations: dict[str, Consultation],
) -> tuple[dict[str, MatchResult], dict[str, dict[str, MatchResult]]]:
    """Precision and recall against the clinicians' own critical findings.

    Each generated note is graded by the model whose fold excluded the
    consultation it came from, so no association used to judge a note was
    learned from that note's own transcript.
    """
    anns = corpus.annotations()
    corpus.require_consultations({a.consultation_id for a in anns})
    by_cid: dict[str, list] = {}
    for a in anns:
        by_cid.setdefault(a.consultation_id, []).append(a)

    lexical = LexicalGrader()
    totals = {"lexical": [0, 0, 0, 0, 0], "statistical": [0, 0, 0, 0, 0]}
    # Also kept per consultation, because the bootstrap resamples whole
    # consultations and needs each one's contribution separately.
    per_cid: dict[str, dict[str, MatchResult]] = {}
    for fm in models:
        grader = StatisticalGrader(fm.model)
        for cid in fm.fold.test:
            c = consultations[cid]
            index = grader.index_for(c.utterances)
            local = {"lexical": [0, 0, 0, 0, 0], "statistical": [0, 0, 0, 0, 0]}
            for a in by_cid.get(cid, []):
                assertions = segment(a.generated_note)
                for key, g in (
                    ("lexical", lexical.ground_note(cid, a.model_id, assertions, c.utterances)),
                    (
                        "statistical",
                        grader.ground_note(cid, a.model_id, assertions, c.utterances, index),
                    ),
                ):
                    for i, v in enumerate(match_note(a, g)):
                        totals[key][i] += v
                        local[key][i] += v
            per_cid[cid] = {k: MatchResult(*v) for k, v in local.items()}
    return {k: MatchResult(*v) for k, v in totals.items()}, per_cid


# -- held-out gate calibration -------------------------------------------


def _thresholds_from(
    groundings: Iterable[NoteGrounding],
    omissions: Iterable[int],
    percentile: float,
) -> tuple[Thresholds, int]:
    """Derive thresholds from a set of already-computed groundings.

    :func:`witness.calibration.calibrate` re-grades the corpus internally,
    which would refit nothing but would also re-run the grader in-sample. Here
    the groundings are supplied, so the calibration consumes out-of-fold
    scores.
    """
    from collections import defaultdict

    unsup: list[float] = []
    safety: list[float] = []
    contra: list[float] = []
    per_slot: dict[str, list[int]] = defaultdict(list)
    for g in groundings:
        unsup.append(g.unsupported_rate)
        safety.append(len(g.safety_failures))
        contra.append(g.count(Verdict.CONTRADICTED))
        counts: dict[str, int] = defaultdict(int)
        for a in g.safety_failures:
            counts[a.assertion.slot.value] += 1
        for slot in ("ALLERGY", "DOSE", "MEDICATION", "DIAGNOSIS"):
            per_slot[slot].append(counts.get(slot, 0))

    ceilings = {
        slot: int(Distribution(slot, [float(v) for v in vals]).percentile(percentile))
        for slot, vals in per_slot.items()
    }
    thresholds = Thresholds(
        max_unsupported_rate=round(Distribution("u", unsup).percentile(percentile), 3),
        max_safety_unsupported=int(Distribution("s", safety).percentile(percentile)),
        max_contradicted=int(Distribution("c", contra).percentile(percentile)),
        slot_ceilings=",".join(f"{k}:{v}" for k, v in sorted(ceilings.items())),
    )
    max_omit = int(
        Distribution("o", [float(o) for o in omissions]).percentile(percentile)
    )
    return thresholds, max_omit


# -- held-out defect detection -------------------------------------------


@dataclass
class GradedSabotage:
    """One injected defect, graded once and judgeable at any threshold."""

    key: str
    should_block: bool
    expect_rule: str
    grounding: NoteGrounding
    critical_omissions: int
    #: The fingerprint case changes no note, only the bar. It is detected by
    #: the ruleset hash rather than by any threshold, so it is decided once.
    threshold_independent: bool = False
    #: The injection matched nothing, so this case tested nothing. Excluded
    #: from both numerator and denominator rather than counted as a pass.
    vacuous: bool = False


def grade_sabotages(
    consultation: Consultation,
    grader: Grader,
) -> list[GradedSabotage]:
    """Inject every defect into a baseline this grader considers clean.

    The baseline is rebuilt per grader on purpose. A note that the lexical
    grader can fully witness is not the same note the statistical grader can,
    and injecting into the other grader's baseline would measure the mismatch
    between the two rather than either one's sensitivity.
    """
    from witness.prove import SABOTAGES, clean_baseline

    clean = clean_baseline(consultation, grader)
    # The baseline's own findings, so each case can be scored as a delta.
    # clean_baseline strips ungrounded claims, and stripped content is
    # missing content -- on roughly half these consultations the "clean"
    # note fails the gate on its own. Without this subtraction a case gets
    # credit for a rule it never triggered. See WIT-013.
    baseline = GradedSabotage(
        key="__baseline__",
        should_block=False,
        expect_rule="",
        grounding=grader.ground_note(
            consultation.consultation_id, "baseline", segment(clean), consultation.utterances
        ),
        critical_omissions=len(
            find_omissions(consultation, clean, "baseline").omitted_at(Tier.CRITICAL)
        ),
    )
    out: list[GradedSabotage] = []
    for s in SABOTAGES:
        if s.key == "weakened_threshold":
            # Detected by the fingerprint covering the threshold values, which
            # holds at every calibration percentile.
            out.append(
                GradedSabotage(s.key, s.should_block, s.expect_rule,
                               NoteGrounding(consultation.consultation_id, "sabotage",
                                             grader.name, []),
                               0, threshold_independent=True)
            )
            continue
        note = s.inject(clean, consultation)
        if note == clean:
            out.append(
                GradedSabotage(s.key, s.should_block, s.expect_rule,
                               baseline.grounding, baseline.critical_omissions,
                               vacuous=True)
            )
            continue
        out.append(
            GradedSabotage(
                key=s.key,
                should_block=s.should_block,
                expect_rule=s.expect_rule,
                grounding=grader.ground_note(
                    consultation.consultation_id, "sabotage", segment(note),
                    consultation.utterances,
                ),
                critical_omissions=len(
                    find_omissions(consultation, note, "sabotage").omitted_at(Tier.CRITICAL)
                ),
            )
        )
    return [baseline, *out]


def _rules(g: GradedSabotage, gate: Gate, max_omissions: int) -> set[str]:
    rules = {f.rule for f in gate.evaluate(g.grounding).unacknowledged_blocking}
    if g.critical_omissions > max_omissions:
        rules.add("critical_omission")
    return rules


def _blocked(
    g: GradedSabotage, gate: Gate, max_omissions: int, baseline: set[str]
) -> bool:
    """Did the *named* mechanism fire, and was it the injection that fired it?

    Two subtractions, each from an incident. Only the named rule counts
    (WIT-006: a defect stopped by an unrelated rule is a lucky catch). And
    only rules the baseline was not already failing count (WIT-013: a baseline
    that fails on its own hands every case a catch it did not make).
    """
    if g.threshold_independent:
        return True
    added = _rules(g, gate, max_omissions) - baseline
    return g.expect_rule in added if g.expect_rule else bool(added)


@dataclass
class OperatingPoint:
    percentile: float
    grader: str
    #: Held-out clinician notes the gate accepts.
    human_pass_rate: float
    #: Defects in the *note* that the named mechanism caught. The
    #: ruleset-fingerprint case is excluded: it modifies no note, it is
    #: detected by construction rather than by any threshold, and counting it
    #: put a guaranteed hit into every percentile of the headline figure.
    #: See WIT-015.
    detection: float
    #: Paraphrase controls wrongly flagged. A harness that fires on rewording
    #: gets switched off, and a switched-off gate is worse than none.
    false_alarm_rate: float
    notes: int
    defects: int
    #: Harness-integrity cases, reported separately because "the gate cannot
    #: be weakened without the fingerprint changing" and "the grader finds a
    #: fabricated allergy" are different claims, and averaging them states
    #: neither.
    integrity_detection: float = 0.0
    integrity_cases: int = 0


def operating_points(
    fold_list: list[Fold],
    groundings: dict[str, NoteGrounding],
    omissions: dict[str, int],
    sabotages: dict[str, list[GradedSabotage]],
    grader_name: str,
    percentiles: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99),
) -> list[OperatingPoint]:
    """The full trade, held out on both sides.

    Thresholds come from each fold's training consultations; acceptance and
    detection are measured on its held-out ones. The README's negative result
    rests on this curve, so it is the last place an in-sample number belongs.
    """
    points: list[OperatingPoint] = []
    for p in percentiles:
        passed = total = caught = detectable = alarms = controls = 0
        integrity_hit = integrity_total = 0
        for f in fold_list:
            thresholds, max_omit = _thresholds_from(
                (groundings[c] for c in f.train if c in groundings),
                (omissions[c] for c in f.train if c in omissions),
                p,
            )
            gate = Gate(thresholds)
            for cid in f.test:
                if cid in groundings:
                    total += 1
                    r = gate.evaluate(groundings[cid])
                    if not r.unacknowledged_blocking and omissions.get(cid, 0) <= max_omit:
                        passed += 1
                cases = sabotages.get(cid, [])
                base_rules: set[str] = set()
                for g in cases:
                    if g.key == "__baseline__":
                        base_rules = _rules(g, gate, max_omit)
                        break
                for g in cases:
                    if g.key == "__baseline__" or g.vacuous:
                        # A case that tested nothing is excluded from both
                        # sides rather than scored as a pass.
                        continue
                    hit = _blocked(g, gate, max_omit, base_rules)
                    if g.threshold_independent:
                        # Detected by the fingerprint, not by the gate. Kept
                        # out of the note-defect rate it would otherwise
                        # inflate at every operating point.
                        integrity_total += 1
                        integrity_hit += int(hit)
                    elif g.should_block:
                        detectable += 1
                        caught += int(hit)
                    else:
                        controls += 1
                        alarms += int(hit)
        points.append(
            OperatingPoint(
                percentile=p,
                grader=grader_name,
                human_pass_rate=passed / total if total else 0.0,
                detection=caught / detectable if detectable else 0.0,
                false_alarm_rate=alarms / controls if controls else 0.0,
                integrity_detection=(
                    integrity_hit / integrity_total if integrity_total else 0.0
                ),
                integrity_cases=integrity_total,
                notes=total,
                defects=detectable,
            )
        )
    return points


# -- association radius ---------------------------------------------------


@dataclass
class RadiusPoint:
    floor: float
    median_grounded: float
    recall: float
    precision: float
    f1: float


def radius_sweep(
    corpus: PriMock57,
    consultations: dict[str, Consultation],
    fold_list: list[Fold],
    models: list[FoldModels],
    floors: tuple[float, ...] = (0.45, 0.55, 0.65, 0.75, 0.85),
) -> list[RadiusPoint]:
    """Trade note coverage against error detection by widening the match radius.

    The association floor is the one knob that controls how far from a claim's
    literal wording the grader will look for support. Lowering it is the whole
    proposition of a learned lexicon, so the question is not whether it raises
    coverage -- it does -- but what that coverage costs.

    Sweeping it held-out answers that, and the answer is the point of this
    module. Every percentage point of extra grounding is paid for in recall
    against the errors clinicians actually found, and at the tight end the
    grader converges exactly onto the lexical baseline. Soft matching cannot
    tell paraphrase from fabrication, because to a term-matcher they are the
    same event: a claim whose words are not in the transcript.
    """
    import dataclasses

    anns = corpus.annotations()
    corpus.require_consultations({a.consultation_id for a in anns})
    by_cid: dict[str, list] = {}
    for a in anns:
        by_cid.setdefault(a.consultation_id, []).append(a)

    points: list[RadiusPoint] = []
    for floor in floors:
        grounded: list[float] = []
        totals = [0, 0, 0, 0, 0]
        for fm in models:
            model = dataclasses.replace(fm.model, floor=floor, _neighbours=None)
            grader = StatisticalGrader(model)
            for cid in fm.fold.test:
                c = consultations[cid]
                index = grader.index_for(c.utterances)
                g = grader.ground_note(
                    cid, "clinician", segment(c.clinician_note), c.utterances, index
                )
                if g.n:
                    grounded.append(
                        (g.count(Verdict.SUPPORTED) + g.count(Verdict.INFERRED)) / g.n
                    )
                for a in by_cid.get(cid, []):
                    ga = grader.ground_note(
                        cid, a.model_id, segment(a.generated_note), c.utterances, index
                    )
                    for i, v in enumerate(match_note(a, ga)):
                        totals[i] += v
        m = MatchResult(*totals)
        points.append(
            RadiusPoint(
                floor=floor,
                median_grounded=statistics.median(grounded) if grounded else 0.0,
                recall=m.recall,
                precision=m.precision,
                f1=m.f1,
            )
        )
    return points


# -- top level ------------------------------------------------------------


@dataclass
class CrossValReport:
    k: int
    lexical_profile: GroundingProfile
    statistical_profile: GroundingProfile
    detection: dict[str, MatchResult]
    lexical_points: list[OperatingPoint]
    statistical_points: list[OperatingPoint]
    radius: list[RadiusPoint] = field(default_factory=list)
    intervals: dict[str, Interval] = field(default_factory=dict)
    differences: list[Difference] = field(default_factory=list)

    #: Above this, a gate is switched off in week two and its detection rate
    #: stops mattering. The project makes exactly that argument about other
    #: people's systems, so it is applied here as a constraint on candidacy
    #: rather than as a term to be traded away: an operating point that fires
    #: on a third of rewordings is not a candidate, whatever it catches.
    MAX_USABLE_FALSE_ALARM: ClassVar[float] = 0.10

    def best(self, points: list[OperatingPoint]) -> OperatingPoint:
        """The most balanced operating point a clinician would tolerate.

        A gate is only as good as its worst side. Scoring on acceptance and
        detection alone selected p=0.50 -- 14% of clinician notes accepted,
        15% of defects caught, and a false alarm on 32% of rewordings. That is
        a gate nobody would leave switched on, and it was being presented as
        the best available.

        So false alarms gate candidacy, and among the survivors the weaker of
        the two remaining rates decides, with the sum as a tie-break so a
        dominated point is never reported. If nothing is under the bound the
        full set is used and the printed false-alarm figure says so, because
        silently returning nothing would hide the situation rather than state
        it.
        """
        usable = [
            o for o in points if o.false_alarm_rate <= self.MAX_USABLE_FALSE_ALARM
        ] or points
        return max(
            usable,
            key=lambda o: (
                min(o.human_pass_rate, o.detection),
                o.human_pass_rate + o.detection,
            ),
        )

    def render(self) -> str:
        lines = [
            f"\n  {self.k}-fold cross-validation over 57 consultations",
            "  " + "=" * 68,
            "  Every number below is held out. The association model that graded a",
            "  note never saw that consultation, and the thresholds that judged it",
            "  were calibrated on the other folds.",
            "",
            "  How much of a clinician-written note can each grader witness?",
            "  " + "-" * 68,
            f"  {'grader':<36}{'median grounded':>17}{'ungrounded':>15}",
        ]
        for prof in (self.lexical_profile, self.statistical_profile):
            sm = prof.summary()
            lines.append(
                f"  {prof.grader:<36}{sm['median_grounded']:>16.1%}"
                f"{sm['median_unsupported']:>15.1%}"
            )
        lines += [
            "",
            "  Against the clinicians' own critical findings:",
            "  " + "-" * 68,
            f"  {'grader':<36}{'recall':>10}{'precision':>12}{'f1':>8}",
        ]
        for name in ("lexical", "statistical"):
            m = self.detection[name]
            lines.append(f"  {name:<36}{m.recall:>9.1%}{m.precision:>12.1%}{m.f1:>8.1%}")

        lines += [
            "",
            "  The trade, both sides held out:",
            "  " + "-" * 68,
            f"  {'':>12}{'lexical':>26}{'statistical':>26}",
            f"  {'percentile':>12}{'notes pass':>13}{'defects':>13}"
            f"{'notes pass':>13}{'defects':>13}",
        ]
        for lp, sp in zip(self.lexical_points, self.statistical_points, strict=True):
            lines.append(
                f"  {lp.percentile:>12.2f}{lp.human_pass_rate:>12.0%}{lp.detection:>13.0%}"
                f"{sp.human_pass_rate:>13.0%}{sp.detection:>13.0%}"
            )
        bl, bs = self.best(self.lexical_points), self.best(self.statistical_points)
        lines += [
            "",
            f"  best balance   lexical      {bl.human_pass_rate:.0%} accepted / "
            f"{bl.detection:.0%} caught   (p={bl.percentile:.2f}, "
            f"{bl.false_alarm_rate:.0%} false alarms)",
            f"                 statistical  {bs.human_pass_rate:.0%} accepted / "
            f"{bs.detection:.0%} caught   (p={bs.percentile:.2f}, "
            f"{bs.false_alarm_rate:.0%} false alarms)",
            "",
            f"  {bl.defects} note-level defects across {bl.notes} held-out consultations.",
            f"  Separately, {bl.integrity_cases} harness-integrity cases (a silently",
            f"  weakened gate) are caught at {bl.integrity_detection:.0%} by the ruleset",
            "  fingerprint. Those are excluded from the rates above: the fingerprint",
            "  detects by construction rather than by any threshold, so including it",
            "  put a guaranteed hit into every operating point. See WIT-015.",
        ]
        if self.radius:
            lines += [
                "",
                "  What a wider match radius buys, and what it costs:",
                "  " + "-" * 68,
                f"  {'assoc. floor':>13}{'median grounded':>18}{'recall':>10}{'f1':>8}",
            ]
            for r in self.radius:
                lines.append(
                    f"  {r.floor:>13.2f}{r.median_grounded:>17.1%}"
                    f"{r.recall:>10.1%}{r.f1:>8.1%}"
                )
            wide, tight = self.radius[0], self.radius[-1]
            lines += [
                "",
                f"  Loosening the floor from {tight.floor:.2f} to {wide.floor:.2f} raises "
                f"grounding by {100 * (wide.median_grounded - tight.median_grounded):+.1f} "
                "points",
                f"  and moves recall by {100 * (wide.recall - tight.recall):+.1f}. The trade "
                "is monotone in both directions and",
                "  at the tight end the learned grader converges onto the lexical one.",
                "  Widening the radius cannot separate paraphrase from fabrication: to a",
                "  term matcher they are the same event, a claim whose words are absent.",
            ]
        if self.intervals:
            lines += [
                "",
                "  How precisely does a 57-consultation corpus pin these down?",
                "  " + "-" * 68,
                "  Bootstrap over whole consultations, 2000 resamples, seeded.",
                "",
            ]
            for name, iv in self.intervals.items():
                lines.append(f"  {name:<40}{iv!s:>28}")
            lines += ["", "  Paired on identical resamples:"]
            for d in self.differences:
                lines.append(f"    {d}")
            undecided = [d for d in self.differences if not d.distinguishable]
            if undecided:
                lines += [
                    "",
                    f"  {len(undecided)} of {len(self.differences)} comparisons are inside the",
                    "  noise of this corpus. An interval containing zero means these 57",
                    "  consultations cannot separate the two -- not that they are equal.",
                ]
        return "\n".join(lines)


def compute_intervals(
    lex: dict[str, NoteGrounding],
    stat: dict[str, NoteGrounding],
    per_cid: dict[str, dict[str, MatchResult]],
    resamples: int = 2000,
) -> tuple[dict[str, Interval], list[Difference]]:
    """Resample the headline figures so their precision is stated, not implied.

    The unit is the consultation. Notes from one encounter share a transcript,
    a patient and a clinician, so resampling notes would treat correlated
    observations as independent and report intervals far narrower than the
    corpus supports.
    """
    ids = sorted(set(lex) | set(stat))

    def grounded(src: dict[str, NoteGrounding]) -> dict[str, float]:
        return {
            cid: (g.count(Verdict.SUPPORTED) + g.count(Verdict.INFERRED)) / g.n
            for cid, g in src.items()
            if g.n
        }

    lex_ground = median_of(grounded(lex))
    stat_ground = median_of(grounded(stat))

    # Recall is a pooled ratio, not a mean of per-consultation recalls.
    def hits(key: str) -> dict[str, float]:
        return {cid: float(m[key].matched) for cid, m in per_cid.items()}

    def attributable(key: str) -> dict[str, float]:
        return {
            cid: float(m[key].findings - m[key].unmatchable_findings)
            for cid, m in per_cid.items()
        }

    intervals: dict[str, Interval] = {
        "lexical median grounded": bootstrap(ids, lex_ground, resamples),
        "statistical median grounded": bootstrap(ids, stat_ground, resamples),
    }
    differences: list[Difference] = [
        paired_difference(
            ids, "grounding, statistical - lexical", stat_ground, lex_ground, resamples
        )
    ]

    if per_cid:
        det_ids = sorted(per_cid)
        lex_recall = ratio_of(hits("lexical"), attributable("lexical"))
        stat_recall = ratio_of(hits("statistical"), attributable("statistical"))
        intervals["lexical recall"] = bootstrap(det_ids, lex_recall, resamples)
        intervals["statistical recall"] = bootstrap(det_ids, stat_recall, resamples)
        differences.append(
            paired_difference(
                det_ids, "recall, statistical - lexical", stat_recall, lex_recall, resamples
            )
        )
    return intervals, differences


def run(
    corpus: PriMock57,
    k: int = 5,
    include_detection: bool = True,
    include_sabotage: bool = True,
    include_radius: bool = True,
    include_intervals: bool = True,
    resamples: int = 2000,
    **fit_kwargs: float | int,
) -> CrossValReport:
    ids = corpus.consultation_ids()
    consultations = {c: corpus.load_consultation(c) for c in ids}
    fold_list = folds(ids, k)
    models = fit_folds(corpus, fold_list, consultations, **fit_kwargs)

    lex, stat = out_of_fold_clinician_notes(corpus, fold_list, models, consultations)
    omissions = {
        cid: len(
            find_omissions(c, c.clinician_note, "clinician").omitted_at(Tier.CRITICAL)
        )
        for cid, c in consultations.items()
    }

    if include_detection:
        detection, per_cid = out_of_fold_detection(corpus, models, consultations)
    else:
        empty = MatchResult(0, 0, 0, 0, 0)
        detection = {"lexical": empty, "statistical": empty}
        per_cid = {}

    lex_sab: dict[str, list[GradedSabotage]] = {}
    stat_sab: dict[str, list[GradedSabotage]] = {}
    if include_sabotage:
        lexical = LexicalGrader()
        for fm in models:
            sgrader = StatisticalGrader(fm.model)
            for cid in fm.fold.test:
                c = consultations[cid]
                lex_sab[cid] = grade_sabotages(c, lexical)
                stat_sab[cid] = grade_sabotages(c, sgrader)

    intervals: dict[str, Interval] = {}
    differences: list[Difference] = []
    if include_intervals:
        intervals, differences = compute_intervals(lex, stat, per_cid, resamples)

    lex_name = LexicalGrader.name
    stat_name = StatisticalGrader(models[0].model).name
    return CrossValReport(
        intervals=intervals,
        differences=differences,
        k=k,
        lexical_profile=_profile(lex_name, lex.values()),
        statistical_profile=_profile(stat_name, stat.values()),
        detection=detection,
        radius=(
            radius_sweep(corpus, consultations, fold_list, models)
            if include_radius
            else []
        ),
        lexical_points=operating_points(fold_list, lex, omissions, lex_sab, lex_name),
        statistical_points=operating_points(
            fold_list, stat, omissions, stat_sab, stat_name
        ),
    )


# -- published-figure gate ------------------------------------------------

#: The figures this project publishes in its README, and the tolerance each is
#: allowed to drift before the build fails.
#:
#: This exists because of what the project claims about everyone else. The
#: whole argument is that a number in a document is worthless unless something
#: refuses to ship when it stops being true, and a README asserting held-out
#: results with nothing checking them would be the same defect wearing this
#: project's name. Tolerances are wide enough to absorb a library version and
#: narrow enough that a real regression cannot pass.
PUBLISHED: dict[str, tuple[float, float]] = {
    "lexical_median_grounded": (0.400, 0.03),
    "statistical_median_grounded": (0.667, 0.03),
    "lexical_recall": (0.483, 0.03),
    "statistical_recall": (0.316, 0.03),
    "lexical_best_acceptance": (0.39, 0.05),
    "lexical_best_detection": (0.12, 0.04),
    # Tolerance is one-sided in effect: a rise is a regression, and a fall to
    # zero is the WIT-013 signature rather than an improvement. The companion
    # check is `control_live_rate` below -- a false-alarm rate of 0% earned by
    # a control that stopped rewording anything would otherwise sail through.
    "statistical_false_alarm": (0.04, 0.04),
    "lexical_false_alarm": (0.04, 0.04),
}


#: Share of consultations on which the false-positive control must actually
#: reword the note. Below this the control is not controlling for anything and
#: its 0% false-alarm rate means nothing. WIT-013 sat at 2%.
MIN_CONTROL_LIVE_RATE = 0.8


def check_control_is_live(
    corpus: PriMock57, consultations: dict[str, Consultation]
) -> list[str]:
    """Verify the false-positive control still changes the notes it is given.

    A control that matches nothing reports zero false alarms, which reads as
    the harness behaving impeccably. That is exactly how WIT-013 survived: the
    number was good, so nobody asked whether it had been earned.
    """
    from witness.prove import SABOTAGES, clean_baseline

    control = next((s for s in SABOTAGES if not s.should_block), None)
    if control is None:
        return ["no false-positive control is defined"]
    live = 0
    for cons in consultations.values():
        base = clean_baseline(cons)
        live += int(control.inject(base, cons) != base)
    rate = live / len(consultations) if consultations else 0.0
    if rate < MIN_CONTROL_LIVE_RATE:
        return [
            f"the false-positive control rewords only {rate:.0%} of notes "
            f"({live}/{len(consultations)}); below {MIN_CONTROL_LIVE_RATE:.0%} it is "
            "not a control and its false-alarm rate is meaningless (WIT-013)"
        ]
    return []


def check_published(report: CrossValReport) -> list[str]:
    """Compare a fresh run against the figures the README prints."""
    bl = report.best(report.lexical_points)
    bs = report.best(report.statistical_points)
    actual = {
        "lexical_median_grounded": report.lexical_profile.summary()["median_grounded"],
        "statistical_median_grounded": report.statistical_profile.summary()[
            "median_grounded"
        ],
        "lexical_recall": report.detection["lexical"].recall,
        "statistical_recall": report.detection["statistical"].recall,
        "lexical_best_acceptance": bl.human_pass_rate,
        "lexical_best_detection": bl.detection,
        "statistical_false_alarm": bs.false_alarm_rate,
        "lexical_false_alarm": bl.false_alarm_rate,
    }
    drift: list[str] = []
    for key, (expected, tol) in PUBLISHED.items():
        got = actual[key]
        if abs(got - expected) > tol:
            drift.append(
                f"{key}: README says {expected:.1%}, this run gives {got:.1%} "
                f"(tolerance {tol:.0%})"
            )
    return drift


def monotone_violations(points: list[RadiusPoint]) -> list[str]:
    """The radius trade is the finding, so its shape is checked, not just its values.

    A published claim of "monotone in both directions" that stops being true
    should fail the build rather than sit in a paragraph nobody re-reads.
    """
    out: list[str] = []
    for a, b in zip(points, points[1:], strict=False):
        if b.median_grounded > a.median_grounded:
            out.append(
                f"grounding rose from floor {a.floor:.2f} to {b.floor:.2f} "
                f"({a.median_grounded:.1%} -> {b.median_grounded:.1%})"
            )
        if b.recall < a.recall:
            out.append(
                f"recall fell from floor {a.floor:.2f} to {b.floor:.2f} "
                f"({a.recall:.1%} -> {b.recall:.1%})"
            )
    return out
