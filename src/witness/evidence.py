"""Does any existing metric tell you whether a clinical note is safe?

This module exists to test the premise of the whole project before anything is
built on top of it. If ROUGE, BERTScore or editing effort already predicted
clinician-judged critical errors, Witness would be redundant and the honest
thing would be to say so and stop.

The test is possible because PriMock57 ships both halves: 24 automatic metrics
computed by the corpus authors, and 1,425 notes graded by five clinicians for
incorrect statements and omissions, each marked critical or not.

Two numbers matter.

**Rank correlation** asks whether a metric orders notes the way a clinician
would. Spearman is used rather than Pearson because none of these metrics is
linear in error count and we only care about ordering.

**Top-decile contamination** asks the operational question instead: if you
shipped the notes your best metric liked most, how many would contain a
critical error anyway? A metric can correlate respectably and still be useless
as a release gate, which is the distinction that matters when the output is a
medical record.

The comparison is deliberately generous to the incumbent metrics. They are
given their best reference type, and the corpus authors' own computed values
are used rather than reimplementations, so nothing here can be dismissed as a
badly tuned baseline.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats

from witness.corpus import PriMock57

#: Metrics whose sign is inverted in the corpus (stored negative so that
#: "higher is better" holds uniformly). Recorded explicitly rather than
#: inferred from the values, because a sign error here would silently reverse
#: a correlation and produce a confident, wrong conclusion.
_ALREADY_NEGATED = {"Levenshtein dist.", "WER", "MER", "WIL", "WMD"}

#: Reference types the corpus scored each note against. ``human_note`` is the
#: treating clinician's own note -- the only one available before a reviewer
#: has touched the output, and therefore the only one a vendor could actually
#: use as a live quality signal. The others require the post-edit that the
#: metric is supposed to predict the need for, so using them to argue a metric
#: works would be circular.
DEPLOYABLE_REFERENCE = "human_note"


@dataclass
class MetricResult:
    name: str
    reference: str
    n: int
    spearman_rho: float
    p_value: float
    #: Of the notes this metric scored in its top decile, the fraction that
    #: contain at least one clinician-labelled critical error.
    top_decile_contamination: float
    #: Mean critical errors per note in that same decile.
    top_decile_mean_critical: float
    #: Mean critical errors per note across all notes.
    base_mean_critical: float

    @property
    def label(self) -> str:
        return f"{self.reference} | {self.name}"

    @property
    def error_reduction(self) -> float:
        """Proportional drop in mean critical errors from selecting on this metric.

        Contamination alone is misleading in this corpus: 98.4% of notes carry
        at least one critical error, so *any* selection looks contaminated and a
        metric can appear useless when it is merely operating on a saturated
        base rate. This asks the fairer question -- if you kept the notes the
        metric liked best, how much did the error burden actually fall?
        """
        if self.base_mean_critical == 0:
            return 0.0
        return 1.0 - (self.top_decile_mean_critical / self.base_mean_critical)


def _split_metric_key(key: str) -> tuple[str, str] | None:
    if " | " not in key:
        return None
    ref, name = key.split(" | ", 1)
    return ref, name.rstrip("*")


def analyse(
    corpus: PriMock57,
    reference: str = DEPLOYABLE_REFERENCE,
    exclude_human_baseline: bool = True,
) -> list[MetricResult]:
    """Correlate every automatic metric against clinician-labelled criticals."""
    rows = corpus.metric_scores()
    if exclude_human_baseline:
        rows = [r for r in rows if r.get("model_id") != "doctor"]

    criticals = [
        int(r.get("incorrect_critical", 0)) + int(r.get("omissions_critical", 0)) for r in rows
    ]

    metric_keys = sorted(
        {
            k
            for r in rows
            for k in r
            if (parsed := _split_metric_key(k)) is not None and parsed[0] == reference
        }
    )

    results: list[MetricResult] = []
    for key in metric_keys:
        _, name = _split_metric_key(key)
        pairs = [
            (float(r[key]), c)
            for r, c in zip(rows, criticals, strict=True)
            if r.get(key) is not None
        ]
        if len(pairs) < 30:
            continue
        scores = [p[0] for p in pairs]
        errs = [p[1] for p in pairs]
        rho, p = stats.spearmanr(scores, errs)
        contamination, mean_crit = _top_decile(scores, errs)
        results.append(
            MetricResult(
                name=name,
                reference=reference,
                n=len(pairs),
                spearman_rho=float(rho),
                p_value=float(p),
                top_decile_contamination=contamination,
                top_decile_mean_critical=mean_crit,
                base_mean_critical=sum(errs) / len(errs),
            )
        )
    # Sort by how strongly the metric predicts *fewer* errors, which is the
    # direction a quality metric is supposed to run.
    results.sort(key=lambda r: r.spearman_rho)
    return results


def _top_decile(
    scores: Sequence[float], errors: Sequence[int], top_frac: float = 0.10
) -> tuple[float, float]:
    """(contamination, mean critical errors) among the best-scoring notes."""
    n = max(1, int(len(scores) * top_frac))
    best = sorted(zip(scores, errors, strict=True), key=lambda t: -t[0])[:n]
    contamination = sum(1 for _, e in best if e > 0) / len(best)
    mean_critical = sum(e for _, e in best) / len(best)
    return contamination, mean_critical


def post_edit_effort(corpus: PriMock57) -> dict[str, float]:
    """Correlate post-edit time against critical errors.

    This is the closest public proxy for the family of metrics ambient-scribe
    vendors actually report -- "edits per 100 notes", edit distance, a trust
    score derived from how much users change the output. It measures whether
    the clinician bothered to fix the note. Whether it measures whether the
    note was *safe* is the question.

    The correlation here is also partly circular and that must be said: more
    errors cause more editing, so effort trails quality rather than predicting
    it. A vendor cannot know the edit time before shipping the note. What the
    number can honestly show is whether the notes clinicians left alone were
    in fact the safe ones.
    """
    rows = [
        r
        for r in corpus.metric_scores()
        if r.get("model_id") != "doctor" and r.get("time_sec") is not None
    ]
    # Negated so that higher = less editing = "better" by the vendor metric,
    # keeping the sign convention identical to every other metric here.
    effort = [-float(r["time_sec"]) for r in rows]
    errs = [int(r.get("incorrect_critical", 0)) + int(r.get("omissions_critical", 0)) for r in rows]
    rho, p = stats.spearmanr(effort, errs)
    contamination, mean_crit = _top_decile(effort, errs)
    return {
        "n": len(rows),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "least_edited_decile_contamination": contamination,
        "least_edited_decile_mean_critical": mean_crit,
        "base_mean_critical": sum(errs) / len(errs),
        "base_contamination": sum(1 for e in errs if e > 0) / len(errs),
    }


def error_composition(corpus: PriMock57) -> dict[str, int | float]:
    """How the labelled errors divide between fabrication and omission.

    Reported because the split is lopsided in a direction that shapes what a
    detector should be built to do, and most published work measures only the
    smaller half.
    """
    anns = corpus.annotations()
    ci = sum(a.critical_incorrect for a in anns)
    co = sum(a.critical_omissions for a in anns)
    clean = sum(1 for a in anns if a.is_clean)
    return {
        "notes": len(anns),
        "critical_incorrect": ci,
        "critical_omissions": co,
        "omission_share": co / (ci + co) if (ci + co) else 0.0,
        "clean_notes": clean,
        "clean_rate": clean / len(anns) if anns else 0.0,
        "median_critical_per_note": statistics.median(a.critical_total for a in anns),
    }
