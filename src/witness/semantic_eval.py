"""Measure the semantic grader against the two that need no model.

Everything else in this project runs in ninety seconds on a laptop with no
network. This does not: it makes one model call per batch of twelve claims, and
a full pass is tens of minutes on a local 7B. So it is a separate command
rather than part of `make check`, and its results are recorded to cassettes so
that the second run is free and the numbers are reproducible by someone who
never installs a model at all.

Three rules hold this to the same standard as the rest of the harness.

**The comparison is on identical inputs.** The same consultations, the same
sampled annotations, the same matcher. A grader evaluated on a different subset
than the baseline it claims to beat is not evidence of anything, and with a
sampled subset that error is very easy to make by accident.

**Thresholds are still held out.** The semantic grader learns nothing from the
corpus so it cannot leak into its own lexicon, but the gate can still be
calibrated on the notes it judges. The fold structure from :mod:`witness.crossval`
applies unchanged.

**Replay is disclosed.** A figure produced from a cassette and a figure
produced from a live model are different claims. The report says which, per
run, and refuses to average them into one number silently.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from witness.corpus import Consultation, PriMock57
from witness.crossval import fit_folds, folds
from witness.graders.lexical import LexicalGrader
from witness.graders.semantic import SemanticGrader
from witness.graders.statistical import StatisticalGrader
from witness.grounding import Grader, NoteGrounding, Verdict
from witness.matching import MatchResult, match_note
from witness.providers import Completion, Provider
from witness.segment import segment


@dataclass
class GraderResult:
    name: str
    median_grounded: float
    median_unsupported: float
    recall: float
    precision: float
    f1: float
    notes_graded: int
    annotations_graded: int
    #: Model calls that came from a recorded cassette rather than a live model.
    replayed_calls: int = 0
    live_calls: int = 0
    seconds: float = 0.0


@dataclass
class SemanticReport:
    corpus: str
    consultations: int
    annotations_per_consultation: int
    results: list[GraderResult] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "\n  Semantic grader vs the graders that need no model",
            "  " + "=" * 82,
            f"  {self.consultations} consultations, "
            f"{self.annotations_per_consultation} generated note(s) each.",
            "  Identical inputs for all three. Thresholds held out by fold.",
            "",
            f"  {'grader':<46}{'grounded':>10}{'recall':>9}{'prec':>8}{'f1':>8}",
            "  " + "-" * 82,
        ]
        for r in self.results:
            lines.append(
                f"  {r.name[:45]:<46}{r.median_grounded:>9.1%}{r.recall:>9.1%}"
                f"{r.precision:>8.1%}{r.f1:>8.1%}"
            )
        for r in self.results:
            if r.live_calls or r.replayed_calls:
                lines += [
                    "",
                    f"  {r.name}: {r.live_calls} live call(s), "
                    f"{r.replayed_calls} replayed, {r.seconds:.0f}s",
                ]
        lines += [
            "",
            "  'grounded' is the median share of a clinician-written note the",
            "  grader can witness. 'recall' is against the critical errors the",
            "  corpus clinicians flagged in generated notes. The two move in",
            "  opposite directions for every grader measured so far; whether a",
            "  model breaks that trade is the entire question.",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "corpus": self.corpus,
                "consultations": self.consultations,
                "annotations_per_consultation": self.annotations_per_consultation,
                "results": [asdict(r) for r in self.results],
            },
            indent=2,
        )


class _CountingProvider:
    """Wraps a provider to count live calls against replayed ones.

    The report states that a figure from a cassette and a figure from a live
    model are different claims. Stating it and then printing zeros for both
    would make the disclosure decorative, which is the failure mode this
    project keeps finding elsewhere.
    """

    def __init__(self, inner: Provider):
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.live = 0
        self.replayed = 0

    def available(self) -> bool:
        return self.inner.available()

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> Completion:
        out = self.inner.complete(system, user, max_tokens)
        if out.replayed:
            self.replayed += 1
        else:
            self.live += 1
        return out


def _profile(groundings: list[NoteGrounding]) -> tuple[float, float]:
    grounded, unsup = [], []
    for g in groundings:
        if not g.n:
            continue
        grounded.append((g.count(Verdict.SUPPORTED) + g.count(Verdict.INFERRED)) / g.n)
        unsup.append(g.unsupported_rate)
    return (
        statistics.median(grounded) if grounded else 0.0,
        statistics.median(unsup) if unsup else 0.0,
    )


def _sample_annotations(corpus: PriMock57, per_consultation: int) -> dict[str, list]:
    """A fixed slice of the annotation set, chosen without randomness.

    Sorted by model id and taken from the front, so the same consultations
    contribute the same notes on every run and on every machine. Sampling with
    an RNG would make two runs incomparable, which for a measurement that takes
    an hour is a costly kind of irreproducibility.
    """
    by_cid: dict[str, list] = {}
    for a in corpus.annotations():
        by_cid.setdefault(a.consultation_id, []).append(a)
    return {
        cid: sorted(anns, key=lambda a: (a.model_id, a.generated_note))[:per_consultation]
        for cid, anns in by_cid.items()
    }


def evaluate(
    corpus: PriMock57,
    graders: dict[str, Grader] | None = None,
    provider: Provider | None = None,
    consultation_limit: int | None = None,
    annotations_per_consultation: int = 1,
    k: int = 5,
    progress: bool = True,
) -> SemanticReport:
    """Run every grader over identical inputs and report them side by side."""
    ids = corpus.consultation_ids()
    if consultation_limit:
        ids = ids[:consultation_limit]
    consultations: dict[str, Consultation] = {c: corpus.load_consultation(c) for c in ids}
    sampled = _sample_annotations(corpus, annotations_per_consultation)

    fold_list = folds(corpus.consultation_ids(), k)
    models = fit_folds(corpus, fold_list, {c: corpus.load_consultation(c)
                                           for c in corpus.consultation_ids()})
    stat_for: dict[str, StatisticalGrader] = {}
    for fm in models:
        grader = StatisticalGrader(fm.model)
        for cid in fm.fold.test:
            stat_for[cid] = grader

    lineup: dict[str, Grader] = dict(graders or {})
    if not lineup:
        lineup = {"lexical": LexicalGrader()}
        if provider is not None:
            lineup["semantic"] = SemanticGrader(provider)

    # Count model traffic per grader. Only the semantic one has any.
    counters: dict[str, _CountingProvider] = {}
    for label, grader in list(lineup.items()):
        if isinstance(grader, SemanticGrader):
            counter = _CountingProvider(grader.provider)
            counters[label] = counter
            lineup[label] = SemanticGrader(counter, batch_size=grader.batch_size)

    results: list[GraderResult] = []
    # The statistical grader varies by fold, so it is evaluated by hand rather
    # than through the same loop.
    for label, grader in [("statistical", None), *lineup.items()]:
        started = time.time()
        groundings: list[NoteGrounding] = []
        totals = [0, 0, 0, 0, 0]
        n_ann = 0
        for n, cid in enumerate(ids, 1):
            c = consultations[cid]
            g = grader or stat_for[cid]
            if progress:
                # Overwrite in place on a terminal; one line per consultation
                # when piped to a file, because a log full of carriage returns
                # is unreadable and this run is long enough to be logged.
                tty = sys.stdout.isatty()
                print(
                    f"    {label:<12} {n:>3}/{len(ids)}  {cid}",
                    end="\r" if tty else "\n",
                    flush=True,
                )
            groundings.append(
                g.ground_note(cid, "clinician", segment(c.clinician_note), c.utterances)
            )
            for a in sampled.get(cid, []):
                n_ann += 1
                ga = g.ground_note(cid, a.model_id, segment(a.generated_note), c.utterances)
                for i, v in enumerate(match_note(a, ga)):
                    totals[i] += v
        if progress and sys.stdout.isatty():
            print(" " * 70, end="\r")
        m = MatchResult(*totals)
        med_g, med_u = _profile(groundings)
        name = getattr(g, "name", label)
        counter = counters.get(label)
        results.append(
            GraderResult(
                name=f"{label} ({name})" if name != label else label,
                replayed_calls=counter.replayed if counter else 0,
                live_calls=counter.live if counter else 0,
                median_grounded=med_g,
                median_unsupported=med_u,
                recall=m.recall,
                precision=m.precision,
                f1=m.f1,
                notes_graded=len(groundings),
                annotations_graded=n_ann,
                seconds=time.time() - started,
            )
        )

    return SemanticReport(
        corpus="PriMock57",
        consultations=len(ids),
        annotations_per_consultation=annotations_per_consultation,
        results=results,
    )


def write_report(report: SemanticReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    return path
