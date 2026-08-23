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
    #: Share of model responses the parser could make nothing of. Every claim
    #: in an unparsed batch becomes UNSUPPORTED, so a high rate here produces a
    #: table that looks like a measurement of the grader and is a measurement
    #: of the parser. See WIT-016.
    unparsed_rate: float = 0.0
    #: The same figures over only the notes where every batch was read. The
    #: contaminated notes are not noise -- they bias the two headline numbers
    #: in *opposite* directions, so the full-set row cannot be corrected by
    #: reasoning about it and has to be recomputed without them.
    clean_notes: int = 0
    clean_median_grounded: float = 0.0
    clean_recall: float = 0.0
    clean_precision: float = 0.0


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
        dropped = [
            r for r in self.results if r.clean_notes and r.clean_notes < r.notes_graded
        ]
        if dropped:
            lines += [
                "",
                "  Recomputed over only the notes where every batch was read",
                "  " + "-" * 82,
                "  An unreadable batch refuses every claim in it, which pushes grounding",
                "  DOWN and detection UP. The two biases point opposite ways, so the row",
                "  above cannot be corrected by argument -- only by dropping those notes.",
                "  All three graders are rescored on the same surviving notes; comparing",
                "  a clean subset against a full set would be a different corpus.",
                "",
                f"  {'grader':<46}{'notes':>8}{'grounded':>10}{'recall':>9}{'prec':>8}",
            ]
            for r in dropped:
                lines.append(
                    f"  {r.name[:45]:<46}{r.clean_notes:>4}/{r.notes_graded:<3}"
                    f"{r.clean_median_grounded:>10.1%}{r.clean_recall:>9.1%}"
                    f"{r.clean_precision:>8.1%}"
                )
            lines += [
                "",
                "  This subset is not a random sample of the corpus. The notes that",
                "  survive are the ones the model stayed on-format for, and if it drops",
                "  format on the harder encounters these figures are optimistic. It is a",
                "  better estimate than the contaminated row, not a clean one.",
            ]
        flagged = [(r, degenerate_reasons(r)) for r in self.results]
        flagged = [(r, why) for r, why in flagged if why]
        if flagged:
            lines += ["", "  NOT A MEASUREMENT", "  " + "-" * 82]
            for r, why in flagged:
                lines.append(f"  {r.name}")
                for reason in why:
                    lines.append(f"    - {reason}")
            lines += [
                "",
                "  Rows above are printed so the failure is visible, not because they",
                "  mean anything. Fix the pipeline and re-run; cassettes make the",
                "  second run free.",
            ]
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


#: A grader that witnesses nothing flags everything, so its recall is 100% by
#: construction and its precision is the base rate. That is the signature of a
#: broken pipeline, not a result, and it is what a live qwen2.5 produced when
#: the parser was discarding three responses in four.
def degenerate_reasons(r: GraderResult) -> list[str]:
    """Why this row is not a measurement. Empty means it is one."""
    out: list[str] = []
    if r.notes_graded and r.median_grounded == 0.0:
        out.append(
            "grounded 0.0% -- the grader witnessed nothing in any note, so every "
            "claim was refused"
        )
    if r.annotations_graded and r.recall >= 0.999:
        out.append(
            f"recall {r.recall:.1%} -- flagging everything makes recall 100% by "
            "construction; it is not detection"
        )
    if r.unparsed_rate > 0.2:
        out.append(
            f"{r.unparsed_rate:.0%} of model responses were unreadable, and every "
            "claim in those batches was scored UNSUPPORTED"
        )
    return out


def _was_contaminated(grader: object, consultation_id: str, model_id: str) -> bool:
    """Did any batch for this note fail to parse?

    Only the semantic grader can be contaminated -- the offline graders read no
    model output -- so for anything else this is always False and the clean
    subset is the full set.
    """
    notes = getattr(grader, "unparsed_notes", None)
    return bool(notes) and (consultation_id, model_id) in notes


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

    # Per-note results are kept rather than accumulated, because the clean
    # subset is not known until every grader has run: it is defined by which
    # notes the *semantic* grader could read, and the offline graders have to
    # be scored on that same subset or the comparison is between different
    # corpora. Aggregating twice at the end is the only way to get that right.
    per_note: dict[str, dict[str, tuple]] = {}
    profiles: dict[str, dict[str, NoteGrounding]] = {}
    meta: dict[str, tuple[str, object]] = {}
    contaminated: set[str] = set()

    for label, grader in [("statistical", None), *lineup.items()]:
        started = time.time()
        profiles[label] = {}
        per_note[label] = {}
        for n, cid in enumerate(ids, 1):
            c = consultations[cid]
            g = grader or stat_for[cid]
            if progress:
                tty = sys.stdout.isatty()
                print(
                    f"    {label:<12} {n:>3}/{len(ids)}  {cid}",
                    end="\r" if tty else "\n",
                    flush=True,
                )
            profiles[label][cid] = g.ground_note(
                cid, "clinician", segment(c.clinician_note), c.utterances
            )
            if _was_contaminated(g, cid, "clinician"):
                contaminated.add(cid)
            rows = [0, 0, 0, 0, 0]
            for a in sampled.get(cid, []):
                ga = g.ground_note(cid, a.model_id, segment(a.generated_note), c.utterances)
                if _was_contaminated(g, cid, a.model_id):
                    contaminated.add(cid)
                for i, v in enumerate(match_note(a, ga)):
                    rows[i] += v
            per_note[label][cid] = tuple(rows)
        if progress and sys.stdout.isatty():
            print(" " * 70, end="\r")
        meta[label] = (getattr(g, "name", label), g, time.time() - started)

    clean_ids = [cid for cid in ids if cid not in contaminated]

    def _aggregate(label: str, subset: list[str]) -> tuple[float, float, MatchResult]:
        med_g, med_u = _profile([profiles[label][c] for c in subset])
        totals = [0, 0, 0, 0, 0]
        for c in subset:
            for i, v in enumerate(per_note[label][c]):
                totals[i] += v
        return med_g, med_u, MatchResult(*totals)

    results: list[GraderResult] = []
    for label in ["statistical", *lineup]:
        name, grader_obj, seconds = meta[label]
        med_g, med_u, m = _aggregate(label, list(ids))
        clean_g, _, cm = _aggregate(label, clean_ids)
        counter = counters.get(label)
        n_ann = sum(len(sampled.get(c, [])) for c in ids)
        results.append(
            GraderResult(
                name=f"{label} ({name})" if name != label else label,
                median_grounded=med_g,
                median_unsupported=med_u,
                recall=m.recall,
                precision=m.precision,
                f1=m.f1,
                notes_graded=len(ids),
                annotations_graded=n_ann,
                replayed_calls=counter.replayed if counter else 0,
                live_calls=counter.live if counter else 0,
                unparsed_rate=(
                    grader_obj.unparsed_rate
                    if isinstance(grader_obj, SemanticGrader)
                    else 0.0
                ),
                clean_notes=len(clean_ids),
                clean_median_grounded=clean_g,
                clean_recall=cm.recall,
                clean_precision=cm.precision,
                seconds=seconds,
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


# -- published-figure gate ------------------------------------------------

#: The semantic figures this project publishes, with tolerances. Enforced by
#: `make semantic` exactly as the offline figures are enforced by
#: `make crossval`. Replay makes this free: the 148 recorded responses ship
#: with the repository, so a reviewer re-derives every number here without a
#: model, a key or a network.
#: WITHDRAWN pending re-measurement. The figures that were here -- 38 clean
#: notes, 82.1% grounded, 69.2% recall -- were produced under a segmentation
#: defect that silently dropped 143 clinical claims before grading (WIT-018).
#: Fixing it changes the claims put to the model, which changes the prompts,
#: which invalidates 137 of the 164 recorded responses. They cannot be
#: re-derived by replay and are not restated here.
#:
#: An empty dict means `make semantic` enforces nothing. That is correct for
#: now and must not become permanent: refill it from a live run on the fixed
#: segmentation, and restore the README section at the same time.
PUBLISHED_SEMANTIC: dict[str, tuple[float, float]] = {}


def check_published(report: SemanticReport) -> list[str]:
    """Compare a replayed run against the figures the README prints."""
    by_label = {r.name.split(" (")[0]: r for r in report.results}
    sem, lex = by_label.get("semantic"), by_label.get("lexical")
    if sem is None or lex is None:
        return []
    actual = {
        "clean_notes": float(sem.clean_notes),
        "semantic_clean_grounded": sem.clean_median_grounded,
        "semantic_clean_recall": sem.clean_recall,
        "lexical_clean_grounded": lex.clean_median_grounded,
        "lexical_clean_recall": lex.clean_recall,
        "semantic_unparsed_rate": sem.unparsed_rate,
    }
    drift: list[str] = []
    for key, (expected, tol) in PUBLISHED_SEMANTIC.items():
        got = actual[key]
        if abs(got - expected) > tol:
            drift.append(f"{key}: README says {expected}, this run gives {got:.4g}")
    return drift
