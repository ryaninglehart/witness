"""Loading PriMock57 -- and the reason this project has ground truth at all.

PriMock57 (Korfiatis et al., ACL 2022, CC BY 4.0) is usually cited for its 57
mock consultations with real audio. The part that matters here is the folder
almost nobody uses: ``human_eval_data``.

Five clinicians each reviewed model-generated notes for every consultation and
post-edited them, producing 1,425 annotated rows. Each row carries the errors
they found, split two ways that no automatic metric captures:

* **incorrect statements** -- the note asserts something the encounter did not
* **omissions** -- the encounter contained something the note dropped

and each individual finding is marked ``!`` for critical or ``-`` for
non-critical *by the clinician who found it*.

That is a labelled test set for an error detector, built by the people whose
judgement the detector is trying to approximate. Everything in Witness is
validated against it.

One honest caveat, stated here because it belongs next to the loader rather
than in a footnote: the model notes in this corpus were generated in 2022. They
are worse than what a 2026 system produces. This corpus is not evidence about
the state of the art -- it is a graded set of notes spanning a wide quality
range, which is exactly what you want when the thing under test is the *grader*.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------

CRITICAL = "CRITICAL"
NONCRITICAL = "NONCRITICAL"
UNMARKED = "UNMARKED"

#: The annotators prefixed each finding with ``!`` or ``-``. A finding with
#: neither is real data, not a parse failure -- an annotator who wrote a note
#: without grading it. It is kept as UNMARKED rather than silently promoted or
#: demoted, because guessing which they meant would be inventing ground truth.
_SEVERITY_PREFIX = {"!": CRITICAL, "-": NONCRITICAL}


@dataclass(frozen=True)
class Finding:
    """One error a clinician found in one note."""

    severity: str
    text: str


@dataclass(frozen=True)
class Utterance:
    """One turn of speech, with the timings that make provenance possible.

    ``start_s``/``end_s`` come from a human transcriber who listened to the
    audio, not from forced alignment. When Witness cites this utterance as the
    source of a claim, the citation points at a real moment someone can play.
    """

    speaker: str
    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Annotation:
    """One clinician's review of one model's note for one consultation."""

    consultation_id: str
    model_id: str
    evaluator: str
    generated_note: str
    human_note: str
    edited_note: str
    post_edit_time_s: int | None
    incorrect: list[Finding] = field(default_factory=list)
    omissions: list[Finding] = field(default_factory=list)

    @property
    def critical_incorrect(self) -> int:
        return sum(1 for f in self.incorrect if f.severity == CRITICAL)

    @property
    def critical_omissions(self) -> int:
        return sum(1 for f in self.omissions if f.severity == CRITICAL)

    @property
    def critical_total(self) -> int:
        return self.critical_incorrect + self.critical_omissions

    @property
    def is_clean(self) -> bool:
        """No critical error of either kind.

        This is the label Witness is ultimately trying to predict, and the one
        a clinician actually cares about: not "how similar is this note to a
        reference" but "is there anything in here that could hurt someone".
        """
        return self.critical_total == 0


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_findings(cell: str | None) -> list[Finding]:
    """Split an annotator's free-text findings cell into graded findings.

    The annotators wrote one finding per line, prefixed with ``!`` or ``-``.
    They were not machines about it: blank lines, trailing whitespace and the
    occasional unprefixed line all occur. Everything here is deliberate about
    not tidying the data into a cleaner shape than it has.
    """
    if not cell:
        return []
    out: list[Finding] = []
    for line in cell.splitlines():
        s = line.strip()
        if not s:
            continue
        sev = _SEVERITY_PREFIX.get(s[0])
        if sev is not None:
            text = s[1:].strip()
            # A bare "!" with nothing after it carries no information about
            # what was wrong. Counting it would inflate the error count with a
            # finding nobody can act on or verify.
            if text:
                out.append(Finding(sev, text))
        else:
            out.append(Finding(UNMARKED, s))
    return out


#: The human-evaluation CSV and the media files disagree about zero-padding:
#: the CSV writes ``day1_consultation010`` where the files are named
#: ``day1_consultation10``. Consultations 1-9 happen to agree, so the mismatch
#: only bites from 10 onwards -- which is why it survives a spot-check of the
#: first few rows and then silently drops 19% of the corpus.
_ID_PADDING_RE = re.compile(r"^(day\d+_consultation)0(\d{2,})$")


def canonical_id(consultation_id: str) -> str:
    """Map a human-eval consultation id onto the media file naming.

    Witness resolves this rather than skipping unmatched rows. Skipping was the
    original behaviour and it cost 220 of 1,425 annotations without raising
    anything -- see docs/INCIDENTS.md, WIT-001.
    """
    m = _ID_PADDING_RE.match(consultation_id)
    return f"{m.group(1)}{m.group(2)}" if m else consultation_id


_TEXTGRID_INTERVAL = re.compile(
    r"intervals\s*\[\d+\]:\s*"
    r"xmin\s*=\s*([\d.]+)\s*"
    r"xmax\s*=\s*([\d.]+)\s*"
    r'text\s*=\s*"((?:[^"]|"")*)"',
    re.DOTALL,
)


def parse_textgrid(path: Path, speaker: str) -> list[Utterance]:
    """Parse a Praat TextGrid into timed utterances.

    Only the interval tier is read. Empty intervals are silence between turns
    and are dropped -- but note that they are *where Whisper hallucinates*
    (Koenecke et al., FAccT 2024, found hallucination correlates with non-vocal
    duration), so :mod:`witness.asr` re-derives them rather than relying on
    this function to preserve them.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: list[Utterance] = []
    for m in _TEXTGRID_INTERVAL.finditer(raw):
        text = m.group(3).replace('""', '"').strip()
        if not text:
            continue
        out.append(Utterance(speaker, float(m.group(1)), float(m.group(2)), text))
    return out


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@dataclass
class Consultation:
    """One encounter: the speech, and what the treating clinician wrote."""

    consultation_id: str
    utterances: list[Utterance]
    presenting_complaint: str
    clinician_note: str
    #: Terms the treating clinician marked as important in their own note.
    #: A salience signal authored at the point of care, which is a much better
    #: basis for "what mattered here" than any heuristic applied afterwards.
    highlights: list[str]

    def transcript_text(self) -> str:
        return "\n".join(f"[{u.speaker}] {u.text}" for u in self.utterances)

    @property
    def duration_s(self) -> float:
        return max((u.end_s for u in self.utterances), default=0.0)


class PriMock57:
    """Reader for a PriMock57 checkout.

    Audio is optional on purpose. The transcripts, notes and human evaluation
    live in git; the .wav files live in git-LFS and may not be present. Every
    part of Witness that does not strictly need audio works without it, so the
    harness stays runnable for a reviewer who did not want to pull 1.8 GB.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "human_eval_data").is_dir():
            raise FileNotFoundError(
                f"{self.root} does not look like a PriMock57 checkout "
                "(no human_eval_data/). Run: make corpus"
            )

    # -- consultations ----------------------------------------------------

    def consultation_ids(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "notes").glob("*.json"))

    def load_consultation(self, cid: str) -> Consultation:
        note = json.loads((self.root / "notes" / f"{cid}.json").read_text())
        utterances: list[Utterance] = []
        for speaker in ("doctor", "patient"):
            tg = self.root / "transcripts" / f"{cid}_{speaker}.TextGrid"
            if tg.exists():
                utterances.extend(parse_textgrid(tg, speaker))
        # Interleaving by start time reconstructs the conversation. The corpus
        # records each speaker on a separate channel, which is why speaker
        # attribution here is ground truth rather than diarization output --
        # worth knowing when comparing against a single-channel pipeline.
        utterances.sort(key=lambda u: (u.start_s, u.speaker))
        return Consultation(
            consultation_id=cid,
            utterances=utterances,
            presenting_complaint=note.get("presenting_complaint", ""),
            clinician_note=note.get("note", ""),
            highlights=list(note.get("highlights", [])),
        )

    def require_consultations(self, ids: set[str]) -> None:
        """Raise unless every id resolves to a consultation on disk.

        Called by anything that joins annotations to transcripts. A join that
        silently drops its right-hand side produces a smaller, cleaner-looking
        dataset and a number nobody can reproduce, so the failure is made loud
        and early rather than left to be noticed in a summary statistic.
        """
        known = set(self.consultation_ids())
        missing = sorted(ids - known)
        if missing:
            raise KeyError(
                f"{len(missing)} consultation id(s) in the human evaluation do not "
                f"resolve to files on disk: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )

    def audio_path(self, cid: str, speaker: str) -> Path | None:
        p = self.root / "audio" / f"{cid}_{speaker}.wav"
        if not p.exists():
            return None
        # An LFS pointer is a ~130-byte text file that is *present* but is not
        # audio. Treating it as audio produces a baffling downstream failure in
        # the decoder, so it is caught here where the cause is obvious.
        if p.stat().st_size < 1024:
            return None
        return p

    def has_audio(self) -> bool:
        return any(
            self.audio_path(cid, "doctor") is not None for cid in self.consultation_ids()
        )

    # -- human evaluation -------------------------------------------------

    def annotations(self, include_human_baseline: bool = False) -> list[Annotation]:
        """The 1,425 clinician reviews.

        ``include_human_baseline`` controls whether the rows where the "model"
        is another clinician are returned. They are excluded by default: they
        are a different question (how good are human notes) and mixing them
        into a model-error rate would flatter the models.
        """
        path = self.root / "human_eval_data" / "results.csv"
        out: list[Annotation] = []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                model = row["Model"]
                if model == "doctor" and not include_human_baseline:
                    continue
                try:
                    t = int(row["Post-edit time"])
                except (ValueError, TypeError):
                    t = None
                out.append(
                    Annotation(
                        consultation_id=canonical_id(row["Consultation"]),
                        model_id=model,
                        evaluator=row["Evaluator"],
                        generated_note=row.get("Model Note", "") or "",
                        human_note=row.get("Evaluator Note", "") or "",
                        edited_note=row.get("Post-edited note", "") or "",
                        post_edit_time_s=t,
                        incorrect=parse_findings(row.get("Incorrect statements")),
                        omissions=parse_findings(row.get("Omissions")),
                    )
                )
        return out

    def metric_scores(self) -> list[dict]:
        """The 24 automatic metrics the PriMock57 authors computed per note.

        Shipped with the corpus, against several reference types. Witness does
        not recompute them -- using the corpus authors' own numbers removes any
        suspicion that the comparison in :mod:`witness.evidence` was tuned.
        """
        path = self.root / "human_eval_data" / "metric-scores.json"
        return json.loads(path.read_text())
