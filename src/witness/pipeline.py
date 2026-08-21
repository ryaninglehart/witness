"""Orchestration: run the checks, decide, publish only if allowed, then re-read.

The ordering here is the part that matters. ``publish`` writes the artifacts and
``verify`` re-reads them from disk afterwards. That looks redundant -- the code
that just wrote a file surely knows what is in it -- and it is not: the whole
premise of this project is that a system's own account of what it did is the
least reliable evidence available. A check that trusts the writer cannot catch
the writer being wrong.

Publication is Write-Audit-Publish with a snapshot. If ``verify`` fails, the
previous build is restored rather than left half-overwritten, because the
failure mode of a partial publish is worse than no publish: downstream readers
get numbers the pipeline has already decided not to stand behind, under a
console line saying nothing was published.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from witness.corpus import Consultation, PriMock57
from witness.gate import Acknowledgement, Gate, GateResult
from witness.graders.lexical import LexicalGrader
from witness.grounding import Grader, NoteGrounding, Verdict
from witness.omission import OmissionReport, Tier, find_omissions
from witness.segment import segment


@dataclass
class RunResult:
    consultation_id: str
    model_id: str
    grounding: NoteGrounding
    omissions: OmissionReport
    gate: GateResult
    published: bool = False
    restored: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "consultation_id": self.consultation_id,
            "model_id": self.model_id,
            "grounding": self.grounding.summary(),
            "omissions": self.omissions.summary(),
            "gate": self.gate.to_dict(),
            "published": self.published,
            "restored_previous_build": self.restored,
            "assertions": [
                {
                    "text": g.assertion.text,
                    "slot": g.assertion.slot.value,
                    "negated": g.assertion.negated,
                    "verdict": g.verdict.value,
                    "rationale": g.rationale,
                    "confidence": g.confidence,
                    "spans": [
                        {
                            "utterance_index": s.utterance_index,
                            "speaker": s.speaker,
                            "start_s": s.start_s,
                            "end_s": s.end_s,
                            "text": s.text,
                        }
                        for s in g.spans
                    ],
                }
                for g in self.grounding.assertions
            ],
            "omitted_facts": [
                {
                    "text": f.text,
                    "tier": f.tier,
                    "basis": f.basis,
                    "speaker": f.speaker,
                    "start_s": f.start_s,
                    "end_s": f.end_s,
                }
                for f in self.omissions.omitted
            ],
        }


class Pipeline:
    def __init__(
        self,
        grader: Grader | None = None,
        gate: Gate | None = None,
        out_dir: Path | None = None,
        max_critical_omissions: int = 0,
    ):
        self.grader = grader or LexicalGrader()
        self.gate = gate or Gate()
        self.out_dir = Path(out_dir or "data/out")
        #: Calibrated alongside the gate thresholds. Zero is the correct value
        #: only if the reference population never omits anything, which no
        #: population of real clinical notes satisfies.
        self.max_critical_omissions = max_critical_omissions

    def run(
        self,
        consultation: Consultation,
        note: str,
        model_id: str,
        acknowledgements: list[Acknowledgement] | None = None,
        publish: bool = True,
    ) -> RunResult:
        grounding = self.grader.ground_note(
            consultation.consultation_id, model_id, segment(note), consultation.utterances
        )
        omissions = find_omissions(consultation, note, model_id)

        # A critical omission is a blocking condition, and it is invisible to
        # the grounding pass by construction -- nothing in the note points at
        # what is missing from it. Folding it in here rather than inside the
        # gate keeps the gate operating on one uniform notion of a finding.
        extra: list[Acknowledgement] = list(acknowledgements or [])
        result = self.gate.evaluate(grounding, extra)
        if omissions.critical_omissions > self.max_critical_omissions:
            from witness.gate import Finding, Severity

            result.findings.append(
                Finding(
                    "critical_omission",
                    Severity.BLOCK,
                    f"{omissions.critical_omissions} salient fact(s) absent from the note, "
                    f"e.g. {omissions.omitted_at(Tier.CRITICAL)[0].text[:60]!r}",
                    omissions.critical_omissions,
                )
            )
            result.published = not result.unacknowledged_blocking

        run = RunResult(
            consultation_id=consultation.consultation_id,
            model_id=model_id,
            grounding=grounding,
            omissions=omissions,
            gate=result,
        )

        if publish and result.published:
            self._publish(run)
        elif publish:
            run.notes.append(
                "gate refused; previous build left serving and nothing was overwritten"
            )
        return run

    # -- write / audit / publish ------------------------------------------

    def _publish(self, run: RunResult) -> None:
        target = self.out_dir / run.consultation_id
        snapshot = self.out_dir / f".{run.consultation_id}.previous"

        # Snapshot before overwriting. Without this a failed verify leaves the
        # rejected build in place, which is worse than publishing nothing.
        if target.exists():
            if snapshot.exists():
                shutil.rmtree(snapshot)
            shutil.copytree(target, snapshot)

        target.mkdir(parents=True, exist_ok=True)
        payload = run.to_dict()
        (target / "report.json").write_text(json.dumps(payload, indent=2))
        (target / "gate.json").write_text(json.dumps(run.gate.to_dict(), indent=2))

        problems = self._verify(target, payload)
        if problems:
            run.notes.extend(problems)
            run.published = False
            if snapshot.exists():
                shutil.rmtree(target)
                shutil.copytree(snapshot, target)
                run.restored = True
                run.notes.append("previous build restored")
            else:
                shutil.rmtree(target)
                run.notes.append("no previous build existed; output removed")
            return

        run.published = True
        if snapshot.exists():
            shutil.rmtree(snapshot)

    def _verify(self, target: Path, expected: dict) -> list[str]:
        """Re-read what was written and check it says what we think it says."""
        problems: list[str] = []
        report = target / "report.json"
        if not report.exists():
            return ["verify: report.json was not written"]
        try:
            got = json.loads(report.read_text())
        except json.JSONDecodeError as exc:
            return [f"verify: report.json is not valid JSON on re-read ({exc})"]

        if got.get("gate", {}).get("published") is not True:
            problems.append("verify: published artifact records a non-published gate decision")
        if got.get("gate", {}).get("ruleset_hash") != expected["gate"]["ruleset_hash"]:
            problems.append("verify: ruleset hash on disk differs from the one just applied")

        # Every supported or inferred claim must carry a span *in the published
        # bytes*, not merely in memory. This is the egress check: it does not
        # trust the serialiser any more than the grader.
        for a in got.get("assertions", []):
            if a["verdict"] in (Verdict.SUPPORTED.value, Verdict.INFERRED.value) and not a["spans"]:
                problems.append(
                    f"verify: published {a['verdict']} claim has no source span "
                    f"({a['text'][:50]!r})"
                )
                break
        return problems


def load_and_run(
    corpus: PriMock57,
    consultation_id: str,
    note: str,
    model_id: str = "unknown",
    **kwargs,
) -> RunResult:
    return Pipeline(**kwargs).run(corpus.load_consultation(consultation_id), note, model_id)
