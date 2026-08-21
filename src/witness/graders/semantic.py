"""A grader that reads for meaning, because term overlap demonstrably cannot.

This module exists because of a measurement, not a hunch. The lexical grader
was swept across every calibration percentile from 0.50 to 0.99, and at no
point did it both accept notes written by clinicians and catch a single
injected defect. Its best operating point was 44% of human notes passing at 29%
of defects caught. The cause is visible in the calibration table: **it cannot
ground 60% of a median human-written note**, because clinicians write
"3/7 hx of diarrhoea, mainly watery" and term matching has no idea that is the
same fact as "it's been about three days, it's quite runny".

So the failure is semantic, and no threshold fixes it.

The prompt does three things the lexical grader cannot:

**It separates inference from fabrication.** "She's up and about most of the day
but tires easily" supports "ECOG 1". A term matcher sees no shared vocabulary
and calls it a hallucination. The model is asked to distinguish the two and to
name the reasoning when it claims inference.

**It demands a span.** Every SUPPORTED or INFERRED verdict must cite the
utterance index it relied on. The response is rejected if it does not, which
makes an uncheckable citation impossible to produce rather than merely
discouraged.

**It is told to prefer UNSUPPORTED when unsure.** A grader that resolves doubt
towards "fine" is worse than no grader, because it converts uncertainty into
false assurance. The asymmetry is deliberate and it costs precision.

Runs against a local model by default. Nothing here needs an API key.
"""

from __future__ import annotations

import json
import re

from witness.corpus import Utterance
from witness.grounding import (
    Assertion,
    GroundedAssertion,
    NoteGrounding,
    Span,
    Verdict,
)
from witness.providers import Provider, ProviderError

SYSTEM = """You are a clinical documentation auditor. You are given a transcript \
of a doctor-patient consultation, numbered by utterance, and a list of claims \
taken from a note generated from that consultation.

For each claim, decide exactly one verdict:

SUPPORTED    - the transcript states this. Cite the utterance number.
INFERRED     - the transcript does not state it, but a clinician would \
reasonably write it from what was said (e.g. "up and about most of the day but \
tires easily" -> "ECOG 1"; "loose watery stool six times a day" -> \
"diarrhoea"). Cite the utterance you reasoned from AND name the inference.
UNSUPPORTED  - nothing in the transcript supports it and no clinical reasoning \
derives it.
CONTRADICTED - the transcript says the opposite. Cite the utterance.

Rules you must follow:
- SUPPORTED and INFERRED REQUIRE a real utterance number. If you cannot point \
to one, the verdict is UNSUPPORTED. Never invent an utterance number.
- Abbreviations are normal clinical shorthand, not fabrication. "3/7" means \
three days, "PMH" past medical history, "nil" none, "LLQ" left lower quadrant, \
"o/e" on examination, "bd" twice daily.
- Pay close attention to negation. "No blood in stool" is SUPPORTED only if the \
patient denied blood. If the patient reported blood, it is CONTRADICTED.
- Patients correct themselves. If an early utterance conflicts with a later \
one, the later one governs.
- When you are unsure, choose UNSUPPORTED. Do not resolve doubt in favour of \
the note.

Reply with JSON only, no prose:
{"verdicts":[{"i":<claim index>,"v":"SUPPORTED|INFERRED|UNSUPPORTED|CONTRADICTED",\
"u":<utterance number or null>,"why":"<short reason>"}]}"""

GRADER_VERSION = "semantic/1.0"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class SemanticGrader:
    """Ground assertions with a language model, span required."""

    def __init__(self, provider: Provider, batch_size: int = 12):
        self.provider = provider
        #: Claims per request. Small batches keep the transcript in context and
        #: stop a long note from pushing the early claims out of attention;
        #: large batches are cheaper. Twelve is a compromise, not a finding.
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{GRADER_VERSION}+{self.provider.name}/{self.provider.model}"

    def ground_note(
        self,
        consultation_id: str,
        model_id: str,
        assertions: list[Assertion],
        utterances: list[Utterance],
    ) -> NoteGrounding:
        graded: list[GroundedAssertion] = []
        transcript = "\n".join(
            f"[{i}] {u.speaker}: {u.text}" for i, u in enumerate(utterances)
        )
        for start in range(0, len(assertions), self.batch_size):
            batch = assertions[start : start + self.batch_size]
            graded.extend(self._grade_batch(batch, transcript, utterances))
        return NoteGrounding(consultation_id, model_id, self.name, graded)

    def _grade_batch(
        self, batch: list[Assertion], transcript: str, utterances: list[Utterance]
    ) -> list[GroundedAssertion]:
        claims = "\n".join(f"{a.index}. {a.text}" for a in batch)
        user = f"TRANSCRIPT\n{transcript}\n\nCLAIMS\n{claims}"
        try:
            out = self.provider.complete(SYSTEM, user, max_tokens=2048)
        except ProviderError as exc:
            # A provider failure must not silently become a clean bill of
            # health. Every claim in the batch is returned UNSUPPORTED with the
            # reason attached, so the gate blocks and the report says why.
            return [
                GroundedAssertion(
                    a, Verdict.UNSUPPORTED, [], f"grader unavailable: {exc}", 0.0, self.name
                )
                for a in batch
            ]

        parsed = self._parse(out.text)
        by_index = {v.get("i"): v for v in parsed}
        graded: list[GroundedAssertion] = []
        for a in batch:
            v = by_index.get(a.index)
            graded.append(self._to_verdict(a, v, utterances, out.replayed))
        return graded

    @staticmethod
    def _parse(text: str) -> list[dict]:
        """Recover the verdict list from whatever the model actually emitted.

        Every branch here returns ``[]`` rather than raising, and an empty list
        becomes UNSUPPORTED for every claim in the batch. That asymmetry is the
        point: a grader that cannot read its own model's output must block, not
        pass. A 7B model asked for JSON will wrap it in prose, truncate it
        mid-object, or answer ``{"verdicts": null}`` -- the last of which
        crashed this method the first time it was run against a scripted
        provider, because ``.get("verdicts", [])`` returns ``None`` when the
        key is present and null.
        """
        m = _JSON_RE.search(text)
        if not m:
            return []
        try:
            doc = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(doc, dict):
            return []
        verdicts = doc.get("verdicts")
        if not isinstance(verdicts, list):
            return []
        return [v for v in verdicts if isinstance(v, dict)]

    def _to_verdict(
        self,
        assertion: Assertion,
        raw: dict | None,
        utterances: list[Utterance],
        replayed: bool,
    ) -> GroundedAssertion:
        note = " (replayed)" if replayed else ""
        if raw is None:
            return GroundedAssertion(
                assertion, Verdict.UNSUPPORTED, [], f"grader returned no verdict{note}",
                0.0, self.name,
            )
        try:
            verdict = Verdict(str(raw.get("v", "")).strip().upper())
        except ValueError:
            return GroundedAssertion(
                assertion, Verdict.UNSUPPORTED, [],
                f"grader returned an unrecognised verdict {raw.get('v')!r}{note}",
                0.0, self.name,
            )

        spans: list[Span] = []
        u = raw.get("u")
        if isinstance(u, int) and 0 <= u < len(utterances):
            utt = utterances[u]
            spans = [Span(u, utt.speaker, utt.start_s, utt.end_s, utt.text)]
        elif u is not None:
            # A cited index that does not exist is a fabricated citation, which
            # is worse than no citation: it looks like evidence. Downgrade and
            # say so rather than dropping the span quietly.
            return GroundedAssertion(
                assertion, Verdict.UNSUPPORTED, [],
                f"grader cited utterance {u!r}, which does not exist{note}",
                0.0, self.name,
            )

        if verdict in (Verdict.SUPPORTED, Verdict.INFERRED) and not spans:
            return GroundedAssertion(
                assertion, Verdict.UNSUPPORTED, [],
                f"grader claimed {verdict.value} without citing an utterance{note}",
                0.0, self.name,
            )

        return GroundedAssertion(
            assertion, verdict, spans,
            f"{str(raw.get('why', '')).strip()[:200]}{note}",
            0.75, self.name,
        )
