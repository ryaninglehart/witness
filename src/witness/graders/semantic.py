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



class SemanticGrader:
    """Ground assertions with a language model, span required."""

    #: Appended to the prompt on a retry. Deliberately blunt: the first
    #: attempt already carried a full format specification and the model
    #: ignored it, so restating the same thing politely is not a retry
    #: strategy.
    RETRY_NUDGE = (
        "\n\nYour previous reply could not be read. Reply with the JSON array "
        "only. No preamble, no explanation, no prose. Begin with [ and end "
        "with ]."
    )

    def __init__(self, provider: Provider, batch_size: int = 12, retries: int = 1):
        self.provider = provider
        #: Extra attempts when a response yields no verdict at all. One is
        #: enough to be worth having and few enough not to hide a provider
        #: that is simply broken -- and every attempt is still counted, so a
        #: high retry rate stays visible instead of being smoothed away.
        self.retries = retries
        #: Batches sent, and batches whose response yielded no verdict at all.
        #: Counted because the alternative is what happened in WIT-016: 74% of
        #: responses were discarded by the parser, every claim in them became
        #: UNSUPPORTED, and the resulting table looked like a measurement. A
        #: grader that cannot read its model has to say so out loud -- blocking
        #: quietly is indistinguishable from a note that deserved to be
        #: blocked.
        self.batches = 0
        self.unparsed_batches = 0
        #: Batches that needed a retry, and batches the retry rescued. Both
        #: printed, because "we retry on failure" is a claim about how often
        #: it works, and a retry that never succeeds is just a slower failure.
        self.retried_batches = 0
        self.rescued_batches = 0
        #: Notes with at least one unparsed batch, as (consultation_id,
        #: model_id). A note is contaminated by a single failed batch: every
        #: claim in it was refused, which drags the note's grounding down and
        #: pushes its detection up. Tracked so those notes can be excluded
        #: rather than reasoned about. See WIT-016.
        self.unparsed_notes: set[tuple[str, str]] = set()
        #: Claims per request. Small batches keep the transcript in context and
        #: stop a long note from pushing the early claims out of attention;
        #: large batches are cheaper. Twelve is a compromise, not a finding.
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"{GRADER_VERSION}+{self.provider.name}/{self.provider.model}"

    @property
    def unparsed_rate(self) -> float:
        """Share of batches unreadable after every retry."""
        return self.unparsed_batches / self.batches if self.batches else 0.0

    @property
    def retry_rate(self) -> float:
        """Share of batches whose first attempt was unreadable."""
        return self.retried_batches / self.batches if self.batches else 0.0

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
        before = self.unparsed_batches
        for start in range(0, len(assertions), self.batch_size):
            batch = assertions[start : start + self.batch_size]
            graded.extend(self._grade_batch(batch, transcript, utterances))
        if self.unparsed_batches > before:
            self.unparsed_notes.add((consultation_id, model_id))
        return NoteGrounding(consultation_id, model_id, self.name, graded)

    def _grade_batch(
        self, batch: list[Assertion], transcript: str, utterances: list[Utterance]
    ) -> list[GroundedAssertion]:
        claims = "\n".join(f"{a.index}. {a.text}" for a in batch)
        user = f"TRANSCRIPT\n{transcript}\n\nCLAIMS\n{claims}"
        try:
            out = self.provider.complete(SYSTEM, user, max_tokens=2048)
            parsed = self._parse(out.text)
            attempt = 0
            while not parsed and attempt < self.retries:
                attempt += 1
                if attempt == 1:
                    self.retried_batches += 1
                out = self.provider.complete(
                    SYSTEM, user + self.RETRY_NUDGE, max_tokens=2048
                )
                parsed = self._parse(out.text)
                if parsed:
                    self.rescued_batches += 1
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

        self.batches += 1
        if not parsed:
            self.unparsed_batches += 1
        by_index = {v.get("i"): v for v in parsed}
        graded: list[GroundedAssertion] = []
        for a in batch:
            v = by_index.get(a.index)
            graded.append(self._to_verdict(a, v, utterances, out.replayed))
        return graded

    @staticmethod
    def _parse(text: str) -> list[dict]:
        """Recover the verdict list from whatever the model actually emitted.

        The prompt asks for ``{"verdicts":[...]}``. Against a live qwen2.5:7b
        that envelope arrived 5 times out of 19: the model replied with a bare
        array on 11 and with a prose preamble on 3. The first version of this
        method required the envelope and returned ``[]`` for everything else,
        so **74% of batches were silently zeroed** and the run measured the
        parser rather than the grader. See WIT-016.

        The block-don't-pass asymmetry below is still the right default -- a
        grader that cannot read its model must not report a clean note -- but
        it only works if genuine output is not being thrown away first. A
        refusal that fires on well-formed answers is not caution, it is a bug
        wearing caution's clothes.

        So this scans for every JSON value in the response and keeps the
        verdict objects, whatever they were wrapped in: the envelope, a bare
        array, a single object, several concatenated objects, or any of those
        buried in prose or a code fence. It never raises. Anything genuinely
        unreadable still yields ``[]`` and still blocks.
        """
        decoder = json.JSONDecoder()
        found: list[dict] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch not in "[{":
                i += 1
                continue
            try:
                value, end = decoder.raw_decode(text, i)
            except ValueError:
                # Not the start of a complete JSON value -- a truncated final
                # object, or a stray brace in prose. Step over it.
                i += 1
                continue
            found.extend(SemanticGrader._verdicts_in(value))
            i = end
        # Deduplicate on claim index, keeping the first. A model that restates
        # its answer in prose and then in JSON would otherwise vote twice.
        seen: set[object] = set()
        out: list[dict] = []
        for v in found:
            key = v.get("i")
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out

    @staticmethod
    def _verdicts_in(value: object) -> list[dict]:
        """Pull verdict-shaped dicts out of one decoded JSON value."""
        if isinstance(value, dict):
            if SemanticGrader._is_verdict(value):
                return [value]
            # Any key, not just "verdicts". Models nest the answer under
            # "result", "response", "output" or the day's favourite, and the
            # shape of a verdict is specific enough to find without being told
            # where to look.
            out: list[dict] = []
            for inner in value.values():
                out.extend(SemanticGrader._verdicts_in(inner))
            return out
        if isinstance(value, list):
            items: list[dict] = []
            for item in value:
                items.extend(SemanticGrader._verdicts_in(item))
            return items
        return []

    @staticmethod
    def _is_verdict(value: object) -> bool:
        """A claim index and a verdict. Anything else is not an answer.

        Deliberately structural rather than permissive: accepting any dict
        would let a model's own commentary object become a verdict, and a
        parser that accepts too much reintroduces the failure it was written
        to fix, pointing the other way.
        """
        return isinstance(value, dict) and "i" in value and "v" in value

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
