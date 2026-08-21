"""A grader with no model in it.

This exists for three reasons, in order of importance.

**It is the floor.** Any claim that an LLM grader is good needs something to be
good *against*. Comparing a language model to nothing is how a field ends up
with numbers that cannot be interpreted.

**It runs everywhere.** No API key, no BAA, no network, no GPU. A reviewer can
clone the repository and reproduce every number in the README on a laptop in
under a minute. That property is worth more than a couple of points of
accuracy, and it is the same reasoning that keeps the whole pipeline runnable
without cloud credentials.

**It is auditable.** When this grader says a claim is contradicted, the reason
is a term and a polarity, and a human can check it in one glance. LLM graders
are more accurate and less inspectable, and knowing exactly what the cheap
transparent method already catches tells you what the expensive opaque one is
actually buying.

The method is deliberately unclever: clinical term overlap, with negation
polarity treated as a first-class signal rather than an afterthought. It will
miss paraphrase. It will not miss "no blood in stool" written over an encounter
in which the patient described blood in their stool, and that second failure is
the one that hurts people.
"""

from __future__ import annotations

import re
from collections import defaultdict

from witness.corpus import Utterance
from witness.grounding import (
    Assertion,
    GroundedAssertion,
    NoteGrounding,
    Span,
    Verdict,
)
from witness.segment import is_negated

GRADER_NAME = "lexical/1.0"

#: Words carrying no clinical discriminative power. Kept deliberately short --
#: an aggressive stop list strips the very terms that distinguish two findings
#: from each other.
#: Kept as prose rather than a 90-item list literal: it is read far more
#: often than it is edited, and the point of a stop list is to be scannable.
_STOP_WORDS = """
    a an and the of to in on at for with is are was were be been being has have had he
    she they it his her their patient pt this that these those there here as by from or
    but if then so than also very much some any all no not nil none which who whom what
    when where how why do does did done will would could should may might can about into
    over under again further once more most other such only own same too s t re ve ll d
    m o e g
"""
_STOP = frozenset(_STOP_WORDS.split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Clinical synonyms and the abbreviations that dominate real notes. Without
#: this the grader marks "LLQ pain" unsupported against a patient who said
#: "pain in my lower left tummy", which is a false alarm on a true claim -- and
#: false alarms are how a gate gets switched off.
_SYNONYMS: dict[str, str] = {}
_SYNONYM_GROUPS = [
    ("abdo", "abdomen", "abdominal", "tummy", "stomach", "belly"),
    ("llq", "lif", "left lower quadrant", "lower left"),
    ("rlq", "rif", "right lower quadrant", "lower right"),
    ("diarrhoea", "diarrhea", "loose stools", "runny"),
    ("vomit", "vomiting", "vomited", "sick", "throwing up", "puking"),
    ("nausea", "nauseous", "queasy"),
    ("pyrexia", "fever", "feverish", "temperature", "hot"),
    ("dyspnoea", "dyspnea", "breathless", "short of breath", "sob"),
    ("lethargic", "lethargy", "tired", "fatigue", "fatigued", "weak", "weakness"),
    ("etoh", "alcohol", "drink", "drinking"),
    ("smoking", "smoker", "smoke", "cigarettes"),
    ("adls", "daily activities", "day to day activities"),
    ("hx", "history"),
    ("pmh", "past medical history"),
    ("dh", "drug history", "medication", "medications", "meds"),
    ("sh", "social history"),
    ("bowels", "bowel", "stool", "stools", "poo", "faeces"),
    ("analgesia", "painkiller", "painkillers", "paracetamol"),
    ("inhaler", "inhalers", "puffer"),
    ("asthma", "asthmatic"),
]
for _group in _SYNONYM_GROUPS:
    _canon = _group[0]
    for _term in _group:
        _SYNONYMS[_term.replace(" ", "_")] = _canon


def _normalise(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        t = _SYNONYMS.get(t, t)
        # Crude depluralisation. A stemmer would be better; it would also be a
        # dependency and a source of surprises on clinical vocabulary.
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(_SYNONYMS.get(t, t))
    return out


def content_terms(text: str) -> set[str]:
    """The clinically meaningful tokens of a fragment."""
    lowered = text.lower()
    # Fold multi-word synonyms before tokenising, so "lower left" survives as
    # one concept instead of two useless tokens.
    for phrase in _SYNONYMS:
        if "_" in phrase:
            lowered = lowered.replace(phrase.replace("_", " "), phrase)
    toks = [t for t in _TOKEN_RE.findall(lowered) if t not in _STOP and len(t) > 1]
    return set(_normalise(toks))


class LexicalGrader:
    """Ground assertions by clinical term overlap and negation polarity."""

    name = GRADER_NAME

    def __init__(
        self,
        support_threshold: float = 0.5,
        weak_threshold: float = 0.34,
        min_terms: int = 1,
    ):
        #: Fraction of an assertion's clinical terms that must appear in one
        #: utterance for it to count as support. Tuned on a held-out slice --
        #: see docs/THRESHOLDS.md. Set too high and paraphrase reads as
        #: fabrication; too low and the grader supports anything.
        self.support_threshold = support_threshold
        #: Below support but above this, evidence is partial. Partial evidence
        #: with matching polarity is INFERRED rather than SUPPORTED: the claim
        #: is probably fine but the grader cannot fully witness it, and saying
        #: so is more useful than rounding up.
        self.weak_threshold = weak_threshold
        self.min_terms = min_terms

    # -- public ----------------------------------------------------------

    def ground_note(
        self,
        consultation_id: str,
        model_id: str,
        assertions: list[Assertion],
        utterances: list[Utterance],
    ) -> NoteGrounding:
        index = _UtteranceIndex(utterances)
        graded = [self._ground_one(a, index) for a in assertions]
        return NoteGrounding(
            consultation_id=consultation_id,
            model_id=model_id,
            grader=self.name,
            assertions=graded,
        )

    # -- internals -------------------------------------------------------

    def _ground_one(self, assertion: Assertion, index: _UtteranceIndex) -> GroundedAssertion:
        terms = content_terms(assertion.text)
        best = index.best_matches(terms, top_k=3)
        if not best:
            return GroundedAssertion(
                assertion=assertion,
                verdict=Verdict.UNSUPPORTED,
                rationale=f"no utterance shares any of: {sorted(terms)[:6]}",
                confidence=0.6,
                grader=self.name,
            )

        top_score, top_utt, top_i = best[0]

        # Polarity check first. An assertion that overlaps strongly with an
        # utterance but disagrees about whether the finding is present is the
        # dangerous case, and it outranks every other reading of the evidence.
        if top_score >= self.weak_threshold:
            utt_negated = is_negated(top_utt.text)
            if utt_negated != assertion.negated:
                # Before calling it contradicted, check whether some *other*
                # utterance carries the same polarity. Patients contradict
                # themselves constantly ("no chest pain... well, a bit of
                # tightness"), and a note that records the later, corrected
                # statement is right, not wrong.
                for score, utt, i in best[1:]:
                    if score >= self.weak_threshold and is_negated(utt.text) == assertion.negated:
                        return GroundedAssertion(
                            assertion=assertion,
                            verdict=Verdict.SUPPORTED,
                            spans=[_span(utt, i)],
                            rationale="polarity matches a later utterance on the same finding",
                            confidence=round(score, 3),
                            grader=self.name,
                        )
                return GroundedAssertion(
                    assertion=assertion,
                    verdict=Verdict.CONTRADICTED,
                    spans=[_span(top_utt, top_i)],
                    rationale=(
                        f"note asserts {'absence' if assertion.negated else 'presence'}; "
                        f"encounter asserts {'absence' if utt_negated else 'presence'}"
                    ),
                    confidence=round(top_score, 3),
                    grader=self.name,
                )

        if top_score >= self.support_threshold:
            return GroundedAssertion(
                assertion=assertion,
                verdict=Verdict.SUPPORTED,
                spans=[_span(top_utt, top_i)],
                rationale=f"{top_score:.0%} of clinical terms present in one utterance",
                confidence=round(top_score, 3),
                grader=self.name,
            )

        if top_score >= self.weak_threshold:
            return GroundedAssertion(
                assertion=assertion,
                verdict=Verdict.INFERRED,
                spans=[_span(top_utt, top_i)],
                rationale=(
                    f"partial evidence ({top_score:.0%} of terms); consistent polarity. "
                    "Lexical grading cannot distinguish clinical inference from "
                    "paraphrase -- an LLM grader should adjudicate this one"
                ),
                confidence=round(top_score, 3),
                grader=self.name,
            )

        return GroundedAssertion(
            assertion=assertion,
            verdict=Verdict.UNSUPPORTED,
            rationale=f"best overlap {top_score:.0%}, below {self.weak_threshold:.0%}",
            confidence=round(1.0 - top_score, 3),
            grader=self.name,
        )


def _span(u: Utterance, i: int) -> Span:
    return Span(
        utterance_index=i, speaker=u.speaker, start_s=u.start_s, end_s=u.end_s, text=u.text
    )


class _UtteranceIndex:
    """Inverted index over utterance terms.

    Scoring every assertion against every utterance is O(assertions x
    utterances) and a long consultation has several hundred turns. The index
    is not an optimisation for its own sake: it keeps a full 57-consultation
    run inside a few seconds, which is what makes the harness usable as a CI
    gate rather than a nightly job.
    """

    def __init__(self, utterances: list[Utterance]):
        self.utterances = utterances
        self.terms: list[set[str]] = [content_terms(u.text) for u in utterances]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, ts in enumerate(self.terms):
            for t in ts:
                self.postings[t].append(i)

    def best_matches(
        self, terms: set[str], top_k: int = 3
    ) -> list[tuple[float, Utterance, int]]:
        hits: dict[int, int] = defaultdict(int)
        for t in terms:
            for i in self.postings.get(t, ()):
                hits[i] += 1
        if not hits:
            return []
        scored = [(n / len(terms), self.utterances[i], i) for i, n in hits.items()]
        scored.sort(key=lambda x: (-x[0], x[2]))
        return scored[:top_k]
