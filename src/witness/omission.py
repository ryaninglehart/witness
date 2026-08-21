"""What the encounter contained and the note dropped.

Omission is 62.5% of the critical errors clinicians found in this corpus and it
is the half almost nobody measures. The reason is structural: hallucination is
visible in the note, so you can check a note against a transcript and find it.
Omission is *absent* from the note, so there is nothing to check. You have to
first decide what should have been there.

That decision is the entire problem. A transcript of a fifteen-minute
consultation contains several hundred utterances and perhaps thirty things a
clinician would write down. Get the salience judgement wrong and the module
either reports nothing useful or reports two hundred "omissions" per note,
which is the same as reporting nothing.

Three signals are used, in descending order of trustworthiness.

**What the treating clinician actually wrote.** PriMock57 ships the note the
consulting clinician produced during or shortly after the visit, plus the terms
they chose to highlight in it. That is a salience judgement made at the point of
care by the person who was in the room. Nothing derived afterwards beats it.

**Safety categories.** Allergies, medications and doses, red-flag symptoms.
These are salient whether or not any particular clinician wrote them down,
because omitting them is dangerous rather than merely untidy.

**Answered questions.** A doctor asking "any blood in the stool?" has decided
the answer matters. A substantive patient reply to a direct clinical question
is a candidate fact, and this is what catches *pertinent negatives* -- "no chest
pain" -- which are clinically load-bearing and among the most commonly dropped
things in a generated note.

The module reports what it could not judge rather than assuming it was
unimportant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from witness.corpus import Consultation
from witness.graders.lexical import content_terms
from witness.segment import is_negated

EXTRACTOR_NAME = "salience/1.0"


class Tier:
    """How much it matters that a fact is missing."""

    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


#: Categories that are salient regardless of whether a given clinician recorded
#: them. Dropping an allergy is not a stylistic difference.
_SAFETY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("allergy", re.compile(r"\baller(gy|gies|gic)\b|\bnkda\b|\breact(ion|ed) to\b", re.I)),
    ("medication", re.compile(
        r"\b(inhaler|tablet|medication|prescri|antibiotic|steroid|painkiller|"
        r"paracetamol|ibuprofen|salbutamol|omeprazole|statin|metformin|insulin|"
        r"warfarin|aspirin|codeine|morphine)\b", re.I)),
    ("red_flag", re.compile(
        r"\b(chest pain|blood in (my |the )?(stool|urine|vomit|sputum)|"
        r"coughing up blood|weight loss|night sweats|short(ness)? of breath|"
        r"can'?t breathe|fainted|passed out|collapse|suicid|self.?harm|"
        r"worst headache|neck stiff|photophob|rash that|unconscious)\b", re.I)),
    ("pregnancy", re.compile(r"\b(pregnan|breastfeed|last period|lmp)\b", re.I)),
]

#: A doctor turn ending in a question mark, or opening with an interrogative,
#: is eliciting something. What follows is by definition information the
#: clinician decided to ask for.
_QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(do|does|did|are|is|was|were|have|has|had|any|can|could|would|"
    r"when|where|how|what|which|why|tell me|describe)\b",
    re.I,
)

#: Backchannel and filler. A patient saying "yeah" in reply to a question is
#: not a recordable fact, and counting it would bury the real ones.
_FILLER_RE = re.compile(
    r"^\s*(yeah|yes|no|ok(ay)?|mm+|hmm+|uh|um|er|right|sure|alright|"
    r"i see|of course|thank you|thanks|sorry|pardon)[\s.,!?]*$",
    re.I,
)

_UNINTELLIGIBLE_RE = re.compile(r"<UNIN/?>|<UNSURE>|</UNSURE>")


@dataclass(frozen=True)
class SalientFact:
    """Something the encounter established that a note ought to carry."""

    text: str
    tier: str
    utterance_index: int
    speaker: str
    start_s: float
    end_s: float
    #: Why this was judged salient. Printed in the report so a reader can
    #: disagree with the judgement rather than only with the conclusion.
    basis: str
    negated: bool = False
    #: The *clinical core* of the utterance, not every token in it. A patient
    #: saying "Oh hey, um, I've just had some diarrhoea for three days" carries
    #: one fact. Scoring coverage over all eleven tokens asks the note to
    #: reproduce the filler, so nothing is ever judged covered -- which is
    #: precisely what happened before this was narrowed (WIT-005).
    terms: frozenset[str] = field(default_factory=frozenset)


@dataclass
class OmissionReport:
    consultation_id: str
    model_id: str
    extractor: str
    covered: list[SalientFact] = field(default_factory=list)
    omitted: list[SalientFact] = field(default_factory=list)

    def omitted_at(self, tier: str) -> list[SalientFact]:
        return [f for f in self.omitted if f.tier == tier]

    @property
    def critical_omissions(self) -> int:
        return len(self.omitted_at(Tier.CRITICAL))

    @property
    def coverage(self) -> float:
        total = len(self.covered) + len(self.omitted)
        return len(self.covered) / total if total else 1.0

    def summary(self) -> dict[str, float | int]:
        return {
            "salient_facts": len(self.covered) + len(self.omitted),
            "covered": len(self.covered),
            "omitted": len(self.omitted),
            "omitted_critical": self.critical_omissions,
            "omitted_major": len(self.omitted_at(Tier.MAJOR)),
            "coverage": round(self.coverage, 4),
        }


def _clean(text: str) -> str:
    return _UNINTELLIGIBLE_RE.sub(" ", text).strip()


def extract_salient(
    consultation: Consultation,
    min_terms: int = 2,
) -> list[SalientFact]:
    """Decide what this encounter established that a note should carry."""
    facts: list[SalientFact] = []
    utts = consultation.utterances

    # Terms the treating clinician chose to record. The strongest available
    # signal of what mattered, because a clinician made it in the room.
    clinician_terms: set[str] = content_terms(consultation.clinician_note)
    highlight_terms: set[str] = set()
    for h in consultation.highlights:
        highlight_terms |= content_terms(h)

    for i, u in enumerate(utts):
        text = _clean(u.text)
        if len(text) < 8 or _FILLER_RE.match(text):
            continue
        terms = content_terms(text)
        if len(terms) < min_terms:
            continue

        # The clinical core: terms the treating clinician also recorded, plus
        # anything a safety pattern matched. A doctor's question shares terms
        # with the answer, so restricting facts to what was *reported* keeps
        # "any blood in the stool?" from becoming a fact in its own right.
        core = terms & (clinician_terms | highlight_terms)
        safety_hit = None
        for label, pat in _SAFETY_PATTERNS:
            m = pat.search(text)
            if m:
                safety_hit = label
                # Only the terms the pattern actually matched. Taking the whole
                # utterance here reintroduced WIT-005 for safety facts alone:
                # "allergic to penicillin" became a five-term fact including
                # "um" and "since", which no note reproduces, so a correctly
                # recorded allergy was still reported as omitted.
                core |= content_terms(m.group(0))
                break
        if not core:
            continue

        tier: str | None = None
        basis = ""
        is_doctor_question = u.speaker == "doctor" and bool(_QUESTION_RE.search(text))
        if is_doctor_question:
            # An elicitation, not a finding. The answer that follows is the fact.
            continue

        if safety_hit:
            tier, basis = Tier.CRITICAL, f"safety category: {safety_hit}"
        elif terms & highlight_terms and u.speaker == "patient":
            # MAJOR, not CRITICAL. The treating clinician highlighting a term is
            # strong evidence it mattered, but CRITICAL is reserved for the
            # categories whose omission is dangerous rather than merely
            # incomplete -- allergies, drugs, red flags. Tiering highlights as
            # CRITICAL made every note in the corpus fail the omission rule,
            # including notes written by clinicians. See WIT-007.
            tier, basis = Tier.MAJOR, "clinician highlighted this in their note"

        if tier is None and u.speaker == "patient" and i > 0:
            prev = _clean(utts[i - 1].text)
            if utts[i - 1].speaker == "doctor" and _QUESTION_RE.search(prev):
                overlap = len(terms & clinician_terms) / len(terms)
                # A direct answer whose content the clinician also recorded is
                # a fact. A direct answer they ignored is context.
                if overlap >= 0.3:
                    tier, basis = Tier.MAJOR, "answer to a direct clinical question"
                elif is_negated(text):
                    # Pertinent negatives are load-bearing and are dropped from
                    # generated notes constantly. "No chest pain" is a finding.
                    tier, basis = Tier.MAJOR, "pertinent negative, elicited by question"

        if tier is None and u.speaker == "patient":
            overlap = len(terms & clinician_terms) / len(terms)
            if overlap >= 0.5:
                tier, basis = Tier.MINOR, "content the clinician also recorded"

        if tier is None:
            continue

        facts.append(
            SalientFact(
                text=text,
                tier=tier,
                utterance_index=i,
                speaker=u.speaker,
                start_s=u.start_s,
                end_s=u.end_s,
                basis=basis,
                negated=is_negated(text),
                terms=frozenset(core),
            )
        )

    return _dedupe(facts)


def _dedupe(facts: list[SalientFact]) -> list[SalientFact]:
    """Collapse restatements of the same fact.

    Patients repeat themselves and clinicians summarise back to confirm. Three
    utterances about the same abdominal pain are one omission if the note drops
    it, not three, and counting them separately would make a single miss look
    like a systemic failure.
    """
    kept: list[SalientFact] = []
    _RANK = {Tier.CRITICAL: 0, Tier.MAJOR: 1, Tier.MINOR: 2}
    for f in facts:
        dup = False
        for j, k in enumerate(kept):
            shared = len(f.terms & k.terms) / max(1, min(len(f.terms), len(k.terms)))
            if shared >= 0.7 and f.negated == k.negated:
                # Keep whichever statement was judged more serious.
                if _RANK[f.tier] < _RANK[k.tier]:
                    kept[j] = f
                dup = True
                break
        if not dup:
            kept.append(f)
    return kept


def find_omissions(
    consultation: Consultation,
    note: str,
    model_id: str = "",
    coverage_threshold: float = 0.65,
) -> OmissionReport:
    """Which salient facts the note failed to carry."""
    facts = extract_salient(consultation)
    note_terms = content_terms(note)
    note_negated_spans = [
        content_terms(seg) for seg in re.split(r"[.;\n]", note) if is_negated(seg)
    ]

    covered: list[SalientFact] = []
    omitted: list[SalientFact] = []
    for f in facts:
        overlap = len(f.terms & note_terms) / len(f.terms) if f.terms else 0.0
        if overlap < coverage_threshold:
            omitted.append(f)
            continue
        # A negative finding is only carried if the note also states it
        # negatively. A note mentioning "chest pain" when the patient denied it
        # has not covered the pertinent negative -- it has inverted it, which
        # the grounding pass will separately catch as a contradiction.
        if f.negated and not any(
            len(f.terms & seg) / len(f.terms) >= coverage_threshold
            for seg in note_negated_spans
        ):
            omitted.append(f)
        else:
            covered.append(f)

    return OmissionReport(
        consultation_id=consultation.consultation_id,
        model_id=model_id,
        extractor=EXTRACTOR_NAME,
        covered=covered,
        omitted=omitted,
    )
