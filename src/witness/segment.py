"""Splitting a note into claims, and working out what kind of claim each is.

Clinical notes are not prose. They are dense, abbreviated, semicolon-delimited
lists written under time pressure, and splitting them on sentence boundaries
loses most of the claims. A real PriMock57 note line reads:

    Associated LLQ pain - crampy, intermittent, nil radiation.

That is four separate assertions -- there is left-lower-quadrant pain, it is
crampy, it is intermittent, it does not radiate -- and the last one is a
*negative* finding. A sentence splitter returns one item and the negation
disappears into the middle of it, which means a grader can mark the whole line
"supported" while the note is wrong about radiation.

So segmentation splits on the delimiters clinicians actually use, and each
fragment inherits the subject of its clause. Getting this wrong quietly caps
the accuracy of everything downstream, which is why it lives in its own module
with its own tests rather than being three lines inside the grader.
"""

from __future__ import annotations

import re

from witness.grounding import Assertion, Slot

# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

#: Cues that negate what follows. "nil" and "denies" are ubiquitous in UK
#: primary-care notes and absent from most general-purpose negation lists,
#: which is one reason off-the-shelf NLI does poorly on this text.
_NEGATION_CUES = (
    r"\bno\b", r"\bnot\b", r"\bnil\b", r"\bnone\b", r"\bdenies\b", r"\bdenied\b",
    r"\bnegative for\b", r"\bwithout\b", r"\babsent\b", r"\bfree of\b",
    r"\bnever\b", r"\bunremarkable\b", r"\bn/?a\b", r"\bruled out\b",
    r"\bno evidence of\b", r"\bdoesn'?t\b", r"\bdidn'?t\b", r"\bhasn'?t\b",
    r"\bisn'?t\b", r"\bcan'?t\b", r"\bwon'?t\b",
)
_NEGATION_RE = re.compile("|".join(_NEGATION_CUES), re.IGNORECASE)

#: Phrases that look negative but are not, and would otherwise flip a verdict.
#: "no longer smoking" asserts a change, not an absence; "not only" is a
#: connective. Each of these was added after it produced a wrong verdict.
_PSEUDO_NEGATION_RE = re.compile(
    r"\bno longer\b|\bnot only\b|\bno doubt\b|\bnot to mention\b|\bnone the less\b",
    re.IGNORECASE,
)


def is_negated(text: str) -> bool:
    """Whether a fragment asserts the absence of something."""
    stripped = _PSEUDO_NEGATION_RE.sub(" ", text)
    return bool(_NEGATION_RE.search(stripped))


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------

_SLOT_PATTERNS: list[tuple[Slot, re.Pattern[str]]] = [
    (
        Slot.ALLERGY,
        re.compile(r"\baller(gy|gies|gic)\b|\bnkda\b|\banaphyla", re.IGNORECASE),
    ),
    (
        Slot.DOSE,
        # A dose is a number bound to a unit or a frequency. Matching bare
        # numbers would sweep in "6 times a day" for stool frequency, which is
        # a symptom, not a prescription.
        re.compile(
            r"\b\d+\s?(mg|mcg|µg|ug|g|ml|units?|iu|puffs?|tablets?|caps?)\b"
            r"|\b(od|bd|tds|qds|prn|nocte|mane)\b"
            r"|\b\d+\s?(mg|mcg|g|ml)\s?/\s?(day|kg|dose)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Slot.MEDICATION,
        re.compile(
            r"\b(inhaler|salbutamol|paracetamol|ibuprofen|amoxicillin|omeprazole|"
            r"statin|metformin|insulin|steroid|antibiotic|analgesi|medication|"
            r"prescri|dh:|drug history|takes?\s+\w+ol\b)",
            re.IGNORECASE,
        ),
    ),
    (
        Slot.DIAGNOSIS,
        re.compile(
            r"\b(imp|impression|diagnosis|dx|likely|probable|consistent with|"
            r"suggestive of|pmh|past medical history)\b"
            r"|\b\w+(itis|osis|aemia|emia|pathy|oma)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Slot.PLAN,
        re.compile(
            r"\b(plan|advise|advised|refer|referral|review|safety net|follow.?up|"
            r"recommend|management|to see|book|arrange)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Slot.SOCIAL,
        re.compile(
            r"\b(sh:|social history|smok|alcohol|etoh|occupation|works? as|lives? "
            r"(with|alone)|accountant|drinks?)\b",
            re.IGNORECASE,
        ),
    ),
]


def classify_slot(text: str) -> Slot:
    """Which clinical category a fragment belongs to.

    Order matters and is deliberate: allergy before medication (an allergy to a
    drug names the drug), dose before medication (a dose names the drug too).
    First match wins, so the most safety-consequential reading is preferred
    when a fragment could be read several ways.
    """
    for slot, pattern in _SLOT_PATTERNS:
        if pattern.search(text):
            return slot
    if is_negated(text):
        return Slot.NEGATIVE_FINDING
    return Slot.SYMPTOM if _looks_clinical(text) else Slot.OTHER


#: Stems, not words. The trailing ``\b`` this pattern used to carry made every
#: entry an exact match, so ``weak`` did not match "weakness", ``swell`` did not
#: match "swelling", and ``abdo`` did not match "abdominal" -- nine of the
#: thirteen stems never fired on their own inflected forms. 143 clinical claims
#: were dropped before grading, most of them pertinent negatives.
#:
#: Restoring them moved lexical grounding *up*, 40.0% to 45.0%. That is the
#: opposite of what I predicted when I found this -- I reasoned that a dropped
#: claim cannot be ungrounded, so the defect must have inflated grounding. The
#: discarded claims turned out to be easier to ground than average, because the
#: stems identifying them are content words patients say out loud. See WIT-018.
_CLINICAL_HINT = re.compile(
    r"\b(pain|ache|fever|cough|nausea|vomit|diarrh|blood|stool|urine|rash|"
    r"swell|breath|chest|abdo|head|dizz|weak|tired|lethargic|appetite|sleep|"
    r"bowel|bladder|sweat|weight|lump|discharge|itch|sore|cramp|numb|tingl|"
    r"nil|history)",
    re.IGNORECASE,
)


def _looks_clinical(text: str) -> bool:
    return bool(_CLINICAL_HINT.search(text))


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

#: Delimiters clinicians actually use. The comma is deliberately absent: it
#: separates modifiers of one finding ("crampy, intermittent") as often as it
#: separates findings, and splitting on it fragments single claims into
#: unjudgeable pieces. Dashes and semicolons are much more reliable boundaries.
_SPLIT_RE = re.compile(r"(?:\.\s+|\n+|;\s*|\s+-\s+|\s+–\s+|\s*\|\s*)")

_HEADER_RE = re.compile(
    r"^\s*(pmh|dh|sh|fh|imp|plan|o/e|hpc|pc|ice|allergies)\s*:\s*", re.IGNORECASE
)


def _SLOT_PATTERNS_MATCH(text: str) -> bool:
    """Whether any slot pattern fires -- i.e. the fragment is a clinical claim."""
    return any(p.search(text) for _, p in _SLOT_PATTERNS)


def segment(note: str) -> list[Assertion]:
    """Split a generated note into individually judgeable clinical claims."""
    assertions: list[Assertion] = []
    for raw in _SPLIT_RE.split(note or ""):
        text = raw.strip().strip(".,;:-– ")
        if len(text) < 3:
            continue
        # A bare section header ("PMH:") is a label, not a claim. A header with
        # content after it is a claim and keeps its header for context, because
        # "asthma" under PMH means something different from "asthma" under Plan.
        if _HEADER_RE.fullmatch(text + ":") or _HEADER_RE.fullmatch(text):
            continue
        # A fragment carrying no clinical content is not a claim and must not
        # reach the grader. Grading them as UNSUPPORTED put a failure in every
        # note and drove the note-level flag rate to 100%, which made the
        # harness useless as a discriminator while still looking strict.
        if not _looks_clinical(text) and not _SLOT_PATTERNS_MATCH(text):
            continue
        assertions.append(
            Assertion(
                index=len(assertions),
                text=text,
                slot=classify_slot(text),
                negated=is_negated(text),
            )
        )
    return assertions
