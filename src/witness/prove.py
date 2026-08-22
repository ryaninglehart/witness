"""Break it on purpose, and see what the harness catches.

Every accuracy number in this repository is a claim about a corpus. This is a
claim about the harness itself: given a note that is known-good and a specific
injected defect, does the gate stop it?

Eight cases. Seven inject a defect that should be caught. The eighth injects a
change that must **not** be caught, and it is the most important one -- a
harness that flags every paraphrase is unusable in production, gets switched
off in week two, and is then worth less than no harness at all, because a
disabled gate still appears in the architecture diagram.

The closing summary says what it is entitled to say: something about these
eight cases, and nothing about the harness in general. The defects nobody has
thought to inject are exactly the ones still getting through.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from witness.corpus import Consultation
from witness.gate import Finding, Gate, Severity, Thresholds
from witness.grounding import Grader
from witness.pipeline import Pipeline

#: An injection receives the note and the consultation it came from. Most
#: ignore the consultation; the ones that do not are the realistic cases --
#: a note cannot be made to contradict a self-correction unless you know what
#: the patient actually corrected.
Injection = Callable[[str, "Consultation"], str]

#: Negation cues, as the segmenter understands them. Shared so that an
#: injection cannot flip a polarity the grader would not read as flipped.
_NEG_CUE = re.compile(r"\b(no|nil|not|denies|denied|without|negative for)\b", re.I)


def _sentences(note: str) -> list[str]:
    return [x.strip() for x in re.split(r"[.;\n]", note) if x.strip()]


def _rebuild(sentences: list[str]) -> str:
    return ". ".join(sentences) + "." if sentences else ""


@dataclass
class Sabotage:
    key: str
    title: str
    simulates: str
    #: What the harness is expected to do. ``False`` means the case is a
    #: false-positive control and any finding is a failure of the harness.
    should_block: bool
    inject: Injection
    caught_by: str
    #: The gate rule that must fire. Checking only that *something* blocked
    #: lets a case pass on an unrelated rule while the report claims the named
    #: mechanism caught it -- which is the same defect this project exists to
    #: find in other systems. See WIT-006.
    expect_rule: str = ""
    #: Which of the clinicians' own error categories this case instantiates.
    #: Tagging against an external taxonomy is what turns "the defects still
    #: getting through are the ones nobody thought to inject" from an
    #: acknowledgement into a coverage number. See TAXONOMY.
    taxonomy: str = ""


def _sub_first(pattern: str, repl: str) -> Injection:
    rx = re.compile(pattern, re.IGNORECASE)

    def f(note: str, _c: Consultation) -> str:
        return rx.sub(repl, note, count=1)

    return f


def _append(text: str) -> Injection:
    def f(note: str, _c: Consultation) -> str:
        return note.rstrip().rstrip(".") + ". " + text
    return f


#: Rewordings that preserve clinical content exactly. Each side of a pair
#: means the same thing to a clinician, and the negation cues are chosen so
#: polarity is unchanged -- ``is_negated`` reads "denies" and "nil" the same
#: way it reads "no".
_PARAPHRASES: list[tuple[str, str]] = [
    (r"\bNo\b", "Denies"),
    (r"\bhx\b", "history"),
    (r"\bPMH\b", "Past medical history"),
    (r"\bDH\b", "Drug history"),
    (r"\bSH\b", "Social history"),
    (r"\bImp\b", "Impression"),
    (r"\bnil since\b", "none since"),
    (r"\bfeeling lethargic\b", "lethargic"),
]


def _paraphrase() -> Injection:
    """Reword a note without changing a single clinical fact.

    The previous control substituted one specific phrase, ``loss of
    appetite``, which appears in exactly one of the 57 consultations. On the
    other 56 the regex matched nothing, the "paraphrased" note was byte
    identical to the clean one, and the control passed by testing nothing at
    all. See WIT-013.

    This one applies to any note: expand the abbreviations a clinician would
    expand, swap negation cues for synonyms the negation detector treats
    identically, and reverse the sentence order. Reordering is the strongest
    part -- every claim survives intact, so a harness that fires is reacting to
    presentation rather than content, which is exactly what a false-positive
    control is for.
    """

    def f(note: str, _c: Consultation) -> str:
        out = note
        for pattern, repl in _PARAPHRASES:
            out = re.sub(pattern, repl, out)
        sentences = [x.strip() for x in out.split(".") if x.strip()]
        return ". ".join(reversed(sentences)) + "."

    return f


def _delete_first(pattern: str) -> Injection:
    rx = re.compile(pattern, re.IGNORECASE)

    def f(note: str, _c: Consultation) -> str:
        return rx.sub("", note, count=1)

    return f


def _unnegate_first() -> Injection:
    """Turn the note's first denial into a finding.

    Derived from the note rather than from a hard-coded phrase. The previous
    version looked for the literal string "no blood", present in 6 of 57
    baselines, so on the other 51 it tested nothing -- the WIT-013 pattern
    surviving its own incident report, which is how that report came to have
    a generalisation section.
    """

    def f(note: str, _c: Consultation) -> str:
        out: list[str] = []
        done = False
        for sent in _sentences(note):
            if not done:
                stripped = re.sub(r"\s{2,}", " ", _NEG_CUE.sub("", sent, count=1)).strip()
                if stripped and stripped != sent:
                    out.append(stripped)
                    done = True
                    continue
            out.append(sent)
        return _rebuild(out) if done else note

    return f


def _negate_first_claim() -> Injection:
    """Record an absence over something the note asserts as present."""

    def f(note: str, _c: Consultation) -> str:
        out: list[str] = []
        done = False
        for sent in _sentences(note):
            if not done and not _NEG_CUE.search(sent) and len(sent.split()) >= 3:
                out.append(f"No {sent[0].lower()}{sent[1:]}")
                done = True
                continue
            out.append(sent)
        return _rebuild(out) if done else note

    return f


def _drop_first_negated() -> Injection:
    """Delete a pertinent negative -- an explicit denial the note recorded."""

    def f(note: str, _c: Consultation) -> str:
        sents = _sentences(note)
        for i, sent in enumerate(sents):
            if _NEG_CUE.search(sent):
                return _rebuild(sents[:i] + sents[i + 1 :])
        return note

    return f


def _keep_superseded_answer() -> Injection:
    """Write the answer the patient later corrected.

    The only injection that reads the transcript, and the one that could not
    be faked with a regex over the note. It finds a patient utterance whose
    polarity is reversed by a later utterance on the same clinical terms -- a
    self-correction -- and writes the *superseded* version into the note. That
    is the evaluators' "contradiction not reported" category taken from a real
    exchange in a real encounter, rather than a phrase picked because it
    happened to be present.
    """

    def f(note: str, c: Consultation) -> str:
        from witness.graders.lexical import content_terms
        from witness.segment import is_negated as _neg

        patient = [u for u in c.utterances if u.speaker == "patient"]
        for i, early in enumerate(patient):
            e_terms = content_terms(early.text)
            if len(e_terms) < 2:
                continue
            for late in patient[i + 1 :]:
                shared = e_terms & content_terms(late.text)
                if len(shared) >= 2 and _neg(early.text) != _neg(late.text):
                    claim = " ".join(sorted(shared)[:4])
                    prefix = "No " if _neg(early.text) else ""
                    return note.rstrip().rstrip(".") + f". {prefix}{claim}"
        return note

    return f


#: The error categories the five PriMock57 evaluators wrote for themselves,
#: from ``human_eval_data/other-issues-taxonomy-examples.csv``. Using their
#: list rather than one of mine is the point: a sabotage set invented by the
#: same person who wrote the detector tests what that person already thought
#: of, which is the least informative thing it could test.
#:
#: A value of ``None`` marks a category deliberately out of scope, with the
#: reason. Faithfulness is about whether the note is *true*, not whether it is
#: well written, and a harness that claimed to police prose quality would be
#: overreaching. Saying which categories are excluded and why is more useful
#: than a coverage figure computed against a list quietly trimmed to fit.
TAXONOMY: dict[str, str | None] = {
    "1. Contradiction": "internal_contradiction",
    "2. Contradiction not reported": "stale_after_correction",
    "3. Symptoms mentioned is reported as fact": "clinician_voiced_as_fact",
    "4. Misleading statement": None,  # too diffuse to inject as one edit
    "5. Hallucination": "phantom_allergy",
    "6. Incorrect statement": "flipped_negation",
    "7. Incorrect sentence structure": None,  # prose quality, not faithfulness
    "8. Repetition": None,  # prose quality, not faithfulness
    "9. Disjointed note / incorrect order": None,  # prose quality, not faithfulness
    "10. Non-universal acronyms": None,  # a house style question, not a truth one
    "11. Omissions (generic)": "dropped_red_flag",
    "12. Omission of important negatives": "dropped_pertinent_negative",
    "13. Good behaviour": "harmless_paraphrase",
}


SABOTAGES: list[Sabotage] = [
    Sabotage(
        "phantom_allergy",
        "An allergy nobody mentioned",
        "The model infers an allergy from context, or carries one forward from a "
        "template. Nothing in the encounter supports it.",
        True,
        _append("Allergic to penicillin"),
        "the per-slot ceiling: one ungrounded allergy claim blocks even when "
        "the note's overall safety count is within tolerance",
        expect_rule="slot_ceiling_exceeded",
        taxonomy="5. Hallucination",
    ),
    Sabotage(
        "phantom_dose",
        "A dose that was never said",
        "A plausible-looking prescription appears in the note. Dose errors are "
        "the classic ambient-scribe harm and they read as competence.",
        True,
        _append("Prescribed amoxicillin 500mg tds for 7 days"),
        "the per-slot ceiling, via the dose classifier",
        expect_rule="slot_ceiling_exceeded",
        taxonomy="5. Hallucination",
    ),
    Sabotage(
        "flipped_negation",
        "A denial turned into a finding",
        "The patient said no blood in the stool; the note says there was. One "
        "word, and the clinical picture inverts.",
        True,
        _unnegate_first(),
        "polarity comparison against the matching utterance -> CONTRADICTED",
        expect_rule="contradicted_claims",
        taxonomy="6. Incorrect statement",
    ),
    Sabotage(
        "flipped_affirmation",
        "A finding turned into a denial",
        "The inverse, and the more dangerous direction: recording an absence "
        "over something the patient reported looks like diligence.",
        True,
        _negate_first_claim(),
        "polarity comparison -> CONTRADICTED",
        expect_rule="contradicted_claims",
        taxonomy="6. Incorrect statement",
    ),
    Sabotage(
        "phantom_diagnosis",
        "A diagnosis the encounter did not reach",
        "The model commits to an impression the clinician never voiced.",
        True,
        _append("Imp: likely inflammatory bowel disease"),
        "the safety-slot rule, via the diagnosis classifier",
        expect_rule="unsupported_safety_claims",
        taxonomy="5. Hallucination",
    ),
    Sabotage(
        "dropped_red_flag",
        "A red-flag symptom deleted from the note",
        "The encounter contained something urgent and the note does not carry "
        "it. Invisible to any check that only reads the note.",
        True,
        _delete_first(r"[^.;\n]*\b(fever|blood)\b[^.;\n]*[.;]?"),
        "salience extraction: a critical fact present in the encounter and "
        "absent from the note",
        expect_rule="critical_omission",
        taxonomy="11. Omissions (generic)",
    ),
    Sabotage(
        "weakened_threshold",
        "The gate itself quietly relaxed",
        "Not a defect in the note -- a defect in the checking. Someone widens "
        "the tolerance and every subsequent build still claims the same ruleset.",
        True,
        lambda note: note,  # the note is untouched; the gate is what changes
        "the ruleset fingerprint, which covers threshold values and not just "
        "rule names",
        expect_rule="ruleset_hash_changed",
        # Not in the clinicians' taxonomy, and it should not be: they were
        # grading notes. This is a defect in the checking rather than in the
        # thing checked, a category only the harness itself can see.
        taxonomy="(harness integrity -- not a note defect)",
    ),
    Sabotage(
        "dropped_pertinent_negative",
        "An important negative deleted from the note",
        "The patient explicitly denied blood in the stool and the note no "
        "longer says so. A reader cannot tell whether it was asked and denied "
        "or never asked, which is the whole clinical value of recording it. "
        "The evaluators made this its own category, separate from omission in "
        "general, and until now the harness had no case for it despite "
        "omission.py claiming to catch it.",
        True,
        _drop_first_negated(),
        "salience extraction: a pertinent negative elicited by a direct "
        "question and absent from the note",
        "critical_omission",
        taxonomy="12. Omission of important negatives",
    ),
    Sabotage(
        "internal_contradiction",
        "A note that disagrees with itself",
        "The note asserts both the presence and the absence of the same "
        "finding. The evaluators' most frequent complaint by volume. Neither "
        "claim needs to be ungrounded for the note to be unusable -- what "
        "makes it dangerous is that a reader who scans only one half comes "
        "away confident.",
        True,
        _append("Blood present in stool"),
        "polarity comparison against the matching utterance -> CONTRADICTED",
        "contradicted_claims",
        taxonomy="1. Contradiction",
    ),
    Sabotage(
        "stale_after_correction",
        "A superseded answer kept after the patient corrected it",
        "Patients revise themselves constantly, and the evaluators recorded "
        "models that never updated: the note keeps the first answer after the "
        "patient replaced it. The grader has explicit handling for later "
        "utterances governing earlier ones, and this is the case that decides "
        "whether that handling actually works or merely exists.",
        True,
        _keep_superseded_answer(),
        "polarity comparison finds the later, governing utterance",
        "contradicted_claims",
        taxonomy="2. Contradiction not reported",
    ),
    Sabotage(
        "clinician_voiced_as_fact",
        "Something the doctor suggested, recorded as the patient's report",
        "The clinician floats a possibility the patient never confirms or "
        "denies, and the note records it as an established finding. The "
        "evaluators flagged this repeatedly and it is the subtlest case here: "
        "the words really were said, by the wrong speaker, so every "
        "term-matching check finds its evidence and passes.",
        True,
        _append("Patient reports no kidney problems"),
        "grounding requires a patient-attributable utterance, not merely a "
        "matching one",
        "unsupported_safety_claims",
        taxonomy="3. Symptoms mentioned is reported as fact",
    ),
    Sabotage(
        "harmless_paraphrase",
        "A rewording that changes nothing (control)",
        "Same clinical content, different words. If the harness fires here it "
        "is unusable: clinicians will switch it off, and a disabled gate is "
        "worse than none because it still looks like a control.",
        False,
        _paraphrase(),
        "nothing should fire",
        taxonomy="13. Good behaviour",
    ),
]


@dataclass
class ProofResult:
    sabotage: Sabotage
    blocked: bool
    findings: list[str]
    evidence: str
    #: Rules the *uninjected* baseline already triggers. A case is credited
    #: only for rules the injection adds, because a baseline that already
    #: fails would otherwise let every case claim a catch it did not make.
    baseline_findings: list[str] = field(default_factory=list)
    #: True when the injection left the note byte-identical, so the case
    #: tested nothing. Reported rather than scored -- a vacuous control that
    #: reads as a pass is worse than a missing one. See WIT-013.
    vacuous: bool = False

    @property
    def as_expected(self) -> bool:
        if self.vacuous:
            return False
        return self.blocked == self.sabotage.should_block


def clean_baseline(consultation: Consultation, grader: Grader | None = None) -> str:
    """A note the harness passes, built only from claims it can ground.

    The first version of this harness injected defects into the treating
    clinician's own note. That note does not pass: the lexical grader cannot
    ground roughly a third of it, because clinicians write in abbreviations and
    inference. So every case "blocked" for the same uninteresting reason and the
    false-positive control could never have passed. See WIT-006.

    Sabotage needs a baseline that is clean *by the harness's own standard*, or
    it measures the baseline rather than the injection.
    """
    from witness.graders.lexical import LexicalGrader
    from witness.grounding import Verdict
    from witness.segment import segment

    grounding = (grader or LexicalGrader()).ground_note(
        consultation.consultation_id, "baseline",
        segment(consultation.clinician_note), consultation.utterances,
    )
    kept = [
        g.assertion.text
        for g in grounding.assertions
        if g.verdict in (Verdict.SUPPORTED, Verdict.INFERRED)
    ]
    return ". ".join(kept) + "."


def run_proof(
    consultation: Consultation,
    clean_note: str,
    thresholds: Thresholds | None = None,
    max_critical_omissions: int = 0,
    grader: Grader | None = None,
) -> list[ProofResult]:
    """Inject each defect and check the *named* mechanism fires.

    ``thresholds`` should be the calibrated set. Running sabotage at
    hand-picked thresholds measures the thresholds, not the harness.
    """
    results: list[ProofResult] = []
    gate = Gate(thresholds) if thresholds else Gate()
    pipeline = Pipeline(gate=gate, max_critical_omissions=max_critical_omissions, grader=grader)

    # What the baseline already fails on, before anything is injected.
    # clean_baseline is built from claims the grader can witness, which was
    # supposed to make it pass. It does not always: stripping ungrounded
    # claims removes content, and removed content is an omission. Measuring
    # each injection as a delta from this makes the comparison valid whether
    # or not the baseline is clean, instead of depending on it. See WIT-013.
    base_run = pipeline.run(consultation, clean_note, "baseline", publish=False)
    base_rules = {f.rule for f in base_run.gate.unacknowledged_blocking}
    if base_run.omissions.critical_omissions > max_critical_omissions:
        base_rules.add("critical_omission")

    for s in SABOTAGES:
        if s.key == "weakened_threshold":
            results.append(_prove_threshold(consultation, clean_note, s, thresholds))
            continue

        injected = s.inject(clean_note, consultation)
        if injected == clean_note:
            # The injection matched nothing. Whatever this case reports, it is
            # not about the defect it names.
            results.append(
                ProofResult(
                    sabotage=s,
                    blocked=False,
                    findings=[],
                    evidence="injection matched nothing; this case tested nothing",
                    baseline_findings=sorted(base_rules),
                    vacuous=True,
                )
            )
            continue

        run = pipeline.run(consultation, injected, "sabotage", publish=False)
        blocking = run.gate.unacknowledged_blocking
        # Only rules the injection *added*. A rule the baseline already fires
        # is not evidence that this defect was detected.
        rules = [f.rule for f in blocking if f.rule not in base_rules]
        if (
            run.omissions.critical_omissions > max_critical_omissions
            and "critical_omission" not in base_rules
        ):
            rules.append("critical_omission")
        named = [f for f in blocking if f.rule == s.expect_rule] if s.expect_rule else []
        if s.expect_rule and s.expect_rule in rules and not named:
            # critical_omission is raised by the pipeline rather than the gate,
            # so it has no Finding object to point at.
            named = [Finding(s.expect_rule, Severity.BLOCK, "critical omission added", 0)]
        named = [f for f in named if f.rule not in base_rules]
        # Credit is given only for the mechanism the case claims. A defect
        # stopped by an unrelated rule is a lucky catch, and reporting it as a
        # designed one would be the exact failure this harness looks for.
        evidence = (
            named[0].detail
            if named
            else (
                f"blocked, but by {rules} rather than the expected "
                f"{s.expect_rule!r}" if rules else "no finding raised"
            )
        )
        results.append(
            ProofResult(
                sabotage=s,
                blocked=bool(named) if s.expect_rule else bool(rules),
                findings=rules,
                evidence=evidence,
                baseline_findings=sorted(base_rules),
            )
        )
    return results


def _prove_threshold(
    consultation: Consultation,
    clean_note: str,
    s: Sabotage,
    thresholds: Thresholds | None = None,
) -> ProofResult:
    """Detecting a weakened gate is not a note-level check.

    Nothing about the note changes. What changes is the bar, and the only thing
    that can notice is a fingerprint that covers the bar. This case exists
    because a gate that can be silently relaxed provides the appearance of
    assurance and none of the substance.
    """
    base = thresholds or Thresholds()
    strict = Gate(base)
    loose = Gate(
        Thresholds(
            max_unsupported_rate=min(0.99, base.max_unsupported_rate + 0.15),
            max_safety_unsupported=base.max_safety_unsupported + 90,
            max_contradicted=base.max_contradicted + 90,
        )
    )
    changed = strict.ruleset_hash != loose.ruleset_hash
    return ProofResult(
        sabotage=s,
        blocked=changed,
        findings=["ruleset_hash_changed"] if changed else [],
        evidence=(
            f"ruleset {strict.ruleset_hash} -> {loose.ruleset_hash}; the weakened "
            "gate cannot publish under the original fingerprint"
            if changed
            else "FINGERPRINT UNCHANGED -- the gate was weakened invisibly"
        ),
    )


def taxonomy_coverage(results: list[ProofResult]) -> str:
    """Which of the clinicians' error categories this harness has a case for.

    A sabotage set is only as good as its imagination, and the report's own
    closing line has always admitted that the defects still getting through are
    the ones nobody thought to inject. Tagging each case against the taxonomy
    the PriMock57 evaluators wrote for themselves turns that from a graceful
    admission into a number, and names the categories with no case at all --
    which is the part a reader cannot otherwise see.
    """
    by_key = {r.sabotage.key: r for r in results}
    lines = ["", "  Coverage against the evaluators' own error taxonomy", "  " + "-" * 72]
    covered = tested = out_of_scope = 0
    for category, key in TAXONOMY.items():
        if key is None:
            out_of_scope += 1
            lines.append(f"    {category:<44} out of scope")
            continue
        covered += 1
        r = by_key.get(key)
        if r is None:
            lines.append(f"    {category:<44} NO CASE")
            continue
        if r.sabotage.should_block:
            mark = "caught" if r.blocked else "not caught"
            tested += int(r.blocked)
        else:
            mark = "correctly ignored" if not r.blocked else "FALSE ALARM"
        lines.append(f"    {category:<44} {mark}  ({key})")
    lines += [
        "",
        f"  {covered} of {len(TAXONOMY)} categories have a case; {tested} of those are"
        " caught at this",
        f"  operating point. {out_of_scope} are excluded as prose quality rather than",
        "  faithfulness -- structure, repetition, ordering, acronym choice. A harness",
        "  that policed those would be overreaching, and trimming them out of the",
        "  denominator instead would flatter the coverage figure.",
    ]
    return "\n".join(lines)


def _summary_lines(results: list[ProofResult]) -> list[str]:
    base = results[0].baseline_findings if results else []
    lines: list[str] = []
    vacuous = [r for r in results if r.vacuous]
    if base:
        lines += [
            f"  The uninjected baseline already fails on: {', '.join(base)}.",
            "  Every case above is scored as a delta from that -- credit is given",
            "  only for a rule the injection added. See WIT-013.",
        ]
    if vacuous:
        lines += [
            f"  {len(vacuous)} case(s) tested nothing: "
            f"{', '.join(r.sabotage.key for r in vacuous)}.",
            "  An injection that matches nothing is reported, never counted as a pass.",
        ]
    return lines


def format_report(results: list[ProofResult]) -> str:
    out: list[str] = []
    w = 74
    out.append("=" * w)
    out.append(f"  make prove -- {len(results)} deliberate defects")
    out.append("=" * w)
    for i, r in enumerate(results, 1):
        s = r.sabotage
        verdict = "STOPPED" if r.blocked else "not stopped"
        if not s.should_block:
            verdict = "correctly ignored" if not r.blocked else "FALSE ALARM"
        status = "" if r.as_expected else "   <-- UNEXPECTED"
        out.append("")
        out.append(f"{i}. {s.title}{status}")
        out.append(f"   simulates   {s.simulates}")
        out.append(f"   result      {verdict}")
        out.append(f"   evidence    {r.evidence[:200]}")
        out.append(f"   caught by   {s.caught_by}")

    stopped = sum(1 for r in results if r.blocked and r.sabotage.should_block)
    missed = sum(1 for r in results if not r.blocked and r.sabotage.should_block)
    false_alarms = sum(1 for r in results if r.blocked and not r.sabotage.should_block)
    out.append("")
    out.append("=" * w)
    out.append(
        f"  {len(results)} cases. {stopped} defects stopped, {missed} missed, "
        f"{false_alarms} false alarm(s)."
    )
    out.extend(_summary_lines(results))
    out.append("  This is a statement about these cases and not about the")
    out.append("  harness. The defects still getting through are the ones nobody")
    out.append("  has thought to inject.")
    out.append("=" * w)
    return "\n".join(out)
