# Incidents

Defects found while building Witness. Each entry pairs the mistake with the
mechanism that now catches its class, because a fix that only closes one
instance leaves the class open.

The log exists for the same reason the harness does. A tool that checks other
systems for confident wrongness has no standing unless it is willing to publish
its own.

---

## WIT-001 — A join silently dropped 19% of the corpus and the run still printed a number

**What happened.** The human-evaluation CSV identifies consultations as
`day1_consultation010`. The transcript and note files are named
`day1_consultation10`. Ids 1 through 9 agree, so the mismatch is invisible in
any spot-check of the first rows.

`validate()` resolved consultations with `dict.get()` and skipped rows that did
not match. **220 of 1,140 annotations were dropped.** The run completed, exited
zero, and reported a validation result computed on 81% of the data. Nothing in
the output said so.

**Why it survived.** The output was a plausible number. There was no reason to
doubt it, because the failure produced a *smaller, cleaner* dataset rather than
an error. It was found only because a separate diagnostic printed the row count
and it did not match the corpus documentation.

**What catches the class now.** `PriMock57.require_consultations()` raises on
any unresolvable id, and every join calls it before grading. A join that cannot
resolve its right-hand side now fails loudly rather than quietly returning
less.

**The generalisation.** A filter and a bug look identical from the outside. Any
step that can silently reduce a dataset must be able to say how much it
dropped, or must refuse to drop anything.

---

## WIT-002 — Every note failed, so the harness looked strict and discriminated nothing

**What happened.** The segmenter emitted every fragment of a note, including
"Hello", "Thanks very much" and section labels. The grader marked any fragment
with no clinical content as `UNSUPPORTED`.

Consequently **every note contained at least one failure**. The note-level flag
rate was 100%, detection was 100%, and the false-alarm rate was 100%. The
harness appeared rigorous and separated nothing.

**Why it survived.** Each component behaved as specified. The segmenter split
text; the grader graded what it was given; the gate blocked on failures. The
defect was in the seam — a fragment that is not a clinical claim should never
have become an assertion, and no component owned that judgement.

**What catches the class now.** Non-clinical fragments are dropped during
segmentation and never reach the grader. `test_non_clinical_fragments_are_dropped`
pins it.

**The generalisation.** A detector that fires on everything is indistinguishable
from one that fires on nothing. Any flag rate at or near 100% is a bug report
about the detector, not a finding about the data.

---

## WIT-003 — Note-level ranking was the wrong evaluation, and it flattered the result

**What happened.** Witness was first evaluated by ranking notes by risk and
measuring the error reduction in the safest decile — the same shape as the
comparison against text-similarity metrics.

The number was 15.9%, below the 22.9% baseline. Both figures are close to
meaningless. **98.4% of notes in this corpus contain at least one critical
error**, so every decile of every ranking is contaminated and the metric mostly
measures the base rate.

**Why it survived.** It produced a defensible-looking comparison against a
published baseline, and the first instinct on seeing 15.9% was to tune the
grader — which would have optimised against a measurement that could not
distinguish a good detector from a bad one.

**What catches the class now.** Evaluation is assertion-level. The annotators
recorded the *text* of each error, so precision and recall are computed against
3,471 specific labelled claims rather than against note-level counts.
`witness.matching` does the alignment and reports how many findings could not
be attributed to a claim rather than absorbing them.

**The generalisation.** When the base rate approaches 100%, ranking metrics
saturate and stop carrying information. Check the base rate before choosing the
evaluation, not after the number disappoints.

---

## WIT-004 — Apparent imprecision was mostly annotator disagreement

**What happened.** Assertion-level precision came out at 21.4%, which reads as
four false alarms for every true one.

Measured against the union of all five annotators' findings on the same note
rather than one annotator's, precision is **45.6%**. Roughly **21 points of the
apparent error was one clinician not writing down what another one did.**

**Why it matters.** The five clinicians agree exactly on critical-error counts
only 27.5% of the time, and agree within one only 55% of the time. Any
precision figure quoted against a single annotator is measuring that annotator's
recall as much as the detector's precision.

**What catches the class now.** Both figures are reported, always, with the
inter-annotator agreement beside them as the ceiling. The scoping review
(medRxiv, Jan 2025) names unreported inter-rater reliability as one of the
field's four systematic failures; it costs nothing to report and it changes how
every other number should be read.

**The generalisation.** Before treating disagreement with ground truth as model
error, measure how much the ground truth disagrees with itself.

---

## WIT-005 — Coverage was computed over filler, so nothing was ever covered

**What happened.** A salient fact's identity was every content token in the
utterance it came from. A patient saying *"Oh hey, um, I've just had some
diarrhoea for the last three days"* produced a fact with eleven terms, and
coverage asked whether the note reproduced 60% of them.

No note does. **Coverage came out at 0.0 on every consultation** and the module
reported 24 omissions per note, against a clinician-labelled ground truth of
roughly five.

**Why it survived.** The number was not obviously absurd — omission detectors
are expected to be noisy — and 0.0 coverage reads like a strict tool rather than
a broken one.

**What catches the class now.** A fact's identity is its *clinical core*: the
terms the treating clinician also recorded, plus anything a safety pattern
matched. Filler cannot enter the comparison.

**The generalisation.** When a rate comes out at exactly 0.0 or exactly 1.0
across an entire corpus, the measurement is broken. Saturation is a bug report.

---

## WIT-006 — The sabotage harness gave itself credit for mechanisms that never fired

**What happened.** Seven defects were injected and the harness reported all
seven "STOPPED", against named mechanisms: polarity comparison, the safety-slot
rule, salience extraction.

Six of them were actually stopped by the generic unsupported-rate rule. The
report named a mechanism; the evidence line beside it named a different one, and
the two were never compared.

**Why it survived.** Every case was blocked, so the summary said `7 stopped, 0
missed`. The harness was asked "did something block?" when the question it
claimed to answer was "did *this* block?".

**What catches the class now.** Each sabotage declares `expect_rule`, and a case
counts as caught only when that specific rule fires. Everything else is reported
as *"blocked, but by X rather than the expected Y"*.

**The generalisation.** This is the project's own thesis turned on itself: a
check verifies the value it names, not the value the reader sees. A test suite
that asserts "an error occurred" rather than "this error occurred" passes for
the wrong reasons and keeps passing after the mechanism it describes has stopped
working.

Related: the harness had also been injecting defects into the treating
clinician's own note, which the grader already fails. Every case blocked before
any injection, so the false-positive control could not have passed. A sabotage
baseline must be clean *by the harness's own standard*.

---

## WIT-007 — Tiering made every clinician-written note fail

**What happened.** Any utterance sharing a term with the treating clinician's
highlights was tiered `CRITICAL`. Highlights are broad (`3/7 hx of diarrhea`),
so most of a consultation qualified, and `critical_omission` fired on all 57
notes — including the ones clinicians wrote themselves.

**What catches the class now.** `CRITICAL` is reserved for categories whose
omission is dangerous: allergies, medications and doses, red flags, pregnancy.
Clinician highlights are strong evidence of salience and are tiered `MAJOR`.

**The generalisation.** A severity scale where most things are the top severity
is not a scale.

---

## WIT-008 — Thresholds chosen by judgement rejected 57 of 57 clinician notes

**What happened.** The gate's opening thresholds — no contradictions, no
ungrounded safety claims, under 15% ungrounded — were picked because they
sounded defensible. Measured against notes written by the consulting clinicians,
**they rejected every single one.**

**Why it matters more than the other four.** A gate that fails everything gets
switched off in week two, and a switched-off gate is worse than no gate: it
still appears in the architecture diagram and the compliance narrative while
doing nothing.

**What catches the class now.** `witness.calibration` derives every threshold
from a percentile of the human-written distribution and publishes the resulting
operating point — the fraction of clinician notes the gate would flag. A
threshold with no reference population is a guess with a decimal point.

**A second finding, from the same fix.** One aggregate safety ceiling is the
wrong shape. Calibrated against real notes it lands at five, and at five a
single fabricated allergy publishes. Allergy claims are rare, so their own
ceiling can be zero while the aggregate stays loose enough to accept human
writing. Ceilings are now derived per slot, and they are inside the ruleset
fingerprint.

**And the finding that outranks all of it.** Sweeping the calibration percentile
from 0.50 to 0.99 shows there is **no operating point** at which the lexical
grader both accepts clinician-written notes and catches a single injected
defect. Its best balance is 44% of human notes accepted at 29% detection. The
cause is in the calibration table: it cannot ground 60% of a median human note,
because clinicians write in abbreviation and inference.

That is a property of the grader, not of the thresholds, and no amount of
tuning fixes it. It is the measured argument for `graders/semantic.py`, and it
is why the lexical grader is documented as a floor rather than shipped as an
answer.

---

## WIT-009 — The fix for WIT-005 was not applied to the facts that mattered most

**What happened.** WIT-005 narrowed a salient fact's identity to its clinical
core, so coverage was no longer scored over filler. The safety branch was
exempted from that narrowing and kept `core |= terms` — every token in the
utterance.

So for allergies, medications and red flags — the categories where a missed
omission is dangerous rather than untidy — the original bug survived. A patient
saying *"Um, I have been allergic to penicillin since I was a child"* produced a
five-term fact including `um`, `since` and `child`. A note reading *"Allergic to
penicillin"* covers two of those five, falls under the 65% threshold, and the
correctly recorded allergy was reported as omitted.

**Why it survived.** The aggregate numbers moved in the right direction when
WIT-005 was fixed — omissions per note fell from 24 to 8.3 — so the fix looked
complete. The exemption was one line inside the branch that the aggregate
improvement was hiding.

**What catches the class now.** The safety branch contributes only the terms its
pattern actually matched, not the whole utterance.
`test_a_carried_fact_is_not_reported_as_omitted` pins it, and that test is what
found this.

**The generalisation.** When a fix improves an aggregate, check whether it
applied to every branch or only to the common one. The uncommon branch here was
the safety-critical one, which is the usual way round: the rare path is both the
one that matters most and the one least exercised by summary statistics.

---

## WIT-010 — A learned lexicon whose nearest neighbour for `hx` was `yeah`

**What happened.** The first association model built one document per
consultation — the union of every utterance plus the clinician's note — took
PPMI over the term co-occurrence matrix, and reduced it with a truncated SVD.
On inspection it had learned real clinical structure: `inhaler` → `salbutamol`
at 0.91, `abdo` → `bowels` at 0.85.

It had also learned that `hx` means `yeah` (0.87), `you're` (0.87) and `uh`
(0.86).

**Why.** Two causes, both structural rather than accidental.

With 57 documents, a term appearing in most consultations co-occurs with
everything, because in a document that size it genuinely does. Filler is the
most frequent thing in a transcript, so filler ended up adjacent to every term
in the space. There was no document-frequency ceiling.

And raw PMI is biased toward rare terms: a term seen twice earns an enormous
score with whatever it happened to sit beside. `hx` appears in a handful of
notes, so its row was almost entirely noise, and the SVD faithfully preserved
the noise.

**Why it nearly shipped.** The terms anyone checks first are the interesting
ones. `inhaler` → `salbutamol` is a striking result and it is genuine, and
having seen it there is very little pull toward typing in a boring term to see
whether the answer is garbage. The failure was not in the method, it was in the
inspection: the sample was chosen by what would look good.

**What changed.** A document-frequency ceiling (a term in more than half the
consultations is dropped rather than learned), context distribution smoothing
at 0.75 on the context marginal — the standard correction for the rare-term
bias — and two co-occurrence views rather than one: cross-view counts pairing
note terms with transcript terms, which is the parallel-corpus signal, plus
local counts over a sliding window of utterances, which supplies the
observations 57 documents cannot.

`test_frequent_terms_are_excluded_from_the_lexicon` pins the ceiling.
`AssociationModel.nearest` exists so the lexicon can be read; a learned mapping
nobody inspects is a hand-written one with worse provenance.

**The generalisation.** When inspecting something learned, pick the sample
before looking at the output. Terms chosen after the fact select for the
result you hoped for.

---

## WIT-011 — Every published number was scored in-sample

**What happened.** Nothing broke. The README said, in the limitations section:
*"Thresholds are calibrated on the same 57 consultations they are evaluated on.
There is no held-out split."* That was true, disclosed, and still wrong to
report numbers from.

It became untenable rather than merely untidy when the statistical grader was
added, because that grader *learns* from the corpus. Fitted on all 57 and
scored on the same 57, it would have reported grounding far above what it can
actually do, and the figure would have been indistinguishable from a real one.

**What changed.** `witness/crossval.py`, and `make crossval` in the default
gate. Five folds by stride over sorted consultation ids — deterministic, no
seed, and striding rather than blocking so a recording day is not confined to
one fold. For each fold the association model is fitted on the training
consultations only, and thresholds are calibrated from the training
consultations' out-of-fold groundings. Every consultation is evaluated exactly
once, by a model and a threshold set derived only from the others.

The lexical grader is run over the same partition despite having nothing to
leak, because comparing a held-out number to an in-sample one is the error this
module exists to prevent. Its held-out recall came out at 48.3% — identical to
the previously published in-sample figure, which is the expected result for a
grader that learns nothing and is the check that the harness itself is sound.

**What it found.** The statistical grader grounds 66.7% of a median
clinician-written note against the lexical grader's 40.0%, held out. It also
drops recall against clinician-identified critical errors from 48.3% to 31.6%.
Sweeping the association floor traces a monotone trade between the two, and at
the tight end the learned grader converges exactly onto the lexical baseline.

Widening the match radius cannot separate paraphrase from fabrication, because
to a term matcher they are the same event: a claim whose words are not in the
transcript. That is now measured rather than argued.

**The generalisation.** A disclosed limitation is still a limitation. Writing
it down converts a hidden error into an honest one; it does not convert it into
a result.
