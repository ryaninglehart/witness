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
