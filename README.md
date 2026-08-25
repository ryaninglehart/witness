# Witness

**A faithfulness harness for ambient clinical documentation.** Every clinical
claim in a generated note must name the utterance that witnessed it, or be
refused.

> Built and validated entirely on [PriMock57](https://github.com/babylonhealth/primock57)
> (CC BY 4.0) — 57 mock consultations recorded with actors, and 228 generated
> notes each graded by five clinicians, for 1,140 annotation rows. **No real
> patient information was used and none is present.**

```bash
git clone https://github.com/<user>/witness.git
cd witness
python3 -m venv .venv && source .venv/bin/activate   # needs Python 3.11+
pip install -r requirements.txt
make corpus      # PriMock57, text only — audio is git-lfs and optional
make check       # lint, 156 tests, and the enforced figures
```

About a minute. No API key, no network egress, no GPU.

`make check` is the whole gate: lint, tests, the premise analysis, validation,
calibration, the sabotage harness, and the held-out cross-validation — which
**exits non-zero if any figure printed below has drifted past its tolerance**.
CI additionally runs the evaluation twice in separate processes and diffs the
output, because a number can be stable inside one run and drift between two.

## The result, in one line

Term matching cannot gate clinical notes — measured twice, two different
graders, no operating point, and coverage always paid for in recall. A 7B model
reading for meaning escapes that entirely: **+34.6 points of grounding and
+25.8 points of recall at the same time**, both intervals clear of zero, on 56
of 57 notes. Getting there needed the model's output constrained to a schema —
constraining it merely to valid JSON made things measurably worse. It is a
result about a grader on one corpus, not a shippable component.

The rest of this document is how each of those was established, in the order it
happened.

---

## The problem

Ambient scribe vendors report accuracy as **edit distance** — "edits per 100
notes", or a trust score derived from how much clinicians change the output.
That measures whether the clinician bothered to fix the note. It does not
measure whether the note is true. A note can be edit-free and still omit a
penicillin allergy.

PriMock57 lets this be tested rather than asserted: it ships 24 automatic
metrics computed by the corpus authors *and* every note graded by clinicians
for incorrect statements and omissions, each marked critical or not.

`make evidence`:

| Signal | Rank correlation with critical errors | Error reduction in its best decile |
|---|---|---|
| ROUGE-2-F1 (best of 24) | −0.094 | **22.9%** |
| BertScore | −0.219 | 22.6% |
| ROUGE-3-F1 | −0.078 | 19.5% |
| METEOR | −0.184 | 19.4% |
| WMD | −0.088 | 19.1% |
| WER (worst of 24) | **+0.085** (wrong sign) | −1.6% |
| Post-edit effort | −0.713 | 64.9% — but only knowable *after* review |

Selecting the notes the best text-similarity metric likes most cuts the
critical-error burden by under a quarter. Editing effort does far better and
cannot gate anything, because you only learn it once a clinician has already
read and fixed the note.

**62.5% of the critical errors clinicians found were omissions** — things the
encounter contained and the note dropped. No similarity metric detects those,
and most published work does not measure them.

## What Witness does

Four verdicts per clinical claim, each carrying the utterance that justifies it:

- `SUPPORTED` — someone said it, here
- `INFERRED` — nobody said it, but a clinician would write it anyway
  ("up and about most of the day but tires easily" → ECOG 1). Must still cite
  the span it reasoned *from*
- `UNSUPPORTED` — nothing supports it
- `CONTRADICTED` — the encounter says otherwise

`INFERRED` is why this is not a wrapper around an entailment model: a naive
checker calls correct clinical reasoning a hallucination.

Verdicts and omissions feed a release gate with BLOCK / WARN / INFO severities.
Overrides need a written reason and a named approver. The ruleset fingerprint
covers **the threshold values themselves**, so a gate that has been quietly
weakened cannot produce artifacts claiming an unchanged ruleset.

`make crossval` re-derives the held-out figures with nothing scored in-sample.
It **exits non-zero** if any of eight enforced figures drifts past its
tolerance, if the radius trade stops being monotone, or if the false-positive
control stops rewording the notes it is given. The eight: the two
median-grounded rates, the two recalls, the lexical best-balance pair, and the
two false-alarm rates. Other numbers here are printed by their own commands and
are not threshold-guarded — a smaller claim than "every figure is enforced",
and the true one.

`make review` writes a one-file page: click a claim, and the utterance that
justifies it highlights with its timestamp.

## The main result is negative

`make crossval` splits the 57 consultations into five folds. Thresholds are
calibrated on each fold's training consultations and applied to its held-out
ones; the association model behind the learned grader is fitted on training
consultations only. **Every number in this section is out-of-sample.**

| Calibration percentile | Clinician notes accepted | Injected defects caught |
|---|---|---|
| 0.50 | 12% | 15% |
| 0.70 | 28% | 11% |
| 0.80 | 39% | 11% |
| 0.90 | 63% | 2% |
| 0.95 | 74% | 0% |
| 0.99 | 88% | 0% |

**There is no operating point at which the gate both accepts notes written by
clinicians and catches more than a small minority of injected defects.** The
best usable balance is 39% acceptance at **11%** detection, and at the
percentiles where most clinician notes get through it catches essentially
nothing.

506 note-level defects across 57 held-out consultations: eleven defect types,
each derived from the note or the transcript rather than written against one
encounter, each tagged to the error taxonomy the corpus evaluators wrote for
themselves. They fire on 25–57 of the 57 consultations depending on type —
`dropped_red_flag` needs a red flag to drop, so it is live on 25.

A twelfth case, the silently weakened gate, is **excluded from that rate**. It
changes no note and is caught by the ruleset fingerprint on all 57
consultations by construction. Counting it put a guaranteed hit into every
operating point and roughly doubled the published detection figure; see
WIT-015.

False alarms on paraphrase controls are 4%. An earlier version of this README
said 47%, which was wrong in the harness's favour and against it at the same
time — see WIT-013. The control was a no-op on 56 of 57 consultations, so it
was testing nothing, and the 47% was the *baseline* failing the gate rather
than any reaction to rewording.

What the calibration table shows is that the lexical grader **cannot ground 55%
of a median clinician-written note**. The examples point at clinical shorthand —
clinicians write `3/7 hx of diarrhoea, mainly watery` and term matching has no
idea that is the same fact as *"it's been about three days, it's quite runny"* —
but that attribution is read off examples rather than a measured decomposition
of the 60%.

The sweep does show the thresholds are not the binding constraint, since no
percentile helps. So the obvious fix was tried.

## The obvious fix does not work either, and that is the useful part

`graders/statistical.py` learns the shorthand instead of being told it.
PriMock57 pairs each transcript with the note the treating clinician wrote from
it, which is a parallel corpus, so the mapping is learnable: PPMI over a term
co-occurrence matrix, truncated SVD, no pretrained weights and no network.

Fitted on a training fold it recovers `inhaler` → `salbutamol` at 0.88–0.92
across the five folds, along with `diarrhoea` → `gastroenteritis`, `watery` and
`rash` → `bite`. Nobody wrote those down.

The rest of the lexicon is less tidy and is left that way. `allergic` gets `pv`
and `active` in two folds and `reaction`, `peanut` in others; `rash` →
`cetirizine` appears in the top four of only one fold. Neighbour lists are
fold-specific and the examples above are the stable ones — which is a claim
about which examples were chosen, not about the method.
`AssociationModel.nearest` exists so the lexicon can be read, and WIT-010 is the
incident where the first version of this model was inspected only on the terms
that made it look good.

Held out, it does exactly what it was built to do — and the bill comes with it:

| grader | median note grounded | recall on clinician-flagged critical errors |
|---|---|---|
| lexical/1.0 | 45.0% [37.5, 50.0] | **48.7%** [43.7, 53.4] |
| statistical/1.0 | **66.7%** [57.1, 75.0] | 32.5% [28.1, 36.8] |

Brackets are 95% bootstrap intervals over 2,000 resamples of whole
consultations — the consultation, not the note, because notes from one
encounter share a transcript, a patient and a clinician. Paired on identical
resamples, so the shared difficulty of a given encounter cancels:

```
grounding, statistical - lexical   +21.7%  [+12.7%, +28.6%]   distinguishable
recall,    statistical - lexical   -16.2%  [-19.3%, -13.2%]   distinguishable
```

Neither interval is near zero. **The trade is not a small-sample artifact** —
this corpus is small, and it is still large enough to see both halves of it
clearly.

The association floor is the one knob controlling how far from a claim's literal
wording the grader will look. Sweeping it, held out:

| floor | median note grounded | recall | f1 |
|---|---|---|---|
| 0.45 | 66.7% | 32.5% | 24.8% |
| 0.55 | 58.8% | 36.2% | 26.4% |
| 0.65 | 55.6% | 39.6% | 27.2% |
| 0.75 | 50.0% | 45.8% | 28.9% |
| 0.85 | 47.6% | 47.9% | 29.2% |

Monotone in both directions, and at the tight end it converges onto the lexical
baseline it was meant to beat. **On this corpus, across this family of graders,
widening the match radius does not separate paraphrase from fabrication** — to a
term matcher they are the same event, a claim whose words are not in the
transcript. Every point of coverage is paid for in detection. That is a result
about term matching over 57 UK primary-care encounters, not a proof about
distributional methods in general.

It buys nothing on the joint trade either. Its best usable balance is 28%
acceptance at 11% detection, against the lexical grader's 39% at 11% — no
better, having paid 16 points of recall for it. False alarms are 2–4%, so there is not even a specificity
argument left once the control actually controls for something.

This is the argument for `graders/semantic.py`. Not "an LLM would probably do
better", but: the failure is a *judgement* about whether two differently-worded
statements assert the same clinical fact, and no amount of distributional
similarity makes that judgement. Something has to read them.

## What does work

`make semantic` runs `graders/semantic.py` against `qwen2.5:7b-instruct` over
Ollama, with generation constrained to a verdict schema. 57 consultations, 174
calls, 61 minutes on a laptop. Every response ships with this repository, so
the figures below re-derive in seconds with no model, no key and no network:

```
make semantic-replay
```

**56 of the 57 notes survive**, and all three graders are scored on those same
56:

| grader | note grounded | recall | precision |
|---|---|---|---|
| lexical/1.0 | 44.7% [37.5, 50.0] | 39.8% [28.9, 50.0] | 16.5% |
| statistical/1.0 | 66.7% | 23.7% | 14.6% |
| **semantic/1.0 + qwen2.5:7b** | **79.3%** [69.0, 87.5] | **65.6%** [53.4, 76.4] | 23.5% |

Paired bootstrap over the same resamples:

```
grounding, semantic - lexical   +34.6%  [+21.4%, +46.1%]   distinguishable
recall,    semantic - lexical   +25.8%  [ +9.3%, +41.7%]   distinguishable
```

**Both clear of zero.** The semantic grader verifies nearly twice as much of a
clinician's note *and* catches substantially more of what the clinicians
flagged. Every term-matching grader in this project pays for coverage in recall,
monotonically — the radius sweep prices it at about a point of recall per point
of grounding. The semantic grader takes thirty-five points of grounding and
gains twenty-six points of recall at the same time.

The trade-off that binds term matching does not bind a model reading for
meaning. That is the question the whole harness was built to ask, and this is
the answer.

### Getting here took three attempts at one problem

The model kept abandoning the required output format and narrating instead,
which refused every claim in the affected batches. Each fix moved the failure
rather than removing it, and the sequence is the useful part:

| constraint | responses unreadable |
|---|---|
| none — prompt asks for JSON | 29% |
| `format: "json"` — syntax constrained | **48%** |
| verdict schema — structure constrained | 6% first attempt, **1%** after one retry |

Constraining syntax made it *worse*. Told only to emit JSON, the model stopped
writing prose and started writing `{"claims":[0,1,2,3]}` — valid, correctly
formed, listing the claims it was asked to grade and grading none of them.
Removing its ability to ramble did not give it the ability to comply; it moved
the failure into a shape that looks like an answer. Only constraining the
*structure* fixed it. See WIT-021.

### What is still true

- **The 56 are not a random sample.** They are the notes the model stayed on
  format for. One note is still excluded, so the effect is now small, but the
  direction of any bias is unknown.
- **One model, one corpus, one care setting.** 57 UK primary-care encounters
  and a single 7B model. Nothing here establishes transfer.
- **The residual failure moved again rather than vanishing.** What survives is
  `{"verdicts": []}` — schema-valid and empty. At 1% after retry it is not
  worth another constraint, and it is recorded rather than fixed.
- **Precision is 23.5%**, below the 32.7% measured when 20 notes were excluded.
  The notes that came back are harder. That is the honest direction for a
  figure to move when a sample stops being filtered.

## What the lexical grader does achieve

`make validate`, on 228 generated notes and the 1,140 annotations of them:

<!-- Pasted verbatim from the command. WIT-014: a block that only looks
     like output is how four invented figures survived here. -->
<!-- verify: validate -->
```
  Does Witness flag the claims the clinicians flagged?
  ------------------------------------------------------------------
  grader                              lexical/1.0  (no model, no network)
  clinician critical findings           3471
  not attributable to a claim           1163  (34%, structural comments)
  flags raised                           980

  grounding     recall     48.7%   of attributable critical incorrect statements
                precision  21.0%   against one annotator
                           39.7%   against the union of all five

  omission      recall     77.7%   of attributable critical omissions
                            1666 of 5792 are attributable (29%); the extractor never
                                 nominated the rest, so they are counted
                                 as neither hit nor miss. Of all 5792, 22.4% are found.
                precision  15.0%   against one annotator
                           41.1%   against the union of all five

  For scale -- the five clinicians against each other, all 2280 pairs:
  exact agreement 19.8%   within one 47.1%   rho +0.315
  No detector can be asked to agree with 'the clinicians' more
  than the clinicians agree with one another. The gap between the
  two precision figures above is that disagreement, measured.
```

Held out under five-fold cross-validation, that recall is 48.7% — the same
figure. A grader that learns nothing cannot leak anything, so the out-of-fold
scoring path should reproduce the in-sample one exactly, and it does. That
checks the scoring plumbing. It does not test the fold partition of the fitted
model, which is the only place leakage was ever possible — a stateless grader
is insensitive to that by construction.

The five clinicians agree exactly with each other only **19.8%** of the time,
and within one only 47.1%, across all 2,280 pairs — and that is agreement on the
*count* of critical statements per note, not on which ones, so it is context for
the precision figures rather than a formal bound on them. No detector can be asked to agree with "the clinicians"
more than the clinicians agree with one another, and the ~19-point gap between
the two grounding precision figures is one annotator not writing down what
another one did.

Neither number is the truth. The single-annotator figure understates precision
by charging the detector for disagreement; the union overstates it by accepting
anything anyone flagged. Both are reported, always, with the ceiling beside
them, because picking whichever one flatters the detector is the thing this
project objects to elsewhere.

Omission recall is high and its precision low, and that is one fact rather than
two: the salience extractor nominates generously, so it catches most of what
the clinicians said was missing and nominates a great deal besides.

## Breaking it on purpose

`make prove` injects twelve cases into a known baseline and checks that the
**named** mechanism fires — not merely that something blocked.

The twelfth is a false-positive control: a paraphrase that changes no clinical
content and must **not** be flagged. A harness that fires on every rewording is
switched off in week two, and a disabled gate is worse than none because it
still appears in the architecture diagram.

**On one consultation at the calibrated p=0.90 point, 1 of 11 cases is stopped
— and it is the ruleset-fingerprint case, which is not a defect in a note at
all.** No note-content defect is caught there. The taxonomy table the command
prints says the same thing in its own words: *0 of those are caught at this
operating point*.

The command exits non-zero if the control fires on a rewording, if any
injection matches nothing, or if a silently weakened gate goes undetected. Low
detection is the reported state of the project and does not fail the build;
those three do, because each means a number above is not measuring what it
says.

`make crossval` runs the same cases against all 57 consultations with held-out
thresholds — 506 note-level defects — and gets 2% at p=0.90, rising to 15% at
p=0.50 as acceptance of clinician-written notes collapses to 12%.

Two subtractions make those numbers mean what they say. A case is credited only
when the rule it *names* fires (WIT-006), and only when the uninjected baseline
was not already failing on that rule (WIT-013). The baseline fails on its own
for 21 of 57 consultations at p=0.90 and 31 of 57 at p=0.80 — it is a function
of the threshold, not a fixed property — so without the second subtraction many
cases would claim catches they never made.

Cases are tagged against the 13-category error taxonomy the PriMock57
evaluators wrote for themselves; `make prove` prints coverage against it. Eight
categories have a case, five are out of scope as prose quality rather than
faithfulness — and those five stay in the denominator, because trimming them
out would flatter the number.

## What this does not do

Stated plainly, because a harness that overclaims has no standing to measure
anything else.

- **The semantic grader is measured on 38 of 57 notes, not 57.** It abandons
  the output format on 29% of batches and those notes are excluded rather than
  scored. It is not a deployable component until that is fixed, and the fix
  (constrained decoding) has not been tried.
- **The gate has never been calibrated against the semantic grader.** Every
  operating point, sweep and sabotage figure above comes from the offline
  graders. What the release gate does with a grader this much better is
  unmeasured, and it is the obvious next run.
- **No ASR layer.** The corpus ships human transcripts with utterance timings,
  so grounding is measured against perfect transcription. Real ASR error is a
  second failure surface and is not included. Whisper's documented tendency to
  fabricate on disfluent speech (Koenecke et al., FAccT 2024) matters most for
  the sickest patients, and is not measured here.
- **Primary care, not oncology.** PriMock57 is UK general practice. The mCODE
  extraction layer — staging, ECOG, regimens — is specified in the design and
  not built.
- **The corpus is 2022-era model output.** Modern systems are much better. The
  98.4% contamination figure describes *this corpus*, not the state of the art.
  It is used because it is graded, not because it is representative.
- **Single-annotator ground truth per row.** Mitigated by reporting the
  five-annotator union and the inter-rater ceiling, not solved.
- **57 consultations is a small corpus.** Folds remove the leakage; they do not
  manufacture statistical power. Rather than asking the reader to discount the
  numbers by an unstated amount, every headline figure carries a bootstrap
  interval. They are not symmetric and the widest matters most: statistical
  grounding is 66.7% but its interval runs [57.1, 75.0]. Lexical grounding runs
  [37.5, 50.0] and both recalls are about ±5. The two differences the argument rests on are far
  larger than any of that, and are reported as paired intervals.
- **One corpus, one country, one care setting.** Nothing here establishes that
  these numbers transfer to US primary care, to specialty clinics, or to a
  different scribe vendor's output.

## How this was built

Written with Claude Code, daily, under the conventions in `CLAUDE.md`. Every
rule in that file was added after something went wrong, and each one cites the
incident that produced it.

That is the part worth looking at. The assistant's failure mode is not bad code
— it is plausible code with a silent defect, meeting an author willing to accept
a number that matches what he expected. Of the seventeen incidents in
`docs/INCIDENTS.md`, the ones that survived longest are the ones where the wrong
answer was *flattering*: a false-alarm rate that fit the story being told, a
learned lexicon inspected only on the terms that made it look good, three
counting errors that each moved a figure the helpful way.

So the repository is arranged so that being wrong is expensive: figures carry
tolerances and fail the build when they drift, gates are verified failing as
well as passing, documented output is checked against real output, and a
separate agent was pointed at the documentation twice with instructions to be
adversarial. Those two audits found 23 defects in about an hour, including four
numbers printed as command output that no command produced.

## Incidents

`docs/INCIDENTS.md` records twenty-one defects found while building this. Five are
worth reading:

**WIT-001** — a join dropped 19% of the corpus and the run still printed a
confident number. A filter and a bug look identical from the outside.

**WIT-006** — the sabotage harness reported "7 stopped, 0 missed" while six of
the seven were caught by a rule other than the one the report named. The
project's own thesis turned on itself: a check verifies the value it names, not
the value the reader sees.

**WIT-008** — the first thresholds, chosen by judgement, rejected 57 of 57 notes
written by clinicians. Thresholds now come from a percentile of the human
distribution and the operating point is published.

**WIT-010** — the learned lexicon's nearest neighbour for `hx` was `yeah`. The
method was sound; the inspection was not. `inhaler` → `salbutamol` is a
striking result and it is real, and having seen it there is very little pull
toward typing in a boring term to check whether the answer is garbage. Choose
the sample before looking at the output.

**WIT-011** — every published number was scored in-sample. It was disclosed in
this README, which made it an honest error rather than a hidden one, and not a
result either. `make crossval` now runs in the default gate.

**WIT-015** — an adversarial audit found three counting errors that each moved
a published figure the flattering way: a case hard-coded to "caught" sitting
inside the detection rate, an operating-point selector recommending a gate that
false-alarms on a third of rewordings, and the inter-rater ceiling measured on
one consultation out of 57 and quoted as the corpus. Detection at the best
usable point fell from 21% to 12%. None was a lie and none was careless in
isolation; all five went the same direction, which is what selection pressure
looks like when the person checking is the person hoping.
