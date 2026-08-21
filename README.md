# Witness

**A faithfulness harness for ambient clinical documentation.** Every clinical
claim in a generated note must name the utterance that witnessed it, or be
refused.

> Built and validated entirely on [PriMock57](https://github.com/babylonhealth/primock57)
> (CC BY 4.0) — 57 mock consultations recorded with actors, and 1,140 notes
> graded by five clinicians. **No real patient information was used and none is
> present.**

```bash
git clone https://github.com/<user>/witness.git
cd witness
python3 -m venv .venv && source .venv/bin/activate   # needs Python 3.11+
pip install -r requirements.txt
make corpus      # PriMock57, text only — audio is git-lfs and optional
make check       # lint, 77 tests, and every number below
```

About ninety seconds. No API key, no network egress, no GPU.

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
| ROUGE-L-F1 | −0.092 | 16.3% |
| WER | **+0.085** (wrong sign) | −1.6% |
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

`make crossval` re-derives every figure in this README with nothing scored
in-sample, and **exits non-zero if any of them has drifted more than its stated
tolerance**. It runs inside `make check`. A document making claims about
held-out accuracy with nothing enforcing them would be the same defect this
project exists to find in other people's systems.

`make review` writes a one-file page: click a claim, and the utterance that
justifies it highlights with its timestamp.

## The main result is negative

`make crossval` splits the 57 consultations into five folds. Thresholds are
calibrated on each fold's training consultations and applied to its held-out
ones; the association model behind the learned grader is fitted on training
consultations only. **Every number in this section is out-of-sample.**

| Calibration percentile | Clinician notes accepted | Injected defects caught |
|---|---|---|
| 0.50 | 14% | 41% |
| 0.70 | 23% | 35% |
| 0.80 | 39% | 33% |
| 0.90 | 61% | 20% |
| 0.99 | 88% | 15% |

**There is no operating point at which the gate both accepts notes written by
clinicians and catches most injected defects.** The best balance is 39%
acceptance at 33% detection — and at that setting it also raises a false alarm
on 47% of the paraphrase controls, which is the number that would get it
switched off. 399 defects across 57 held-out consultations.

The cause is in the calibration table: the lexical grader **cannot ground 60% of
a median clinician-written note**, because clinicians write `3/7 hx of
diarrhoea, mainly watery` and term matching has no idea that is the same fact as
*"it's been about three days, it's quite runny"*.

That is a property of the grader, not of the thresholds. So the obvious fix was
tried.

## The obvious fix does not work either, and that is the useful part

`graders/statistical.py` learns the shorthand instead of being told it.
PriMock57 pairs each transcript with the note the treating clinician wrote from
it, which is a parallel corpus, so the mapping is learnable: PPMI over a term
co-occurrence matrix, truncated SVD, no pretrained weights and no network.

Fitted on a training fold it recovers `inhaler` → `salbutamol` (0.89–0.91 in
every fold), `diarrhoea` → `gastroenteritis`, `watery`, and `rash` → `bite`,
`cetirizine`. Nobody wrote those down.

It also thinks `allergic` means `pv` and `active`, and that is left in the
output rather than trimmed out of it. `AssociationModel.nearest` exists so the
lexicon can be read, and WIT-010 is the incident where the first version of
this model was inspected only on the terms that made it look good.

Held out, it does exactly what it was built to do — and the bill comes with it:

| grader | median note grounded | recall on clinician-flagged critical errors |
|---|---|---|
| lexical/1.0 | 40.0% | **48.3%** |
| statistical/1.0 | **66.7%** | 31.6% |

The association floor is the one knob controlling how far from a claim's literal
wording the grader will look. Sweeping it, held out:

| floor | median note grounded | recall | f1 |
|---|---|---|---|
| 0.45 | 66.7% | 31.6% | 24.7% |
| 0.55 | 60.0% | 35.3% | 26.4% |
| 0.65 | 53.3% | 39.3% | 27.6% |
| 0.75 | 44.4% | 45.6% | 29.4% |
| 0.85 | 40.0% | 47.6% | 29.6% |

Monotone in both directions, and at the tight end it converges onto the lexical
baseline it was meant to beat. **Widening the match radius cannot separate
paraphrase from fabrication, because to a term matcher they are the same event:
a claim whose words are not in the transcript.** Every point of coverage is paid
for in detection.

It does buy one real thing — false alarms on paraphrase controls fall from 47%
to 33% — and it is worth having for that. But it does not find an operating
point, and it cannot, for a reason that is now measured rather than asserted.

This is the argument for `graders/semantic.py`. Not "an LLM would probably do
better", but: the failure is a *judgement* about whether two differently-worded
statements assert the same clinical fact, and no amount of distributional
similarity makes that judgement. Something has to read them.

## What the lexical grader does achieve

`make validate`, on 1,140 clinician-graded notes:

```
grader        lexical/1.0   (no model, no network, no API key)

grounding     recall     48.3%   of attributable critical incorrect statements
              precision  21.4%   against one annotator
                         45.6%   against the union of all five

omission      recall     30.0%
              precision  26.0%   against one annotator
                         51.0%   against the union of all five
```

Held out under five-fold cross-validation, that recall is 48.3% — the same
figure. A grader that learns nothing cannot leak anything, and reproducing the
in-sample number out-of-sample is the check that the cross-validation harness
is itself sound.

The five clinicians agree exactly with each other only **27.5%** of the time,
and within one only 55%. No detector can be asked to agree with "the clinicians"
more than the clinicians agree with one another, and roughly 21 points of the
apparent imprecision above is one annotator not writing down what another one
did. Both figures are reported, always, with the ceiling beside them.

## Breaking it on purpose

`make prove` injects eight defects into a known baseline and checks that the
**named** mechanism fires — not merely that something blocked.

The eighth is a false-positive control: a paraphrase that changes no clinical
content and must **not** be flagged. A harness that fires on every rewording is
switched off in week two, and a disabled gate is worse than none because it
still appears in the architecture diagram.

On a single consultation at the calibrated operating point the harness catches
1 of 7 with 0 false alarms. `make crossval` runs the same eight cases against
every one of the 57 consultations with held-out thresholds — 399 detectable
defects — and gets 20% at p=0.90, rising to 41% at p=0.50 as acceptance of
clinician-written notes collapses to 14%. One consultation is not a
measurement; that first number is kept only because it is what `make prove`
prints and the two should not disagree silently.

## What this does not do

Stated plainly, because a harness that overclaims has no standing to measure
anything else.

- **The semantic grader is written but unmeasured.** It needs a model, and the
  container this was built in has no API access, no Ollama and no route to
  pretrained weights. Every number here comes from graders that need neither.
  `make validate` picks up a local Ollama automatically if one is running.
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
- **57 consultations is a small corpus**, and five-fold cross-validation over
  it gives wide confidence intervals on every figure above. Folds remove the
  leakage; they do not manufacture statistical power. Treat differences of a
  few points as noise and the monotone trends as the result.
- **One corpus, one country, one care setting.** Nothing here establishes that
  these numbers transfer to US primary care, to specialty clinics, or to a
  different scribe vendor's output.

## Incidents

`docs/INCIDENTS.md` records eleven defects found while building this. Four are
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
