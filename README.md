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
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make corpus      # PriMock57, text only — audio is git-lfs and optional
make check       # lint, 63 tests, and every number below
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

`make review` writes a one-file page: click a claim, and the utterance that
justifies it highlights with its timestamp.

## The main result is negative

`make sweep` calibrates the gate against notes written by the consulting
clinicians, at every percentile from 0.50 to 0.99:

| Calibration percentile | Clinician notes accepted | Injected defects caught |
|---|---|---|
| 0.50 | 11% | 29% |
| 0.70 | 30% | 29% |
| 0.80 | 44% | 29% |
| 0.90 | 74% | 14% |
| 0.99 | 93% | 14% |

**There is no operating point at which lexical grounding both accepts notes
written by clinicians and catches a single injected defect.** The best balance
is 44% acceptance at 29% detection.

The cause is in the calibration table: the lexical grader **cannot ground 60% of
a median clinician-written note**, because clinicians write `3/7 hx of
diarrhoea, mainly watery` and term matching has no idea that is the same fact as
*"it's been about three days, it's quite runny"*.

That is a property of the grader, not of the thresholds. No amount of tuning
fixes it, and it is the measured argument for `graders/semantic.py` rather than
an asserted one.

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

At the calibrated operating point the harness catches 1 of 7 injected defects
with 0 false alarms. That number is low and it is reported as it is; see the
sweep above for why, and `graders/semantic.py` for what fixes it.

## What this does not do

Stated plainly, because a harness that overclaims has no standing to measure
anything else.

- **The semantic grader is written but unmeasured here.** It needs a model. The
  container this was built in has no API access and no Ollama, so every number
  in this README is the lexical floor. Run `make validate` with Ollama running
  to measure it.
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
- **Thresholds are calibrated on the same 57 consultations they are evaluated
  on.** There is no held-out split. With 57 encounters there was a choice
  between a calibration set and an evaluation set, and neither would have been
  large enough; the honest response is to say so rather than to report a number
  that implies otherwise.

## Incidents

`docs/INCIDENTS.md` records nine defects found while building this. Three are
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
