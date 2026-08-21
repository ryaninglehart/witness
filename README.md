# Witness

**A faithfulness harness for ambient clinical documentation.** Every clinical
claim in a generated note must name the utterance that witnessed it, or be
refused.

> All data here is public and synthetic. Witness is built and validated
> entirely on [PriMock57](https://github.com/babylonhealth/primock57) (CC BY
> 4.0) — 57 mock consultations recorded with actors, and 1,140 notes graded by
> five clinicians. **No real patient information was used and none is present.**

---

## The problem

Ambient scribe vendors report accuracy as **edit distance** — "edits per 100
notes", or a trust score derived from how much clinicians change the output.
That measures whether the clinician bothered to fix the note. It does not
measure whether the note is true. A note can be edit-free and still omit a
penicillin allergy.

This is testable, because PriMock57 ships both halves: 24 automatic metrics
computed by the corpus authors, and every note graded by clinicians for
incorrect statements and omissions, each marked critical or not.

Run `make evidence` to reproduce the following on your own machine.

| Signal | Rank correlation with critical errors | Error reduction in its best decile |
|---|---|---|
| ROUGE-2-F1 (best of 24) | −0.094 | **22.9%** |
| BertScore | −0.219 | 22.6% |
| ROUGE-L-F1 | −0.092 | 16.3% |
| WER | **+0.085** (wrong sign) | −1.6% |
| Post-edit effort | −0.713 | 64.9% — but only knowable *after* review |

Selecting the notes the best text-similarity metric likes most reduces the
critical-error burden by under a quarter. Editing effort does much better and
cannot gate anything, because you only learn it once a clinician has already
read and fixed the note.

**And 62.5% of the critical errors clinicians found were omissions** — things
the encounter contained and the note dropped. No similarity metric detects
those at all, and most published work does not measure them.

---

## What Witness does

Four verdicts per clinical claim, each carrying the utterance that justifies it:

- `SUPPORTED` — someone said it, here
- `INFERRED` — nobody said it, but a clinician would write it anyway
  ("up and about most of the day but tires easily" → ECOG 1). Must still cite
  the span it reasoned *from*
- `UNSUPPORTED` — nothing supports it
- `CONTRADICTED` — the encounter says otherwise

`INFERRED` is the reason this is not a wrapper around an entailment model. A
naive checker calls correct clinical reasoning a hallucination.

Verdicts feed a release gate with BLOCK / WARN / INFO severities. Overrides
require a written reason and a named approver, and the ruleset fingerprint
covers **the threshold values themselves** — so a gate that has been quietly
weakened cannot produce artifacts claiming the same ruleset.

## Results

`make validate`, on 1,140 clinician-graded notes:

```
grader        lexical/1.0   (no model, no network, no API key)
recall        48.3%   of attributable critical errors
precision     21.4%   against one annotator
              45.6%   against the union of all five annotators
```

For scale, the five clinicians agree exactly with each other only **27.5%** of
the time. No detector can be asked to agree with "the clinicians" more than the
clinicians agree with one another, and roughly 21 points of the apparent
imprecision above is one annotator not writing down what another one did.

The grader shipped here contains **no model at all** — clinical term overlap
with negation polarity treated as a first-class signal. It is the floor, not
the ceiling, and it exists so that any later claim about an LLM grader has
something interpretable to be measured against.

## Run it

```bash
git clone https://github.com/<user>/witness.git
cd witness

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make corpus      # PriMock57, text only — audio is git-lfs and optional
make check       # lint, tests, evidence, validation
```

About a minute. `make check` runs everything; `make evidence` and
`make validate` reproduce the two tables above on their own.

## What this does not do

Stated plainly, because a harness that overclaims has no standing to measure
anything else.

- **Omission detection is not built yet.** It is 62.5% of the critical errors
  in this corpus and the harder half. Grounding reads the note against the
  transcript and structurally cannot see what is missing.
- **No LLM grader yet.** The interface exists; the lexical grader is the only
  implementation. Every number above is a floor.
- **No ASR layer yet.** The corpus ships human transcripts with utterance
  timings, so grounding is measured against perfect transcription. Real ASR
  error is a second failure surface and is not included here.
- **The corpus is 2022-era model output.** Modern systems are much better. The
  98.4% contamination figure describes *this corpus*, not the state of the art.
  It is used because it is graded, not because it is representative.
- **Primary care, not oncology.** PriMock57 is UK general practice. The
  oncology layer — mCODE extraction, ECOG, staging — is specified but not
  built.
- **Single-annotator ground truth per row.** Mitigated by reporting the
  five-annotator union and the inter-rater ceiling, not solved.

See `docs/INCIDENTS.md` for four defects found while building this, including
one where a silent join dropped 19% of the corpus and the run still printed a
confident number.
