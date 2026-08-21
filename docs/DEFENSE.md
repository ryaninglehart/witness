# Defending Witness

Notes for talking about this project with engineers who will push on it. Every
number here is reproducible with `make check`; nothing in this document should
be said if the command stops printing it.

---

## The thirty-second version

Ambient scribe vendors report accuracy as edit distance — edits per note, or a
trust score derived from how much the clinician changed the output. I tested
whether that predicts clinical safety, using the one public corpus where both
things are measured, and it does not: the best of 24 text-similarity metrics
cuts the critical-error burden by 22.9% in its best decile, and 62.5% of the
critical errors clinicians found were **omissions**, which no similarity metric
detects at all.

So I built a harness where every clinical claim in a note must name the
utterance that witnessed it or be refused, put a release gate on top of it, and
then measured the gate honestly enough that it failed.

## The result is negative. Lead with that.

There is no operating point where the gate both accepts notes written by
practising clinicians and catches most injected defects. Best balance: 39%
acceptance at 33% detection, with false alarms on 47% of paraphrase controls.

If someone reads that as a failed project, the answer is that a harness which
reported success here would have been wrong, and I would not have known. The
deliverable is the measurement.

## Questions I expect, and the honest answers

**"Isn't this just an entailment model with extra steps?"**

No, and the reason is the `INFERRED` verdict. A clinician writing "up and about
most of the day but tires easily" → ECOG 1 is doing correct clinical reasoning
that no utterance literally supports. A naive entailment checker calls that a
hallucination, floods the reviewer with false positives, and gets switched off.
`INFERRED` claims must still cite the span they reasoned *from*, which is
enforced in the type: a `GroundedAssertion` raises on construction if it is
`SUPPORTED` or `INFERRED` with no spans.

**"Your precision is 21%. That's unusable."**

It is, and the ceiling matters more than the number. The five clinicians who
graded this corpus agree exactly with each other 27.5% of the time and within
one grade 55% of the time. Measured against the union of all five annotators
rather than one, precision is 45.6%. Roughly 21 points of the apparent
imprecision is one annotator not writing down what another one did. Both
figures are reported everywhere, always with the ceiling beside them, because
reporting only the flattering one is the thing this project is about.

**"Why not use a real model?"**

I should, and `graders/semantic.py` is written for it. I could not run it: the
build environment had no API access, no local model server, and no route to
pretrained weights. So rather than shipping an unmeasured module and describing
it as the solution, I asked whether the gap could be closed without one.

That question turned out to be the most useful thing in the project.

**"So what did you do instead?"**

PriMock57 pairs 57 transcripts with the notes clinicians wrote from them. That
is a parallel corpus, so the shorthand is learnable rather than hand-written:
PPMI over a term co-occurrence matrix, truncated SVD. Latent semantic analysis,
1990 technology, no weights and no network. It recovers `inhaler` →
`salbutamol` at 0.90, `diarrhoea` → `gastroenteritis`, `watery`, `rash` →
`bite`, `cetirizine`. Nobody wrote those down.

Held out, it grounds 66.7% of a median clinician note against the lexical
grader's 40.0% — it closes half the gap the README complains about.

And it drops recall on clinician-flagged critical errors from 48.3% to 31.6%.

**"Then it didn't work."**

It worked exactly as designed and the design was insufficient, which is a
different and more useful finding. Sweeping the association floor — the one
knob controlling how far from a claim's literal wording the grader will look —
traces a clean monotone trade: 66.7%/31.6% at the loose end, 40.0%/47.6% at the
tight end, converging onto the lexical baseline it was meant to beat.

Every point of coverage is paid for in detection, because widening the match
radius cannot separate paraphrase from fabrication. To a term matcher those are
the same event: a claim whose words are not in the transcript. Distinguishing
them requires a judgement about whether two differently-worded statements
assert the same clinical fact, and no amount of distributional similarity makes
that judgement. Something has to read them.

That is the argument for a semantic grader — measured, not asserted. It is a
stronger argument than the one it replaced.

**"How do I know your numbers aren't in-sample?"**

They were, and it is written up as WIT-011. Every figure is now produced by
five-fold cross-validation: the association model is fitted on training folds
only, and thresholds are calibrated from training-fold groundings and applied
to held-out ones. Folds are assigned by stride over sorted consultation ids —
deterministic, no seed, and striding rather than blocking because PriMock57 ids
encode the recording day, and contiguous blocks would confound the split with
whatever varied between days.

The internal check: the lexical grader learns nothing and therefore cannot
leak, so its held-out recall should equal its in-sample recall. It does —
48.3% both ways. That is how I know the harness itself is sound.

**"What stops the README from going stale?"**

`make crossval` exits non-zero if any published figure drifts past its stated
tolerance, or if the monotone trade stops being monotone. It runs inside `make
check`. I verified it fails as well as passes — a gate only tested in the
passing direction is decoration, which is WIT-006 in this project's own
incident log.

**"57 consultations is nothing."**

Correct, and it bounds every claim here. Five-fold cross-validation over 57
encounters gives wide intervals; folds remove the leakage, they do not
manufacture power. Treat differences of a few points as noise and the monotone
trends as the result. It is also one country, one care setting, and 2022-era
model output — nothing here establishes transfer to US primary care, to
specialty clinics, or to a current vendor's system.

**"Did you use AI to write this?"**

Yes, daily, under a maintained `CLAUDE.md` that defines conventions, protects
test fixtures from modification, and sets release rules. I wrote those rules
after three releases broke in ways I should have caught.

The verification layer is the engineering contribution. Eleven incidents in
`docs/INCIDENTS.md` are defects that got past me and were caught by a test, a
gate, or a number that stopped making sense — several of them in code an
assistant wrote and I approved. WIT-001 is a join that silently dropped 19% of
the corpus while the run printed a confident number. WIT-010 is a learned
lexicon whose nearest neighbour for `hx` was `yeah`, which survived because I
inspected it on the terms that made it look good.

## What I would do next, in order

1. **Run the semantic grader.** Everything above is the floor. The specific
   question it should answer: does an LLM adjudicating `INFERRED` claims break
   the coverage/detection trade that distributional similarity cannot?
2. **An ASR layer.** Grounding is currently measured against perfect human
   transcripts. Whisper's documented tendency to fabricate on disfluent speech
   (Koenecke et al., FAccT 2024) matters most for the sickest patients and is a
   second failure surface entirely.
3. **More annotators, or a second corpus.** The inter-rater ceiling is doing a
   lot of work in the precision story and it deserves to be tested rather than
   relied on.

## What I will not claim

- That this is production-ready. It is a measurement harness.
- That the numbers transfer beyond UK primary care.
- That the semantic grader works. It has never been run.
- That AI-assisted code needs no review. Eleven incidents say otherwise.
