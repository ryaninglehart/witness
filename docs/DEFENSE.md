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
practising clinicians and catches more than a small minority of injected
defects. Best usable balance: 39% acceptance at **11%** detection, over 506
note-level defects spanning eleven types drawn from the error taxonomy the
corpus evaluators wrote for themselves. At the percentiles where most clinician
notes get through, it catches essentially nothing.

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
graded this corpus agree exactly with each other 19.8% of the time and within
one grade 47.1%, across all 2,280 pairs. (An earlier version of this document
said 27.5% and 55%. Those are one consultation's figures — 40 pairs — quoted as
a property of the corpus, which is the friendlier of the two numbers reported
as though it were the general case. WIT-015.)

One caveat travels with it: that is agreement on the *count* of critical
statements per note, not on which statements they are. Two clinicians who each
find three different errors agree perfectly by this measure. So it is context
for an item-level precision figure, not a formal bound on it, and I do not
present it as one. Measured against the union of all five annotators
rather than one, precision is 39.7%. The ~19-point gap is one annotator not
writing down what another one did.

Neither figure is the truth. The single-annotator number understates precision
by charging the detector for disagreement; the union overstates it by accepting
anything anyone flagged. Both are reported everywhere with the ceiling beside
them, because reporting only the flattering one is the thing this project is
about.

Omission is the other half, and the denominator matters more than the headline:
recall 77.7% **of attributable omissions**, where only 1,666 of 5,792 clinician-
reported critical omissions are attributable at all. The extractor never
nominated the other 71%, so they are counted as neither hit nor miss. Against
everything the clinicians said was missing, **22.4%** is found. Precision is
15.0% against one annotator and 41.1% against the union.

So the extractor nominates generously and still misses most of the problem.
`make validate` prints all of that, because 77.7% on its own reads as three
quarters of the problem solved and it is not.

**"Why not use a real model?"**

I did, eventually — `qwen2.5:7b-instruct` over Ollama, 148 batches, 47 minutes
on a laptop. But not first. The build environment had no model access, so
before reaching for one I asked whether the gap could be closed without it.

That question turned out to be the most useful thing in the project, because
the answer was no, and establishing *why* is what makes the eventual positive
result mean anything.

**"So what did you do instead?"**

PriMock57 pairs 57 transcripts with the notes clinicians wrote from them. That
is a parallel corpus, so the shorthand is learnable rather than hand-written:
PPMI over a term co-occurrence matrix, truncated SVD. Latent semantic analysis,
1990 technology, no weights and no network. It recovers `inhaler` → `salbutamol` at
0.88–0.92 across the five folds, `diarrhoea` → `gastroenteritis`, `watery`, and
`rash` → `bite`. Nobody wrote those down. The rest of the lexicon is noisier
and fold-dependent, and the examples above are the stable ones — a claim about
which examples I chose, not about the method.

Held out, it grounds 66.7% of a median clinician note against the lexical
grader's 45.0% — about 22 of the 55 ungrounded points.

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

**"So did the model fix it?"**

Half of it, decisively — and I want to be precise about which half.

On the 37 notes where the model held its output format, all three graders
rescored on those same 37: grounding goes 42.9% → 83.3%, recall 39.7% → 57.1%,
precision 15.4% → 32.7%. Paired bootstrap: **+40.5% [+30.0, +52.2]** on
grounding, **+17.5% [+0.0, +36.5]** on recall.

The recall interval touches zero. On 37 notes this corpus cannot establish that
the semantic grader catches more errors, and I do not claim it does.

What it does establish is the more useful thing. Every term-matching grader here
pays for coverage in recall, monotonically — the radius sweep prices it at about
a point of recall per point of grounding. The semantic grader takes forty points
of grounding and does *not* pay that bill. The trade-off that binds term
matching does not bind a model reading for meaning. That is the finding.

Worth knowing how I got here: the first version of this answer said 82.1% and
69.2%, and I withdrew it. A segmentation defect had been dropping 143 clinical
claims — mostly pertinent negatives — before anything was graded, which
invalidated the recorded responses those figures came from. Re-measured, the
grounding held and recall came in twelve points lower. Correcting it is exactly
what moved the recall claim from "distinguishable" to "not".

Both term-matching graders sit on the trade-off curve. The semantic one steps
off it. That is exactly what the negative result predicted, and the prediction
was published before the test was run.

**"Then why isn't it in production?"**

Because it dropped the output format on 29% of batches — 43 of 148 responses
stopped classifying and started narrating, with no verdict anywhere in them.
Not a parsing problem; there is nothing in those to recover.

I refuse the full-corpus row rather than publishing it, because an unreadable
batch refuses every claim in it, which pushes grounding down and detection up.
The two biases point opposite ways, so that row cannot be corrected by
argument — only by dropping those notes. The harness prints it under NOT A
MEASUREMENT and exits non-zero.

And the 38-note subset is not random: it is the encounters the model stayed on
format for. If it drops format on harder consultations, my figures are
optimistic. I would fix the format compliance with constrained decoding and
re-run before claiming anything about deployment.

**"You withdrew your headline result. Why should I trust the rest?"**

Because the same machinery that produced it is what caught the defect and what
refused to let the old number stand. The drift gate failed the build the moment
segmentation changed and named the figure that had moved. Nothing quietly
updated itself.

I would rather show you a withdrawn result than a number I cannot reproduce. The
offline figures in this document were all re-derived on the fixed code before I
sent it, and the semantic section says withdrawn instead of restating what it
used to say.

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
48.7% both ways.

I want to be precise about what that proves, because it is less than it looks.
It shows the out-of-fold scoring path reproduces the in-sample path for a
stateless grader. It does not test the fold partition of the *fitted* model,
which is the only place leakage was ever possible — a stateless grader is
insensitive to that by construction.

**"How do you know the sabotage numbers mean anything?"**

Because they did not, and I found out. WIT-013: the false-positive control
substituted a phrase that appears in one of 57 consultations, so on the other 56
it compared a note to itself and passed. And `clean_baseline` — which strips
ungrounded claims and was therefore supposed to pass the gate — fails on its own
for 21 of 57 at p=0.90 and 31 of 57 at p=0.80, because stripped content is
missing content. The published "47% false alarms" was neither a paraphrase nor
an alarm. It was the baseline.

Two fixes. The control now reorders sentences and expands abbreviations rather
than substituting one phrase, which takes it from 1 of 57 notes to 50 — not all
57, and the gate requires 80%. An injection that changes nothing is marked
vacuous and excluded from both sides rather than scored as a pass. And every case is credited only for rules the injection
*added* to the baseline's own findings.

The honest figure is 4%. Fixing the injections so they fire on most
consultations rather than on the one they were written against raised the
defect count from 399 to 552 and dropped detection from 33% to 21% — the extra
cases are ones the harness does not catch. Both corrections moved the numbers
against the project, and one of them killed a selling point of the learned
grader.

What I take from it: the number was bad, and it agreed with the story I was
already telling, so it read as confirmation. A figure that had come out
implausibly good would have been audited the same day.

**"What stops the README from going stale?"**

`make crossval` exits non-zero if any of eight enforced figures drifts past its
stated tolerance, or if the monotone trade stops being monotone. It runs inside `make
check`. I verified it fails as well as passes — a gate only tested in the
passing direction is decoration, which is WIT-006 in this project's own
incident log. It has since earned its keep: it is what refused the build when
the WIT-013 correction moved the false-alarm figure, rather than leaving a
stale number in the README.

**"57 consultations is nothing."**

Correct, and it bounds every claim here — so the bound is stated rather than
gestured at. Every headline figure carries a 95% bootstrap interval over 2,000
resamples of whole consultations. The unit is the consultation, not the note:
notes from one encounter share a transcript, a patient and a clinician, and
resampling notes would treat correlated observations as independent and report
intervals far narrower than the corpus supports.

They are not symmetric, and the widest one matters most: statistical grounding
is 66.7% with an interval of [57.1, 75.0]. Lexical grounding runs [37.5, 50.0]
and both recalls are about ±5. Differences smaller than that are not readable here, and I do
not read them.

The two the argument rests on are much bigger. Paired on identical resamples,
so the shared difficulty of a given encounter cancels: grounding +26.7%
[+15.6%, +31.4%], recall −16.8% [−20.1%, −13.2%]. Neither is near zero. **The
trade-off is not a small-sample artifact.**

One thing I am careful about in how that is phrased: an interval containing
zero would mean these 57 consultations cannot separate two graders. It would
not mean they are equal, and reporting it that way is the same overclaim
pointed in the other direction.

It is also one country, one care setting, and 2022-era model output — nothing
here establishes transfer to US primary care, to specialty clinics, or to a
current vendor's system.

**"Did you use AI to write this?"**

Yes, daily, under a maintained `CLAUDE.md` that defines conventions, protects
test fixtures from modification, and sets release rules. I wrote those rules
after three releases broke in ways I should have caught.

The verification layer is the engineering contribution. Nineteen incidents in
`docs/INCIDENTS.md` are defects that got past me and were caught by a test, a
gate, or a number that stopped making sense — several of them in code an
assistant wrote and I approved. WIT-001 is a join that silently dropped 19% of
the corpus while the run printed a confident number. WIT-010 is a learned
lexicon whose nearest neighbour for `hx` was `yeah`, which survived because I
inspected it on the terms that made it look good. WIT-014 is four numbers
printed in the README as command output that no command produced.

## What I would do next, in order

1. **Fix format compliance and re-run.** Ollama's constrained JSON mode, or
   retry-on-unparsed. That takes the 38-note result to a 57-note one and is the
   difference between a finding and a component.
2. **Calibrate the gate against the semantic grader.** Every operating point
   and sabotage figure in this project comes from the offline graders. What a
   release gate does with a grader this much better is unmeasured, and it is
   the question the whole harness was built to answer.
3. **An ASR layer.** Grounding is currently measured against perfect human
   transcripts. Whisper's documented tendency to fabricate on disfluent speech
   (Koenecke et al., FAccT 2024) matters most for the sickest patients and is a
   second failure surface entirely.
4. **More annotators, or a second corpus.** The inter-rater ceiling is doing a
   lot of work in the precision story and it deserves to be tested rather than
   relied on.

## What I will not claim

- That this is production-ready. It is a measurement harness.
- That the numbers transfer beyond UK primary care.
- That the semantic grader beats the baseline. The measurement is withdrawn
  pending a re-run (WIT-018), and it drops its output format on 29% of batches
  regardless.
- That AI-assisted code needs no review. Nineteen incidents say otherwise.
