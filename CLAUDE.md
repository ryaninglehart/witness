# Working rules for this repository

Conventions for AI-assisted work on Witness. Every rule below was written after
something went wrong; the incident number is the citation. `docs/INCIDENTS.md`
has the full accounts.

This file exists because the assistant is fast and confident and I approve its
work. The verification layer around it is the engineering, not the generation.

---

## Numbers

**Never write a figure into documentation that no command prints.** Four numbers
in this README were formatted as `make validate` output and nothing computed
them; one of them carried an entire argument. Generate output blocks, do not
retype them. A block tagged `<!-- verify: <command> -->` is checked against real
output by `tests/test_documented_numbers.py`; an untagged block is an
illustration and must read as one. **WIT-014**

**A published figure gets a tolerance and a gate.** `crossval.PUBLISHED` and
`semantic_eval.PUBLISHED_SEMANTIC` hold the numbers this project asserts. Change
a number and the build fails until the README and the tolerance move together.
Never widen a tolerance to make a run pass. **WIT-011**

**Report the denominator.** 77.7% recall meant 77.7% *of attributable
omissions*, and 71% were not attributable. If a metric excludes cases, print how
many. **WIT-015**

**Both bounds, always.** Precision against one annotator and against the union
of five, with the inter-rater ceiling beside them. Picking whichever flatters
the detector is the practice this project objects to elsewhere. **WIT-015**

## Measurement

**Scope a statistic to what it measures.** Inter-rater agreement was computed on
one consultation and quoted as a property of the corpus. Default arguments that
silently narrow a population are how that happens. **WIT-015**

**Nothing is scored in-sample.** Any grader that learns from the corpus is
fitted on training folds only, and `StatisticalGrader` raises rather than score
a consultation it trained on. Thresholds come from the training folds too.
**WIT-011**

**Compare on identical inputs.** A clean subset scored against a full corpus is
two different corpora. When notes are dropped, every grader is rescored on the
survivors. **WIT-017**

**A statistic whose denominator is its numerator is not a statistic.** Check
that a metric *can* return something other than 100%. **WIT-014**

## Checks

**A check that cannot fail is decoration.** Verify every gate in both
directions: break the thing deliberately, confirm the named message and a
non-zero exit, then restore. Applies to the drift gate, the sabotage harness,
the control-liveness check, and the degenerate-result guard. **WIT-006**

**Credit only the named mechanism.** A defect stopped by an unrelated rule is a
lucky catch, and a rule the baseline already fires is not a catch at all.
**WIT-006, WIT-013**

**Assert the manipulation happened.** An injection that matches nothing tests
nothing and must be reported vacuous, never scored as a pass. Content-dependent
injections derive from the note or transcript, never from a phrase that happens
to appear in one consultation. **WIT-013**

**Exclude what is detected by construction.** The ruleset-fingerprint case is
caught at every threshold by definition. Counting it put a guaranteed hit inside
the detection rate and roughly doubled it. **WIT-015**

## Model output

**Fixtures for a parser must come from the model, not from the parser's
author.** The test suite covered null, truncation, prose and code fences — every
fixture using the envelope the prompt asks for, because I wrote both. The live
model used that envelope 5 times in 19. **WIT-016**

**Failure to parse is never silent.** Count it, print the rate, and refuse the
run above a threshold. Blocking quietly is indistinguishable from a note that
deserved to be blocked. **WIT-016**

**Refuse a degenerate result instead of printing it.** 0% grounding with 100%
recall is a broken pipeline, not a measurement. `degenerate_reasons` names the
class; extend it rather than relying on a reader to notice. **WIT-016**

**Contamination that biases two ways cannot be argued away.** Recompute without
it. **WIT-017**

## Data and safety

**No PHI, ever.** PriMock57 is public, CC BY 4.0, acted, and synthetic. Nothing
else enters this repository.

**Escape at the boundary, not only at the leaves.** The review page escaped
every field it rendered and still executed injected script, because the payload
one level up was not escaped. Anything embedding data into markup is a boundary.
**WIT-012**

**Treat every input as hostile.** Transcripts, generated notes and evaluator
comments are model or third-party output. That is the premise of the project.

## Test fixtures

**Do not edit a fixture to make a test pass.** Two tests failed after a rule
change; the tests were stale and the padding was the fix, but that determination
comes first and is written down. If a fixture must change, say why in the commit
message.

**Corpus files are read-only.** Nothing writes to `primock57/`. The recorded
responses in `cassettes/` are evidence — they are committed so the published
figures reproduce without a model, and they are not regenerated casually.

## Releases

**`make check` is the gate.** Lint, tests, evidence, validation, calibration,
sabotage, cross-validation. It must exit 0 and take about a minute.

**Reproducibility is checked across processes, not just within one.** CI runs
the evaluation twice and diffs it. **WIT-011**

**Never claim something works without having run it.** Not "this should pass" —
run it, paste the output.

**When a constraint removes a failure mode, find where the failure went.**
It rarely leaves. Forbidding prose produced valid JSON containing no answer,
and the unreadable rate went up, not down. The replacement failure is normally
quieter than the one removed, which is exactly why the constraint feels like it
worked. **WIT-021**

**Read a long run while it runs.** The intermediate state is on disk and costs
nothing to sample. WIT-021 was visible in 120 stored responses 45 minutes
before the summary would have shown it. **WIT-021**

**A cache key must cover every input that changes the output**, settings
included. A replayed answer to a different question is silent, fast, and
agrees with what you expected. When an experiment changes a request, check the
cache can tell the difference before trusting a run that finishes early.
**WIT-020**

**Never skip a test file wholesale for an optional dependency.**
`pytest.importorskip` at module scope makes every test in the file conditional
on something only a few of them need. The WIT-012 injection guards were skipped
in CI for two weeks that way. Put the import in the fixture. **WIT-019**

**A test count that moves without a cause is the signal.** Find the commit that
changed it before accepting the explanation you invented for it. **WIT-019**

**Confirm CI actually ran, not that it exists.** The first push landed on
`master` while `ci.yml` triggered on `main`, so no job fired and GitHub never
registered the workflow. The repository looked published and healthy while
testing nothing — the same shape as WIT-006, moved up a level: an unregistered
workflow and a passing one are indistinguishable from the repository page. After
any push that changes branches, workflow files or triggers, check `gh run list`
for a run against that commit rather than assuming one happened.

---

## On the assistant

Claude Code writes most of the lines here. The failure mode is not bad code; it
is plausible code with a silent defect, plus my willingness to accept a number
that agrees with what I expected.

Of the seventeen incidents, the ones that took longest to find are the ones
where the wrong answer was *flattering*: a 47% false-alarm rate that fit the
story, a lexicon inspected only on the terms that made it look good, three
counting errors that each moved a figure the helpful way. None was a lie. All
went the same direction.

Two audits by a separate agent, told to be adversarial and to classify anything
unverifiable as unverifiable, found 23 documentation defects in about an hour —
including the four fabricated figures. That is now part of how this repo works:
**arrange for someone else to do the counting.**
