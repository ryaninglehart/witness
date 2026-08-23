"""Command line entry points.

Each subcommand prints a table and exits non-zero if the result it reports
should stop a build. Printing a number and exiting 0 regardless is how a check
becomes decoration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from witness import calibration as cal
from witness import crossval as cx
from witness import evidence as ev
from witness import validate as val
from witness.corpus import PriMock57
from witness.gate import Gate
from witness.graders.lexical import LexicalGrader
from witness.matching import MatchResult, match_note, match_note_union, match_omissions
from witness.pipeline import Pipeline
from witness.prove import format_report, run_proof, taxonomy_coverage
from witness.providers import OllamaProvider, default_provider
from witness.segment import segment
from witness.viewer import export

CORPUS = Path(__file__).resolve().parents[2] / "primock57"


def _corpus() -> PriMock57:
    try:
        return PriMock57(CORPUS)
    except FileNotFoundError as exc:
        print(f"  {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_evidence(_: argparse.Namespace) -> int:
    c = _corpus()
    comp = ev.error_composition(c)
    print("\n  What the clinicians found, across 1,140 graded notes")
    print("  " + "-" * 66)
    print(f"  critical incorrect statements       {comp['critical_incorrect']:>6}")
    print(f"  critical omissions                  {comp['critical_omissions']:>6}")
    print(
        f"  omissions as a share of criticals   {comp['omission_share']:>6.1%}"
        "   <- the half most work skips"
    )
    print(f"  notes with no critical error        {comp['clean_notes']:>6}"
          f"  ({comp['clean_rate']:.1%})")

    e = ev.post_edit_effort(c)
    print("\n  Do the metrics the industry reports predict clinical safety?")
    print("  " + "-" * 66)
    per_note = (comp["critical_incorrect"] + comp["critical_omissions"]) / comp["notes"]
    print(f"  base rate: {per_note:.2f} critical errors per note")
    effort_reduction = 1 - e["least_edited_decile_mean_critical"] / e["base_mean_critical"]
    print(
        f"  post-edit effort (edit-distance proxy)   rho {e['spearman_rho']:+.3f}"
        f"   error reduction {effort_reduction:.1%}"
    )
    print("     ...but effort is only knowable after a clinician has already")
    print("        read and fixed the note, so it cannot gate a release.")

    res = ev.analyse(c)
    best = max(res, key=lambda r: r.error_reduction)
    worst = min(res, key=lambda r: r.error_reduction)
    print("\n  Text-similarity metrics, ranked by error reduction in their top decile")
    print("  " + "-" * 66)
    print(f"  {'metric':<20} {'rho':>7} {'reduction':>11}")
    for r in sorted(res, key=lambda r: -r.error_reduction)[:5]:
        print(f"  {r.name:<20} {r.spearman_rho:+.3f} {r.error_reduction:>10.1%}")
    print(f"  {'...':<20}")
    print(f"  {worst.name:<20} {worst.spearman_rho:+.3f} {worst.error_reduction:>10.1%}")
    print(f"\n  best available signal: {best.name} at {best.error_reduction:.1%}")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    from collections import defaultdict

    from witness.omission import find_omissions

    c = _corpus()
    grader = LexicalGrader()
    anns = c.annotations()
    c.require_consultations({a.consultation_id for a in anns})
    consultations = {cid: c.load_consultation(cid) for cid in c.consultation_ids()}

    # Grouped by note, because the union figures pool the five clinicians who
    # each reviewed the same note.
    by_note: dict[tuple[str, str], list] = defaultdict(list)
    for a in anns:
        by_note[(a.consultation_id, a.model_id)].append(a)

    single = [0, 0, 0, 0, 0]
    union = [0, 0, 0, 0, 0]
    omit_single = [0, 0, 0, 0, 0]
    omit_union = [0, 0, 0, 0, 0]

    for (cid, model_id), group in by_note.items():
        cons = consultations[cid]
        note = group[0].generated_note
        g = grader.ground_note(cid, model_id, segment(note), cons.utterances)
        report = find_omissions(cons, note, model_id)
        reported = [f.text for f in report.omitted]
        # Everything the extractor judged salient, reported or not. This is
        # the universe an annotator's omission has to be found in before the
        # question "did Witness report it" can be asked at all.
        salient = reported + [f.text for f in report.covered]

        for a in group:
            for i, v in enumerate(match_note(a, g)):
                single[i] += v
            for i, v in enumerate(match_omissions([a], salient, reported)):
                omit_single[i] += v
        for i, v in enumerate(match_note_union(group, g)):
            union[i] += v
        for i, v in enumerate(match_omissions(group, salient, reported)):
            omit_union[i] += v

    m = MatchResult(*single)
    mu = MatchResult(*union)
    om = MatchResult(*omit_single)
    omu = MatchResult(*omit_union)

    print("\n  Does Witness flag the claims the clinicians flagged?")
    print("  " + "-" * 66)
    print(f"  grader                              {grader.name}  (no model, no network)")
    print(f"  clinician critical findings         {m.findings:>6}")
    print(
        f"  not attributable to a claim         {m.unmatchable_findings:>6}"
        f"  ({m.unmatchable_findings / m.findings:.0%}, structural comments)"
    )
    # From the union path, which counts each note once. The per-annotator
    # accumulation multiplies flags by five -- harmless in the ratios, where
    # it cancels, and badly misleading printed next to a finding count.
    print(f"  flags raised                        {mu.flags:>6}")
    print()
    print(f"  grounding     recall     {m.recall:>5.1%}   of attributable critical "
          "incorrect statements")
    print(f"                precision  {m.precision:>5.1%}   against one annotator")
    print(f"                           {mu.precision:>5.1%}   against the union of all five")
    print()
    # The attributability share is printed for omissions for the same reason
    # it is printed for grounding above: 77.7% "of attributable" reads as
    # 77.7% of the problem unless the reader is told what was excluded, and
    # here far more is excluded than on the grounding side.
    print(
        f"  omission      recall     {om.recall:>5.1%}   of attributable critical "
        "omissions"
    )
    print(
        f"                           {omu.findings - omu.unmatchable_findings:>5}"
        f" of {omu.findings} are attributable "
        f"({1 - omu.unmatchable_findings / omu.findings:.0%}); the extractor never"
    )
    print("                                 nominated the rest, so they are counted")
    print(
        f"                                 as neither hit nor miss. Of all "
        f"{omu.findings}, {omu.matched / omu.findings:.1%} are found."
    )
    print(f"                precision  {om.precision:>5.1%}   against one annotator")
    print(f"                           {omu.precision:>5.1%}   against the union of all five")

    a1 = val.agreement_on(c)
    if "exact_agreement" in a1:
        print(
            f"\n  For scale -- the five clinicians against each other, "
            f"all {int(a1['pairs'])} pairs:"
        )
        print(
            f"  exact agreement {a1['exact_agreement']:.1%}"
            f"   within one {a1['within_one']:.1%}   rho {a1['spearman_rho']:+.3f}"
        )
        print("  No detector can be asked to agree with 'the clinicians' more")
        print("  than the clinicians agree with one another. The gap between the")
        print("  two precision figures above is that disagreement, measured.")
    return 0


def cmd_calibrate(_: argparse.Namespace) -> int:
    c = _corpus()
    print()
    print(cal.calibrate(c, 0.90).report())
    return 0


def cmd_sweep(_: argparse.Namespace) -> int:
    c = _corpus()
    print("\n  Calibration sweep -- is there any threshold that works?")
    print("  " + "-" * 66)
    print(f"  {'percentile':>11}{'human notes pass':>19}{'defects caught':>17}")
    points = cal.sweep(c)
    for op in points:
        print(f"  {op.percentile:>11.2f}{op.human_pass_rate:>18.0%}{op.sabotage_detection:>17.0%}")
    best = max(points, key=lambda o: min(o.human_pass_rate, o.sabotage_detection))
    print()
    print(
        f"  Best balance: {best.human_pass_rate:.0%} of clinician notes accepted at "
        f"{best.sabotage_detection:.0%} detection."
    )
    print("  Lexical grounding has no usable operating point. That is a property")
    print("  of the grader, not of the thresholds -- see graders/semantic.py.")
    return 0


def cmd_crossval(ns: argparse.Namespace) -> int:
    c = _corpus()
    report = cx.run(c, k=ns.k)
    print(report.render())

    if ns.k != 5:
        print(f"\n  (published figures are 5-fold; k={ns.k} not checked against them)")
        return 0

    drift = cx.check_published(report)
    shape = cx.monotone_violations(report.radius)
    live = cx.check_control_is_live(
        c, {cid: c.load_consultation(cid) for cid in c.consultation_ids()}
    )
    shape = shape + live
    if not drift and not shape:
        print(
            f"\n  The {len(cx.PUBLISHED)} enforced figures reproduce within tolerance, "
            "the radius trade"
        )
        print("  is still monotone, and the false-positive control still rewords the")
        print("  notes it is given. Other numbers in the README are printed by their")
        print("  own commands and are not threshold-guarded.")
        return 0

    # Exit non-zero. A README whose numbers have quietly stopped being true is
    # the exact defect this project was built to find in other people's
    # systems, and printing a warning would make this one an instance of it.
    print("\n  PUBLISHED FIGURES HAVE DRIFTED")
    print("  " + "-" * 68)
    for line in drift + shape:
        print(f"    {line}")
    print("\n  Update README.md and witness.crossval.PUBLISHED together, or fix")
    print("  the regression. Do not widen the tolerance to make this pass.")
    return 1


def cmd_semantic(ns: argparse.Namespace) -> int:
    from pathlib import Path as _Path

    from witness import semantic_eval as se
    from witness.graders.lexical import LexicalGrader
    from witness.graders.semantic import SemanticGrader

    c = _corpus()
    provider = default_provider(prefer=ns.provider)
    if provider is None:
        # Say exactly what is missing. "No provider available" sends a reviewer
        # looking for a bug in the harness when the answer is a missing model.
        ollama = OllamaProvider()
        installed = ollama.installed_models()
        print("\n  No model provider is reachable.")
        print("  " + "-" * 68)
        if installed:
            print(f"  Ollama is running at {ollama.host} and serves: {', '.join(installed)}")
            print(f"  ...but the harness wants {ollama.model!r}. Either:")
            print(f"      ollama pull {ollama.model}")
            print(f"      OLLAMA_MODEL={installed[0]} make semantic")
        else:
            print(f"  Ollama is not answering at {ollama.host}.")
            print(f"      ollama serve  &&  ollama pull {ollama.model}")
        print("  Or export ANTHROPIC_API_KEY / OPENAI_API_KEY.")
        return 2

    print(f"\n  provider: {provider.name} / {provider.model}")
    print(f"  {ns.consultations or 57} consultation(s), "
          f"{ns.annotations} generated note(s) each")
    print("  Responses are cached to cassettes/, so a second run is free and")
    print("  an interrupted run resumes where it stopped.\n")

    report = se.evaluate(
        c,
        graders={"lexical": LexicalGrader(), "semantic": SemanticGrader(provider)},
        consultation_limit=ns.consultations,
        annotations_per_consultation=ns.annotations,
    )
    print(report.render())
    out = se.write_report(report, _Path("data/out/semantic_report.json"))
    print(f"\n  wrote {out}")

    # A degenerate row is not a result and must not exit zero. WIT-016: a run
    # reporting grounded 0.0% / recall 100.0% was a parser failure wearing a
    # measurement's clothes, and it took a careful reader to notice rather
    # than the harness saying so.
    drift = se.check_published(report)
    if drift:
        print("\n  PUBLISHED SEMANTIC FIGURES HAVE DRIFTED")
        print("  " + "-" * 68)
        for line in drift:
            print(f"    {line}")
        return 1

    broken = [r for r in report.results if se.degenerate_reasons(r)]
    if broken:
        # The contaminated row is refused, but the clean-subset row beside it
        # is a real comparison and the reason this command still has a point.
        print("\n  The full-corpus row is contaminated and is refused. The")
        print("  clean-subset row above is the measurement; see WIT-016.")
        return 1
    return 0


def cmd_prove(_: argparse.Namespace) -> int:
    c = _corpus()
    calib = cal.calibrate(c, 0.90)
    cons = c.load_consultation("day1_consultation01")
    results = run_proof(
        cons, cons.clinician_note, calib.thresholds, calib.max_critical_omissions
    )
    print(format_report(results))
    print(taxonomy_coverage(results))

    # This command used to return 0 unconditionally. Ten missed defects
    # produced a passing build, in a project whose entire argument is that a
    # check exiting zero regardless is decoration. Low detection is the
    # published state and must not fail the build; these three things must.
    failures: list[str] = []
    for r in results:
        if r.vacuous:
            failures.append(
                f"{r.sabotage.key}: injection matched nothing, so this case tested nothing"
            )
        elif not r.sabotage.should_block and r.blocked:
            failures.append(
                f"{r.sabotage.key}: the harness fired on a rewording that changed no "
                "clinical content"
            )
    caught = {r.sabotage.key for r in results if r.sabotage.should_block and r.blocked}
    if "weakened_threshold" not in caught:
        failures.append(
            "weakened_threshold: a silently relaxed gate was not detected by the "
            "ruleset fingerprint"
        )
    if not failures:
        return 0
    print("\n  SABOTAGE HARNESS INVALID")
    print("  " + "-" * 68)
    for line in failures:
        print(f"    {line}")
    print("\n  Low detection is the reported state of this project and does not")
    print("  fail the build. A control that cannot fire, a false alarm on a")
    print("  paraphrase, and an undetectable weakened gate all do -- each one")
    print("  means a number printed above is not measuring what it says.")
    return 1


def cmd_review(ns: argparse.Namespace) -> int:
    c = _corpus()
    calib = cal.calibrate(c, 0.90)
    pipe = Pipeline(
        gate=Gate(calib.thresholds), max_critical_omissions=calib.max_critical_omissions
    )
    cons = c.load_consultation(ns.consultation)
    anns = [a for a in c.annotations() if a.consultation_id == ns.consultation]
    note = anns[0].generated_note if anns else cons.clinician_note
    model = anns[0].model_id if anns else "clinician"
    run = pipe.run(cons, note, model, publish=False)
    out = export(run, cons, Path("data/out") / f"{ns.consultation}.html")
    print(f"\n  wrote {out}")
    print(f"  {run.grounding.summary()}")
    print(f"  gate: {'passed' if run.gate.published else 'refused'}")
    print("  open it and click any claim to see the utterance that justifies it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="witness")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evidence", help="do existing metrics predict clinical safety?").set_defaults(
        fn=cmd_evidence
    )
    sub.add_parser("validate", help="does Witness flag what clinicians flagged?").set_defaults(
        fn=cmd_validate
    )
    sub.add_parser("calibrate", help="derive thresholds from clinician-written notes").set_defaults(
        fn=cmd_calibrate
    )
    sub.add_parser("sweep", help="trace the sensitivity/specificity trade").set_defaults(
        fn=cmd_sweep
    )
    sub.add_parser("prove", help="break it on purpose; see what is caught").set_defaults(
        fn=cmd_prove
    )
    xv = sub.add_parser("crossval", help="held-out evaluation; no number scored in-sample")
    xv.add_argument("--k", type=int, default=5, help="number of folds (default 5)")
    xv.set_defaults(fn=cmd_crossval)

    sm = sub.add_parser("semantic", help="measure the LLM grader against the offline ones")
    sm.add_argument("--consultations", type=int, default=None,
                    help="limit to the first N consultations (default: all 57)")
    sm.add_argument("--annotations", type=int, default=1,
                    help="generated notes per consultation (default 1)")
    sm.add_argument("--provider", default=None, choices=["ollama", "anthropic", "openai"],
                    help="preferred provider (default: local first)")
    sm.set_defaults(fn=cmd_semantic)
    rv = sub.add_parser("review", help="export a one-file review page")
    rv.add_argument("--consultation", default="day1_consultation01")
    rv.set_defaults(fn=cmd_review)
    ns = p.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    raise SystemExit(main())
