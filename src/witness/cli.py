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
from witness.matching import MatchResult, match_note
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
    c = _corpus()
    grader = LexicalGrader()
    anns = c.annotations()
    c.require_consultations({a.consultation_id for a in anns})
    consultations = {cid: c.load_consultation(cid) for cid in c.consultation_ids()}

    totals = [0, 0, 0, 0, 0]
    for a in anns:
        g = grader.ground_note(
            a.consultation_id,
            a.model_id,
            segment(a.generated_note),
            consultations[a.consultation_id].utterances,
        )
        for i, v in enumerate(match_note(a, g)):
            totals[i] += v
    m = MatchResult(*totals)

    print("\n  Does Witness flag the claims the clinicians flagged?")
    print("  " + "-" * 66)
    print(f"  grader                              {grader.name}  (no model, no network)")
    print(f"  clinician critical findings         {m.findings:>6}")
    print(
        f"  not attributable to a claim         {m.unmatchable_findings:>6}"
        f"  ({m.unmatchable_findings / m.findings:.0%}, structural comments)"
    )
    print(f"  flags raised                        {m.flags:>6}")
    print(f"\n  recall     {m.recall:>6.1%}   of attributable critical errors")
    print(f"  precision  {m.precision:>6.1%}   of flags landing on a labelled error")
    print(f"  f1         {m.f1:>6.1%}")

    a1 = val.agreement_on(c, "day1_consultation01")
    if "exact_agreement" in a1:
        print("\n  For scale -- the five clinicians, against each other:")
        print(
            f"  exact agreement {a1['exact_agreement']:.1%}"
            f"   within one {a1['within_one']:.1%}   rho {a1['spearman_rho']:+.3f}"
        )
        print("  No detector can be asked to agree with 'the clinicians' more")
        print("  than the clinicians agree with one another.")
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
    if not drift and not shape:
        print("\n  every figure the README publishes reproduces within tolerance.")
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
    return 0


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
