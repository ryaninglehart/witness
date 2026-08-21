"""Command line entry points.

Each subcommand prints a table and exits non-zero if the result it reports
should stop a build. Printing a number and exiting 0 regardless is how a check
becomes decoration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from witness import evidence as ev
from witness import validate as val
from witness.corpus import PriMock57
from witness.graders.lexical import LexicalGrader
from witness.matching import MatchResult, match_note
from witness.segment import segment

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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="witness")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evidence", help="do existing metrics predict clinical safety?").set_defaults(
        fn=cmd_evidence
    )
    sub.add_parser("validate", help="does Witness flag what clinicians flagged?").set_defaults(
        fn=cmd_validate
    )
    ns = p.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    raise SystemExit(main())
