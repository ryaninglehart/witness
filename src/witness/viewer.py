"""Export a run as a self-contained review page.

The point of this file is the click. A reviewer reads a claim in the note,
clicks it, and the utterance that justifies it highlights in the transcript with
its timestamp. No score to interpret, no trust required -- the evidence is one
click away and they can check it themselves.

That is deliberately the same idea as DeepScribe's "Clinical Moments", which
does it for prose. Here it also covers the *verdict*: clicking an unsupported
claim shows you the nearest thing the encounter contained, so a reviewer can see
why the grader could not ground it rather than being told a number.

One page, no build step, no server. A reviewer who has to run `npm install` to
look at a demonstration will not look at the demonstration.
"""

from __future__ import annotations

import json
from pathlib import Path

from witness.corpus import Consultation
from witness.pipeline import RunResult

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Witness &middot; __TITLE__</title>
<style>
:root{
  --bg:#fbfaf8; --panel:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e6e3de;
  --supported:#2f7d5d; --inferred:#8a6d1f; --unsupported:#a8451f; --contradicted:#9b1d20;
  --hit:#fff3cd;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#16161a; --panel:#1e1e23; --ink:#eceae5; --muted:#9a9a9a; --line:#32323a;
  --supported:#6cc39a; --inferred:#d4ae4e;
  --unsupported:#e08a63; --contradicted:#e56b6f; --hit:#3d3520;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:17px;letter-spacing:-.01em}
h1 small{color:var(--muted);font-weight:400;margin-left:8px}
.bar{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;font-size:13px;color:var(--muted)}
.bar b{color:var(--ink);font-weight:600}
.verdictpill{padding:1px 7px;border-radius:99px;font-size:11px;
  font-weight:600;letter-spacing:.03em;color:#fff}
.gate{margin-top:12px;padding:10px 12px;border-radius:6px;font-size:13px;
  border:1px solid var(--line)}
.gate.blocked{border-color:var(--contradicted)}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:0;min-height:calc(100vh - 150px)}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.col{padding:18px 24px;overflow-y:auto;max-height:calc(100vh - 150px)}
.col+.col{border-left:1px solid var(--line)}
@media(max-width:900px){.col+.col{border-left:0;border-top:1px solid var(--line)}}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin:0 0 12px}
.claim{padding:8px 10px;margin-bottom:5px;border-radius:5px;cursor:pointer;
  border-left:3px solid transparent;background:var(--panel);
  border:1px solid var(--line);border-left-width:3px}
.claim:hover{border-color:var(--muted)}
.claim.sel{outline:2px solid var(--muted);outline-offset:1px}
.claim .txt{display:block}
.claim .meta{display:block;font-size:11px;color:var(--muted);margin-top:3px}
.utt{padding:6px 9px;margin-bottom:3px;border-radius:5px;font-size:13.5px}
.utt.hit{background:var(--hit);font-weight:500}
.utt .who{font-size:10px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);margin-right:6px}
.utt .t{float:right;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.omit .meta{display:block;font-size:11px;color:var(--muted);margin-top:3px}
.omit .txt{display:block}
.omit{padding:8px 10px;margin-bottom:5px;border-radius:5px;background:var(--panel);
  border:1px solid var(--line);border-left:3px solid var(--unsupported);cursor:pointer}
.tier{font-size:10px;font-weight:700;letter-spacing:.05em;color:var(--unsupported)}
.empty{color:var(--muted);font-style:italic;font-size:13px}
</style></head><body>
<div id="root"></div>
<script>window.__WITNESS__ = __DATA__;</script>
<script>
/* No framework, and that is a decision rather than a shortcut.
   This page renders clinical content. A build step keeps a reviewer from
   opening it, and a CDN dependency means the page breaks with no network --
   in a tool whose privacy argument is that nothing leaves the machine, an
   outbound request on page load is the wrong default. Zero dependencies, one
   file, opens by double-click, works offline and forever. */
(function () {
  var d = window.__WITNESS__;
  var COLOR = {
    SUPPORTED: 'var(--supported)', INFERRED: 'var(--inferred)',
    UNSUPPORTED: 'var(--unsupported)', CONTRADICTED: 'var(--contradicted)'
  };
  var pad = function (n) { return String(n).padStart(2, '0'); };
  var time = function (s) { return pad(Math.floor(s / 60)) + ':' + pad(Math.floor(s % 60)); };
  /* Transcriber markup. Shown raw it looks like a rendering bug; stripped
     from the stored text it would destroy provenance, since <UNIN/> marks a
     genuinely unintelligible span a reviewer needs to know about. So it is
     cleaned for display only, and the marker is kept as a visible ellipsis. */
  var clean = function (s) {
    return String(s).replace(/<UNIN\\/?>/g, '\u2026').replace(/<\\/?UNSURE>/g, '');
  };
  var esc = function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };

  function render() {
    var counts = d.summary;
    var pills = Object.keys(COLOR).map(function (k) {
      return '<span><span class="verdictpill" style="background:' + COLOR[k] + '">' +
        (counts[k.toLowerCase()] || 0) + '</span> ' + k.toLowerCase() + '</span>';
    }).join('');

    var gate = d.published
      ? '\u2713 gate passed \u00b7 ruleset ' + esc(d.ruleset_hash)
      : '\u2715 gate refused \u2014 ' + esc(d.blocking.join('; ')) +
        ' \u00b7 ruleset ' + esc(d.ruleset_hash);

    var claims = d.assertions.map(function (a, i) {
      return '<div class="claim" data-claim="' + i + '" style="border-left-color:' +
        COLOR[a.verdict] + '"><span class="txt">' + esc(a.text) + '</span>' +
        '<span class="meta"><span style="color:' + COLOR[a.verdict] +
        ';font-weight:600">' + a.verdict + '</span> \u00b7 ' + esc(a.slot) +
        (a.rationale ? ' \u00b7 ' + esc(a.rationale) : '') + '</span></div>';
    }).join('');

    var omitted = d.omitted.length === 0
      ? '<p class="empty">Nothing salient was dropped.</p>'
      : d.omitted.map(function (o, i) {
          return '<div class="omit" data-omit="' + i + '"><span class="tier">' +
            esc(o.tier) + '</span> <span class="txt">' + esc(clean(o.text)) +
            '</span><span class="meta">' + esc(o.basis) + ' \u00b7 ' +
            time(o.start_s) + '</span></div>';
        }).join('');

    var utts = d.utterances.map(function (u, i) {
      return '<div class="utt" id="u' + i + '"><span class="t">' + time(u.start_s) +
        '</span><span class="who">' + esc(u.speaker) + '</span>' + esc(clean(u.text)) + '</div>';
    }).join('');

    document.getElementById('root').innerHTML =
      '<header><h1>Witness <small>' + esc(d.consultation_id) + ' \u00b7 ' +
      esc(d.model_id) + '</small></h1><div class="bar">' + pills +
      '<span><b>' + d.omitted.length + '</b> omitted</span>' +
      '<span>grader <b>' + esc(d.grader) + '</b></span></div>' +
      '<div class="gate' + (d.published ? '' : ' blocked') + '">' + gate + '</div></header>' +
      '<div class="wrap"><div class="col"><h2>Generated note \u2014 click any claim</h2>' +
      claims + '<h2 style="margin-top:24px">Present in the encounter, absent from the note</h2>' +
      omitted + '</div><div class="col"><h2>Transcript</h2>' + utts + '</div></div>';
  }

  function clearMarks() {
    Array.prototype.forEach.call(document.querySelectorAll('.utt.hit'), function (e) {
      e.classList.remove('hit');
    });
    Array.prototype.forEach.call(document.querySelectorAll('.sel'), function (e) {
      e.classList.remove('sel');
    });
  }

  function focusUtterances(indices, source) {
    clearMarks();
    if (source) source.classList.add('sel');
    var first = null;
    indices.forEach(function (i) {
      var el = document.getElementById('u' + i);
      if (el) { el.classList.add('hit'); if (first === null) first = el; }
    });
    if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  render();
  document.addEventListener('click', function (ev) {
    var claim = ev.target.closest('[data-claim]');
    if (claim) {
      var a = d.assertions[+claim.dataset.claim];
      focusUtterances((a.spans || []).map(function (s) { return s.utterance_index; }), claim);
      return;
    }
    var omit = ev.target.closest('[data-omit]');
    if (omit) {
      focusUtterances([d.omitted[+omit.dataset.omit].utterance_index], omit);
    }
  });
})();
</script></body></html>
"""


def export(run: RunResult, consultation: Consultation, path: Path) -> Path:
    """Write a one-file review page for a single run."""
    payload = run.to_dict()
    data = {
        "consultation_id": run.consultation_id,
        "model_id": run.model_id,
        "grader": run.grounding.grader,
        "published": run.gate.published,
        "ruleset_hash": run.gate.ruleset_hash,
        "blocking": [f.detail for f in run.gate.unacknowledged_blocking],
        "summary": run.grounding.summary(),
        "assertions": payload["assertions"],
        "omitted": [
            dict(o, utterance_index=_index_for(consultation, o["start_s"]))
            for o in payload["omitted_facts"]
        ],
        "utterances": [
            {"speaker": u.speaker, "start_s": u.start_s, "end_s": u.end_s, "text": u.text}
            for u in consultation.utterances
        ],
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
        "__TITLE__", f"{run.consultation_id} / {run.model_id}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _index_for(consultation: Consultation, start_s: float) -> int:
    """Map a fact's timestamp back to its utterance.

    Omitted facts carry timings rather than indices because they come from the
    salience extractor, which works on time. Recovering the index here keeps
    the click-to-highlight behaviour identical for both panels.
    """
    for i, u in enumerate(consultation.utterances):
        if abs(u.start_s - start_s) < 0.01:
            return i
    return 0
