"""The README's claims, checked against what the commands actually print.

WIT-014: four numbers sat in a fenced block under a `make validate` heading,
formatted exactly as terminal output, and no command produced them. One of them
carried the entire precision defence. They survived every read because a code
fence reads as evidence — it has the authority of having been executed, and
nothing checked whether it had been.

So this compares them. Any block in README.md tagged with the command that
produced it is re-derived by running that command, and every percentage in the
block must appear in the real output.

This is deliberately narrow. It does not verify prose, and a number stated in a
sentence can still drift; `witness.crossval.PUBLISHED` guards the eight figures
the argument rests on. What this closes is the specific hole WIT-014 went
through: output that was never output.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: A fenced block in README.md may declare the command that produced it with a
#: trailing marker comment. Blocks without one are not checked -- they are
#: illustrations, not transcripts, and the distinction is the point.
_BLOCK = re.compile(
    r"<!--\s*verify:\s*(?P<cmd>[\w\s-]+?)\s*-->\s*\n```(?:\w+)?\n(?P<body>.*?)```",
    re.DOTALL,
)
_PERCENT = re.compile(r"\d+\.\d+%")


def _run(subcommand: str) -> str:
    if not (ROOT / "primock57").exists():
        pytest.skip("PriMock57 not fetched; run `make corpus`")
    proc = subprocess.run(
        [sys.executable, "-m", "witness.cli", *subcommand.split()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        check=False,
    )
    return proc.stdout


def _blocks() -> list[tuple[str, str]]:
    return [(m.group("cmd"), m.group("body")) for m in _BLOCK.finditer(README.read_text())]


def test_at_least_one_block_is_verified():
    # If the marker is removed or renamed, this file would pass by checking
    # nothing -- which is WIT-013's failure mode applied to WIT-014's fix.
    assert _blocks(), "no verify-tagged blocks found in README.md"


@pytest.mark.parametrize("command,body", _blocks(), ids=lambda x: str(x)[:24])
def test_every_documented_figure_appears_in_real_output(command: str, body: str):
    actual = _run(command)
    claimed = set(_PERCENT.findall(body))
    assert claimed, f"block for `{command}` contains no figures to check"
    missing = sorted(v for v in claimed if v not in actual)
    assert not missing, (
        f"README shows these under `witness {command}` but the command does not "
        f"print them: {missing}"
    )
