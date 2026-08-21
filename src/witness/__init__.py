"""Witness -- a faithfulness harness for ambient clinical documentation."""

import sys

# Checked here rather than left to fail later. On 3.10 the first symptom is
# `ImportError: cannot import name 'StrEnum'` from a module three levels down,
# which reads like a broken repository rather than an old interpreter --
# macOS still ships 3.9 as `python3`, so this is the likeliest way a reviewer
# meets this project for the first time.
if sys.version_info < (3, 11):  # noqa: UP036
    raise RuntimeError(
        "Witness needs Python 3.11 or newer; this is "
        f"{sys.version_info.major}.{sys.version_info.minor}. "
        "On macOS, `brew install python@3.12` then rebuild the venv with "
        "`python3.12 -m venv .venv`."
    )

