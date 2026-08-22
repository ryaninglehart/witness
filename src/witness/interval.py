"""How much of a difference is a difference.

The README says 57 consultations is a small corpus and that differences of a
few points should be read as noise. That sentence is doing a lot of work for
something with no number attached to it, and a reader has no way to tell
whether "a few" means two points or fifteen. Nor did I, when I wrote it.

So it is resampled instead. The consultation is the unit -- notes from the same
encounter share a transcript, a patient and a clinician, so resampling
individual notes would treat 20 correlated observations as 20 independent ones
and produce intervals far too narrow. Sampling whole consultations with
replacement keeps that structure intact.

Two things are reported, and the second matters more:

**A percentile interval for each figure**, which says how precisely the corpus
pins it down.

**A paired interval on the difference between two graders**, computed on the
same resamples. This is the question anyone actually has -- is the learned
grader better -- and it is not answerable by checking whether two separate
intervals overlap. Non-overlapping intervals imply a difference; *overlapping*
intervals do not imply its absence, and the two graders are evaluated on
identical consultations, so the paired comparison is both more powerful and
the correct one.

The resampling is seeded and the seed is part of the result. An interval that
moves when you re-run it is not an interval.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range the corpus actually supports."""

    point: float
    low: float
    high: float
    level: float = 0.95
    resamples: int = 2000

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


@dataclass(frozen=True)
class Difference:
    """A paired comparison, and whether the corpus can tell the two apart."""

    label: str
    point: float
    low: float
    high: float
    level: float = 0.95

    @property
    def distinguishable(self) -> bool:
        """Does the interval exclude zero?

        Phrased as a question about the corpus rather than about truth. An
        interval containing zero means these 57 consultations cannot separate
        the two graders; it does not mean they are equal, and reporting it as
        "no difference" would be the same overclaim in the other direction.
        """
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        verdict = "distinguishable" if self.distinguishable else "not distinguishable"
        return f"{self.label}: {self.point:+.1%} [{self.low:+.1%}, {self.high:+.1%}] -- {verdict}"


def _percentiles(draws: list[float], level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    ordered = sorted(draws)
    return (
        float(np.quantile(ordered, tail)),
        float(np.quantile(ordered, 1.0 - tail)),
    )


def bootstrap(
    units: Sequence[str],
    statistic: Callable[[Sequence[str]], float],
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 20260821,
) -> Interval:
    """Resample whole consultations with replacement and recompute.

    ``statistic`` receives a list of consultation ids -- possibly with repeats
    -- and returns one number. Keeping it a callable over ids rather than over
    precomputed values is what lets a ratio like recall be recomputed correctly
    on each draw: recall is not the mean of per-consultation recalls, and
    averaging those instead would quietly answer a different question.
    """
    rng = np.random.default_rng(seed)
    n = len(units)
    draws: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        draws.append(statistic([units[i] for i in idx]))
    low, high = _percentiles(draws, level)
    return Interval(statistic(list(units)), low, high, level, resamples)


def paired_difference(
    units: Sequence[str],
    label: str,
    a: Callable[[Sequence[str]], float],
    b: Callable[[Sequence[str]], float],
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 20260821,
) -> Difference:
    """Bootstrap ``a - b`` on identical resamples.

    Both statistics see the same draw every iteration. That pairing is the
    whole point: it removes the variance that comes from some consultations
    being harder than others, which is variance both graders share and which
    tells you nothing about the difference between them.
    """
    rng = np.random.default_rng(seed)
    n = len(units)
    draws: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        sample = [units[i] for i in idx]
        draws.append(a(sample) - b(sample))
    low, high = _percentiles(draws, level)
    return Difference(label, a(list(units)) - b(list(units)), low, high, level)


def median_of(values: dict[str, float]) -> Callable[[Sequence[str]], float]:
    """A statistic over per-consultation values, safe under repeated ids."""

    def f(sample: Sequence[str]) -> float:
        vals = [values[c] for c in sample if c in values]
        return statistics.median(vals) if vals else 0.0

    return f


def ratio_of(
    numerator: dict[str, float], denominator: dict[str, float]
) -> Callable[[Sequence[str]], float]:
    """A pooled ratio -- summed across the draw, not averaged per consultation.

    Recall over a resample is total hits over total findings. Taking the mean
    of per-consultation recalls weights a consultation with two findings the
    same as one with forty, which is a different statistic than the one the
    README reports.
    """

    def f(sample: Sequence[str]) -> float:
        num = sum(numerator.get(c, 0.0) for c in sample)
        den = sum(denominator.get(c, 0.0) for c in sample)
        return num / den if den else 0.0

    return f
