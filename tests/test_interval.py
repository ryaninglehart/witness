"""Tests for the resampling.

A bootstrap is easy to write and easy to write wrongly, and both versions
produce plausible intervals. The three ways to get it wrong here are all
tested: resampling the wrong unit, averaging a ratio instead of pooling it, and
comparing two graders through separate intervals instead of a paired one.
"""

from __future__ import annotations

import statistics

from witness.interval import (
    bootstrap,
    median_of,
    paired_difference,
    ratio_of,
)

IDS = [f"c{i:02d}" for i in range(40)]


class TestBootstrap:
    def test_the_point_estimate_is_the_unresampled_statistic(self):
        vals = {c: float(i) for i, c in enumerate(IDS)}
        iv = bootstrap(IDS, median_of(vals), resamples=200)
        assert iv.point == statistics.median(vals.values())

    def test_the_interval_brackets_the_point(self):
        vals = {c: float(i % 7) for i, c in enumerate(IDS)}
        iv = bootstrap(IDS, median_of(vals), resamples=500)
        assert iv.low <= iv.point <= iv.high

    def test_it_is_seeded_and_reproducible(self):
        # An interval that moves when you re-run it is not an interval.
        vals = {c: float(i % 5) for i, c in enumerate(IDS)}
        a = bootstrap(IDS, median_of(vals), resamples=300)
        b = bootstrap(IDS, median_of(vals), resamples=300)
        assert (a.low, a.high) == (b.low, b.high)

    def test_a_different_seed_gives_a_similar_but_not_identical_interval(self):
        vals = {c: float(i % 5) for i, c in enumerate(IDS)}
        a = bootstrap(IDS, median_of(vals), resamples=500, seed=1)
        b = bootstrap(IDS, median_of(vals), resamples=500, seed=2)
        assert abs(a.width - b.width) < 1.0

    def test_more_data_narrows_the_interval(self):
        # The property that makes an interval mean anything.
        small = [f"c{i}" for i in range(10)]
        large = [f"c{i}" for i in range(400)]
        vals_s = {c: float(i % 9) for i, c in enumerate(small)}
        vals_l = {c: float(i % 9) for i, c in enumerate(large)}
        a = bootstrap(small, median_of(vals_s), resamples=500)
        b = bootstrap(large, median_of(vals_l), resamples=500)
        assert b.width < a.width

    def test_no_variation_gives_a_degenerate_interval(self):
        vals = dict.fromkeys(IDS, 0.5)
        iv = bootstrap(IDS, median_of(vals), resamples=200)
        assert iv.low == iv.high == 0.5


class TestPooledRatio:
    def test_a_ratio_is_pooled_not_averaged(self):
        # One consultation with 1/1 and one with 0/99 is 1% pooled and 50%
        # averaged. Averaging per-consultation recalls weights an encounter
        # with two findings the same as one with forty, which is a different
        # statistic than the one the README reports.
        num = {"a": 1.0, "b": 0.0}
        den = {"a": 1.0, "b": 99.0}
        pooled = ratio_of(num, den)(["a", "b"])
        averaged = statistics.fmean([1 / 1, 0 / 99])
        assert abs(pooled - 0.01) < 1e-9
        assert abs(averaged - 0.5) < 1e-9
        assert pooled != averaged

    def test_repeated_ids_in_a_resample_are_counted_repeatedly(self):
        # Bootstrap draws contain duplicates by construction. A statistic that
        # deduplicates them silently would resample nothing.
        num, den = {"a": 1.0}, {"a": 2.0}
        f = ratio_of(num, den)
        assert f(["a"]) == f(["a", "a", "a"]) == 0.5
        assert ratio_of({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0})(["a", "a", "b"]) == 2 / 3

    def test_an_empty_denominator_does_not_divide_by_zero(self):
        assert ratio_of({}, {})(["a"]) == 0.0


class TestPairedDifference:
    def test_pairing_is_tighter_than_two_separate_intervals(self):
        # The reason the comparison is paired. Both graders find the same
        # consultations hard; that shared variance says nothing about the
        # difference between them, and leaving it in hides real effects.
        hard = {c: (0.9 if i % 2 else 0.1) for i, c in enumerate(IDS)}
        a_vals = {c: v + 0.05 for c, v in hard.items()}
        a, b = median_of(a_vals), median_of(hard)
        paired = paired_difference(IDS, "a-b", a, b, resamples=800)
        ia = bootstrap(IDS, a, resamples=800)
        ib = bootstrap(IDS, b, resamples=800)
        separate_width = (ia.high - ia.low) + (ib.high - ib.low)
        assert paired.high - paired.low < separate_width

    def test_a_real_difference_is_reported_distinguishable(self):
        base = {c: float(i % 10) / 10 for i, c in enumerate(IDS)}
        shifted = {c: v + 0.5 for c, v in base.items()}
        d = paired_difference(
            IDS, "shift", median_of(shifted), median_of(base), resamples=800
        )
        assert d.distinguishable
        assert d.point > 0

    def test_identical_statistics_are_not_distinguishable(self):
        vals = {c: float(i % 6) for i, c in enumerate(IDS)}
        f = median_of(vals)
        d = paired_difference(IDS, "same", f, f, resamples=400)
        assert not d.distinguishable
        assert d.point == 0.0

    def test_the_wording_does_not_claim_equality(self):
        # "Not distinguishable" is a statement about the corpus. Rendering it
        # as "no difference" would be the same overclaim in the other
        # direction, which is the failure this whole module guards against.
        vals = {c: float(i % 6) for i, c in enumerate(IDS)}
        f = median_of(vals)
        text = str(paired_difference(IDS, "same", f, f, resamples=200))
        assert "not distinguishable" in text
        assert "no difference" not in text
        assert "equal" not in text
