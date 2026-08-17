"""Tests for selection_engine.classification — percentile computation and
rank_value -> (historical_percentile, label, bet_eligible) classification.

historical_percentile must be derived EXCLUSIVELY from comparing rank_value
(p_cal_adj) against the historical p_cal_adj distribution breakpoints — never
from hit rate, a 0-100 score, or any other derived value."""
from __future__ import annotations

import pytest

from selection_engine.classification import (
    LABEL_HIGH,
    LABEL_LOW,
    LABEL_MEDIUM,
    LABEL_TOP,
    LABEL_VERY_LOW,
    classify,
    compute_percentiles,
)

PERCENTILES = {
    "p0": 0.70, "p20": 0.80, "p40": 0.85, "p60": 0.88, "p80": 0.92,
    "p90": 0.95, "p95": 0.97, "p100": 0.98,
}
ALL_BREAKPOINT_KEYS = {"p0", "p20", "p40", "p60", "p80", "p90", "p95", "p100"}


class TestComputePercentiles:
    def test_returns_all_eight_breakpoints(self):
        values = [0.70 + 0.01 * i for i in range(29)]  # 0.70..0.98
        result = compute_percentiles(values)
        assert set(result.keys()) == ALL_BREAKPOINT_KEYS

    def test_breakpoints_are_monotonic(self):
        values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.72, 0.88, 0.91]
        result = compute_percentiles(values)
        ordered = [result[k] for k in ("p0", "p20", "p40", "p60", "p80", "p90", "p95", "p100")]
        assert ordered == sorted(ordered)

    def test_p0_and_p100_are_min_and_max(self):
        values = [0.9, 0.7, 0.95, 0.8]
        result = compute_percentiles(values)
        assert result["p0"] == pytest.approx(0.7)
        assert result["p100"] == pytest.approx(0.95)


class TestClassify:
    def test_none_rank_value_is_unclassified(self):
        historical_percentile, label, bet_eligible = classify(None, PERCENTILES)
        assert historical_percentile is None
        assert label == ""
        assert bet_eligible is False

    def test_empty_percentiles_is_unclassified(self):
        historical_percentile, label, bet_eligible = classify(0.90, {})
        assert historical_percentile is None
        assert bet_eligible is False

    def test_at_or_above_p80_is_top_quintile_and_eligible(self):
        _, label, bet_eligible = classify(0.92, PERCENTILES)
        assert label == LABEL_TOP
        assert bet_eligible is True

    def test_between_p60_and_p80_is_high_not_eligible(self):
        _, label, bet_eligible = classify(0.89, PERCENTILES)
        assert label == LABEL_HIGH
        assert bet_eligible is False

    def test_between_p40_and_p60_is_medium(self):
        _, label, bet_eligible = classify(0.86, PERCENTILES)
        assert label == LABEL_MEDIUM
        assert bet_eligible is False

    def test_between_p20_and_p40_is_low(self):
        _, label, bet_eligible = classify(0.82, PERCENTILES)
        assert label == LABEL_LOW
        assert bet_eligible is False

    def test_below_p20_is_very_low(self):
        _, label, bet_eligible = classify(0.75, PERCENTILES)
        assert label == LABEL_VERY_LOW
        assert bet_eligible is False

    def test_being_best_available_is_not_the_same_as_eligible(self):
        # A value clearly the "best" of a hypothetical small batch can still
        # be far below the real historical p80 -- and must NOT be eligible.
        _, label, bet_eligible = classify(0.885, PERCENTILES)
        assert label == LABEL_HIGH
        assert bet_eligible is False

    def test_historical_percentile_interpolates_between_breakpoints(self):
        # Halfway between p60 (0.88) and p80 (0.92) -> ~70th percentile
        historical_percentile, _, _ = classify(0.90, PERCENTILES)
        assert 65.0 <= historical_percentile <= 75.0

    def test_value_below_p0_clamps_to_zero_percentile_no_extrapolation(self):
        historical_percentile, _, _ = classify(0.50, PERCENTILES)
        assert historical_percentile == pytest.approx(0.0)

    def test_value_above_p100_clamps_to_hundred_percentile_no_extrapolation(self):
        historical_percentile, _, _ = classify(1.0, PERCENTILES)
        assert historical_percentile == pytest.approx(100.0)

    def test_percentile_derived_only_from_p_cal_adj_distribution(self):
        # Two different rank_values straddling p80 with an IDENTICAL
        # percentiles dict must classify purely off rank_value vs
        # breakpoints -- nothing else can enter this computation.
        just_below, _, not_eligible = classify(PERCENTILES["p80"] - 1e-9, PERCENTILES)
        just_above, _, eligible = classify(PERCENTILES["p80"] + 1e-9, PERCENTILES)
        assert not_eligible is False
        assert eligible is True
        assert just_below < 80.0 <= just_above


class TestBetThresholdPercentile:
    def test_custom_threshold_changes_eligibility_not_label(self):
        # label bands are fixed at 80/60/40/20; bet_threshold_percentile is a
        # separate, configurable eligibility knob that defaults to 80.
        _, label, eligible_default = classify(0.89, PERCENTILES)  # HIGH, ~70th pct
        _, label_custom, eligible_custom = classify(0.89, PERCENTILES, bet_threshold_percentile=60.0)
        assert label == label_custom == LABEL_HIGH
        assert eligible_default is False
        assert eligible_custom is True
