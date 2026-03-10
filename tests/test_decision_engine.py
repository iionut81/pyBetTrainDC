"""Tests for decision_engine.py — market evaluation, edge, variance classification."""
from __future__ import annotations

import pytest

from decision_engine import implied_probability, classify_variance, evaluate_market


class TestImpliedProbability:
    def test_normal_odds(self):
        assert implied_probability(2.0) == pytest.approx(0.5)

    def test_evens(self):
        assert implied_probability(1.0) == pytest.approx(1.0)

    def test_none(self):
        assert implied_probability(None) is None

    def test_zero(self):
        assert implied_probability(0) is None

    def test_negative(self):
        assert implied_probability(-1.5) is None


class TestClassifyVariance:
    def test_low(self):
        assert classify_variance(0.10) == "LOW"

    def test_low_boundary(self):
        assert classify_variance(0.145) == "LOW"

    def test_low_medium(self):
        assert classify_variance(0.16) == "LOW-MEDIUM"

    def test_medium(self):
        assert classify_variance(0.20) == "MEDIUM"

    def test_high(self):
        assert classify_variance(0.25) == "HIGH"


class TestEvaluateMarket:
    def test_recommended_when_all_pass(self):
        result = evaluate_market(
            market="1X",
            model_probability=0.85,
            variance_value=0.12,       # LOW
            upset_frequency=0.15,
            offered_odds=1.30,
            min_dc_probability=0.78,
            min_odds=1.25,
            max_odds=1.35,
        )
        assert result["recommended"] is True
        assert result["edge"] > 0

    def test_not_recommended_low_probability(self):
        result = evaluate_market(
            market="1X",
            model_probability=0.70,    # below 0.78
            variance_value=0.12,
            upset_frequency=0.30,
            offered_odds=1.30,
        )
        assert result["recommended"] is False

    def test_not_recommended_odds_out_of_range(self):
        result = evaluate_market(
            market="X2",
            model_probability=0.85,
            variance_value=0.12,
            upset_frequency=0.15,
            offered_odds=1.50,         # above max_odds=1.35
        )
        assert result["recommended"] is False

    def test_not_recommended_no_odds(self):
        result = evaluate_market(
            market="1X",
            model_probability=0.85,
            variance_value=0.12,
            upset_frequency=0.15,
            offered_odds=None,
        )
        assert result["recommended"] is False
        assert result["odds_source"] == "missing"

    def test_not_recommended_negative_edge(self):
        result = evaluate_market(
            market="1X",
            model_probability=0.80,
            variance_value=0.12,
            upset_frequency=0.20,
            offered_odds=1.15,         # implied=0.87 > model=0.80 → negative edge
            min_odds=1.10,
            max_odds=1.35,
        )
        assert result["recommended"] is False
        assert result["edge"] < 0

    def test_not_recommended_high_variance(self):
        result = evaluate_market(
            market="1X",
            model_probability=0.85,
            variance_value=0.23,       # HIGH
            upset_frequency=0.15,
            offered_odds=1.30,
            allowed_variance_classes={"LOW"},
        )
        assert result["recommended"] is False

    def test_fair_odds_calculation(self):
        result = evaluate_market(
            market="1X",
            model_probability=0.80,
            variance_value=0.15,
            upset_frequency=0.20,
            offered_odds=1.30,
        )
        assert result["fair_odds"] == pytest.approx(1.25)