"""Tests for simulation.py — Monte Carlo output shape and bounds."""
from __future__ import annotations

import pytest

from simulation import run_monte_carlo


class TestRunMonteCarlo:
    def test_keys_present(self):
        result = run_monte_carlo(1.5, 1.2)
        expected_keys = {"mc_1X", "mc_X2", "upset_1X", "upset_X2", "variance_1X", "variance_X2"}
        assert set(result.keys()) == expected_keys

    def test_probabilities_in_01(self):
        result = run_monte_carlo(1.5, 1.2)
        for key in ("mc_1X", "mc_X2", "upset_1X", "upset_X2"):
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]}"

    def test_variance_non_negative(self):
        result = run_monte_carlo(1.5, 1.2)
        assert result["variance_1X"] >= 0
        assert result["variance_X2"] >= 0

    def test_deterministic_with_seed(self):
        r1 = run_monte_carlo(1.5, 1.2, seed=99)
        r2 = run_monte_carlo(1.5, 1.2, seed=99)
        for key in r1:
            assert r1[key] == r2[key]

    def test_strong_home_gives_high_1x(self):
        result = run_monte_carlo(3.0, 0.5, iterations=100_000)
        assert result["mc_1X"] > 0.9

    def test_1x_plus_upset_1x_near_one(self):
        # mc_1X = P(home_win | draw), upset_1X = P(away_win)
        # They should sum to 1
        result = run_monte_carlo(1.5, 1.2, iterations=100_000)
        assert result["mc_1X"] + result["upset_1X"] == pytest.approx(1.0, abs=1e-3)