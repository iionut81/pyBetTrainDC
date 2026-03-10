"""Tests for fhg_weibull.py — Weibull FHG probability and k estimation."""
from __future__ import annotations

import math

import pytest

from fhg_weibull import clamp_probability, k_from_fh_share, p_goal_before_45


class TestClampProbability:
    def test_normal_value(self):
        assert clamp_probability(0.5) == 0.5

    def test_clamp_zero(self):
        assert clamp_probability(0.0) > 0

    def test_clamp_one(self):
        assert clamp_probability(1.0) < 1.0

    def test_clamp_negative(self):
        assert clamp_probability(-0.5) > 0


class TestKFromFhShare:
    def test_share_half_gives_k_one(self):
        # 0.5^k = 0.5 → k = 1
        assert k_from_fh_share(0.5) == pytest.approx(1.0)

    def test_share_quarter_gives_k_two(self):
        # 0.5^k = 0.25 → k = 2
        assert k_from_fh_share(0.25) == pytest.approx(2.0)

    def test_invalid_share_returns_default(self):
        assert k_from_fh_share(0.0) == 1.2
        assert k_from_fh_share(1.0) == 1.2
        assert k_from_fh_share(-0.1) == 1.2

    def test_clamped_within_bounds(self):
        k = k_from_fh_share(0.99)  # very high share → low k
        assert k >= 0.6
        k = k_from_fh_share(0.01)  # very low share → high k
        assert k <= 2.0


class TestPGoalBefore45:
    def test_output_between_0_and_1(self):
        p = p_goal_before_45(2.5, 1.0)
        assert 0 < p < 1

    def test_higher_xg_higher_probability(self):
        p_low = p_goal_before_45(1.0, 1.0)
        p_high = p_goal_before_45(3.0, 1.0)
        assert p_high > p_low

    def test_zero_xg_near_zero(self):
        p = p_goal_before_45(0.0, 1.0)
        assert p < 0.01

    def test_known_value(self):
        # H(45) = xg * 0.5^k = 2.5 * 0.5^1.0 = 1.25
        # P = 1 - exp(-1.25) ≈ 0.7135
        p = p_goal_before_45(2.5, 1.0)
        assert p == pytest.approx(1.0 - math.exp(-1.25), abs=1e-6)