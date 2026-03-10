"""Tests for corners logic — NB/Poisson probability, lambda calculation."""
from __future__ import annotations

import pytest
from scipy.stats import nbinom, poisson


def _prob_under_12_5(lam: float, model: str, k: float) -> float:
    """Reproduce _prob_under_12_5 from run_corners_daily.py."""
    lam = max(1e-6, float(lam))
    if model == "poisson":
        return float(poisson.cdf(12, lam))
    p = k / (k + lam)
    return float(nbinom.cdf(12, k, p))


class TestProbUnder12_5:
    def test_output_bounded(self):
        for lam in (5.0, 10.0, 15.0):
            p = _prob_under_12_5(lam, "nb", k=12.0)
            assert 0.0 <= p <= 1.0

    def test_poisson_model(self):
        p = _prob_under_12_5(10.0, "poisson", k=0)
        assert 0.5 < p < 1.0  # mean 10, CDF at 12 should be well above 0.5

    def test_lower_lambda_higher_prob(self):
        p_low = _prob_under_12_5(8.0, "nb", k=12.0)
        p_high = _prob_under_12_5(14.0, "nb", k=12.0)
        assert p_low > p_high

    def test_very_low_lambda(self):
        p = _prob_under_12_5(3.0, "nb", k=12.0)
        assert p > 0.99

    def test_very_high_lambda(self):
        p = _prob_under_12_5(20.0, "nb", k=12.0)
        assert p < 0.15

    def test_poisson_vs_nb_similar_for_large_k(self):
        # As k → ∞, NB converges to Poisson
        p_pois = _prob_under_12_5(10.0, "poisson", k=0)
        p_nb = _prob_under_12_5(10.0, "nb", k=1000.0)
        assert abs(p_pois - p_nb) < 0.01


class TestLambdaCalculation:
    def test_lambda_formula(self):
        # Reproduce the lambda calculation from run_corners_daily.py
        h_for, h_against = 6.0, 5.0
        a_for, a_against = 5.5, 5.5
        tempo = 1.05
        mu = 10.0

        lam_base = (h_for + a_against + a_for + h_against) / 2.0
        lam = 0.8 * (lam_base * tempo) + 0.2 * mu
        lam = max(2.5, min(18.0, lam))

        assert lam_base == pytest.approx(11.0)
        expected = 0.8 * (11.0 * 1.05) + 0.2 * 10.0
        assert lam == pytest.approx(expected)

    def test_lambda_clipping_low(self):
        lam = max(2.5, min(18.0, 1.0))
        assert lam == 2.5

    def test_lambda_clipping_high(self):
        lam = max(2.5, min(18.0, 25.0))
        assert lam == 18.0