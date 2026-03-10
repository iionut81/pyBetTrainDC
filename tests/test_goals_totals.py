"""Tests for goals totals logic — O/U probability from score matrix."""
from __future__ import annotations

import numpy as np
import pytest

from dixon_coles import score_matrix


def _ou_probs(mat: np.ndarray) -> dict[str, float]:
    """Reproduce the _ou_probs function from run_goals_totals_daily.py."""
    n = mat.shape[0]
    idx = np.arange(n)
    ig, jg = np.meshgrid(idx, idx, indexing="ij")
    total = ig + jg
    return {
        "over_2_5": float(mat[total >= 3].sum()),
        "under_3_5": float(mat[total <= 3].sum()),
        "under_4_5": float(mat[total <= 4].sum()),
    }


class TestOUProbs:
    def test_probabilities_bounded(self):
        mat = score_matrix(1.5, 1.2, -0.05, max_goals=10)
        probs = _ou_probs(mat)
        for key, val in probs.items():
            assert 0.0 <= val <= 1.0, f"{key}={val}"

    def test_over_under_complement(self):
        # P(over 2.5) + P(under 2.5) = 1
        # P(under 2.5) = P(total <= 2) = 1 - P(over 2.5) ... but we have under_3_5 = P(total <= 3)
        # So: over_2_5 + P(total <= 2) = 1 — let's verify the non-overlap
        mat = score_matrix(1.5, 1.2, -0.05, max_goals=10)
        probs = _ou_probs(mat)
        n = mat.shape[0]
        idx = np.arange(n)
        ig, jg = np.meshgrid(idx, idx, indexing="ij")
        total = ig + jg
        p_under_2_5 = float(mat[total <= 2].sum())
        assert probs["over_2_5"] + p_under_2_5 == pytest.approx(1.0, abs=1e-6)

    def test_higher_xg_means_more_over(self):
        mat_low = score_matrix(0.8, 0.7, -0.05, max_goals=10)
        mat_high = score_matrix(2.0, 1.8, -0.05, max_goals=10)
        assert _ou_probs(mat_high)["over_2_5"] > _ou_probs(mat_low)["over_2_5"]

    def test_under_4_5_gte_under_3_5(self):
        mat = score_matrix(1.5, 1.2, -0.05, max_goals=10)
        probs = _ou_probs(mat)
        assert probs["under_4_5"] >= probs["under_3_5"]

    def test_extreme_low_xg(self):
        mat = score_matrix(0.3, 0.3, -0.05, max_goals=10)
        probs = _ou_probs(mat)
        assert probs["under_3_5"] > 0.95
        assert probs["under_4_5"] > 0.99