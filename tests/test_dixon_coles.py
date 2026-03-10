"""Tests for dixon_coles.py — score matrix, probabilities, team name resolution."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dixon_coles import (
    TeamStrength,
    LeagueParams,
    _canonical_team_name,
    _resolve_team_name,
    _tau,
    expected_goals,
    score_matrix,
    market_probabilities,
    resolve_team_strength,
)


# ---------- _tau correction ----------

class TestTau:
    def test_no_correction_high_scores(self):
        assert _tau(2, 3, 1.0, 1.0, -0.05) == 1.0

    def test_00_negative_rho(self):
        result = _tau(0, 0, 1.2, 0.8, -0.1)
        assert result == pytest.approx(1.0 - (1.2 * 0.8 * -0.1))

    def test_01(self):
        result = _tau(0, 1, 1.2, 0.8, -0.1)
        assert result == pytest.approx(1.0 + (1.2 * -0.1))

    def test_10(self):
        result = _tau(1, 0, 1.2, 0.8, -0.1)
        assert result == pytest.approx(1.0 + (0.8 * -0.1))

    def test_11(self):
        result = _tau(1, 1, 1.2, 0.8, -0.1)
        assert result == pytest.approx(1.0 - (-0.1))

    def test_rho_zero_is_identity(self):
        for x in range(3):
            for y in range(3):
                assert _tau(x, y, 1.5, 1.0, 0.0) == 1.0


# ---------- expected_goals ----------

class TestExpectedGoals:
    def test_positive_output(self):
        h, a = expected_goals(TeamStrength(0.2, -0.1), TeamStrength(0.1, -0.2), 0.3)
        assert h > 0
        assert a > 0

    def test_home_advantage_increases_home_goals(self):
        h0, _ = expected_goals(TeamStrength(0, 0), TeamStrength(0, 0), 0.0)
        h1, _ = expected_goals(TeamStrength(0, 0), TeamStrength(0, 0), 0.3)
        assert h1 > h0

    def test_formula(self):
        home = TeamStrength(attack=0.3, defence=-0.1)
        away = TeamStrength(attack=0.1, defence=-0.2)
        ha = 0.25
        lam_h, lam_a = expected_goals(home, away, ha)
        assert lam_h == pytest.approx(math.exp(0.3 + (-0.2) + 0.25))
        assert lam_a == pytest.approx(math.exp(0.1 + (-0.1)))


# ---------- score_matrix ----------

class TestScoreMatrix:
    def test_sums_to_one(self):
        mat = score_matrix(1.5, 1.2, -0.05)
        assert mat.sum() == pytest.approx(1.0, abs=1e-6)

    def test_shape(self):
        mat = score_matrix(1.0, 1.0, 0.0, max_goals=8)
        assert mat.shape == (9, 9)

    def test_all_non_negative(self):
        mat = score_matrix(1.5, 1.2, -0.1)
        assert (mat >= 0).all()

    def test_rho_zero_is_independent_poisson(self):
        mat = score_matrix(1.5, 1.2, 0.0, max_goals=6)
        from scipy.stats import poisson
        h = poisson.pmf(np.arange(7), 1.5)
        a = poisson.pmf(np.arange(7), 1.2)
        expected = np.outer(h, a)
        expected /= expected.sum()
        np.testing.assert_allclose(mat, expected, atol=1e-10)


# ---------- market_probabilities ----------

class TestMarketProbabilities:
    def test_probabilities_sum_to_one(self):
        mat = score_matrix(1.5, 1.2, -0.05)
        probs = market_probabilities(mat)
        total = probs["home_win"] + probs["draw"] + probs["away_win"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_1x_equals_home_plus_draw(self):
        mat = score_matrix(1.5, 1.2, -0.05)
        probs = market_probabilities(mat)
        assert probs["1X"] == pytest.approx(probs["home_win"] + probs["draw"])

    def test_x2_equals_draw_plus_away(self):
        mat = score_matrix(1.5, 1.2, -0.05)
        probs = market_probabilities(mat)
        assert probs["X2"] == pytest.approx(probs["draw"] + probs["away_win"])

    def test_strong_home_has_high_1x(self):
        mat = score_matrix(2.5, 0.5, -0.05)
        probs = market_probabilities(mat)
        assert probs["1X"] > 0.9


# ---------- _canonical_team_name ----------

class TestCanonicalTeamName:
    def test_strips_fc(self):
        assert _canonical_team_name("FC Porto") == "porto"

    def test_lowercase_and_normalize(self):
        assert _canonical_team_name("São Paulo") == "sao paulo"

    def test_removes_digits(self):
        assert _canonical_team_name("1860 Munich") == "munich"


# ---------- _resolve_team_name ----------

class TestResolveTeamName:
    KNOWN = {
        "hull city": {"attack": 0.1, "defence": -0.1},
        "fc porto": {"attack": 0.2, "defence": -0.2},
        "manchester united": {"attack": 0.3, "defence": -0.1},
    }

    def test_exact_match(self):
        assert _resolve_team_name(self.KNOWN, "hull city") == "hull city"

    def test_canonical_match(self):
        assert _resolve_team_name(self.KNOWN, "Hull City FC") == "hull city"

    def test_substring_match(self):
        assert _resolve_team_name(self.KNOWN, "hull") == "hull city"

    def test_no_match(self):
        assert _resolve_team_name(self.KNOWN, "zzz unknown team") is None


# ---------- resolve_team_strength ----------

class TestResolveTeamStrength:
    RATINGS_V2 = {
        "leagues": {
            "E0": {
                "home_advantage": 0.25,
                "rho": -0.04,
                "teams": {
                    "arsenal": {"attack": 0.4, "defence": -0.3},
                    "chelsea": {"attack": 0.2, "defence": -0.1},
                },
            }
        }
    }

    def test_v2_format(self):
        result = resolve_team_strength(self.RATINGS_V2, "E0", "arsenal", "chelsea", 0.0, -0.05)
        assert result is not None
        home, away, params = result
        assert home.attack == pytest.approx(0.4)
        assert away.defence == pytest.approx(-0.1)
        assert params.home_advantage == pytest.approx(0.25)

    def test_missing_league(self):
        result = resolve_team_strength(self.RATINGS_V2, "XX", "arsenal", "chelsea", 0.0, -0.05)
        assert result is None

    def test_missing_team(self):
        result = resolve_team_strength(self.RATINGS_V2, "E0", "arsenal", "nonexistent", 0.0, -0.05)
        assert result is None

    def test_v1_format(self):
        flat = {
            "teamA": {"attack": 0.1, "defence": -0.2},
            "teamB": {"attack": 0.3, "defence": -0.1},
        }
        result = resolve_team_strength(flat, "E0", "teamA", "teamB", 0.2, -0.06)
        assert result is not None
        _, _, params = result
        assert params.home_advantage == pytest.approx(0.2)