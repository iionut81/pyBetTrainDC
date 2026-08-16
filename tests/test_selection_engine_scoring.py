"""Tests for the TENNIS_SET1_OVER_7_5 category scorers and contradiction detection."""
from __future__ import annotations

import pytest

from selection_engine.contradiction import detect_contradiction
from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    score_form,
    score_market_compatibility,
    score_matchup,
    score_statistics,
    score_stability,
)
from selection_engine.types import CategoryScore, MarketProfile, MatchInput


def _match(**stats) -> MatchInput:
    return MatchInput(
        match_id="X",
        market=MARKET_ID,
        sport="tennis",
        competitors=("A", "B"),
        stats=stats,
    )


class TestScoreForm:
    def test_missing_variance_is_neutral_with_risk_note(self):
        result = score_form(_match())
        assert result.value == 14.0
        assert any(n.startswith("-") for n in result.notes)

    def test_low_variance_scores_high(self):
        result = score_form(_match(recent_form_variance_a=0.05, recent_form_variance_b=0.05))
        assert result.value > 15.0
        assert any(n.startswith("+") for n in result.notes)

    def test_high_variance_scores_low(self):
        result = score_form(_match(recent_form_variance_a=0.5, recent_form_variance_b=0.5))
        assert result.value < 12.0
        assert any(n.startswith("-") for n in result.notes)


class TestScoreMatchup:
    def test_balanced_sweet_spot_scores_max(self):
        result = score_matchup(_match(p_hold_a=0.70, p_hold_b=0.69))
        assert result.value == pytest.approx(20.0)

    def test_imbalanced_holds_score_lower(self):
        result = score_matchup(_match(p_hold_a=0.82, p_hold_b=0.55))
        low = score_matchup(_match(p_hold_a=0.70, p_hold_b=0.69))
        assert result.value < low.value
        assert any(n.startswith("-") for n in result.notes)


class TestScoreStatistics:
    def test_high_expected_games_scores_high(self):
        result = score_statistics(_match(expected_total_games=13.0))
        assert result.value == pytest.approx(20.0)
        assert any(n.startswith("+") for n in result.notes)

    def test_low_expected_games_scores_low(self):
        result = score_statistics(_match(expected_total_games=9.0))
        assert result.value == pytest.approx(0.0)
        assert any(n.startswith("-") for n in result.notes)


class TestScoreMarketCompatibility:
    def test_high_tiebreak_rate_scores_high(self):
        result = score_market_compatibility(_match(surface="Grass", tiebreak_rate=0.50))
        assert result.value > 14.0
        assert any(n.startswith("+") for n in result.notes)

    def test_low_tiebreak_rate_scores_lower(self):
        high = score_market_compatibility(_match(surface="Grass", tiebreak_rate=0.30))
        low = score_market_compatibility(_match(surface="Grass", tiebreak_rate=0.05))
        assert low.value < high.value
        assert any(n.startswith("-") for n in low.notes)

    def test_surface_no_longer_affects_score(self):
        # Removed 2026-08-16 after backtest showed the surface bonus made no
        # measurable difference to hit rate — same tiebreak_rate must now
        # score identically regardless of surface.
        grass = score_market_compatibility(_match(surface="Grass", tiebreak_rate=0.20))
        hard = score_market_compatibility(_match(surface="Hard", tiebreak_rate=0.20))
        clay = score_market_compatibility(_match(surface="Clay", tiebreak_rate=0.20))
        assert grass.value == hard.value == clay.value


class TestScoreStability:
    def test_no_flags_scores_max(self):
        result = score_stability(_match(p_hold_a=0.70, p_hold_b=0.68))
        assert result.value == pytest.approx(20.0)

    def test_weak_server_and_gap_penalized(self):
        result = score_stability(_match(p_hold_a=0.80, p_hold_b=0.50))
        assert result.value < 10.0
        assert len(result.notes) >= 1


class TestDetectContradiction:
    def _profile(self) -> MarketProfile:
        return MarketProfile(market_id=MARKET_ID, sport="tennis")

    def test_spec_example_flags_contradiction(self):
        # statistics=19 (high) alongside matchup=8 and stability=7 (low)
        scores = {
            "form": CategoryScore(18),
            "matchup": CategoryScore(8),
            "statistics": CategoryScore(19),
            "market_compatibility": CategoryScore(19),
            "stability": CategoryScore(7),
        }
        contradiction, penalty, notes = detect_contradiction(scores, self._profile())
        assert contradiction is True
        assert penalty > 0.0
        assert notes

    def test_consistent_scores_no_contradiction(self):
        scores = {name: CategoryScore(15) for name in
                  ("form", "matchup", "statistics", "market_compatibility", "stability")}
        contradiction, penalty, notes = detect_contradiction(scores, self._profile())
        assert contradiction is False
        assert penalty == 0.0
        assert notes == []
