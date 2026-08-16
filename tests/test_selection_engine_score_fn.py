"""Tests for MarketProfile.score_fn — the opt-in override that lets a market
drive ranking off a single trusted category instead of summing all 5 (with
the contradiction check bypassed when it's set)."""
from __future__ import annotations

import pytest

from selection_engine.engine import run_selection
from selection_engine.markets.tennis_set1_over_7_5 import score_from_p_cal_adj
from selection_engine.types import CategoryScore, MarketProfile, MatchInput


def _match(match_id="X") -> MatchInput:
    return MatchInput(match_id=match_id, market="TEST", sport="tennis", competitors=("A", "B"), stats={})


def _always(value):
    return lambda match: CategoryScore(value=value)


class TestScoreFromPCalAdj:
    def test_uses_statistics_category_rescaled_to_100(self):
        scores = {
            "form": CategoryScore(2.0),
            "matchup": CategoryScore(2.0),
            "statistics": CategoryScore(18.0),
            "market_compatibility": CategoryScore(2.0),
            "stability": CategoryScore(2.0),
        }
        assert score_from_p_cal_adj(scores) == pytest.approx(90.0)

    def test_ignores_other_categories(self):
        low_others = {
            "form": CategoryScore(0.0),
            "matchup": CategoryScore(0.0),
            "statistics": CategoryScore(16.0),
            "market_compatibility": CategoryScore(0.0),
            "stability": CategoryScore(0.0),
        }
        high_others = {
            "form": CategoryScore(20.0),
            "matchup": CategoryScore(20.0),
            "statistics": CategoryScore(16.0),
            "market_compatibility": CategoryScore(20.0),
            "stability": CategoryScore(20.0),
        }
        assert score_from_p_cal_adj(low_others) == score_from_p_cal_adj(high_others) == pytest.approx(80.0)


class TestEngineScoreFnHook:
    def _profile(self, score_fn=None) -> MarketProfile:
        return MarketProfile(
            market_id="TEST",
            sport="tennis",
            minimum_score=0.0,
            top_n=1,
            category_scorers={
                "form": _always(18.0),
                "matchup": _always(6.0),
                "statistics": _always(18.0),
                "market_compatibility": _always(18.0),
                "stability": _always(6.0),
            },
            score_fn=score_fn,
        )

    def test_default_sums_all_categories_and_applies_contradiction(self):
        # 18+6+18+18+6 = 66; high(18)+low(6) co-occur -> contradiction -> -8
        profile = self._profile(score_fn=None)
        result = run_selection([_match()], profile)
        match = result.qualified[0]
        assert match.contradiction is True
        assert match.final_score == pytest.approx(66.0 - profile.contradiction_penalty)

    def test_score_fn_bypasses_sum_and_contradiction(self):
        profile = self._profile(score_fn=lambda scores: scores["statistics"].value * 5.0)
        result = run_selection([_match()], profile)
        match = result.qualified[0]
        assert match.contradiction is False
        assert match.contradiction_penalty == 0.0
        assert match.final_score == pytest.approx(90.0)
        # other categories still attached as diagnostics
        assert match.category_scores["matchup"].value == pytest.approx(6.0)
