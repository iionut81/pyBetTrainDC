"""Tests for selection_engine veto, ranking and confidence."""
from __future__ import annotations

from selection_engine.confidence import compute_confidence
from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    TENNIS_SET1_OVER_7_5_PROFILE,
)
from selection_engine.ranking import rank_matches
from selection_engine.types import CategoryScore, MatchInput
from selection_engine.veto import apply_vetoes


def _match(**stats) -> MatchInput:
    return MatchInput(
        match_id="X",
        market=MARKET_ID,
        sport="tennis",
        competitors=("A", "B"),
        stats=stats,
    )


def _result(match_id, final_score, stability=15.0, confidence="MEDIUM",
            contradiction=False, data_quality=1.0):
    from selection_engine.types import MatchResult

    return MatchResult(
        match_id=match_id,
        competitors=("A", "B"),
        status="QUALIFIED",
        category_scores={"stability": CategoryScore(stability)},
        final_score=final_score,
        contradiction=contradiction,
        confidence=confidence,
        data_quality=data_quality,
    )


class TestApplyVetoes:
    def test_blowout_risk_vetoed(self):
        match = _match(p_hold_a=0.70, p_hold_b=0.52, expected_total_games=11.0, surface="Grass")
        reason = apply_vetoes(match, {}, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason == "VETO:CRITICAL_CONTRADICTION"

    def test_healthy_match_no_veto(self):
        match = _match(p_hold_a=0.70, p_hold_b=0.68, expected_total_games=11.0, surface="Hard")
        reason = apply_vetoes(match, {}, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason is None


class TestRankMatches:
    def test_sorts_by_final_score_desc(self):
        results = [_result("low", 60.0), _result("high", 90.0), _result("mid", 75.0)]
        ranked = rank_matches(results)
        assert [r.match_id for r in ranked] == ["high", "mid", "low"]

    def test_tiebreak_by_stability_then_confidence(self):
        a = _result("a", 80.0, stability=18.0, confidence="HIGH")
        b = _result("b", 80.0, stability=10.0, confidence="HIGH")
        c = _result("c", 80.0, stability=18.0, confidence="LOW")
        ranked = rank_matches([b, c, a])
        assert ranked[0].match_id == "a"  # highest stability wins the tie
        assert ranked[1].match_id == "c"  # same stability as a, lower confidence

    def test_contradiction_pushes_match_down(self):
        clean = _result("clean", 80.0, stability=15.0, confidence="HIGH", contradiction=False)
        flagged = _result("flagged", 80.0, stability=15.0, confidence="HIGH", contradiction=True)
        ranked = rank_matches([flagged, clean])
        assert ranked[0].match_id == "clean"


class TestComputeConfidence:
    def test_strong_agreement_high_data_quality_is_high(self):
        scores = {name: CategoryScore(18) for name in
                  ("form", "matchup", "statistics", "market_compatibility", "stability")}
        confidence = compute_confidence(scores, data_quality=1.0, contradiction=False)
        assert confidence == "HIGH"

    def test_contradiction_lowers_confidence(self):
        scores = {name: CategoryScore(15) for name in
                  ("form", "matchup", "statistics", "market_compatibility", "stability")}
        with_contradiction = compute_confidence(scores, data_quality=1.0, contradiction=True)
        without = compute_confidence(scores, data_quality=1.0, contradiction=False)
        assert without == "HIGH"
        assert with_contradiction == "MEDIUM"

    def test_poor_data_quality_and_spread_is_low(self):
        scores = {
            "form": CategoryScore(2),
            "matchup": CategoryScore(3),
            "statistics": CategoryScore(1),
            "market_compatibility": CategoryScore(2),
            "stability": CategoryScore(1),
        }
        confidence = compute_confidence(scores, data_quality=0.0, contradiction=True)
        assert confidence == "LOW"
