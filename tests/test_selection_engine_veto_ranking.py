"""Tests for selection_engine veto and ranking."""
from __future__ import annotations

from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    TENNIS_SET1_OVER_7_5_PROFILE,
)
from selection_engine.ranking import rank_matches
from selection_engine.types import CategoryScore, MatchInput, MatchResult
from selection_engine.veto import apply_vetoes


def _match(**stats) -> MatchInput:
    return MatchInput(
        match_id="X",
        market=MARKET_ID,
        sport="tennis",
        competitors=("A", "B"),
        stats=stats,
    )


def _result(match_id, rank_value, data_quality=1.0, stability=None):
    category_scores = {"stability": CategoryScore(stability)} if stability is not None else {}
    return MatchResult(
        match_id=match_id,
        competitors=("A", "B"),
        status="QUALIFIED",
        rank_value=rank_value,
        data_quality=data_quality,
        category_scores=category_scores,
    )


class TestApplyVetoes:
    def test_blowout_risk_vetoed(self):
        match = _match(p_hold_a=0.70, p_hold_b=0.52, surface="Grass")
        reason = apply_vetoes(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason == "VETO:CRITICAL_CONTRADICTION"

    def test_healthy_match_no_veto(self):
        match = _match(p_hold_a=0.70, p_hold_b=0.68, surface="Hard")
        reason = apply_vetoes(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason is None


class TestRankMatches:
    def test_sorts_by_rank_value_desc(self):
        results = [_result("low", 0.80), _result("high", 0.95), _result("mid", 0.88)]
        ranked = rank_matches(results)
        assert [r.match_id for r in ranked] == ["high", "mid", "low"]

    def test_tiebreak_by_data_quality(self):
        a = _result("a", 0.90, data_quality=1.0)
        b = _result("b", 0.90, data_quality=0.5)
        ranked = rank_matches([b, a])
        assert ranked[0].match_id == "a"

    def test_missing_rank_value_sorts_last(self):
        has_signal = _result("has_signal", 0.80)
        no_signal = _result("no_signal", None)
        ranked = rank_matches([no_signal, has_signal])
        assert ranked[0].match_id == "has_signal"
        assert ranked[1].match_id == "no_signal"

    def test_tiebreak_by_stability_after_data_quality(self):
        a = _result("a", 0.90, data_quality=1.0, stability=18.0)
        b = _result("b", 0.90, data_quality=1.0, stability=10.0)
        ranked = rank_matches([b, a])
        assert ranked[0].match_id == "a"

    def test_tiebreak_by_deterministic_match_id_as_last_resort(self):
        a = _result("aaa", 0.90, data_quality=1.0, stability=15.0)
        b = _result("zzz", 0.90, data_quality=1.0, stability=15.0)
        ranked = rank_matches([b, a])
        assert ranked[0].match_id == "aaa"
