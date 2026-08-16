"""Tests for selection_engine.output.format_report rendering."""
from __future__ import annotations

from selection_engine.output import format_report
from selection_engine.types import CategoryScore, EngineResult, MatchResult


def _qualified(match_id, competitors, score, strengths=(), risks=()):
    notes = [f"+ {s}" for s in strengths] + [f"- {r}" for r in risks]
    return MatchResult(
        match_id=match_id,
        competitors=competitors,
        status="QUALIFIED",
        category_scores={"form": CategoryScore(15.0, notes)},
        final_score=score,
        confidence="HIGH",
    )


class TestFormatReportBet:
    def test_contains_top_picks_and_decision(self):
        pick1 = _qualified("A", ("Alice", "Bea"), 92.0, strengths=["High combined hold"])
        pick2 = _qualified("B", ("Carol", "Dana"), 88.0, risks=["Moderate volatility"])
        result = EngineResult(
            market_id="TENNIS_SET1_OVER_7_5",
            total_analyzed=10,
            eliminated=[MatchResult("E1", ("X", "Y"), "ELIMINATED", "INSUFFICIENT_DATA")] * 6,
            qualified=[pick1, pick2],
            top_picks=[pick1, pick2],
            decision="BET",
        )
        report = format_report("TENNIS SET 1 OVER 7.5", result)
        assert "MATCHES ANALYZED: 10" in report
        assert "ELIMINATED: 6" in report
        assert "QUALIFIED: 2" in report
        assert "#1 Alice vs Bea" in report
        assert "Rank Score: 92/100" in report
        assert "+ High combined hold" in report
        assert "- Moderate volatility" in report
        assert "TOP PICK: Alice vs Bea" in report
        assert "SECOND PICK: Carol vs Dana" in report


class TestFormatReportNoBet:
    def test_shows_no_bet_when_no_top_picks(self):
        result = EngineResult(
            market_id="TENNIS_SET1_OVER_7_5",
            total_analyzed=5,
            eliminated=[],
            qualified=[_qualified("A", ("Alice", "Bea"), 60.0)],
            top_picks=[],
            decision="NO_BET",
        )
        report = format_report("TENNIS SET 1 OVER 7.5", result)
        assert "FINAL DECISION: NO BET" in report
        assert "TOP PICK" not in report
