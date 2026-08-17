"""Tests for selection_engine.output.format_report rendering."""
from __future__ import annotations

from selection_engine.output import format_report
from selection_engine.types import EngineResult, MatchResult


def _candidate(match_id, competitors, rank_value, label, bet_eligible, historical_percentile=75.0):
    return MatchResult(
        match_id=match_id,
        competitors=competitors,
        status="QUALIFIED",
        rank_value=rank_value,
        historical_percentile=historical_percentile,
        label=label,
        bet_eligible=bet_eligible,
    )


class TestFormatReportBet:
    def test_contains_top_picks_and_decision(self):
        pick1 = _candidate("A", ("Alice", "Bea"), 0.94, "TOP_HISTORICAL_QUINTILE", True)
        pick2 = _candidate("B", ("Carol", "Dana"), 0.925, "TOP_HISTORICAL_QUINTILE", True)
        result = EngineResult(
            market_id="TENNIS_SET1_OVER_7_5",
            total_analyzed=10,
            eliminated=[MatchResult("E1", ("X", "Y"), "ELIMINATED", "INSUFFICIENT_DATA")] * 6,
            qualified=[pick1, pick2],
            top_picks=[pick1, pick2],
            decision="BET",
            historical_p80=0.9196,
        )
        report = format_report("TENNIS SET 1 OVER 7.5", result)
        assert "MATCHES ANALYZED: 10" in report
        assert "ELIMINATED: 6" in report
        assert "ELIMINATED BY VETO: 0" in report
        assert "QUALIFIED: 2" in report
        assert "BET ELIGIBLE: 2" in report
        assert "HISTORICAL P80: 91.96%" in report
        assert "#1 Alice vs Bea" in report
        assert "p_cal_adj: 94.00%" in report
        assert "label: TOP_HISTORICAL_QUINTILE" in report
        assert "bet_eligible: YES" in report
        assert "FINAL DECISION: BET" in report
        assert "TOP PICK: Alice vs Bea" in report
        assert "SECOND PICK: Carol vs Dana" in report


class TestFormatReportNoBet:
    def test_still_shows_best_candidates_with_bet_eligible_no(self):
        best = _candidate("A", ("Parry", "Boisson"), 0.8767, "HIGH", False, historical_percentile=76.4)
        second = _candidate("B", ("Frech", "Rybakina"), 0.8671, "HIGH", False)
        result = EngineResult(
            market_id="TENNIS_SET1_OVER_7_5",
            total_analyzed=5,
            eliminated=[],
            qualified=[best, second],
            top_picks=[],
            decision="NO_BET",
            historical_p80=0.92,
        )
        report = format_report("TENNIS SET 1 OVER 7.5", result)
        # Never say "Parry-Boisson is a bad match" -- best-available context
        # must stay visible even though nothing was BET_ELIGIBLE.
        assert "ELIMINATED BY VETO: 0" in report
        assert "BET ELIGIBLE: 0" in report
        assert "#1 Parry vs Boisson" in report
        assert "p_cal_adj: 87.67%" in report
        assert "bet_eligible: NO" in report
        assert "FINAL DECISION: NO BET" in report
        assert "REASON: Parry vs Boisson is the best available candidate" in report
        assert "TOP PICK" not in report

    def test_no_qualified_candidates_at_all(self):
        result = EngineResult(
            market_id="TENNIS_SET1_OVER_7_5",
            total_analyzed=3,
            eliminated=[MatchResult("E1", ("X", "Y"), "ELIMINATED", "INSUFFICIENT_DATA")] * 3,
            qualified=[],
            top_picks=[],
            decision="NO_BET",
            historical_p80=0.92,
        )
        report = format_report("TENNIS SET 1 OVER 7.5", result)
        assert "(no qualified candidates)" in report
        assert "FINAL DECISION: NO BET" in report
