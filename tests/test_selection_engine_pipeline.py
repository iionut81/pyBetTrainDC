"""Integration tests for selection_engine.engine.run_selection using the
demo's 10 mock TENNIS_SET1_OVER_7_5 matches (see demo_selection_engine.py)."""
from __future__ import annotations

import dataclasses

import pytest

from demo_selection_engine import MOCK_MATCHES
from selection_engine.engine import run_selection
from selection_engine.markets.tennis_set1_over_7_5 import TENNIS_SET1_OVER_7_5_PROFILE


class TestRunSelectionBet:
    def setup_method(self):
        self.result = run_selection(MOCK_MATCHES, TENNIS_SET1_OVER_7_5_PROFILE)

    def test_total_analyzed(self):
        assert self.result.total_analyzed == 10

    def test_eliminates_bad_data_hard_filters_and_veto(self):
        reasons = {r.match_id: r.elimination_reason for r in self.result.eliminated}
        assert reasons["M4"] == "INSUFFICIENT_DATA"
        assert reasons["M5"] == "HARD_FILTER:LOW_EXPECTED_GAMES"
        assert reasons["M6"] == "HARD_FILTER:EXTREME_HOLD_GAP"
        assert reasons["M7"] == "VETO:CRITICAL_CONTRADICTION"

    def test_decision_is_bet_with_top_two_picks(self):
        # Ranking is STATISTICS-only (p_cal_adj proxy) since 2026-08-16 — M8's
        # weak MATCHUP/STABILITY diagnostics no longer keep it out of #1, it
        # just has the highest expected_total_games of the batch.
        assert self.result.decision == "BET"
        assert len(self.result.top_picks) == 2
        assert self.result.top_picks[0].match_id == "M8"
        assert self.result.top_picks[1].match_id == "M1"

    def test_qualified_exceeds_top_picks(self):
        # M2, M3, M9, M10 all pass elimination (qualified) but sit below
        # minimum_score=80 once ranking is STATISTICS-only, so none of them
        # make top_picks.
        assert len(self.result.qualified) > len(self.result.top_picks)
        qualified_ids = [r.match_id for r in self.result.qualified]
        assert "M2" in qualified_ids
        assert "M2" not in [r.match_id for r in self.result.top_picks]

    def test_contradiction_never_flagged_for_this_market(self):
        # score_fn is set for TENNIS_SET1_OVER_7_5 -> contradiction detection
        # is bypassed entirely (see engine.py), regardless of how mixed the
        # diagnostic categories look (M8, M10 both have a poor MATCHUP score
        # alongside a strong/weak STATISTICS score respectively).
        assert all(not r.contradiction for r in self.result.qualified)


class TestRunSelectionNoBet:
    def test_strict_threshold_forces_no_bet(self):
        # 100.5 is above the max achievable score (STATISTICS caps at 20 -> *5 = 100)
        strict_profile = dataclasses.replace(TENNIS_SET1_OVER_7_5_PROFILE, minimum_score=100.5)
        result = run_selection(MOCK_MATCHES, strict_profile)
        assert result.decision == "NO_BET"
        assert result.top_picks == []
        # matches still get evaluated even though none clear the bar
        assert len(result.qualified) > 0

    def test_allow_no_bet_false_forces_a_pick(self):
        forced_profile = dataclasses.replace(
            TENNIS_SET1_OVER_7_5_PROFILE, minimum_score=100.5, allow_no_bet=False
        )
        result = run_selection(MOCK_MATCHES, forced_profile)
        assert result.decision == "BET"
        assert len(result.top_picks) == 2
