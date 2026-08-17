"""Integration tests for selection_engine.engine.run_selection using the
demo's 10 mock TENNIS_SET1_OVER_7_5 matches (see demo_selection_engine.py)."""
from __future__ import annotations

import dataclasses

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
        # M1 (0.94) and M2 (0.925) both clear historical p80 (0.9163,
        # POST_HARD_FILTER/PRE_VETO population) -> BET
        assert self.result.decision == "BET"
        assert len(self.result.top_picks) == 2
        assert self.result.top_picks[0].match_id == "M1"
        assert self.result.top_picks[1].match_id == "M2"

    def test_full_label_spread_across_the_batch(self):
        labels = {r.match_id: r.label for r in self.result.qualified}
        assert labels["M1"] == "TOP_HISTORICAL_QUINTILE"
        assert labels["M2"] == "TOP_HISTORICAL_QUINTILE"
        assert labels["M3"] == "HIGH"
        assert labels["M8"] == "MEDIUM"
        assert labels["M10"] == "LOW"
        assert labels["M9"] == "VERY_LOW"  # fallback signal, below p0

    def test_top_picks_are_bet_eligible_top_quintile(self):
        for pick in self.result.top_picks:
            assert pick.bet_eligible is True
            assert pick.label == "TOP_HISTORICAL_QUINTILE"

    def test_qualified_exceeds_top_picks(self):
        # M3, M8, M9, M10 all pass elimination (qualified) but sit below the
        # historical p80 threshold, so none of them make top_picks.
        assert len(self.result.qualified) > len(self.result.top_picks)
        qualified_ids = [r.match_id for r in self.result.qualified]
        assert "M3" in qualified_ids
        assert "M3" not in [r.match_id for r in self.result.top_picks]

    def test_best_available_below_threshold_is_labeled_not_eligible(self):
        # M3 (p_cal_adj 0.91) is HIGH but explicitly not BET_ELIGIBLE.
        m3 = next(r for r in self.result.qualified if r.match_id == "M3")
        assert m3.label == "HIGH"
        assert m3.bet_eligible is False

    def test_veto_wins_even_with_a_strong_p_cal_adj(self):
        # M7 has p_cal_adj=0.95 (would be TOP_HISTORICAL_QUINTILE) but its
        # hold profile trips the veto regardless.
        assert all(r.match_id != "M7" for r in self.result.qualified)


class TestRunSelectionNoBet:
    def test_no_candidate_above_p80_forces_no_bet(self):
        forced_percentiles = {**TENNIS_SET1_OVER_7_5_PROFILE.historical_percentiles, "p80": 0.99}
        strict_profile = dataclasses.replace(TENNIS_SET1_OVER_7_5_PROFILE, historical_percentiles=forced_percentiles)
        result = run_selection(MOCK_MATCHES, strict_profile)
        assert result.decision == "NO_BET"
        assert result.top_picks == []
        # matches still get evaluated, ranked, and labeled even though none
        # clear the bar -- NO BET must not hide the fact that M1/M2 were
        # still the best available.
        assert len(result.qualified) > 0
        assert result.qualified[0].match_id == "M1"
        assert result.qualified[0].bet_eligible is False
