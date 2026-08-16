"""Tests for selection_engine data_validation + hard_filters."""
from __future__ import annotations

from selection_engine.data_validation import validate
from selection_engine.hard_filters import apply_hard_filters
from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    TENNIS_SET1_OVER_7_5_PROFILE,
)
from selection_engine.types import MatchInput


def _match(**stats) -> MatchInput:
    return MatchInput(
        match_id="X",
        market=MARKET_ID,
        sport="tennis",
        competitors=("A", "B"),
        stats=stats,
    )


class TestValidate:
    def test_missing_required_field_is_insufficient_data(self):
        match = _match(p_hold_a=0.70, expected_total_games=11.0, surface="Hard")
        reason, data_quality = validate(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason == "INSUFFICIENT_DATA"
        assert data_quality == 0.0

    def test_all_required_present_passes(self):
        match = _match(p_hold_a=0.70, p_hold_b=0.68, expected_total_games=11.0, surface="Hard")
        reason, data_quality = validate(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason is None
        assert 0.0 <= data_quality <= 1.0

    def test_data_quality_reflects_optional_fields(self):
        full = _match(
            p_hold_a=0.70,
            p_hold_b=0.68,
            surface="Hard",
            p_cal_adj=0.88,
            expected_total_games=11.0,
            recent_form_variance_a=0.1,
            recent_form_variance_b=0.1,
            tiebreak_rate=0.2,
        )
        partial = _match(p_hold_a=0.70, p_hold_b=0.68, surface="Hard")
        _, dq_full = validate(full, TENNIS_SET1_OVER_7_5_PROFILE)
        _, dq_partial = validate(partial, TENNIS_SET1_OVER_7_5_PROFILE)
        assert dq_full == 1.0
        assert dq_partial == 0.0
        assert dq_full > dq_partial


class TestHardFilters:
    def test_low_expected_games_eliminated(self):
        match = _match(p_hold_a=0.75, p_hold_b=0.74, expected_total_games=7.0, surface="Hard")
        reason = apply_hard_filters(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason == "HARD_FILTER:LOW_EXPECTED_GAMES"

    def test_extreme_hold_gap_eliminated(self):
        match = _match(p_hold_a=0.82, p_hold_b=0.45, expected_total_games=10.5, surface="Hard")
        reason = apply_hard_filters(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason == "HARD_FILTER:EXTREME_HOLD_GAP"

    def test_healthy_match_survives(self):
        match = _match(p_hold_a=0.72, p_hold_b=0.70, expected_total_games=12.5, surface="Hard")
        reason = apply_hard_filters(match, TENNIS_SET1_OVER_7_5_PROFILE)
        assert reason is None
