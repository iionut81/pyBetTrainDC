from __future__ import annotations

"""
demo_selection_engine.py
Runs the selection engine over 10 mock TENNIS_SET1_OVER_7_5 matches and prints
the formatted report. Mock data only — no external API calls.

Usage:
    python demo_selection_engine.py
"""

import dataclasses

from selection_engine.engine import run_selection
from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    TENNIS_SET1_OVER_7_5_PROFILE,
)
from selection_engine.output import format_report
from selection_engine.types import MatchInput

MOCK_MATCHES = [
    MatchInput(
        match_id="M1",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Alice Model", "Bea Baseline"),
        stats={
            "p_hold_a": 0.72,
            "p_hold_b": 0.70,
            "p_cal_adj": 0.94,  # >= p80 (0.9163) -> TOP_HISTORICAL_QUINTILE, bet_eligible
            "surface": "Hard",
            "recent_form_variance_a": 0.10,
            "recent_form_variance_b": 0.12,
            "tiebreak_rate": 0.30,
        },
    ),
    MatchInput(
        match_id="M2",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Carol Contender", "Dana Underdog"),
        stats={
            "p_hold_a": 0.68,
            "p_hold_b": 0.65,
            "p_cal_adj": 0.925,  # also >= p80 -> second BET_ELIGIBLE pick
            "surface": "Grass",
            "recent_form_variance_a": 0.15,
            "recent_form_variance_b": 0.18,
            "tiebreak_rate": 0.20,
        },
    ),
    MatchInput(
        match_id="M3",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Ellen Edge", "Fiona Flux"),
        stats={
            "p_hold_a": 0.66,
            "p_hold_b": 0.63,
            "p_cal_adj": 0.91,  # p60-p80 -> HIGH, but not bet_eligible
            "surface": "Hard",
            "recent_form_variance_a": 0.30,
            "recent_form_variance_b": 0.35,
            "tiebreak_rate": 0.12,
        },
    ),
    MatchInput(
        match_id="M4",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Grace Gap", "Hana Hidden"),
        stats={
            "p_hold_a": 0.70,
            # p_hold_b missing -> INSUFFICIENT_DATA
            "p_cal_adj": 0.93,
            "surface": "Clay",
        },
    ),
    MatchInput(
        match_id="M5",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Ivy Ice", "Jill Jolt"),
        stats={
            "p_hold_a": 0.75,
            "p_hold_b": 0.74,
            "expected_total_games": 7.0,  # too low -> HARD_FILTER
            "surface": "Hard",
        },
    ),
    MatchInput(
        match_id="M6",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Kara Krush", "Lily Lowrank"),
        stats={
            "p_hold_a": 0.82,
            "p_hold_b": 0.45,  # gap 0.37 -> HARD_FILTER
            "p_cal_adj": 0.90,
            "surface": "Hard",
        },
    ),
    MatchInput(
        match_id="M7",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Mona Mid", "Nora Novice"),
        stats={
            "p_hold_a": 0.70,
            "p_hold_b": 0.52,  # gap 0.18, min_hold 0.52 -> VETO, regardless of p_cal_adj below
            "p_cal_adj": 0.95,
            "surface": "Grass",
            "recent_form_variance_a": 0.05,
            "recent_form_variance_b": 0.05,
            "tiebreak_rate": 0.30,
        },
    ),
    MatchInput(
        match_id="M8",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Olga Outlier", "Petra Paradox"),
        stats={
            "p_hold_a": 0.80,
            # mixed diagnostics (large hold gap -> weak MATCHUP/STABILITY) but
            # those no longer feed ranking -> MEDIUM purely on p_cal_adj (p40-p60)
            "p_hold_b": 0.58,
            "p_cal_adj": 0.895,
            "surface": "Hard",
            "recent_form_variance_a": 0.05,
            "recent_form_variance_b": 0.05,
            "tiebreak_rate": 0.05,
        },
    ),
    MatchInput(
        match_id="M9",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Priya Points", "Quyen Quick"),
        stats={
            "p_hold_a": 0.71,
            "p_hold_b": 0.69,
            # no p_cal_adj -> falls back to expected_total_games pseudo-probability
            # (illustrates the mock-data fallback path; not a real probability)
            "expected_total_games": 12.0,
            "surface": "Hard",
        },
    ),
    MatchInput(
        match_id="M10",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Rachel Rally", "Sofia Steady"),
        stats={
            "p_hold_a": 0.60,
            "p_hold_b": 0.58,
            "p_cal_adj": 0.87,  # between p20 and p40 -> LOW
            "surface": "Clay",
            "recent_form_variance_a": 0.35,
            "recent_form_variance_b": 0.30,
            "tiebreak_rate": 0.08,
        },
    ),
]


def main() -> None:
    result = run_selection(MOCK_MATCHES, TENNIS_SET1_OVER_7_5_PROFILE)
    print(format_report("TENNIS SET 1 OVER 7.5", result))

    print()
    print("--- Same 10 matches, historical p80 pushed above every candidate's p_cal_adj (forces NO BET) ---")
    print()
    forced_percentiles = {**TENNIS_SET1_OVER_7_5_PROFILE.historical_percentiles, "p80": 0.99}
    strict_profile = dataclasses.replace(TENNIS_SET1_OVER_7_5_PROFILE, historical_percentiles=forced_percentiles)
    strict_result = run_selection(MOCK_MATCHES, strict_profile)
    print(format_report("TENNIS SET 1 OVER 7.5 (strict)", strict_result))


if __name__ == "__main__":
    main()
