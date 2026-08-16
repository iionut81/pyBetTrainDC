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
            "expected_total_games": 12.5,
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
            "expected_total_games": 11.5,
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
            "expected_total_games": 10.0,
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
            "expected_total_games": 11.0,
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
            "expected_total_games": 10.5,
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
            "p_hold_b": 0.52,  # gap 0.18, min_hold 0.52 -> VETO
            "expected_total_games": 11.0,
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
            # high STATISTICS (drives ranking) but weak MATCHUP/STABILITY diagnostics
            # (large hold gap) -> since 2026-08-16 ranking is p_cal_adj/STATISTICS-only,
            # so this still ranks near the top despite the mixed diagnostics
            "p_hold_b": 0.58,
            "expected_total_games": 13.0,
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
            "expected_total_games": 12.0,
            "surface": "Hard",
            # no optional fields -> lower data_quality -> MEDIUM confidence despite a good score
        },
    ),
    MatchInput(
        match_id="M10",
        market=MARKET_ID,
        sport="tennis",
        competitors=("Rachel Rally", "Sofia Steady"),
        stats={
            "p_hold_a": 0.60,
            "p_hold_b": 0.58,  # low STATISTICS -> ranks near the bottom regardless of diagnostics
            "expected_total_games": 9.5,
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
    print("--- Same 10 matches, minimum_score raised above the max achievable 100 (forces NO BET) ---")
    print()
    strict_profile = dataclasses.replace(TENNIS_SET1_OVER_7_5_PROFILE, minimum_score=100.5)
    strict_result = run_selection(MOCK_MATCHES, strict_profile)
    print(format_report("TENNIS SET 1 OVER 7.5 (strict)", strict_result))


if __name__ == "__main__":
    main()
