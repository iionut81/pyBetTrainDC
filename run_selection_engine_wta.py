from __future__ import annotations

"""
run_selection_engine_wta.py
Adapter: reads today's TENNIS_SET1_OVER_7_5 candidates produced by the
production WTA daily pipeline (simulations/WTA/evaluations/1.2_WTA_Set1_Over_7_5.csv,
written by run_wta_daily.py) and runs them through the new generic selection
engine (selection_engine/). Read-only against the CSV — does not touch the
production pipeline itself.

Usage:
    PYTHONIOENCODING=utf-8 python run_selection_engine_wta.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

from selection_engine.engine import run_selection
from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    TENNIS_SET1_OVER_7_5_PROFILE,
)
from selection_engine.output import format_report
from selection_engine.types import MatchInput
from wta_recent_form import load_history, recent_form_variance

CSV_PATH = Path(__file__).resolve().parent / "simulations" / "WTA" / "evaluations" / "1.2_WTA_Set1_Over_7_5.csv"


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def load_today_matches(
    csv_path: Path, today: date, history_long: pd.DataFrame | None = None
) -> list[MatchInput]:
    """Filter to today's confirmed-time matches.

    match_date keeps each match's own local offset (e.g. "2026-08-16T11:00-04:00")
    so the calendar date is read straight from the string, NOT via a UTC
    conversion — converting a late-day local time to UTC can roll it into the
    next calendar day and wrongly drop a real match. run_wta_daily.py stamps
    "T23:59" as its own sentinel for "no confirmed time yet" (projected/
    unconfirmed fixture) — per project rule those are excluded here too.
    """
    df = pd.read_csv(csv_path)
    today_str = today.isoformat()
    as_of = pd.Timestamp(today)

    match_date = df["match_date"].astype(str)
    is_today = match_date.str.startswith(today_str)
    has_confirmed_time = ~match_date.str.contains("T23:59")
    today_df = df[is_today & has_confirmed_time]

    skipped_unconfirmed = int((is_today & ~has_confirmed_time).sum())
    if skipped_unconfirmed:
        print(f"Skipping {skipped_unconfirmed} match(es) with no confirmed time (T23:59 placeholder).")

    matches: list[MatchInput] = []
    for i, row in today_df.iterrows():
        stats = {
            "p_hold_a": _to_float(row.get("p_hold_a")),
            "p_hold_b": _to_float(row.get("p_hold_b")),
            "surface": row.get("surface"),
            "p_cal_adj": _to_float(row.get("p_cal_adj")),
            "tiebreak_rate": _to_float(row.get("tb_p_cal")),
            # NOTE: the CSV's "expected_games" is the whole-MATCH expected total games
            # (wta_markov.simulate_match), not set-1 games — intentionally not mapped
            # here since it would silently saturate the STATISTICS category. p_cal_adj
            # (the pipeline's own calibrated Set1 O7.5 probability) is the real signal.
        }
        if history_long is not None:
            stats["recent_form_variance_a"] = recent_form_variance(history_long, str(row["player_a"]), as_of)
            stats["recent_form_variance_b"] = recent_form_variance(history_long, str(row["player_b"]), as_of)
        stats = {k: v for k, v in stats.items() if v is not None and v == v}  # drop NaN/None

        matches.append(
            MatchInput(
                match_id=f"wta-{i}",
                market=MARKET_ID,
                sport="tennis",
                competitors=(str(row["player_a"]), str(row["player_b"])),
                stats=stats,
                meta={
                    "tournament": row.get("tournament"),
                    "level": row.get("level"),
                    "round": row.get("round"),
                    "match_date": row.get("match_date"),
                },
            )
        )
    return matches


def main() -> None:
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH} — run run_wta_daily.py first.")
        sys.exit(1)

    today = date.today()
    print("Loading historical WTA match data for recent-form signal ...")
    history_long = load_history()
    matches = load_today_matches(CSV_PATH, today, history_long)

    if not matches:
        print(f"No matches found for {today.isoformat()} in {CSV_PATH.name}.")
        print("Check that run_wta_daily.py has been run today and match_date values are current.")
        sys.exit(0)

    result = run_selection(matches, TENNIS_SET1_OVER_7_5_PROFILE)
    print(format_report(f"TENNIS SET 1 OVER 7.5 — {today.isoformat()}", result))


if __name__ == "__main__":
    main()
