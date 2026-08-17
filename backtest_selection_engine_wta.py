from __future__ import annotations

"""
backtest_selection_engine_wta.py
Backtests selection_engine's TENNIS_SET1_OVER_7_5 market (percentile-based
ranking/eligibility, see markets/tennis_set1_over_7_5.py) against ~17.6K
historical WTA matches (simulations/WTA/backtests/wta_predictions.csv — the
walk-forward output of train_wta.py, where every row carries both the
production model's own calibrated p_cal_adj and the REAL outcome).

Runs the real run_selection() pipeline end to end (not a hand-rolled copy),
so this always reflects the market's actual current configuration.

IMPORTANT: this script measures how the ALREADY-FIXED historical percentile
breakpoints (P_CAL_ADJ_HISTORICAL_PERCENTILES in the market module) perform.
It must NOT tune p80 (or any other breakpoint) to make the reported numbers
look better — p80 is a description of the historical distribution, not a
parameter to optimize against this same dataset. See
feedback_backtest_holdout_before_optimizing in memory.

LEAKAGE CAVEAT (read this before trusting the main numbers below): the
primary sections (B/D/E/F/G) compare every historical match's p_cal_adj
against ONE static distribution computed from the FULL POST_HARD_FILTER /
PRE_VETO population (2017-2026, i.e. including matches chronologically AFTER
the one being classified). That is look-ahead bias for a strict out-of-sample
claim — a 2017 match is being judged against percentiles that include 2026
data it could not have known about. The WALK-FORWARD section further down
recomputes percentiles per calendar year using ONLY prior years' data and
reports hit rates from that instead — treat THAT section, not the static one,
as the honest out-of-sample check. The static section is still useful (it's
literally what today's production run currently uses), just don't call it
out-of-sample.

PROFITABILITY: wta_predictions.csv has no historical odds column, so ROI is
reported as NOT AVAILABLE rather than approximated or invented.

Usage:
    PYTHONIOENCODING=utf-8 python backtest_selection_engine_wta.py
"""

import pandas as pd

from selection_engine.classification import classify, compute_percentiles
from selection_engine.data_validation import validate
from selection_engine.engine import run_selection
from selection_engine.hard_filters import apply_hard_filters
from selection_engine.markets.tennis_set1_over_7_5 import (
    MARKET_ID,
    TENNIS_SET1_OVER_7_5_PROFILE,
)
from selection_engine.types import MatchInput
from wta_recent_form import build_player_index, load_history, recent_form_variance_indexed

PREDICTIONS_CSV = "simulations/WTA/backtests/wta_predictions.csv"
OUT_CSV = "simulations/WTA/backtests/selection_engine_backtest.csv"

TOP_PCTS = [0.01, 0.02, 0.05, 0.10]
LABEL_BANDS = ["VERY_LOW (<20)", "LOW (20-40)", "MEDIUM (40-60)", "HIGH (60-80)", "TOP (>=80)"]


def build_matches(df: pd.DataFrame, player_index) -> list:
    matches = []
    for i, row in df.iterrows():
        stats = {
            "p_hold_a": float(row["p_hold_w"]),
            "p_hold_b": float(row["p_hold_l"]),
            "surface": row["surface"],
            "p_cal_adj": float(row["p_set1_over_7_5"]),
            "tiebreak_rate": float(row["p_tiebreak"]),
        }
        as_of = row["match_date"]
        va = recent_form_variance_indexed(player_index, row["winner_name"], as_of)
        vb = recent_form_variance_indexed(player_index, row["loser_name"], as_of)
        if va is not None:
            stats["recent_form_variance_a"] = va
        if vb is not None:
            stats["recent_form_variance_b"] = vb

        match = MatchInput(
            match_id=f"bt-{i}",
            market=MARKET_ID,
            sport="tennis",
            competitors=(str(row["winner_name"]), str(row["loser_name"])),
            stats=stats,
            meta={"match_date": row["match_date"]},
        )
        matches.append((match, float(row["y_set1_over_7_5"])))
    return matches


def population_post_hard_filter(df: pd.DataFrame, profile) -> pd.DataFrame:
    """DATA VALIDATION -> HARD FILTER -> STOP BEFORE VETO. This is the
    official population for historical_percentiles -- veto is a selection
    filter and must not also define the statistical universe p_cal_adj is
    measured against."""
    rows = []
    for _, row in df.iterrows():
        stats = {"p_hold_a": float(row["p_hold_w"]), "p_hold_b": float(row["p_hold_l"]), "surface": row["surface"]}
        m = MatchInput(match_id="x", market=MARKET_ID, sport="tennis", competitors=("a", "b"), stats=stats)
        reason, _ = validate(m, profile)
        if reason:
            continue
        reason = apply_hard_filters(m, profile)
        if reason:
            continue
        rows.append({"p_cal_adj": float(row["p_set1_over_7_5"]), "match_date": row["match_date"], "y": float(row["y_set1_over_7_5"])})
    return pd.DataFrame(rows)


def run_pipeline(matches: list, profile) -> tuple:
    inputs = [m for m, _y in matches]
    outcomes = {m.match_id: y for m, y in matches}
    dates = {m.match_id: m.meta["match_date"] for m, _y in matches}

    result = run_selection(inputs, profile)

    rows = []
    for r in result.qualified:
        rows.append({
            "match_id": r.match_id, "match_date": dates[r.match_id], "y": outcomes[r.match_id],
            "rank_value": r.rank_value, "historical_percentile": r.historical_percentile,
            "label": r.label, "bet_eligible": r.bet_eligible, "vetoed": False,
        })
    for r in result.eliminated:
        if (r.elimination_reason or "").startswith("VETO:"):
            rows.append({
                "match_id": r.match_id, "match_date": dates[r.match_id], "y": outcomes[r.match_id],
                "rank_value": None, "historical_percentile": None,
                "label": "", "bet_eligible": False, "vetoed": True,
            })
    return pd.DataFrame(rows), result


def label_band(historical_percentile: float) -> str:
    if historical_percentile >= 80:
        return LABEL_BANDS[4]
    if historical_percentile >= 60:
        return LABEL_BANDS[3]
    if historical_percentile >= 40:
        return LABEL_BANDS[2]
    if historical_percentile >= 20:
        return LABEL_BANDS[1]
    return LABEL_BANDS[0]


def print_band_table(kept: pd.DataFrame) -> None:
    print("\nD. HIT RATE BY HISTORICAL PERCENTILE BAND (post-veto candidates, breakpoints from pre-veto population)")
    print(f"{'Band':<16} {'N':>6} {'Hit Rate':>10}")
    bands = kept["historical_percentile"].apply(label_band)
    for label in LABEL_BANDS:
        sub = kept[bands == label]
        hr = f"{sub['y'].mean():.1%}" if len(sub) else "n/a"
        print(f"{label:<16} {len(sub):>6} {hr:>10}")


def print_top_pct_table(kept: pd.DataFrame) -> None:
    print("\nE. TOP-N% BY p_cal_adj (post-veto candidates)")
    print(f"{'Slice':<10} {'N':>6} {'Hit Rate':>10}")
    ranked = kept.sort_values("rank_value", ascending=False)
    n_total = len(ranked)
    print(f"{'All':<10} {n_total:>6} {ranked['y'].mean():>9.1%}")
    for pct in TOP_PCTS:
        n = max(1, int(round(n_total * pct)))
        top = ranked.head(n)
        print(f"{f'Top {pct:.0%}':<10} {n:>6} {top['y'].mean():>9.1%}")


def print_daily_simulation(matches: list, profile) -> None:
    """G (daily part): group historical matches by calendar day and run each
    day through run_selection() independently, exactly like a real
    'Ruleaza WTA NOU' -- how often would this have produced BET vs NO_BET,
    and how many total picks across the whole history?"""
    by_day: dict = {}
    for m, y in matches:
        day = m.meta["match_date"].date()
        by_day.setdefault(day, []).append((m, y))

    n_bet_days = 0
    n_no_bet_days = 0
    n_picks = 0
    n_pick_wins = 0
    for day, day_matches in by_day.items():
        inputs = [m for m, _y in day_matches]
        outcomes = {m.match_id: y for m, y in day_matches}
        result = run_selection(inputs, profile)
        if result.decision == "BET":
            n_bet_days += 1
            for pick in result.top_picks:
                n_picks += 1
                n_pick_wins += int(outcomes[pick.match_id])
        else:
            n_no_bet_days += 1

    print("\nG. DAILY SIMULATION (each historical calendar day run independently, as production would)")
    print(f"  Days simulated:  {len(by_day)}")
    print(f"  BET days:        {n_bet_days}")
    print(f"  NO_BET days:     {n_no_bet_days}")
    print(f"  Total picks made: {n_picks}")
    if n_picks:
        print(f"  Pick hit rate:    {n_pick_wins / n_picks:.1%}")


def print_walk_forward(post_hf: pd.DataFrame, profile) -> None:
    """Expanding, year-by-year percentile recompute using ONLY prior years'
    data -- the honest out-of-sample check (see module docstring's LEAKAGE
    CAVEAT). Coarser than a fully continuous expanding window (year-level,
    not per-match) to keep this computationally cheap, but avoids the
    look-ahead bias in the main static-distribution sections."""
    print("\nWALK-FORWARD (expanding by year, no look-ahead) -- the out-of-sample-honest check")
    df = post_hf.copy()
    df["year"] = df["match_date"].dt.year
    years = sorted(df["year"].unique())

    print(f"{'Year':<8} {'N (this yr)':>12} {'P80 (prior yrs)':>16} {'Hit Rate >=P80':>16} {'Hit Rate <P80':>16}")
    for year in years:
        prior = df[df["year"] < year]
        current = df[df["year"] == year]
        if len(prior) < 200 or len(current) == 0:
            print(f"{year:<8} {len(current):>12} {'insufficient prior data':>16}")
            continue
        prior_p80 = compute_percentiles(prior["p_cal_adj"])["p80"]
        above = current[current["p_cal_adj"] >= prior_p80]
        below = current[current["p_cal_adj"] < prior_p80]
        hr_above = f"{above['y'].mean():.1%}" if len(above) else "n/a"
        hr_below = f"{below['y'].mean():.1%}" if len(below) else "n/a"
        print(f"{year:<8} {len(current):>12} {prior_p80:>16.4f} {hr_above:>16} {hr_below:>16}")


def main() -> None:
    df = pd.read_csv(PREDICTIONS_CSV)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    n_total_historical = len(df)
    df = df.dropna(subset=["p_hold_w", "p_hold_l", "surface", "p_set1_over_7_5", "p_tiebreak", "y_set1_over_7_5", "match_date"])

    print("Building recent-form player index (data/historical/wta_matches_combined.csv) ...")
    history_long = load_history()
    player_index = build_player_index(history_long)

    matches = build_matches(df, player_index)
    scored, engine_result = run_pipeline(matches, TENNIS_SET1_OVER_7_5_PROFILE)

    post_hf = population_post_hard_filter(df, TENNIS_SET1_OVER_7_5_PROFILE)
    n_post_veto = int((~scored["vetoed"]).sum())

    print("\nA. POPULATION")
    print(f"  Total historical rows (raw CSV):              {n_total_historical}")
    print(f"  Usable (complete required fields):             {len(df)}")
    print(f"  POST_HARD_FILTER / PRE_VETO (percentile pop.): {len(post_hf)}")
    print(f"  POST_VETO (actual candidates in production):   {n_post_veto}")

    live_percentiles = compute_percentiles(post_hf["p_cal_adj"])
    stored_percentiles = TENNIS_SET1_OVER_7_5_PROFILE.historical_percentiles
    print("\nB. DISTRIBUTION (POST_HARD_FILTER / PRE_VETO p_cal_adj, recomputed live from current data)")
    for key in ("p0", "p20", "p40", "p60", "p80", "p90", "p95", "p100"):
        match_flag = "" if abs(live_percentiles[key] - stored_percentiles.get(key, float("nan"))) < 1e-4 else "  <-- differs from stored constant, consider recomputing"
        print(f"  {key:<5} = {live_percentiles[key]:.6f}   (stored: {stored_percentiles.get(key, float('nan')):.6f}){match_flag}")

    print("\nC. HISTORICAL PERCENTILE ROUND-TRIP VALIDATION")
    for key, expected in (("p0", 0.0), ("p20", 20.0), ("p40", 40.0), ("p60", 60.0), ("p80", 80.0), ("p100", 100.0)):
        pct, _, _ = classify(stored_percentiles[key], stored_percentiles)
        ok = "OK" if abs(pct - expected) < 0.5 else "MISMATCH"
        print(f"  classify({key}={stored_percentiles[key]:.6f}) -> historical_percentile={pct:.2f} (expected ~{expected:.0f})  [{ok}]")

    kept = scored[~scored["vetoed"]]
    print_band_table(kept)
    print_top_pct_table(kept)

    print("\nF. VETO")
    n_vetoed = int(scored["vetoed"].sum())
    if n_vetoed:
        print(f"  Vetoed:     n={n_vetoed:5d}  hit rate = {scored[scored['vetoed']]['y'].mean():.1%}")
    else:
        print("  Vetoed: none triggered")
    print(f"  Not vetoed: n={len(kept):5d}  hit rate = {kept['y'].mean():.1%}")

    n_eligible = int(kept["bet_eligible"].sum())
    print("\nG. SELECTION COUNT")
    print(f"  BET_ELIGIBLE matches (pooled, not day-grouped): {n_eligible} / {len(kept)} ({n_eligible/len(kept):.1%})")
    print_daily_simulation(matches, TENNIS_SET1_OVER_7_5_PROFILE)

    print_walk_forward(post_hf, TENNIS_SET1_OVER_7_5_PROFILE)

    print("\nPROFITABILITY: ROI = NOT AVAILABLE (no historical odds column in wta_predictions.csv)")

    scored.to_csv(OUT_CSV, index=False)
    print(f"\nSaved raw per-match scores -> {OUT_CSV}")


if __name__ == "__main__":
    main()
