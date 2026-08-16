from __future__ import annotations

"""
backtest_selection_engine_wta.py
Backtests selection_engine's TENNIS_SET1_OVER_7_5 market against ~17.6K
historical WTA matches (simulations/WTA/backtests/wta_predictions.csv — the
walk-forward output of train_wta.py, where every row carries both the
production model's own calibrated prediction and the REAL outcome).

Read-only diagnostic — does not change selection_engine. Runs the real
run_selection() pipeline (same code path as production), so this always
reflects the market's actual current configuration rather than a hand-rolled
copy of the pipeline stages.

2026-08-16 finding (see markets/tennis_set1_over_7_5.py docstring for detail):
p_cal_adj alone out-ranked the old 5-category composite score, and the
surface bonus tested as noise — both were acted on: ranking is now
STATISTICS-only (score_fn), surface bonus is gone. This script now focuses on
the next question: how well does the engine reduce a big pool of matches down
to the best 1-2%? (Top-N% hit rate, requested 2026-08-16.) The 5 diagnostic
categories' correlations and VETO's impact are kept as an ongoing health
check, not because they're still in question.

KNOWN GAP: this only measures hit rate, not profitability — wta_predictions.csv
has no historical odds column, so ROI/yield can't be computed here. Hit rate
alone does not imply value at any given price; that needs a separate pass
once historical odds are available.

Usage:
    PYTHONIOENCODING=utf-8 python backtest_selection_engine_wta.py
"""

import numpy as np
import pandas as pd

from selection_engine.data_validation import validate
from selection_engine.hard_filters import apply_hard_filters
from selection_engine.markets.tennis_set1_over_7_5 import (
    TENNIS_SET1_OVER_7_5_PROFILE,
    veto_blowout_risk,
)
from selection_engine.types import MatchInput
from wta_recent_form import build_player_index, load_history, recent_form_variance_indexed

PREDICTIONS_CSV = "simulations/WTA/backtests/wta_predictions.csv"
OUT_CSV = "simulations/WTA/backtests/selection_engine_backtest.csv"

# (lo, hi, label) — half-open [lo, hi), matching the production minimum_score=80 threshold
SCORE_BUCKETS = [
    (-np.inf, 80, "<80"),
    (80, 83, "80-82"),
    (83, 86, "83-85"),
    (86, 89, "86-88"),
    (89, 92, "89-91"),
    (92, 100.0001, "92-100"),
]

TOP_PCTS = [0.01, 0.02, 0.05, 0.10]


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
            market=TENNIS_SET1_OVER_7_5_PROFILE.market_id,
            sport="tennis",
            competitors=(str(row["winner_name"]), str(row["loser_name"])),
            stats=stats,
        )
        matches.append((match, float(row["y_set1_over_7_5"])))
    return matches


def score_all(matches: list, profile) -> tuple[pd.DataFrame, int, int]:
    """Runs each match through the real pipeline stages (validate -> hard
    filters -> score -> veto), the same code selection_engine.engine.run_selection
    uses, just keeping the per-match diagnostics for analysis instead of
    building a ranked EngineResult for a single day's candidate pool."""
    rows = []
    n_insufficient = 0
    n_hard_filtered = 0

    for match, y in matches:
        reason, _data_quality = validate(match, profile)
        if reason is not None:
            n_insufficient += 1
            continue
        reason = apply_hard_filters(match, profile)
        if reason is not None:
            n_hard_filtered += 1
            continue

        category_scores = {name: scorer(match) for name, scorer in profile.category_scorers.items()}
        final_score = max(0.0, profile.score_fn(category_scores)) if profile.score_fn else sum(
            cs.value for cs in category_scores.values()
        )
        vetoed = veto_blowout_risk(match, category_scores) is not None

        rows.append({
            "y": y,
            "final_score": final_score,
            "p_cal_adj": match.stats["p_cal_adj"],
            "vetoed": vetoed,
            "form": category_scores["form"].value,
            "matchup": category_scores["matchup"].value,
            "statistics": category_scores["statistics"].value,
            "market_compatibility": category_scores["market_compatibility"].value,
            "stability": category_scores["stability"].value,
        })

    return pd.DataFrame(rows), n_insufficient, n_hard_filtered


def bucket_hit_rates(df: pd.DataFrame, score_col: str) -> list:
    out = []
    for lo, hi, label in SCORE_BUCKETS:
        sub = df[(df[score_col] >= lo) & (df[score_col] < hi)]
        out.append((label, len(sub), sub["y"].mean() if len(sub) else None))
    return out


def print_bucket_table(title: str, buckets: list) -> None:
    print(f"\n{title}")
    print(f"{'Bucket':<10} {'N':>6} {'Hit Rate':>10}")
    for label, n, hr in buckets:
        print(f"{label:<10} {n:>6} {(f'{hr:.1%}' if hr is not None else 'n/a'):>10}")


def print_top_pct_table(kept: pd.DataFrame) -> None:
    print("\nTOP-N% BY SCORE (== p_cal_adj rank, veto-excluded pool) vs actual hit rate")
    print(f"{'Slice':<10} {'N':>6} {'Hit Rate':>10}")
    ranked = kept.sort_values("final_score", ascending=False)
    n_total = len(ranked)
    print(f"{'All':<10} {n_total:>6} {ranked['y'].mean():>9.1%}")
    for pct in TOP_PCTS:
        n = max(1, int(round(n_total * pct)))
        top = ranked.head(n)
        print(f"{f'Top {pct:.0%}':<10} {n:>6} {top['y'].mean():>9.1%}")


def main() -> None:
    df = pd.read_csv(PREDICTIONS_CSV)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["p_hold_w", "p_hold_l", "surface", "p_set1_over_7_5", "p_tiebreak", "y_set1_over_7_5", "match_date"])
    print(f"Loaded {before} historical rows, {len(df)} usable (have ground truth + required fields).")

    print("Building recent-form player index (data/historical/wta_matches_combined.csv) ...")
    history_long = load_history()
    player_index = build_player_index(history_long)

    matches = build_matches(df, player_index)
    scored, n_insufficient, n_hard_filtered = score_all(matches, TENNIS_SET1_OVER_7_5_PROFILE)
    n_vetoed = int(scored["vetoed"].sum())
    print(f"Eliminated: {n_insufficient} INSUFFICIENT_DATA, {n_hard_filtered} HARD_FILTER.")
    print(f"Scored: {len(scored)} matches ({n_vetoed} flagged VETO).")

    kept = scored[~scored["vetoed"]]

    print_bucket_table("ENGINE SCORE (p_cal_adj-driven) vs actual Set1 Over 7.5 (veto excluded)", bucket_hit_rates(kept, "final_score"))
    print_top_pct_table(kept)

    print("\nDIAGNOSTIC categories' correlation with actual outcome (Pearson r) — not used for ranking:")
    for cat in ("form", "matchup", "statistics", "market_compatibility", "stability"):
        r = kept[cat].corr(kept["y"])
        print(f"  {cat:<22} r = {r:+.3f}")

    print("\nVETO impact:")
    if n_vetoed:
        print(f"  Vetoed:     n={n_vetoed:5d}  hit rate = {scored[scored['vetoed']]['y'].mean():.1%}")
    else:
        print("  Vetoed: none triggered")
    print(f"  Not vetoed: n={len(kept):5d}  hit rate = {kept['y'].mean():.1%}")

    scored.to_csv(OUT_CSV, index=False)
    print(f"\nSaved raw per-match scores -> {OUT_CSV}")


if __name__ == "__main__":
    main()
