"""
Backtest Under 4.5 goals model on historical matches.

Runs Dixon-Coles predictions + calibration for every historical match,
then evaluates hit rate across probability bands to find optimal thresholds.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_team_ratings
from dixon_coles import expected_goals, resolve_team_strength, score_matrix
from fhg_calibration import apply_calibration, calibration_from_row


def _ou_probs(mat: np.ndarray) -> dict[str, float]:
    n = mat.shape[0]
    idx = np.arange(n)
    ig, jg = np.meshgrid(idx, idx, indexing="ij")
    total = ig + jg
    return {
        "over_2_5": float(mat[total >= 3].sum()),
        "under_3_5": float(mat[total <= 3].sum()),
        "under_4_5": float(mat[total <= 4].sum()),
        "btts": float(mat[1:, 1:].sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest Under 4.5 goals model.")
    p.add_argument("--history-csv", default="data/historical/historical_matches_transfermarkt.csv")
    p.add_argument("--ratings-pkl", default="data/historical/team_ratings.pkl")
    p.add_argument("--calibration-csv", default="simulations/Goals/data/goals_calibration.csv")
    p.add_argument("--min-date", default="2024-07-01", help="Start date for backtest window")
    p.add_argument("--max-date", default="2026-03-16", help="End date for backtest window")
    p.add_argument("--output-csv", default="simulations/backtests/backtest_under_4_5.csv")
    args = p.parse_args()

    # Load ratings
    ratings = load_team_ratings(args.ratings_pkl)

    # Load calibration
    cal_path = Path(args.calibration_csv)
    cal_df = (
        pd.read_csv(cal_path)
        if cal_path.exists()
        else pd.DataFrame(columns=["league", "market", "method", "a", "b", "temperature"])
    )
    cal_map: dict[tuple[str, str], dict] = {}
    for _, r in cal_df.iterrows():
        lg = str(r["league"]).strip().upper()
        mk = str(r["market"]).strip()
        cal_map[(lg, mk)] = calibration_from_row(dict(r))
    global_cal = cal_map.get(("__GLOBAL__", "under_4_5"), {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0})

    # Load historical matches
    hist = pd.read_csv(args.history_csv)
    hist["match_date"] = pd.to_datetime(hist["match_date"])
    hist = hist[(hist["match_date"] >= args.min_date) & (hist["match_date"] <= args.max_date)].copy()
    hist["total_goals"] = hist["home_goals"] + hist["away_goals"]
    hist["actual_under_4_5"] = (hist["total_goals"] <= 4).astype(int)
    print(f"Historical matches in window: {len(hist)}")

    rows: list[dict] = []
    skipped = 0
    for _, m in hist.iterrows():
        league = str(m["league"]).strip().upper()
        resolved = resolve_team_strength(
            ratings=ratings,
            league=league,
            home_team=str(m["home_team"]),
            away_team=str(m["away_team"]),
            default_home_advantage=0.0,
            default_rho=-0.05,
        )
        if resolved is None:
            skipped += 1
            continue

        home_s, away_s, league_params = resolved
        lam_h, lam_a = expected_goals(home=home_s, away=away_s, home_advantage=league_params.home_advantage)
        mat = score_matrix(lambda_home=lam_h, lambda_away=lam_a, rho=league_params.rho, max_goals=10)
        probs = _ou_probs(mat)

        p_raw = probs["under_4_5"]
        calib = cal_map.get((league, "under_4_5"), global_cal)
        p_cal = float(apply_calibration(np.array([p_raw], dtype=float), calib)[0])
        fair_odds = (1.0 / p_cal) if p_cal > 0 else None

        rows.append({
            "match_date": m["match_date"].strftime("%Y-%m-%d"),
            "league": league,
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_goals": int(m["home_goals"]),
            "away_goals": int(m["away_goals"]),
            "total_goals": int(m["total_goals"]),
            "actual_under_4_5": int(m["actual_under_4_5"]),
            "lam_home": round(lam_h, 4),
            "lam_away": round(lam_a, 4),
            "total_lambda": round(lam_h + lam_a, 4),
            "p_raw": round(p_raw, 4),
            "p_cal": round(p_cal, 4),
            "fair_odds": round(fair_odds, 4) if fair_odds else None,
        })

    print(f"Evaluated: {len(rows)}, Skipped (no ratings): {skipped}")

    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved backtest: {out_path} ({len(df)} rows)")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("UNDER 4.5 BACKTEST RESULTS")
    print("=" * 70)

    overall_rate = df["actual_under_4_5"].mean()
    print(f"\nOverall Under 4.5 rate: {overall_rate:.1%} ({df['actual_under_4_5'].sum()}/{len(df)})")

    # By probability band
    bands = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.93), (0.93, 0.96), (0.96, 1.00)]
    print(f"\n{'p_cal band':<15} {'count':>6} {'hits':>6} {'hit_rate':>9} {'avg_p_cal':>10} {'avg_fair_odds':>14}")
    print("-" * 62)
    for lo, hi in bands:
        mask = (df["p_cal"] >= lo) & (df["p_cal"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        hits = sub["actual_under_4_5"].sum()
        rate = sub["actual_under_4_5"].mean()
        avg_p = sub["p_cal"].mean()
        avg_fo = sub["fair_odds"].mean()
        print(f"[{lo:.2f}, {hi:.2f})  {len(sub):>6} {hits:>6} {rate:>9.1%} {avg_p:>10.4f} {avg_fo:>14.4f}")

    # Cumulative: if we pick all matches with p_cal >= threshold
    print(f"\n{'threshold':>10} {'count':>6} {'hits':>6} {'hit_rate':>9} {'avg_fair_odds':>14}")
    print("-" * 50)
    for thresh in [0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]:
        sub = df[df["p_cal"] >= thresh]
        if len(sub) == 0:
            continue
        hits = sub["actual_under_4_5"].sum()
        rate = sub["actual_under_4_5"].mean()
        avg_fo = sub["fair_odds"].mean()
        print(f"{thresh:>10.2f} {len(sub):>6} {hits:>6} {rate:>9.1%} {avg_fo:>14.4f}")

    # By league
    print(f"\n{'league':<8} {'count':>6} {'hits':>6} {'hit_rate':>9} {'avg_p_cal':>10}")
    print("-" * 42)
    for lg in sorted(df["league"].unique()):
        sub = df[df["league"] == lg]
        hits = sub["actual_under_4_5"].sum()
        rate = sub["actual_under_4_5"].mean()
        avg_p = sub["p_cal"].mean()
        print(f"{lg:<8} {len(sub):>6} {hits:>6} {rate:>9.1%} {avg_p:>10.4f}")

    # Calibration check: is p_cal well calibrated?
    print(f"\n{'p_cal band':<15} {'predicted':>10} {'actual':>10} {'diff':>10}")
    print("-" * 48)
    for lo, hi in bands:
        mask = (df["p_cal"] >= lo) & (df["p_cal"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        predicted = sub["p_cal"].mean()
        actual = sub["actual_under_4_5"].mean()
        diff = actual - predicted
        print(f"[{lo:.2f}, {hi:.2f})  {predicted:>10.4f} {actual:>10.4f} {diff:>+10.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())