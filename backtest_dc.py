"""
Backtest Dixon-Coles Double Chance (1X / X2) on historical matches.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from data_loader import load_team_ratings
from decision_engine import classify_variance
from dixon_coles import expected_goals, market_probabilities, resolve_team_strength, score_matrix
from simulation import run_monte_carlo

STRICT_LOW_VARIANCE_LEAGUES = set(CFG["strict_low_variance_leagues"])


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest DC Double Chance model.")
    p.add_argument("--history-csv", default="data/historical/historical_matches_transfermarkt.csv")
    p.add_argument("--ratings-pkl", default="data/historical/team_ratings.pkl")
    p.add_argument("--min-date", default="2024-07-01")
    p.add_argument("--max-date", default="2026-03-16")
    p.add_argument("--output-csv", default="simulations/backtests/backtest_dc.csv")
    args = p.parse_args()

    ratings = load_team_ratings(args.ratings_pkl)
    dc_cfg = CFG["dc"]
    var_cfg = CFG["variance"]

    hist = pd.read_csv(args.history_csv)
    hist["match_date"] = pd.to_datetime(hist["match_date"])
    hist = hist[(hist["match_date"] >= args.min_date) & (hist["match_date"] <= args.max_date)].copy()
    hist = hist.dropna(subset=["home_goals", "away_goals"])
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
            default_home_advantage=CFG["dixon_coles"]["default_home_advantage"],
            default_rho=CFG["dixon_coles"]["default_rho"],
        )
        if resolved is None:
            skipped += 1
            continue

        home_s, away_s, league_params = resolved
        lam_h, lam_a = expected_goals(home=home_s, away=away_s, home_advantage=league_params.home_advantage)
        mat = score_matrix(lam_h, lam_a, rho=league_params.rho, max_goals=CFG["dixon_coles"]["max_goals"])
        probs = market_probabilities(mat)
        mc = run_monte_carlo(lam_h, lam_a, iterations=CFG["dixon_coles"]["mc_iterations"])

        hg = int(m["home_goals"])
        ag = int(m["away_goals"])
        actual_1x = int(hg >= ag)  # home win or draw
        actual_x2 = int(ag >= hg)  # away win or draw

        for market, p_model, var_key, actual in [
            ("1X", probs["1X"], "variance_1X", actual_1x),
            ("X2", probs["X2"], "variance_X2", actual_x2),
        ]:
            variance_val = mc[var_key]
            variance_class = classify_variance(variance_val)
            fair_odds = (1.0 / p_model) if p_model > 0 else None

            allowed = (
                {"LOW"} if league in STRICT_LOW_VARIANCE_LEAGUES else {"LOW", "MEDIUM"}
            )
            pass_variance = variance_class in allowed
            pass_prob = p_model >= dc_cfg["min_probability"]

            rows.append({
                "match_date": m["match_date"].strftime("%Y-%m-%d"),
                "league": league,
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_goals": hg,
                "away_goals": ag,
                "market": market,
                "model_probability": round(p_model, 4),
                "fair_odds": round(fair_odds, 4) if fair_odds else None,
                "variance": round(variance_val, 4),
                "variance_class": variance_class,
                "pass_variance": pass_variance,
                "pass_prob": pass_prob,
                "would_recommend_no_odds": pass_prob and pass_variance,
                "actual_won": actual,
                "lam_home": round(lam_h, 4),
                "lam_away": round(lam_a, 4),
            })

    print(f"Evaluated: {len(rows)}, Skipped (no ratings): {skipped}")

    df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

    # ===================== ANALYSIS =====================
    print("\n" + "=" * 70)
    print("DOUBLE CHANCE BACKTEST RESULTS")
    print("=" * 70)

    # Overall
    for mkt in ["1X", "X2"]:
        sub = df[df["market"] == mkt]
        print(f"\n--- {mkt} ---")
        print(f"Total: {len(sub)}, Won: {sub['actual_won'].sum()}, Rate: {sub['actual_won'].mean():.1%}")

    # By probability band
    print("\n\n=== HIT RATE BY PROBABILITY BAND ===")
    bands = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.78), (0.78, 0.82),
             (0.82, 0.86), (0.86, 0.90), (0.90, 0.95), (0.95, 1.00)]
    print(f"{'band':<15} {'count':>6} {'hits':>6} {'hit_rate':>9} {'avg_prob':>10} {'calibr_diff':>12}")
    print("-" * 60)
    for lo, hi in bands:
        mask = (df["model_probability"] >= lo) & (df["model_probability"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        hits = sub["actual_won"].sum()
        rate = sub["actual_won"].mean()
        avg_p = sub["model_probability"].mean()
        diff = rate - avg_p
        print(f"[{lo:.2f}, {hi:.2f})  {len(sub):>6} {hits:>6} {rate:>9.1%} {avg_p:>10.4f} {diff:>+12.4f}")

    # Cumulative threshold
    print("\n=== CUMULATIVE HIT RATE (prob >= threshold) ===")
    print(f"{'threshold':>10} {'count':>6} {'hits':>6} {'hit_rate':>9} {'avg_fair_odds':>14}")
    print("-" * 50)
    for thresh in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95]:
        sub = df[df["model_probability"] >= thresh]
        if len(sub) == 0:
            continue
        hits = sub["actual_won"].sum()
        rate = sub["actual_won"].mean()
        avg_fo = sub["fair_odds"].mean() if sub["fair_odds"].notna().any() else 0
        print(f"{thresh:>10.2f} {len(sub):>6} {hits:>6} {rate:>9.1%} {avg_fo:>14.4f}")

    # Variance filter impact
    print("\n=== VARIANCE FILTER IMPACT (prob >= 0.78) ===")
    base = df[df["model_probability"] >= dc_cfg["min_probability"]]
    for vc in ["LOW", "LOW-MEDIUM", "MEDIUM", "HIGH"]:
        sub = base[base["variance_class"] == vc]
        if len(sub) == 0:
            continue
        rate = sub["actual_won"].mean()
        print(f"  {vc:<12} count={len(sub):>5}  hit_rate={rate:.1%}")

    # Combined: pass_prob + pass_variance (what would be recommended if odds existed)
    rec_sim = df[df["would_recommend_no_odds"] == True]
    print(f"\n  Would-recommend (prob+variance): count={len(rec_sim)}, hit_rate={rec_sim['actual_won'].mean():.1%}")

    # By league
    print("\n=== HIT RATE BY LEAGUE (prob >= 0.78, pass_variance) ===")
    rec_base = df[(df["pass_prob"]) & (df["pass_variance"])]
    print(f"{'league':<8} {'count':>6} {'hits':>6} {'hit_rate':>9} {'avg_prob':>10}")
    print("-" * 42)
    for lg in sorted(rec_base["league"].unique()):
        sub = rec_base[rec_base["league"] == lg]
        if len(sub) < 10:
            continue
        hits = sub["actual_won"].sum()
        rate = sub["actual_won"].mean()
        avg_p = sub["model_probability"].mean()
        print(f"{lg:<8} {len(sub):>6} {hits:>6} {rate:>9.1%} {avg_p:>10.4f}")

    # 1X vs X2 breakdown for would-recommend
    print("\n=== 1X vs X2 BREAKDOWN (would-recommend) ===")
    for mkt in ["1X", "X2"]:
        sub = rec_sim[rec_sim["market"] == mkt]
        if len(sub) == 0:
            continue
        print(f"  {mkt}: count={len(sub)}, hit_rate={sub['actual_won'].mean():.1%}")

    # Calibration check
    print("\n=== CALIBRATION (predicted vs actual) ===")
    print(f"{'band':<15} {'predicted':>10} {'actual':>10} {'diff':>10} {'n':>6}")
    print("-" * 55)
    for lo, hi in bands:
        mask = (df["model_probability"] >= lo) & (df["model_probability"] < hi)
        sub = df[mask]
        if len(sub) < 20:
            continue
        predicted = sub["model_probability"].mean()
        actual = sub["actual_won"].mean()
        diff = actual - predicted
        print(f"[{lo:.2f}, {hi:.2f})  {predicted:>10.4f} {actual:>10.4f} {diff:>+10.4f} {len(sub):>6}")

    # Profit simulation: flat 1u stake on all would-recommend at fair odds
    print("\n=== SIMULATED PROFIT (flat 1u at various odds levels) ===")
    rec_only = df[df["would_recommend_no_odds"] == True].copy()
    for target_odds in [1.25, 1.28, 1.30, 1.33, 1.35]:
        profit = rec_only["actual_won"].apply(lambda w: target_odds - 1 if w else -1).sum()
        roi = profit / len(rec_only) * 100 if len(rec_only) > 0 else 0
        print(f"  Odds {target_odds:.2f}: {len(rec_only)} bets, profit={profit:+.1f}u, ROI={roi:+.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())