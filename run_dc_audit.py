from __future__ import annotations

"""
run_dc_audit.py
Walk-forward audit for the Dixon-Coles Double Chance model.

Produces:
  simulations/backtests/dc_audit_summary.csv
  simulations/backtests/dc_audit_by_league.csv
  simulations/backtests/dc_calibration_buckets.csv
  simulations/backtests/dc_backtest_oos.csv

Usage:
  python -X utf8 run_dc_audit.py
  python -X utf8 run_dc_audit.py --history-csv data/historical/historical_matches_transfermarkt_new_leagues.csv
"""

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dc_double_chance import DixonColesModel


# ── helpers ──────────────────────────────────────────────────────────────────

def _score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    mat = np.zeros((max_goals + 1, max_goals + 1))
    from scipy.stats import poisson
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam) * poisson.pmf(j, mu)
            if i == 0 and j == 0:
                p *= 1 - rho * lam * mu
            elif i == 0 and j == 1:
                p *= 1 + rho * lam
            elif i == 1 and j == 0:
                p *= 1 + rho * mu
            elif i == 1 and j == 1:
                p *= 1 - rho
            mat[i, j] = p
    return mat / mat.sum()


def _dc_probs(mat: np.ndarray) -> Dict[str, float]:
    home_win = float(np.tril(mat, -1).sum())
    draw     = float(np.trace(mat))
    away_win = float(np.triu(mat, 1).sum())
    return {
        "1X": home_win + draw,
        "X2": draw + away_win,
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
    }


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _calibration_buckets(p: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, n_buckets + 1)
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_buckets - 1 else (p >= lo) & (p <= hi)
        if mask.sum() > 0:
            rows.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "count": int(mask.sum()),
                "mean_pred": float(p[mask].mean()),
                "mean_obs": float(y[mask].mean()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            })
    return pd.DataFrame(rows)


# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    lookback_days: int,
    retrain_days: int,
    min_train_matches: int,
    min_train_teams: int,
    decay_xi: float,
    min_dc_prob: float,
) -> pd.DataFrame:
    """Walk-forward: train on lookback window, predict next retrain_days."""
    g = df.sort_values("match_date").reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()

    min_date = g["match_date"].min()
    max_date = g["match_date"].max()
    anchor = min_date + pd.Timedelta(days=lookback_days)

    rows: List[dict] = []
    while anchor <= max_date:
        train_start = anchor - pd.Timedelta(days=lookback_days)
        pred_end    = anchor + pd.Timedelta(days=retrain_days)

        train = g[(g["match_date"] >= train_start) & (g["match_date"] < anchor)].copy()
        pred  = g[(g["match_date"] >= anchor) & (g["match_date"] < pred_end)].copy()

        if pred.empty:
            anchor = pred_end
            continue

        teams = set(train["home_team"]).union(set(train["away_team"]))
        if len(train) < min_train_matches or len(teams) < min_train_teams:
            anchor = pred_end
            continue

        model = DixonColesModel(max_goals=10)
        try:
            model.fit(
                train[["home_team", "away_team", "home_goals", "away_goals", "match_date"]],
                decay_xi=decay_xi,
                reference_date=(anchor - pd.Timedelta(days=1)).date(),
            )
        except Exception:
            anchor = pred_end
            continue

        for _, row in pred.iterrows():
            h, a = str(row["home_team"]), str(row["away_team"])
            if h not in model._team_to_idx or a not in model._team_to_idx:
                continue
            p_1x, p_x2 = model.predict_1x_x2(h, a)
            total = int(row["home_goals"]) + int(row["away_goals"])
            home_goals, away_goals = int(row["home_goals"]), int(row["away_goals"])
            actual_1x = float(home_goals >= away_goals)   # home win or draw
            actual_x2 = float(away_goals >= home_goals)   # draw or away win

            for market, p_model, actual in [("1X", p_1x, actual_1x), ("X2", p_x2, actual_x2)]:
                rows.append({
                    "match_date": row["match_date"],
                    "league": str(row["league"]).upper(),
                    "home_team": h,
                    "away_team": a,
                    "market": market,
                    "p_model": p_model,
                    "actual": actual,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                })

        anchor = pred_end

    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dixon-Coles walk-forward audit.")
    p.add_argument("--history-csv", default="data/historical/historical_matches_transfermarkt_new_leagues.csv")
    p.add_argument("--lookback-days",  type=int,   default=365)
    p.add_argument("--retrain-days",   type=int,   default=30)
    p.add_argument("--min-train-matches", type=int, default=35)
    p.add_argument("--min-train-teams",   type=int, default=6)
    p.add_argument("--decay-xi",       type=float, default=0.0025)
    p.add_argument("--min-dc-prob",    type=float, default=0.78,
                   help="Only include best predictions (p >= this threshold)")
    p.add_argument("--oos-start-date", default="",
                   help="OOS production backtest start date YYYY-MM-DD (default: auto last season)")
    p.add_argument("--out-dir", default="simulations/backtests")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load history ─────────────────────────────────────────────────────────
    print(f"Loading {args.history_csv} ...")
    df = pd.read_csv(args.history_csv)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip().str.lower()
    df["away_team"] = df["away_team"].astype(str).str.strip().str.lower()
    df["league"]    = df["league"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    leagues = sorted(df["league"].unique())
    print(f"  Rows: {len(df)} | Leagues: {leagues}")

    # ── walk-forward per league ───────────────────────────────────────────────
    print("\nRunning walk-forward validation ...")
    all_preds: List[pd.DataFrame] = []
    for league in leagues:
        ldf = df[df["league"] == league].copy()
        wf = walk_forward(
            df=ldf,
            lookback_days=args.lookback_days,
            retrain_days=args.retrain_days,
            min_train_matches=args.min_train_matches,
            min_train_teams=args.min_train_teams,
            decay_xi=args.decay_xi,
            min_dc_prob=args.min_dc_prob,
        )
        if wf.empty:
            print(f"  {league}: skipped (insufficient data)")
            continue
        wf["league"] = league
        all_preds.append(wf)
        print(f"  {league}: {len(wf)} predictions")

    if not all_preds:
        print("No predictions generated.")
        return 1

    preds = pd.concat(all_preds, ignore_index=True)

    # ── filter to best predictions (min_dc_prob threshold) ───────────────────
    best = preds[preds["p_model"] >= args.min_dc_prob].copy()
    print(f"\nAll predictions:  {len(preds)}")
    print(f"Best predictions: {len(best)}  (p >= {args.min_dc_prob})")

    # ── overall metrics ───────────────────────────────────────────────────────
    p_arr = best["p_model"].to_numpy()
    y_arr = best["actual"].to_numpy()
    overall = {
        "hit_rate": float(y_arr.mean()),
        "log_loss": _log_loss(p_arr, y_arr),
        "brier":    _brier(p_arr, y_arr),
        "p_mean":   float(p_arr.mean()),
        "y_mean":   float(y_arr.mean()),
        "n":        len(best),
    }

    # ── per-league metrics ────────────────────────────────────────────────────
    league_rows = []
    for league, grp in best.groupby("league"):
        p_l = grp["p_model"].to_numpy()
        y_l = grp["actual"].to_numpy()
        league_rows.append({
            "league":   league,
            "n":        len(grp),
            "hit_rate": float(y_l.mean()),
            "log_loss": _log_loss(p_l, y_l),
            "brier":    _brier(p_l, y_l),
            "p_mean":   float(p_l.mean()),
            "y_mean":   float(y_l.mean()),
        })
    league_df = pd.DataFrame(league_rows).sort_values("hit_rate", ascending=False).reset_index(drop=True)

    # ── calibration buckets ───────────────────────────────────────────────────
    buckets_df = _calibration_buckets(p_arr, y_arr)

    # ── OOS production backtest (last season) ────────────────────────────────
    if args.oos_start_date:
        oos_start = pd.Timestamp(args.oos_start_date)
    else:
        max_date = preds["match_date"].max()
        oos_start = max_date - pd.Timedelta(days=210)   # ~last 7 months

    oos = best[best["match_date"] >= oos_start].copy()
    oos_end = best["match_date"].max()
    oos_n   = len(oos)
    oos_wins = int(oos["actual"].sum())
    oos_hit  = float(oos_wins / oos_n) if oos_n > 0 else 0.0

    # ── save CSVs ─────────────────────────────────────────────────────────────
    summary_row = pd.DataFrame([{
        "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csv": args.history_csv,
        "leagues": ", ".join(leagues),
        "n_leagues": len(leagues),
        "total_matches_evaluated": len(best),
        "min_dc_prob_threshold": args.min_dc_prob,
        "hit_rate":  round(overall["hit_rate"], 4),
        "log_loss":  round(overall["log_loss"], 4),
        "brier":     round(overall["brier"], 4),
        "p_mean":    round(overall["p_mean"], 4),
        "y_mean":    round(overall["y_mean"], 4),
        "oos_start": oos_start.date().isoformat(),
        "oos_end":   oos_end.date().isoformat(),
        "oos_predictions": oos_n,
        "oos_wins":  oos_wins,
        "oos_hit_rate": round(oos_hit, 4),
        "lookback_days": args.lookback_days,
        "decay_xi":  args.decay_xi,
    }])

    summary_path  = out_dir / "dc_audit_summary.csv"
    league_path   = out_dir / "dc_audit_by_league.csv"
    buckets_path  = out_dir / "dc_calibration_buckets.csv"
    preds_path    = out_dir / "dc_predictions_walkforward.csv"

    summary_row.to_csv(summary_path, index=False)
    league_df.round(4).to_csv(league_path, index=False)
    buckets_df.round(4).to_csv(buckets_path, index=False)
    best.to_csv(preds_path, index=False)

    # ── print audit report ────────────────────────────────────────────────────
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    w = 60
    print("\n" + "=" * w)
    print("  Latest Dixon-Coles walk-forward audit")
    print(f"  Timestamp: {ts}")
    print(f"  Source: dc_audit_summary.csv, dc_audit_by_league.csv")
    print("=" * w)
    print()
    print("Coverage:")
    print(f"  - Data: {args.history_csv}")
    print(f"  - Scope: walk-forward validation (best predictions only, p >= {args.min_dc_prob})")
    print(f"  - Leagues: {', '.join(leagues)}")
    print(f"  - Total evaluated matches: {len(best):,}")
    print()
    print("Overall (validation_best_predictions_walk_forward):")
    print(f"  hit_rate  = {overall['hit_rate']:.4f}")
    print(f"  log_loss  = {overall['log_loss']:.4f}")
    print(f"  brier     = {overall['brier']:.4f}")
    print(f"  p_mean    = {overall['p_mean']:.4f}")
    print(f"  y_mean    = {overall['y_mean']:.4f}")
    print()
    print("Per-league breakdown:")
    print(f"  {'League':<6}  {'n':<6}  {'hit_rate':<10}  {'log_loss':<10}  {'brier':<8}  {'p_mean'}")
    for _, r in league_df.iterrows():
        print(f"  {r['league']:<6}  {int(r['n']):<6}  {r['hit_rate']:.4f}      {r['log_loss']:.4f}      {r['brier']:.4f}    {r['p_mean']:.4f}")
    print()
    print("Sharpness (prediction distribution):")
    print(f"  {'Bucket':<10}  {'Count'}")
    for _, b in buckets_df.iterrows():
        if b["count"] > 0:
            print(f"  {b['bucket']:<10}  {int(b['count'])}")
    print()
    print("Calibration gaps (predicted vs observed):")
    print(f"  {'Bucket':<10}  {'mean_pred':<10}  {'mean_obs':<10}  gap")
    for _, b in buckets_df.iterrows():
        if b["count"] > 0:
            gap = b["gap"]
            label = "(well calibrated)" if abs(gap) < 0.02 else ("(slightly over-confident)" if gap > 0 else "(slightly under-confident)")
            sign = "+" if gap >= 0 else ""
            print(f"  {b['bucket']:<10}  {b['mean_pred']:.4f}      {b['mean_obs']:.4f}      {sign}{gap:.4f}  {label}")
    print()
    print(f"Production backtest (OOS, {oos_start.date()} -> {oos_end.date()}):")
    print(f"  predictions = {oos_n}")
    print(f"  wins        = {oos_wins}")
    print(f"  hit_rate    = {oos_hit:.4f}")
    print(f"  config: lookback={args.lookback_days}d, decay_xi={args.decay_xi}, min_dc_prob={args.min_dc_prob}")
    print()
    print(f"Per-league details: {league_path}")
    print(f"Calibration detail: {buckets_path}")
    print("=" * w)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
