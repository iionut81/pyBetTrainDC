from __future__ import annotations

"""
run_goals_audit.py
Walk-forward audit for Goals Totals (over_2_5, under_3_5, under_4_5).
Matches the format of the Goals Totals audit report.

Usage:
  python -X utf8 run_goals_audit.py
"""

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dc_double_chance import DixonColesModel
from fhg_calibration import apply_platt_logit, fit_platt_logit

MARKETS = ["over_2_5", "under_3_5", "under_4_5", "btts"]

ELIGIBLE_BANDS: Dict[str, Tuple[float, float]] = {
    "over_2_5":  (0.63, 1.00),
    "under_3_5": (0.65, 0.85),
    "under_4_5": (0.82, 0.93),
    "btts":      (0.60, 1.00),
}

MARKET_LABELS = {
    "over_2_5":  "OVER 2.5",
    "under_3_5": "UNDER 3.5",
    "under_4_5": "UNDER 4.5",
    "btts":      "BTTS YES",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _ou_probs(mat: np.ndarray) -> Dict[str, float]:
    n = mat.shape[0]
    idx = np.arange(n)
    ig, jg = np.meshgrid(idx, idx, indexing="ij")
    total = ig + jg
    return {
        "over_2_5":  float(mat[total >= 3].sum()),
        "under_3_5": float(mat[total <= 3].sum()),
        "under_4_5": float(mat[total <= 4].sum()),
        "btts":      float(mat[1:, 1:].sum()),
    }


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _gap_label(gap: float) -> str:
    a = abs(gap)
    if a <= 0.002:
        return "near-perfect"
    if a <= 0.025:
        return "good"
    if a <= 0.05:
        return "acceptable"
    return "miscalibrated"


def _calibration_buckets(p: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, n_buckets + 1)
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_buckets - 1 else (p >= lo) & (p <= hi)
        if mask.sum() > 0:
            rows.append({
                "bucket":   f"{lo:.1f}-{hi:.1f}",
                "n":        int(mask.sum()),
                "p_mean":   float(p[mask].mean()),
                "y_mean":   float(y[mask].mean()),
                "gap":      float(p[mask].mean() - y[mask].mean()),
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
) -> pd.DataFrame:
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
            hi = model._team_to_idx[h]
            ai = model._team_to_idx[a]
            lam_h = math.exp(model.attack[hi] + model.defense[ai] + model.home_adv)
            lam_a = math.exp(model.attack[ai] + model.defense[hi])
            mat   = model._score_matrix(lam_h, lam_a, model.rho, model.max_goals)
            probs = _ou_probs(mat)
            total = int(row["home_goals"]) + int(row["away_goals"])
            rows.append({
                "match_date":     row["match_date"],
                "league":         str(row["league"]).upper(),
                "home_team":      h,
                "away_team":      a,
                "p_over_2_5":     probs["over_2_5"],
                "p_under_3_5":    probs["under_3_5"],
                "p_under_4_5":    probs["under_4_5"],
                "p_btts":         probs["btts"],
                "y_over_2_5":     float(total >= 3),
                "y_under_3_5":    float(total <= 3),
                "y_under_4_5":    float(total <= 4),
                "y_btts":         float(int(row["home_goals"]) >= 1 and int(row["away_goals"]) >= 1),
                "total_goals":    total,
            })

        anchor = pred_end

    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Goals Totals walk-forward audit.")
    p.add_argument("--history-csv", default="data/historical/historical_matches_transfermarkt_new_leagues.csv")
    p.add_argument("--calibration-csv", default="simulations/Goals/data/goals_calibration.csv")
    p.add_argument("--lookback-days",     type=int,   default=365)
    p.add_argument("--retrain-days",      type=int,   default=30)
    p.add_argument("--min-train-matches", type=int,   default=35)
    p.add_argument("--min-train-teams",   type=int,   default=6)
    p.add_argument("--decay-xi",          type=float, default=0.0)
    p.add_argument("--min-samples",       type=int,   default=50)
    p.add_argument("--out-dir", default="simulations/Goals/backtests")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load history ──────────────────────────────────────────────────────────
    print(f"Loading {args.history_csv} ...")
    df = pd.read_csv(args.history_csv)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["home_team"]  = df["home_team"].astype(str).str.strip().str.lower()
    df["away_team"]  = df["away_team"].astype(str).str.strip().str.lower()
    df["league"]     = df["league"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    leagues = sorted(df["league"].unique())
    print(f"  Rows: {len(df)} | Leagues: {leagues}")

    # ── load calibration ──────────────────────────────────────────────────────
    cal_df = pd.read_csv(args.calibration_csv)

    def get_cal(league: str, market: str) -> Tuple[float, float]:
        row = cal_df[(cal_df["league"] == league) & (cal_df["market"] == market)]
        if row.empty:
            row = cal_df[(cal_df["league"] == "__GLOBAL__") & (cal_df["market"] == market)]
        if row.empty:
            return 0.0, 1.0
        return float(row.iloc[0]["a"]), float(row.iloc[0]["b"])

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
        )
        if wf.empty or len(wf) < args.min_samples:
            print(f"  {league}: skipped")
            continue
        wf["league"] = league
        all_preds.append(wf)
        print(f"  {league}: {len(wf)} samples")

    if not all_preds:
        print("No predictions generated.")
        return 1

    preds = pd.concat(all_preds, ignore_index=True)

    # ── apply calibration per league ──────────────────────────────────────────
    for market in MARKETS:
        p_cal_col = f"p_cal_{market}"
        preds[p_cal_col] = np.nan
        for league in preds["league"].unique():
            mask = preds["league"] == league
            a, b = get_cal(league, market)
            raw = preds.loc[mask, f"p_{market}"].to_numpy()
            preds.loc[mask, p_cal_col] = apply_platt_logit(raw, a, b)

    # ── print report ──────────────────────────────────────────────────────────
    w = 48
    sep = "─" * w
    print(f"\n{sep}")
    print("CALIBRATION BUCKETS (calibrated, eligible only)")
    print(sep)

    bucket_dfs: Dict[str, pd.DataFrame] = {}
    eligible_dfs: Dict[str, pd.DataFrame] = {}
    raw_dfs: Dict[str, pd.DataFrame] = {}

    for market in MARKETS:
        lo, hi = ELIGIBLE_BANDS[market]
        p_cal = preds[f"p_cal_{market}"].to_numpy()
        p_raw = preds[f"p_{market}"].to_numpy()
        y     = preds[f"y_{market}"].to_numpy()

        elig_mask = (p_cal >= lo) & (p_cal <= hi)
        p_elig = p_cal[elig_mask]
        y_elig = y[elig_mask]
        p_raw_elig = p_raw[elig_mask]

        eligible_dfs[market] = preds[elig_mask].copy()
        raw_dfs[market] = preds.copy()

        bkt = _calibration_buckets(p_elig, y_elig)
        bucket_dfs[market] = bkt

        print(f"\n{market}:")
        print(f"  {'Bucket':<9}  {'n':>6}  {'p_mean':>7}  {'y_mean':>7}  {'gap':>8}")
        for _, r in bkt.iterrows():
            sign = "+" if r["gap"] >= 0 else ""
            label = _gap_label(r["gap"])
            print(f"  {r['bucket']:<9}  {int(r['n']):>6,}  {r['p_mean']:>7.4f}  {r['y_mean']:>7.4f}  {sign}{r['gap']:>7.4f}  ({label})")

    print(f"\n{sep}")
    print("PER-LEAGUE (sorted by brier, best first)")
    print(sep)

    league_rows: Dict[str, List[dict]] = {m: [] for m in MARKETS}
    for market in MARKETS:
        lo, hi = ELIGIBLE_BANDS[market]
        for league in sorted(preds["league"].unique()):
            grp = preds[preds["league"] == league]
            p_cal = grp[f"p_cal_{market}"].to_numpy()
            y     = grp[f"y_{market}"].to_numpy()
            p_raw = grp[f"p_{market}"].to_numpy()
            elig  = (p_cal >= lo) & (p_cal <= hi)
            if elig.sum() < 20:
                continue
            p_e, y_e = p_cal[elig], y[elig]
            league_rows[market].append({
                "league":   league,
                "n":        int(elig.sum()),
                "p_mean":   round(float(p_e.mean()), 4),
                "y_mean":   round(float(y_e.mean()), 4),
                "log_loss": round(_log_loss(p_e, y_e), 4),
                "brier":    round(_brier(p_e, y_e), 4),
            })

    for market in MARKETS:
        lo, hi = ELIGIBLE_BANDS[market]
        rows = sorted(league_rows[market], key=lambda x: x["brier"])
        print(f"\n{MARKET_LABELS[market]} ({lo:.2f}–{hi:.2f}):")
        print(f"  {'League':<6}  {'n':>6}  {'p_mean':>7}  {'y_mean':>7}  {'log_loss':>9}  {'brier':>7}")
        for r in rows:
            print(f"  {r['league']:<6}  {r['n']:>6,}  {r['p_mean']:>7.4f}  {r['y_mean']:>7.4f}  {r['log_loss']:>9.4f}  {r['brier']:>7.4f}")

    # ── before/after summary ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("BEFORE vs AFTER CALIBRATION (eligible band, worst bucket gap)")
    print(sep)
    print(f"\n  {'Market':<12}  {'Raw max gap':>12}  {'Cal max gap':>12}  {'Cal brier':>10}")
    for market in MARKETS:
        lo, hi = ELIGIBLE_BANDS[market]
        p_raw = preds[f"p_{market}"].to_numpy()
        p_cal = preds[f"p_cal_{market}"].to_numpy()
        y     = preds[f"y_{market}"].to_numpy()
        elig  = (p_cal >= lo) & (p_cal <= hi)

        bkt_raw = _calibration_buckets(p_raw[elig], y[elig])
        bkt_cal = _calibration_buckets(p_cal[elig], y[elig])

        max_raw = float(bkt_raw["gap"].abs().max()) if not bkt_raw.empty else 0.0
        max_cal = float(bkt_cal["gap"].abs().max()) if not bkt_cal.empty else 0.0
        brier_cal = _brier(p_cal[elig], y[elig])
        print(f"  {market:<12}  {max_raw:>+12.4f}  {max_cal:>+12.4f}  {brier_cal:>10.4f}")

    print(f"\n{sep}\n")

    # ── save CSVs ─────────────────────────────────────────────────────────────
    preds.to_csv(out_dir / "goals_audit_predictions.csv", index=False)
    for market in MARKETS:
        bucket_dfs[market]["market"] = market
    pd.concat(bucket_dfs.values(), ignore_index=True).to_csv(
        out_dir / "goals_audit_calibration_buckets.csv", index=False
    )
    for market in MARKETS:
        for r in league_rows[market]:
            r["market"] = market
    all_league = [r for rows in league_rows.values() for r in rows]
    pd.DataFrame(all_league).to_csv(out_dir / "goals_audit_by_league.csv", index=False)

    print(f"Saved audit files to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
