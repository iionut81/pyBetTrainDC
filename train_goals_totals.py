from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CFG
from dc_double_chance import DixonColesModel
from fhg_calibration import apply_platt_logit, fit_platt_logit

_TG = CFG["training"]["goals"]

MARKETS = ["over_2_5", "under_3_5", "under_4_5", "btts"]


def _ou_probs(mat: np.ndarray) -> Dict[str, float]:
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


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _calibration_buckets(p: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_buckets):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        mask = (p >= lo) & (p < hi)
        if i == n_buckets - 1:
            mask = (p >= lo) & (p <= hi)
        if mask.sum() > 0:
            rows.append(
                {
                    "bucket": f"{lo:.1f}-{hi:.1f}",
                    "p_mean": float(p[mask].mean()),
                    "y_mean": float(y[mask].mean()),
                    "count": int(mask.sum()),
                    "gap": float(p[mask].mean() - y[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def walk_forward_samples(
    league_df: pd.DataFrame,
    lookback_days: int,
    retrain_days: int,
    min_train_matches: int,
    min_train_teams: int,
) -> pd.DataFrame:
    out: list[dict] = []
    g = league_df.sort_values("match_date").reset_index(drop=True).copy()
    if g.empty:
        return pd.DataFrame()
    min_date = g["match_date"].min()
    max_date = g["match_date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame()

    anchor = min_date + pd.Timedelta(days=lookback_days)
    while anchor <= max_date:
        train_start = anchor - pd.Timedelta(days=lookback_days)
        train_end = anchor
        pred_end = anchor + pd.Timedelta(days=retrain_days)

        train = g[
            (g["match_date"] >= train_start)
            & (g["match_date"] < train_end)
            & g["home_goals"].notna()
            & g["away_goals"].notna()
        ].copy()
        pred = g[
            (g["match_date"] >= anchor)
            & (g["match_date"] < pred_end)
            & g["home_goals"].notna()
            & g["away_goals"].notna()
        ].copy()

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
                decay_xi=0.0,
                reference_date=(anchor - pd.Timedelta(days=1)).date(),
            )
        except Exception:
            anchor = pred_end
            continue

        for _, row in pred.iterrows():
            h = str(row["home_team"])
            a = str(row["away_team"])
            if h not in model._team_to_idx or a not in model._team_to_idx:
                continue
            hi = model._team_to_idx[h]
            ai = model._team_to_idx[a]
            lam_h = math.exp(model.attack[hi] + model.defense[ai] + model.home_adv)
            lam_a = math.exp(model.attack[ai] + model.defense[hi])
            mat = model._score_matrix(lam_h, lam_a, model.rho, model.max_goals)
            probs = _ou_probs(mat)
            total_goals = int(row["home_goals"]) + int(row["away_goals"])
            out.append(
                {
                    "match_date": row["match_date"],
                    "home_team": h,
                    "away_team": a,
                    "lam_home": lam_h,
                    "lam_away": lam_a,
                    "p_over_2_5": probs["over_2_5"],
                    "p_under_3_5": probs["under_3_5"],
                    "p_under_4_5": probs["under_4_5"],
                    "p_btts": probs["btts"],
                    "total_goals": total_goals,
                    "y_over_2_5": float(total_goals >= 3),
                    "y_under_3_5": float(total_goals <= 3),
                    "y_under_4_5": float(total_goals <= 4),
                    "y_btts": float(int(row["home_goals"]) >= 1 and int(row["away_goals"]) >= 1),
                }
            )
        anchor = pred_end

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("match_date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train leakage-free Goals Totals calibration from walk-forward DC predictions."
    )
    p.add_argument("--history-csv", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--out-calibration-csv", default="simulations/Goals/data/goals_calibration.csv")
    p.add_argument("--out-predictions-csv", default="simulations/Goals/backtests/goals_predictions.csv")
    p.add_argument("--out-summary-csv", default="simulations/Goals/backtests/goals_backtest_summary.csv")
    p.add_argument("--out-buckets-csv", default="simulations/Goals/backtests/goals_calibration_buckets.csv")
    p.add_argument("--lookback-days", type=int, default=_TG["lookback_days"])
    p.add_argument("--retrain-days", type=int, default=_TG["retrain_days"])
    p.add_argument("--min-train-matches", type=int, default=_TG["min_train_matches"])
    p.add_argument("--min-train-teams", type=int, default=_TG["min_train_teams"])
    p.add_argument("--min-samples", type=int, default=_TG["min_samples"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    h["match_date"] = pd.to_datetime(h["match_date"], errors="coerce")
    h = h.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    if h.empty:
        raise RuntimeError("No rows with goals for training.")

    all_preds: list[pd.DataFrame] = []
    cal_rows: list[dict] = []
    summary_rows: list[dict] = []
    bucket_dfs: list[pd.DataFrame] = []
    all_cal_pairs: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {m: ([], []) for m in MARKETS}

    for league, g in h.groupby("league"):
        print(f"  Walking forward league={league} rows={len(g)}")
        wf = walk_forward_samples(
            league_df=g,
            lookback_days=args.lookback_days,
            retrain_days=args.retrain_days,
            min_train_matches=args.min_train_matches,
            min_train_teams=args.min_train_teams,
        )
        if wf.empty or len(wf) < args.min_samples:
            print(f"    Skipped (samples={len(wf)})")
            continue

        wf["league"] = str(league)
        all_preds.append(wf)
        print(f"    Samples={len(wf)}")

        for market in MARKETS:
            p_raw = wf[f"p_{market}"].to_numpy(dtype=float)
            y = wf[f"y_{market}"].to_numpy(dtype=float)
            a_val, b_val = fit_platt_logit(p_raw, y)
            p_cal = apply_platt_logit(p_raw, a_val, b_val)

            all_cal_pairs[market][0].append(p_raw)
            all_cal_pairs[market][1].append(y)

            cal_rows.append(
                {
                    "league": str(league).upper(),
                    "market": market,
                    "method": "platt",
                    "a": a_val,
                    "b": b_val,
                    "temperature": 1.0,
                    "n_train": len(p_raw),
                }
            )

            summary_rows.append(
                {
                    "league": str(league).upper(),
                    "market": market,
                    "n": len(p_raw),
                    "p_mean": float(p_raw.mean()),
                    "y_mean": float(y.mean()),
                    "gap_raw": float(p_raw.mean() - y.mean()),
                    "log_loss_raw": _log_loss(p_raw, y),
                    "log_loss_cal": _log_loss(p_cal, y),
                    "brier_raw": _brier(p_raw, y),
                    "brier_cal": _brier(p_cal, y),
                }
            )

            bdf = _calibration_buckets(p_cal, y)
            if not bdf.empty:
                bdf["league"] = str(league).upper()
                bdf["market"] = market
                bucket_dfs.append(bdf)

    # Global calibration per market
    for market in MARKETS:
        ps_list, ys_list = all_cal_pairs[market]
        if ps_list:
            p_all = np.concatenate(ps_list)
            y_all = np.concatenate(ys_list)
            ga, gb = fit_platt_logit(p_all, y_all)
            n_total = int(len(p_all))
        else:
            ga, gb = 0.0, 1.0
            n_total = 0
        cal_rows.append(
            {
                "league": "__GLOBAL__",
                "market": market,
                "method": "platt",
                "a": ga,
                "b": gb,
                "temperature": 1.0,
                "n_train": n_total,
            }
        )

    # Save outputs
    cal_df = pd.DataFrame(cal_rows).sort_values(["league", "market"]).reset_index(drop=True)
    Path(args.out_calibration_csv).parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(args.out_calibration_csv, index=False)
    print(f"Saved calibration: {args.out_calibration_csv} rows={len(cal_df)}")

    if all_preds:
        pred_df = pd.concat(all_preds, ignore_index=True)
        Path(args.out_predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.out_predictions_csv, index=False)
        print(f"Saved predictions: {args.out_predictions_csv} rows={len(pred_df)}")

    if summary_rows:
        summ_df = pd.DataFrame(summary_rows).sort_values(["league", "market"]).reset_index(drop=True)
        Path(args.out_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summ_df.to_csv(args.out_summary_csv, index=False)
        print(f"Saved summary: {args.out_summary_csv} rows={len(summ_df)}")

    if bucket_dfs:
        bkt_df = pd.concat(bucket_dfs, ignore_index=True)
        Path(args.out_buckets_csv).parent.mkdir(parents=True, exist_ok=True)
        bkt_df.to_csv(args.out_buckets_csv, index=False)
        print(f"Saved calibration buckets: {args.out_buckets_csv} rows={len(bkt_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
