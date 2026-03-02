from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dc_double_chance import DixonColesModel
from prepare_season_splits import split_5_seasons


@dataclass
class TuneConfig:
    lookback_days: int
    decay_xi: float
    retrain_days: int


def bin_calibration_error(df: pd.DataFrame, bins: int = 10) -> float:
    if df.empty:
        return float("nan")
    x = df.copy()
    x["bin"] = pd.cut(x["pred_prob"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    g = x.groupby("bin", observed=False).agg(n=("outcome", "size"), p_hat=("pred_prob", "mean"), y=("outcome", "mean"))
    g = g[g["n"] > 0]
    if g.empty:
        return float("nan")
    return float((np.abs(g["p_hat"] - g["y"]) * g["n"]).sum() / g["n"].sum())


def walk_forward_validate(
    all_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TuneConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[dict] = []
    model_cache: Dict[str, Tuple[DixonColesModel, dt.date]] = {}

    val_days = sorted(val_df["match_date"].unique())
    for day in val_days:
        today = val_df[val_df["match_date"] == day]
        train_start = day - dt.timedelta(days=cfg.lookback_days)
        train = all_df[
            (all_df["match_date"] >= train_start)
            & (all_df["match_date"] < day)
            & all_df["home_goals"].notna()
            & all_df["away_goals"].notna()
        ].copy()
        if train.empty or today.empty:
            continue

        for league in sorted(today["league"].dropna().unique()):
            day_matches = today[today["league"] == league].copy()
            train_l = train[train["league"] == league].copy()
            if day_matches.empty or train_l.empty:
                continue

            n_teams = len(set(train_l["home_team"]).union(set(train_l["away_team"])))
            if n_teams < 6 or len(train_l) < 35:
                continue

            need_retrain = True
            cached = model_cache.get(league)
            if cached is not None:
                _, last_fit_day = cached
                need_retrain = (day - last_fit_day).days >= cfg.retrain_days

            if need_retrain:
                model = DixonColesModel(max_goals=10)
                try:
                    model.fit(
                        train_l[["home_team", "away_team", "home_goals", "away_goals", "match_date"]],
                        decay_xi=cfg.decay_xi,
                        reference_date=day - dt.timedelta(days=1),
                    )
                except Exception:
                    continue
                model_cache[league] = (model, day)
            else:
                model = cached[0]

            for _, m in day_matches.iterrows():
                h = m["home_team"]
                a = m["away_team"]
                try:
                    p_1x, p_x2 = model.predict_1x_x2(h, a)
                except KeyError:
                    continue

                if p_1x >= p_x2:
                    market = "1X"
                    p = float(p_1x)
                    y = int(m["home_goals"] >= m["away_goals"])
                else:
                    market = "X2"
                    p = float(p_x2)
                    y = int(m["away_goals"] >= m["home_goals"])

                rows.append(
                    {
                        "match_date": day,
                        "league": league,
                        "home_team": h,
                        "away_team": a,
                        "market": market,
                        "pred_prob": p,
                        "outcome": y,
                    }
                )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return pred_df, {
            "predictions": 0,
            "hit_rate": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    p = np.clip(pred_df["pred_prob"].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9)
    y = pred_df["outcome"].to_numpy(dtype=float)
    log_loss = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    brier = float(np.mean((p - y) ** 2))
    ece = bin_calibration_error(pred_df, bins=10)
    hit_rate = float(pred_df["outcome"].mean())
    return pred_df, {
        "predictions": int(len(pred_df)),
        "hit_rate": hit_rate,
        "log_loss": log_loss,
        "brier": brier,
        "ece": ece,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune DC hyperparameters on validation season via walk-forward.")
    parser.add_argument("--input-csv", default="data/historical/historical_matches_transfermarkt.csv")
    parser.add_argument("--out-dir", default="simulations/validation_tuning")
    parser.add_argument("--lookbacks", default="180,270,365")
    parser.add_argument("--decay-grid", default="0.0,0.0008,0.0015,0.0025")
    parser.add_argument("--retrain-days-grid", default="7,30")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    for c in ("home_team", "away_team", "league", "season_code"):
        df[c] = df[c].astype(str).str.strip()
    df = df.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    df = df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    train_df, val_df, _test_df, last5 = split_5_seasons(df)
    min_val_day = val_df["match_date"].min()
    history_for_val = df[df["match_date"] < min_val_day].copy()
    val_full = df[(df["season_code"] == last5[3])].copy()
    all_for_val = pd.concat([history_for_val, val_full], ignore_index=True)
    all_for_val = all_for_val.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    lookbacks = [int(x.strip()) for x in args.lookbacks.split(",") if x.strip()]
    decay_grid = [float(x.strip()) for x in args.decay_grid.split(",") if x.strip()]
    retrain_days_grid = [int(x.strip()) for x in args.retrain_days_grid.split(",") if x.strip()]

    results: List[dict] = []
    best_pred_df = pd.DataFrame()
    best_cfg = None

    for lb in lookbacks:
        for xi in decay_grid:
            for rd in retrain_days_grid:
                cfg = TuneConfig(lookback_days=lb, decay_xi=xi, retrain_days=rd)
                pred_df, metrics = walk_forward_validate(all_for_val, val_full, cfg)
                row = {
                    "lookback_days": lb,
                    "decay_xi": xi,
                    "retrain_days": rd,
                    **metrics,
                }
                results.append(row)
                pd.DataFrame(results).to_csv(
                    os.path.join(args.out_dir, "validation_tuning_results_partial.csv"), index=False
                )
                print(
                    f"[TUNE] lb={lb} xi={xi} retrain={rd} -> "
                    f"n={metrics['predictions']} ll={metrics['log_loss']:.5f} ece={metrics['ece']:.5f}"
                )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(["log_loss", "ece", "brier"], ascending=[True, True, True]).reset_index(drop=True)
    results_path = os.path.join(args.out_dir, "validation_tuning_results.csv")
    res_df.to_csv(results_path, index=False)

    if not res_df.empty and pd.notna(res_df.loc[0, "log_loss"]):
        best = res_df.iloc[0].to_dict()
        best_cfg = TuneConfig(
            lookback_days=int(best["lookback_days"]),
            decay_xi=float(best["decay_xi"]),
            retrain_days=int(best["retrain_days"]),
        )
        best_pred_df, _ = walk_forward_validate(all_for_val, val_full, best_cfg)
        best_pred_path = os.path.join(args.out_dir, "validation_best_predictions.csv")
        best_pred_df.to_csv(best_pred_path, index=False)

        best_cfg_path = os.path.join(args.out_dir, "validation_best_config.csv")
        pd.DataFrame([best]).to_csv(best_cfg_path, index=False)
        print(f"\nBest config: lb={best_cfg.lookback_days}, xi={best_cfg.decay_xi}, retrain={best_cfg.retrain_days}")
        print(f"Saved: {results_path}, {best_cfg_path}, {best_pred_path}")
    else:
        print(f"No valid configuration found. Saved partial results to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
