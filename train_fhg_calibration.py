from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dc_double_chance import DixonColesModel
from fhg_calibration import fit_isotonic, fit_platt_logit
from fhg_weibull import p_goal_before_45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train leakage-free FHG calibration from walk-forward pre-match predictions.")
    p.add_argument("--history-csv", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--ratios-csv", default="simulations/FHG/data/fhg_league_ratios.csv")
    p.add_argument("--out-csv", default="simulations/FHG/data/fhg_calibration.csv")
    p.add_argument("--method", choices=["platt", "isotonic"], default="platt")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--min-train-matches", type=int, default=35)
    p.add_argument("--min-train-teams", type=int, default=6)
    p.add_argument("--min-samples", type=int, default=150)
    return p.parse_args()


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
        return pd.DataFrame(columns=["match_date", "xg_pre", "y"])
    min_date = g["match_date"].min()
    max_date = g["match_date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame(columns=["match_date", "xg_pre", "y"])

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
            & g["ht_home_goals"].notna()
            & g["ht_away_goals"].notna()
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
            y = float((float(row["ht_home_goals"]) + float(row["ht_away_goals"])) > 0.0)
            out.append(
                {
                    "match_date": row["match_date"],
                    "xg_pre": float(lam_h + lam_a),
                    "y": y,
                }
            )
        anchor = pred_end

    if not out:
        return pd.DataFrame(columns=["match_date", "xg_pre", "y"])
    s = pd.DataFrame(out)
    s = s.sort_values("match_date").reset_index(drop=True)
    return s


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    r = pd.read_csv(args.ratios_csv)
    h["match_date"] = pd.to_datetime(h["match_date"], errors="coerce")
    h = h.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    h = h[(h["ht_home_goals"].notna()) & (h["ht_away_goals"].notna())].copy()
    if h.empty:
        raise RuntimeError("No HT-available rows for calibration training.")

    k_map = {str(x["league"]).upper(): float(x["k_estimate"]) for _, x in r.iterrows()}
    rows = []
    all_p: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for league, g in h.groupby("league"):
        wf = walk_forward_samples(
            league_df=g,
            lookback_days=args.lookback_days,
            retrain_days=args.retrain_days,
            min_train_matches=args.min_train_matches,
            min_train_teams=args.min_train_teams,
        )
        if len(wf) < args.min_samples:
            continue

        k = float(k_map.get(str(league).upper(), 1.2))
        p_raw = np.array([p_goal_before_45(x, k) for x in wf["xg_pre"].to_numpy(dtype=float)], dtype=float)
        y = wf["y"].to_numpy(dtype=float)

        if args.method == "isotonic":
            xb, yv = fit_isotonic(p_raw, y)
            rows.append(
                {
                    "league": str(league).upper(),
                    "method": "isotonic",
                    "a": 0.0,
                    "b": 1.0,
                    "x_breaks": json.dumps(xb.tolist()),
                    "y_values": json.dumps(yv.tolist()),
                    "n_train": int(len(wf)),
                }
            )
        else:
            a, b = fit_platt_logit(p_raw, y)
            rows.append(
                {
                    "league": str(league).upper(),
                    "method": "platt",
                    "a": a,
                    "b": b,
                    "x_breaks": "[]",
                    "y_values": "[]",
                    "n_train": int(len(wf)),
                }
            )
        all_p.append(p_raw)
        all_y.append(y)

    if all_p:
        p_all = np.concatenate(all_p)
        y_all = np.concatenate(all_y)
        if args.method == "isotonic":
            xb, yv = fit_isotonic(p_all, y_all)
            rows.append(
                {
                    "league": "__GLOBAL__",
                    "method": "isotonic",
                    "a": 0.0,
                    "b": 1.0,
                    "x_breaks": json.dumps(xb.tolist()),
                    "y_values": json.dumps(yv.tolist()),
                    "n_train": int(len(p_all)),
                }
            )
        else:
            ga, gb = fit_platt_logit(p_all, y_all)
            rows.append(
                {
                    "league": "__GLOBAL__",
                    "method": "platt",
                    "a": ga,
                    "b": gb,
                    "x_breaks": "[]",
                    "y_values": "[]",
                    "n_train": int(len(p_all)),
                }
            )
    else:
        rows.append(
            {
                "league": "__GLOBAL__",
                "method": "platt",
                "a": 0.0,
                "b": 1.0,
                "x_breaks": "[]",
                "y_values": "[]",
                "n_train": 0,
            }
        )

    out = pd.DataFrame(rows).sort_values("league").reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved leakage-free calibration: {out_path} rows={len(out)} method={args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
