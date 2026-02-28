from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fhg_calibration import apply_platt_logit
from fhg_weibull import p_goal_before_45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train league-level FHG bias factors from historical residual gap.")
    p.add_argument("--history-csv", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--ratios-csv", default="simulations/FHG/data/fhg_league_ratios.csv")
    p.add_argument("--calibration-csv", default="simulations/FHG/data/fhg_calibration.csv")
    p.add_argument("--out-csv", default="simulations/FHG/data/fhg_league_bias.csv")
    p.add_argument("--min-train", type=int, default=120)
    p.add_argument("--min-bias", type=float, default=0.85)
    p.add_argument("--max-bias", type=float, default=1.30)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    r = pd.read_csv(args.ratios_csv)
    c = pd.read_csv(args.calibration_csv)

    h["match_date"] = pd.to_datetime(h["match_date"], errors="coerce")
    h = h.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    h = h[(h["ht_home_goals"].notna()) & (h["ht_away_goals"].notna())].copy()
    if h.empty:
        raise RuntimeError("No HT rows available for league bias training.")

    k_map = {str(x["league"]).upper(): float(x["k_estimate"]) for _, x in r.iterrows()}
    cal_map = {str(x["league"]).upper(): (float(x["a"]), float(x["b"])) for _, x in c.iterrows()}
    global_ab = cal_map.get("__GLOBAL__", (0.0, 1.0))

    rows = []
    for league, g in h.groupby("league"):
        g = g.sort_values("match_date").reset_index(drop=True)
        split = int(len(g) * 0.8)
        train = g.iloc[:split].copy()
        if len(train) < args.min_train:
            continue
        k = k_map.get(str(league).upper(), 1.2)
        a, b = cal_map.get(str(league).upper(), global_ab)
        xg_proxy = (train["home_goals"].astype(float) + train["away_goals"].astype(float)).clip(lower=0.05)
        p_raw = np.array([p_goal_before_45(x, k) for x in xg_proxy], dtype=float)
        p_cal = apply_platt_logit(p_raw, a=a, b=b)
        y = ((train["ht_home_goals"].astype(float) + train["ht_away_goals"].astype(float)) > 0).astype(float).to_numpy()

        mean_pred = float(np.mean(p_cal))
        mean_actual = float(np.mean(y))
        if mean_pred <= 0:
            bias = 1.0
        else:
            bias = mean_actual / mean_pred
        bias = float(np.clip(bias, args.min_bias, args.max_bias))
        rows.append(
            {
                "league": str(league).upper(),
                "bias": bias,
                "mean_actual": mean_actual,
                "mean_pred_cal": mean_pred,
                "n_train": int(len(train)),
            }
        )

    if not rows:
        raise RuntimeError("No leagues met minimum rows for bias estimation.")

    out = pd.DataFrame(rows).sort_values("league").reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved league bias: {out_path} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

