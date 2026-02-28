from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fhg_weibull import p_goal_before_45
from fhg_calibration import fit_platt_logit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-league FHG calibration (Platt on logit).")
    p.add_argument("--history-csv", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--ratios-csv", default="simulations/FHG/data/fhg_league_ratios.csv")
    p.add_argument("--out-csv", default="simulations/FHG/data/fhg_calibration.csv")
    p.add_argument("--min-train", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    r = pd.read_csv(args.ratios_csv)
    h["match_date"] = pd.to_datetime(h["match_date"], errors="coerce")
    h = h.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    h = h[(h["ht_home_goals"].notna()) & (h["ht_away_goals"].notna())].copy()
    if h.empty:
        raise RuntimeError("No HT-available rows for calibration training.")

    rows = []
    all_p, all_y = [], []
    for league, g in h.groupby("league"):
        g = g.sort_values("match_date").reset_index(drop=True)
        split = int(len(g) * 0.8)
        train = g.iloc[:split].copy()
        if len(train) < args.min_train:
            continue
        k = float(r.loc[r["league"] == league, "k_estimate"].iloc[0]) if (r["league"] == league).any() else 1.2
        xg_proxy = (train["home_goals"].astype(float) + train["away_goals"].astype(float)).clip(lower=0.05)
        p_raw = np.array([p_goal_before_45(x, k) for x in xg_proxy], dtype=float)
        y = ((train["ht_home_goals"].astype(float) + train["ht_away_goals"].astype(float)) > 0).astype(float).to_numpy()
        a, b = fit_platt_logit(p_raw, y)
        rows.append({"league": league, "a": a, "b": b, "n_train": int(len(train))})
        all_p.append(p_raw)
        all_y.append(y)

    # global fallback
    if all_p:
        p_all = np.concatenate(all_p)
        y_all = np.concatenate(all_y)
        ga, gb = fit_platt_logit(p_all, y_all)
    else:
        ga, gb = 0.0, 1.0
    rows.append({"league": "__GLOBAL__", "a": ga, "b": gb, "n_train": int(sum(len(x) for x in all_p))})

    out = pd.DataFrame(rows).sort_values("league").reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved calibration: {out_path} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

