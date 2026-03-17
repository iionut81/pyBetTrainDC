"""
Train Platt calibration for Dixon-Coles Double Chance (1X / X2).

Reads the backtest CSV produced by backtest_dc.py and fits per-league +
global Platt scaling, saving results to simulations/DC/data/dc_calibration.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fhg_calibration import fit_platt_logit, fit_temperature, apply_platt_logit


def main() -> int:
    p = argparse.ArgumentParser(description="Train DC calibration from backtest results.")
    p.add_argument("--backtest-csv", default="simulations/backtests/backtest_dc.csv")
    p.add_argument("--output-csv", default="simulations/DC/data/dc_calibration.csv")
    p.add_argument("--min-samples", type=int, default=80, help="Min samples per league/market for per-league calibration")
    args = p.parse_args()

    df = pd.read_csv(args.backtest_csv)
    print(f"Loaded backtest: {len(df)} rows")

    rows: list[dict] = []

    for market in ["1X", "X2"]:
        mdf = df[df["market"] == market]

        # Global calibration
        p_raw = mdf["model_probability"].values
        y = mdf["actual_won"].values
        a, b = fit_platt_logit(p_raw, y)
        p_platt = apply_platt_logit(p_raw, a, b)
        t = fit_temperature(p_platt, y)
        rows.append({
            "league": "__GLOBAL__",
            "market": market,
            "method": "platt",
            "a": a,
            "b": b,
            "temperature": t,
            "n_train": len(mdf),
        })
        print(f"  GLOBAL {market}: a={a:.4f}, b={b:.4f}, T={t:.4f} (n={len(mdf)})")

        # Per-league calibration
        for league in sorted(mdf["league"].unique()):
            ldf = mdf[mdf["league"] == league]
            if len(ldf) < args.min_samples:
                continue
            p_raw_l = ldf["model_probability"].values
            y_l = ldf["actual_won"].values
            a_l, b_l = fit_platt_logit(p_raw_l, y_l)
            p_platt_l = apply_platt_logit(p_raw_l, a_l, b_l)
            t_l = fit_temperature(p_platt_l, y_l)
            rows.append({
                "league": league,
                "market": market,
                "method": "platt",
                "a": a_l,
                "b": b_l,
                "temperature": t_l,
                "n_train": len(ldf),
            })
            print(f"  {league} {market}: a={a_l:.4f}, b={b_l:.4f}, T={t_l:.4f} (n={len(ldf)})")

    out = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
