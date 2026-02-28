from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from fhg_calibration import apply_platt_logit
from fhg_weibull import p_goal_before_45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit FHG probability quality from built historical data.")
    p.add_argument("--history-csv", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--ratios-csv", default="simulations/FHG/data/fhg_league_ratios.csv")
    p.add_argument("--out-by-league", default="simulations/FHG/backtests/fhg_audit_by_league.csv")
    p.add_argument("--out-overall", default="simulations/FHG/backtests/fhg_audit_overall.csv")
    p.add_argument("--out-log", default="simulations/FHG/backtests/fhg_audit_log.csv")
    p.add_argument("--calibration-csv", default="simulations/FHG/data/fhg_calibration.csv")
    p.add_argument("--league-bias-csv", default="simulations/FHG/data/fhg_league_bias.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    r = pd.read_csv(args.ratios_csv)
    c = pd.read_csv(args.calibration_csv) if Path(args.calibration_csv).exists() else pd.DataFrame(columns=["league", "a", "b"])
    bdf = pd.read_csv(args.league_bias_csv) if Path(args.league_bias_csv).exists() else pd.DataFrame(columns=["league", "bias"])
    cal_map = {
        str(x["league"]).strip().upper(): (float(x.get("a", 0.0)), float(x.get("b", 1.0)))
        for _, x in c.iterrows()
    }
    global_ab = cal_map.get("__GLOBAL__", (0.0, 1.0))
    bias_map = {str(x["league"]).strip().upper(): float(x.get("bias", 1.0)) for _, x in bdf.iterrows()}
    if h.empty or r.empty:
        print("No FHG data to audit.")
        return 0
    h["match_date"] = pd.to_datetime(h["match_date"], errors="coerce")
    h = h.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    if h.empty:
        print("No valid rows after cleaning.")
        return 0

    # Holdout: latest 20% per league (chronological)
    rows = []
    for league, g in h.groupby("league"):
        g = g.sort_values("match_date").reset_index(drop=True)
        n = len(g)
        split = int(n * 0.8)
        if split < 20 or n - split < 10:
            continue
        test = g.iloc[split:].copy()
        k = float(r.loc[r["league"] == league, "k_estimate"].iloc[0]) if (r["league"] == league).any() else 1.2
        xg_proxy = (test["home_goals"].astype(float) + test["away_goals"].astype(float)).clip(lower=0.05)
        test["p_fhg"] = xg_proxy.apply(lambda x: p_goal_before_45(x, k))
        has_ht = test["ht_home_goals"].notna() & test["ht_away_goals"].notna()
        test = test[has_ht].copy()
        if test.empty:
            continue
        test["y"] = ((test["ht_home_goals"].astype(float) + test["ht_away_goals"].astype(float)) > 0).astype(float)
        p = test["p_fhg"].to_numpy().clip(1e-9, 1 - 1e-9)
        y = test["y"].to_numpy()
        a, b = cal_map.get(str(league).upper(), global_ab)
        p_cal = apply_platt_logit(p, a=a, b=b)
        bias = float(bias_map.get(str(league).upper(), 1.0))
        p_bias = np.clip(p_cal * bias, 0.05, 0.95)
        ll = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
        brier = float(np.mean((p - y) ** 2))
        ll_cal = float(np.mean(-(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))))
        brier_cal = float(np.mean((p_cal - y) ** 2))
        ll_bias = float(np.mean(-(y * np.log(p_bias) + (1.0 - y) * np.log(1.0 - p_bias))))
        brier_bias = float(np.mean((p_bias - y) ** 2))
        rows.append(
            {
                "league": league,
                "n_test": len(test),
                "hit_rate": float(np.mean((p >= 0.5) == (y >= 0.5))),
                "log_loss": ll,
                "brier": brier,
                "log_loss_cal": ll_cal,
                "brier_cal": brier_cal,
                "log_loss_bias": ll_bias,
                "brier_bias": brier_bias,
                "p_fhg_mean": float(np.mean(p)),
                "p_fhg_cal_mean": float(np.mean(p_cal)),
                "p_fhg_bias_mean": float(np.mean(p_bias)),
                "y_mean": float(np.mean(y)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No league had enough rows for FHG audit.")
        return 0
    out = out.sort_values("log_loss").reset_index(drop=True)
    run_ts = dt.datetime.now(dt.UTC).isoformat()
    out["run_timestamp_utc"] = run_ts

    by_league_path = Path(args.out_by_league)
    by_league_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(by_league_path, index=False)

    w = out["n_test"].to_numpy()
    overall_df = pd.DataFrame(
        [
            {
                "run_timestamp_utc": run_ts,
                "n_test_total": int(out["n_test"].sum()),
                "log_loss": float(np.average(out["log_loss"], weights=w)),
                "brier": float(np.average(out["brier"], weights=w)),
                "log_loss_cal": float(np.average(out["log_loss_cal"], weights=w)),
                "brier_cal": float(np.average(out["brier_cal"], weights=w)),
                "log_loss_bias": float(np.average(out["log_loss_bias"], weights=w)),
                "brier_bias": float(np.average(out["brier_bias"], weights=w)),
                "p_fhg_mean": float(np.average(out["p_fhg_mean"], weights=w)),
                "p_fhg_cal_mean": float(np.average(out["p_fhg_cal_mean"], weights=w)),
                "p_fhg_bias_mean": float(np.average(out["p_fhg_bias_mean"], weights=w)),
                "y_mean": float(np.average(out["y_mean"], weights=w)),
            }
        ]
    )
    overall_path = Path(args.out_overall)
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(overall_path, index=False)

    log_path = Path(args.out_log)
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, overall_df], ignore_index=True)
    else:
        log_df = overall_df.copy()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)

    print("FHG_AUDIT_BY_LEAGUE")
    print(out.to_string(index=False))
    print("\nFHG_AUDIT_OVERALL")
    print(overall_df.to_string(index=False))
    print(f"\nSaved by-league audit: {by_league_path}")
    print(f"Saved overall audit:   {overall_path}")
    print(f"Saved audit log:       {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
