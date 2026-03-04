from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dc_double_chance import DixonColesModel
from fhg_calibration import apply_calibration, calibration_from_row
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
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--min-train-matches", type=int, default=35)
    p.add_argument("--min-train-teams", type=int, default=6)
    return p.parse_args()


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-9, 1 - 1e-9)
    y = y.astype(float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-9, 1 - 1e-9)
    y = y.astype(float)
    return float(np.mean((p - y) ** 2))


def _assert_no_forbidden_feature_columns(columns: list[str]) -> None:
    forbidden_tokens = ("goal", "ft_", "result")
    bad = [c for c in columns if any(tok in c.lower() for tok in forbidden_tokens)]
    if bad:
        raise RuntimeError(f"Forbidden post-match columns in feature inputs: {bad}")


def _walk_forward_eval_frame(
    league_df: pd.DataFrame,
    lookback_days: int,
    retrain_days: int,
    min_train_matches: int,
    min_train_teams: int,
) -> pd.DataFrame:
    cols = [
        "match_date",
        "home_team",
        "away_team",
        "xg_pre",
        "xg_leaky",
        "y",
    ]
    if league_df.empty:
        return pd.DataFrame(columns=cols)

    feature_cols = ["match_date", "home_team", "away_team"]
    _assert_no_forbidden_feature_columns(feature_cols)

    g = league_df.sort_values("match_date").reset_index(drop=True).copy()
    min_date = g["match_date"].min()
    max_date = g["match_date"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame(columns=cols)

    anchor = min_date + pd.Timedelta(days=lookback_days)
    out_rows: list[dict] = []
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
            & g["ht_home_goals"].notna()
            & g["ht_away_goals"].notna()
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
            y = float((float(row["ht_home_goals"]) + float(row["ht_away_goals"])) > 0.0)
            out_rows.append(
                {
                    "match_date": row["match_date"],
                    "home_team": h,
                    "away_team": a,
                    "xg_pre": float(lam_h + lam_a),
                    "xg_leaky": float(row["home_goals"]) + float(row["away_goals"]),
                    "y": y,
                }
            )

        anchor = pred_end

    if not out_rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(out_rows, columns=cols)
    out = out.drop_duplicates(subset=["match_date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _score_league(
    audit_type: str,
    league: str,
    test: pd.DataFrame,
    k: float,
    calib: dict,
    bias: float,
    p_base: np.ndarray,
) -> dict:
    y = test["y"].to_numpy(dtype=float)
    p = np.clip(p_base.astype(float), 1e-9, 1 - 1e-9)
    p_cal = apply_calibration(p, calib)
    p_bias = np.clip(p_cal * bias, 0.05, 0.95)
    return {
        "audit_type": audit_type,
        "league": league,
        "n_test": len(test),
        "hit_rate": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "log_loss": _log_loss(y, p),
        "brier": _brier(y, p),
        "log_loss_cal": _log_loss(y, p_cal),
        "brier_cal": _brier(y, p_cal),
        "log_loss_bias": _log_loss(y, p_bias),
        "brier_bias": _brier(y, p_bias),
        "p_fhg_mean": float(np.mean(p)),
        "p_fhg_cal_mean": float(np.mean(p_cal)),
        "p_fhg_bias_mean": float(np.mean(p_bias)),
        "y_mean": float(np.mean(y)),
        "k_estimate": float(k),
    }


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    r = pd.read_csv(args.ratios_csv)
    c = pd.read_csv(args.calibration_csv) if Path(args.calibration_csv).exists() else pd.DataFrame(columns=["league", "a", "b"])
    bdf = pd.read_csv(args.league_bias_csv) if Path(args.league_bias_csv).exists() else pd.DataFrame(columns=["league", "bias"])
    cal_map = {str(x["league"]).strip().upper(): calibration_from_row(dict(x)) for _, x in c.iterrows()}
    global_cal = cal_map.get("__GLOBAL__", {"method": "platt", "a": 0.0, "b": 1.0})
    bias_map = {str(x["league"]).strip().upper(): float(x.get("bias", 1.0)) for _, x in bdf.iterrows()}
    if h.empty or r.empty:
        print("No FHG data to audit.")
        return 0
    h["match_date"] = pd.to_datetime(h["match_date"], errors="coerce")
    h = h.dropna(subset=["match_date", "home_goals", "away_goals"]).copy()
    if h.empty:
        print("No valid rows after cleaning.")
        return 0

    # Walk-forward temporal audit by league.
    rows = []
    for league, g in h.groupby("league"):
        k = float(r.loc[r["league"] == league, "k_estimate"].iloc[0]) if (r["league"] == league).any() else 1.2
        calib = cal_map.get(str(league).upper(), global_cal)
        bias = float(bias_map.get(str(league).upper(), 1.0))

        wf = _walk_forward_eval_frame(
            league_df=g,
            lookback_days=args.lookback_days,
            retrain_days=args.retrain_days,
            min_train_matches=args.min_train_matches,
            min_train_teams=args.min_train_teams,
        )
        if wf.empty:
            continue

        p_leaky = np.array([p_goal_before_45(x, k) for x in wf["xg_leaky"].to_numpy(dtype=float)], dtype=float)
        rows.append(_score_league("legacy_leaky_audit", league, wf, k, calib, bias, p_leaky))

        p_prod = np.array([p_goal_before_45(x, k) for x in wf["xg_pre"].to_numpy(dtype=float)], dtype=float)
        rows.append(_score_league("production_kpi", league, wf, k, calib, bias, p_prod))

    out = pd.DataFrame(rows)
    if out.empty:
        print("No league had enough rows for FHG audit.")
        return 0
    out = out.sort_values(["audit_type", "log_loss"]).reset_index(drop=True)
    run_ts = dt.datetime.now(dt.UTC).isoformat()
    out["run_timestamp_utc"] = run_ts

    by_league_path = Path(args.out_by_league)
    by_league_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(by_league_path, index=False)

    overall_rows = []
    for audit_type, grp in out.groupby("audit_type"):
        w = grp["n_test"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "audit_type": audit_type,
                "run_timestamp_utc": run_ts,
                "n_test_total": int(grp["n_test"].sum()),
                "log_loss": float(np.average(grp["log_loss"], weights=w)),
                "brier": float(np.average(grp["brier"], weights=w)),
                "log_loss_cal": float(np.average(grp["log_loss_cal"], weights=w)),
                "brier_cal": float(np.average(grp["brier_cal"], weights=w)),
                "log_loss_bias": float(np.average(grp["log_loss_bias"], weights=w)),
                "brier_bias": float(np.average(grp["brier_bias"], weights=w)),
                "p_fhg_mean": float(np.average(grp["p_fhg_mean"], weights=w)),
                "p_fhg_cal_mean": float(np.average(grp["p_fhg_cal_mean"], weights=w)),
                "p_fhg_bias_mean": float(np.average(grp["p_fhg_bias_mean"], weights=w)),
                "y_mean": float(np.average(grp["y_mean"], weights=w)),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("audit_type").reset_index(drop=True)
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
