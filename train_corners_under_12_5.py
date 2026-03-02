from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from fhg_calibration import apply_platt_logit, fit_platt_logit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate Under 12.5 corners model (Poisson / Negative Binomial)."
    )
    p.add_argument("--history-csv", default="simulations/Corners U12.5/data/corners_history.csv")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--min-team-home", type=int, default=5)
    p.add_argument("--min-team-away", type=int, default=5)
    p.add_argument("--model", choices=["poisson", "nb"], default="nb")
    p.add_argument("--out-team-profiles", default="simulations/Corners U12.5/data/corners_team_profiles.csv")
    p.add_argument("--out-league-params", default="simulations/Corners U12.5/data/corners_league_params.csv")
    p.add_argument("--out-predictions", default="simulations/Corners U12.5/backtests/corners_under12_5_predictions.csv")
    p.add_argument("--out-summary", default="simulations/Corners U12.5/backtests/corners_under12_5_summary.csv")
    p.add_argument("--out-calibration", default="simulations/Corners U12.5/backtests/corners_under12_5_calibration.csv")
    p.add_argument("--out-sharpness", default="simulations/Corners U12.5/backtests/corners_under12_5_sharpness.csv")
    p.add_argument("--out-calibration-params", default="simulations/Corners U12.5/data/corners_calibration.csv")
    return p.parse_args()


def _safe_mean(s: pd.Series, fallback: float) -> float:
    if s.empty:
        return float(fallback)
    x = s.dropna()
    if x.empty:
        return float(fallback)
    return float(x.mean())


def _estimate_nb_k(values: np.ndarray, default_k: float = 12.0) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return float(default_k)
    mu = float(np.mean(vals))
    var = float(np.var(vals, ddof=1))
    if var <= mu or mu <= 0:
        return float(default_k)
    k = (mu * mu) / (var - mu)
    return float(np.clip(k, 0.5, 200.0))


def _prob_under_12_5(lam: float, model: str, k: float) -> float:
    lam = max(1e-6, float(lam))
    if model == "poisson":
        return float(poisson.cdf(12, lam))
    # NB parameterization: mean=mu, var=mu+mu^2/k.
    p = k / (k + lam)
    return float(nbinom.cdf(12, k, p))


def _team_profiles(train: pd.DataFrame) -> pd.DataFrame:
    if train.empty:
        return pd.DataFrame(columns=["league", "team", "h_for", "h_against", "a_for", "a_against", "n_home", "n_away"])

    rows: list[dict] = []
    for league, lg in train.groupby("league"):
        teams = sorted(set(lg["home_team"]).union(set(lg["away_team"])))
        # League fallbacks
        lg_h_for = _safe_mean(lg["home_corners"], 5.0)
        lg_h_against = _safe_mean(lg["away_corners"], 5.0)
        lg_a_for = _safe_mean(lg["away_corners"], 5.0)
        lg_a_against = _safe_mean(lg["home_corners"], 5.0)

        for team in teams:
            h = lg[lg["home_team"] == team]
            a = lg[lg["away_team"] == team]
            rows.append(
                {
                    "league": league,
                    "team": team,
                    "h_for": _safe_mean(h["home_corners"], lg_h_for),
                    "h_against": _safe_mean(h["away_corners"], lg_h_against),
                    "a_for": _safe_mean(a["away_corners"], lg_a_for),
                    "a_against": _safe_mean(a["home_corners"], lg_a_against),
                    "n_home": int(len(h)),
                    "n_away": int(len(a)),
                }
            )
    return pd.DataFrame(rows)


def _league_params(train: pd.DataFrame, global_mu: float) -> pd.DataFrame:
    rows: list[dict] = []
    for league, lg in train.groupby("league"):
        totals = (lg["home_corners"].astype(float) + lg["away_corners"].astype(float)).to_numpy(dtype=float)
        mu = float(np.mean(totals)) if len(totals) else global_mu
        var = float(np.var(totals, ddof=1)) if len(totals) > 1 else mu
        k = _estimate_nb_k(totals)
        tempo = (mu / global_mu) if global_mu > 0 else 1.0
        rows.append(
            {
                "league": league,
                "mu_total": mu,
                "var_total": var,
                "k_dispersion": k,
                "tempo_factor": float(np.clip(tempo, 0.75, 1.25)),
                "n_train": int(len(lg)),
            }
        )
    return pd.DataFrame(rows)


def _predict_lambda(
    profiles: pd.DataFrame,
    league_params: pd.DataFrame,
    league: str,
    home: str,
    away: str,
    min_team_home: int,
    min_team_away: int,
) -> Tuple[float, bool]:
    lp = league_params[league_params["league"] == league]
    if lp.empty:
        return np.nan, False
    mu = float(lp.iloc[0]["mu_total"])
    tempo = float(lp.iloc[0]["tempo_factor"])

    league_profiles = profiles[profiles["league"] == league]
    h = league_profiles[league_profiles["team"] == home]
    a = league_profiles[league_profiles["team"] == away]
    if h.empty or a.empty:
        return np.nan, False

    hrow = h.iloc[0]
    arow = a.iloc[0]
    if int(hrow["n_home"]) < min_team_home or int(arow["n_away"]) < min_team_away:
        return np.nan, False

    # Base industry-form lambda:
    # ((home_for + away_against) + (away_for + home_against)) / 2
    lam_base = (
        float(hrow["h_for"]) + float(arow["a_against"]) + float(arow["a_for"]) + float(hrow["h_against"])
    ) / 2.0
    lam = lam_base * tempo
    lam = float(np.clip(lam, 2.5, 18.0))
    # Blend with league mean to avoid unstable extremes.
    lam = float(0.8 * lam + 0.2 * mu)
    return lam, True


def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "n": 0,
            "hit_rate": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "p_mean": float("nan"),
            "y_mean": float("nan"),
        }
    p = np.clip(df["p_under_12_5"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    y = df["under_12_5"].to_numpy(dtype=float)
    ll = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    br = float(np.mean((p - y) ** 2))
    hr = float(np.mean((p >= 0.5) == (y >= 0.5)))
    return {
        "n": int(len(df)),
        "hit_rate": hr,
        "log_loss": ll,
        "brier": br,
        "p_mean": float(np.mean(p)),
        "y_mean": float(np.mean(y)),
    }


def _calibration_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
    x = df.copy()
    x["bucket"] = pd.cut(x["p_under_12_5"], bins=bins, labels=labels, include_lowest=True, right=True)
    cal = (
        x.groupby("bucket", observed=False)
        .agg(
            mean_pred=("p_under_12_5", "mean"),
            mean_obs=("under_12_5", "mean"),
            count=("under_12_5", "size"),
        )
        .reset_index()
    )
    cal["calibration_gap"] = cal["mean_pred"] - cal["mean_obs"]
    sharp = cal[["bucket", "count"]].rename(columns={"count": "count_predictions"}).copy()
    return cal, sharp


def main() -> int:
    args = parse_args()
    hist_path = Path(args.history_csv)
    if not hist_path.exists():
        raise RuntimeError(f"Corners history not found: {hist_path}")

    df = pd.read_csv(hist_path)
    required = {"league", "match_date", "home_team", "away_team", "home_corners", "away_corners"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    for c in ("league", "home_team", "away_team"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_corners", "away_corners", "league", "home_team", "away_team"]).copy()
    if df.empty:
        raise RuntimeError("No valid rows in corners history.")

    df["home_corners"] = pd.to_numeric(df["home_corners"], errors="coerce")
    df["away_corners"] = pd.to_numeric(df["away_corners"], errors="coerce")
    df = df.dropna(subset=["home_corners", "away_corners"]).copy()
    df["total_corners"] = df["home_corners"] + df["away_corners"]
    df["under_12_5"] = (df["total_corners"] <= 12).astype(float)
    df = df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    # Walk-forward backtest-like training/evaluation.
    min_date = df["match_date"].min().date()
    max_date = df["match_date"].max().date()
    anchor = min_date + dt.timedelta(days=args.lookback_days)
    rows: list[dict] = []
    cal_pairs: dict[str, tuple[list, list]] = {}  # league -> (p_raw list, y list)

    while anchor <= max_date:
        train_start = anchor - dt.timedelta(days=args.lookback_days)
        pred_end = anchor + dt.timedelta(days=args.retrain_days)

        train = df[(df["match_date"].dt.date >= train_start) & (df["match_date"].dt.date < anchor)].copy()
        pred = df[(df["match_date"].dt.date >= anchor) & (df["match_date"].dt.date < pred_end)].copy()
        if train.empty or pred.empty:
            anchor = pred_end
            continue

        global_mu = float(np.mean(train["total_corners"])) if not train.empty else 10.0
        profiles = _team_profiles(train)
        league_params = _league_params(train, global_mu=global_mu)
        k_map = {str(x["league"]): float(x["k_dispersion"]) for _, x in league_params.iterrows()}

        for _, m in pred.iterrows():
            lam, ok = _predict_lambda(
                profiles=profiles,
                league_params=league_params,
                league=str(m["league"]),
                home=str(m["home_team"]),
                away=str(m["away_team"]),
                min_team_home=args.min_team_home,
                min_team_away=args.min_team_away,
            )
            if not ok or not np.isfinite(lam):
                continue
            k = float(k_map.get(str(m["league"]), 12.0))
            p_under = _prob_under_12_5(lam=lam, model=args.model, k=k)
            rows.append(
                {
                    "match_date": m["match_date"].date().isoformat(),
                    "league": m["league"],
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "lambda_corners": lam,
                    "k_dispersion": k,
                    "model": args.model,
                    "p_under_12_5": p_under,
                    "total_corners": float(m["total_corners"]),
                    "under_12_5": float(m["under_12_5"]),
                }
            )
            lg_key = str(m["league"])
            if lg_key not in cal_pairs:
                cal_pairs[lg_key] = ([], [])
            cal_pairs[lg_key][0].append(p_under)
            cal_pairs[lg_key][1].append(float(m["under_12_5"]))
        anchor = pred_end

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise RuntimeError("No predictions generated. Increase history or loosen min-team filters.")

    # Fit Platt calibration per league on the walk-forward predictions.
    cal_rows: list[dict] = []
    cal_map: dict[str, tuple[float, float]] = {}
    all_cp: list[np.ndarray] = []
    all_cy: list[np.ndarray] = []
    for lg, (ps, ys) in cal_pairs.items():
        if len(ps) < 30:
            continue
        p_arr = np.array(ps, dtype=float)
        y_arr = np.array(ys, dtype=float)
        a, b = fit_platt_logit(p_arr, y_arr)
        cal_map[lg] = (a, b)
        all_cp.append(p_arr)
        all_cy.append(y_arr)
        cal_rows.append({"league": lg.upper(), "method": "platt", "a": a, "b": b, "n_train": len(ps)})
    ga, gb = (0.0, 1.0)
    if all_cp:
        ga, gb = fit_platt_logit(np.concatenate(all_cp), np.concatenate(all_cy))
    cal_rows.append({
        "league": "__GLOBAL__", "method": "platt", "a": ga, "b": gb,
        "n_train": int(sum(len(v[0]) for v in cal_pairs.values())),
    })
    cal_params_df = pd.DataFrame(cal_rows)
    Path(args.out_calibration_params).parent.mkdir(parents=True, exist_ok=True)
    cal_params_df.to_csv(args.out_calibration_params, index=False)

    # Apply calibration to all walk-forward predictions.
    pred_df["p_under_12_5_cal"] = pred_df["p_under_12_5"].copy()
    for lg in pred_df["league"].unique():
        mask = pred_df["league"] == lg
        p_raw_arr = pred_df.loc[mask, "p_under_12_5"].to_numpy(dtype=float)
        a, b = cal_map.get(lg, (ga, gb))
        pred_df.loc[mask, "p_under_12_5_cal"] = apply_platt_logit(p_raw_arr, a, b)

    # Save walk-forward outputs.
    Path(args.out_predictions).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.out_predictions, index=False)

    cal_df_tmp = pred_df.drop(columns=["p_under_12_5"]).rename(columns={"p_under_12_5_cal": "p_under_12_5"})
    cal_metrics = _metrics(cal_df_tmp)
    summary = pd.DataFrame(
        [
            {
                "scope": "corners_under_12_5_walk_forward",
                "model": args.model,
                "lookback_days": int(args.lookback_days),
                "retrain_days": int(args.retrain_days),
                **_metrics(pred_df),
            },
            {
                "scope": "corners_under_12_5_walk_forward_calibrated",
                "model": args.model,
                "lookback_days": int(args.lookback_days),
                "retrain_days": int(args.retrain_days),
                **cal_metrics,
            },
        ]
    )
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    cal, sharp = _calibration_tables(pred_df)
    Path(args.out_calibration).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_sharpness).parent.mkdir(parents=True, exist_ok=True)
    cal.to_csv(args.out_calibration, index=False)
    sharp.to_csv(args.out_sharpness, index=False)

    # Train latest model artifacts for daily use.
    latest_end = max_date
    latest_start = latest_end - dt.timedelta(days=args.lookback_days)
    latest_train = df[(df["match_date"].dt.date >= latest_start) & (df["match_date"].dt.date <= latest_end)].copy()
    global_mu_latest = float(np.mean(latest_train["total_corners"])) if not latest_train.empty else 10.0
    prof_latest = _team_profiles(latest_train)
    lp_latest = _league_params(latest_train, global_mu=global_mu_latest)
    Path(args.out_team_profiles).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_league_params).parent.mkdir(parents=True, exist_ok=True)
    prof_latest.to_csv(args.out_team_profiles, index=False)
    lp_latest.to_csv(args.out_league_params, index=False)

    print("CORNERS_UNDER_12_5_SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved predictions:   {args.out_predictions}")
    print(f"Saved summary:       {args.out_summary}")
    print(f"Saved calibration:   {args.out_calibration}")
    print(f"Saved sharpness:     {args.out_sharpness}")
    print(f"Saved cal params:    {args.out_calibration_params}")
    print(f"Saved profiles:      {args.out_team_profiles}")
    print(f"Saved league prm:    {args.out_league_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
