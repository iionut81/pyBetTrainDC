"""
train_sot_per_team.py

Per-team Shots on Target (SOT) model — walk-forward training + Platt calibration.

Data note:
  Flashscore SOT count (our source) has consistently lower magnitude than
  bookmaker SOT definition (ratio ~1.6x). We train on raw Flashscore counts,
  then apply CFG["sot"]["scaling_factor"] at inference to align lambda with
  bookmaker scale. Outcomes for Platt calibration are computed in our scale:
  (sot > line_bookmaker / scaling_factor).

Outputs:
  simulations/SOT/data/sot_team_profiles.csv       — latest team profiles
  simulations/SOT/data/sot_league_params.csv       — latest league params
  simulations/SOT/data/sot_calibration.csv         — Platt per (league, side, line)
  simulations/SOT/backtests/sot_predictions.csv    — walk-forward raw predictions
  simulations/SOT/backtests/sot_summary.csv        — hit rate / logloss / brier
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from config import CFG
from fhg_calibration import apply_platt_logit, fit_platt_logit

_TS = CFG["training"]["sot"]
_SC = CFG["sot"]
_LINES = list(_SC["lines"])
_SCALE = float(_SC["scaling_factor"])
_SCALE_PER_LEAGUE: Dict[str, float] = {
    str(k).lower(): float(v) for k, v in (_SC.get("scaling_per_league") or {}).items()
}
_RECENCY = _SC.get("recency", {"enabled": False})
_ELO_CFG = _SC.get("elo", {"enabled": False})


def _scale_for(league: str) -> float:
    """Return scaling_factor for this league (falls back to global)."""
    return _SCALE_PER_LEAGUE.get(str(league).lower(), _SCALE)


def _load_elo_ratings() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load team_ratings.pkl. Returns {league_lower: {team_lower: {attack, defence}}}."""
    path = Path("team_ratings.pkl")
    if not path.exists() or not _ELO_CFG.get("enabled"):
        return {}
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict) or "leagues" not in raw:
        return {}
    leagues = raw["leagues"]
    items = leagues.items() if isinstance(leagues, dict) else leagues
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for lg, payload in items:
        teams_payload = payload.get("teams", {}) if isinstance(payload, dict) else {}
        out[str(lg).lower()] = {
            str(team).lower(): {
                "attack": float(v.get("attack", 0.0)),
                "defence": float(v.get("defence", 0.0)),
            }
            for team, v in teams_payload.items()
        }
    return out


def _elo_multiplier(
    ratings: Dict[str, Dict[str, Dict[str, float]]],
    league: str,
    attacker: str,
    defender: str,
) -> float:
    """Multiplier applied to lambda based on Dixon-Coles attack vs opponent defence.
    defence coefficient: higher = worse at defending (concedes more) per DC convention.
    Strong attack + weak defence → boost lambda; inverse → reduce."""
    if not _ELO_CFG.get("enabled"):
        return 1.0
    lg = str(league).lower()
    league_ratings = ratings.get(lg)
    if not league_ratings:
        return 1.0
    atk = league_ratings.get(str(attacker).lower(), {}).get("attack", 0.0)
    dfc = league_ratings.get(str(defender).lower(), {}).get("defence", 0.0)
    elasticity = float(_ELO_CFG.get("sot_elasticity", 0.15))
    mult = 1.0 + elasticity * (atk + dfc)
    return float(np.clip(mult, _ELO_CFG.get("clip_min", 0.65), _ELO_CFG.get("clip_max", 1.50)))


def _recency_weights(dates: pd.Series, anchor: dt.date) -> np.ndarray:
    """Exponential decay weights: 1.0 for anchor day, e^(-ln2/HL * days_ago)."""
    if not _RECENCY.get("enabled"):
        return np.ones(len(dates), dtype=float)
    hl = float(_RECENCY.get("half_life_days", 140))
    anchor_ts = pd.Timestamp(anchor)
    delta = (anchor_ts - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    delta = np.clip(delta, 0.0, None)
    decay = math.log(2.0) / max(hl, 1.0)
    return np.exp(-decay * delta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-team SOT model (NB, per-side).")
    p.add_argument("--history-csv", default="simulations/SOT/data/sot_history.csv")
    p.add_argument("--lookback-days", type=int, default=_TS["lookback_days"])
    p.add_argument("--retrain-days", type=int, default=_TS["retrain_days"])
    p.add_argument("--min-team-home", type=int, default=_TS["min_team_home"])
    p.add_argument("--min-team-away", type=int, default=_TS["min_team_away"])
    p.add_argument("--model", choices=["poisson", "nb"], default="nb")
    p.add_argument("--out-team-profiles", default="simulations/SOT/data/sot_team_profiles.csv")
    p.add_argument("--out-league-params", default="simulations/SOT/data/sot_league_params.csv")
    p.add_argument("--out-predictions", default="simulations/SOT/backtests/sot_predictions.csv")
    p.add_argument("--out-summary", default="simulations/SOT/backtests/sot_summary.csv")
    p.add_argument("--out-calibration-params", default="simulations/SOT/data/sot_calibration.csv")
    return p.parse_args()


def _safe_mean(s: pd.Series, fallback: float) -> float:
    if s.empty:
        return float(fallback)
    x = s.dropna()
    if x.empty:
        return float(fallback)
    return float(x.mean())


def _estimate_nb_k(values: np.ndarray, default_k: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 20:
        return float(default_k)
    mu = float(np.mean(vals))
    var = float(np.var(vals, ddof=1))
    if var <= mu or mu <= 0:
        return float(default_k)
    k = (mu * mu) / (var - mu)
    return float(np.clip(k, _SC["k_min"], _SC["k_max"]))


def _prob_over_bk(line_bk: float, lam_our: float, model: str, k: float, league: str = "") -> float:
    """P(bookmaker_sot > line_bk) using scaled lambda. Scaling factor is per-league
    (with global fallback). Threshold is already at bookmaker scale."""
    scale = _scale_for(league)
    lam_bk = max(1e-6, float(lam_our) * scale)
    floor_t = int(np.floor(line_bk))
    if model == "poisson":
        return float(1.0 - poisson.cdf(floor_t, lam_bk))
    p = k / (k + lam_bk)
    return float(1.0 - nbinom.cdf(floor_t, k, p))


def _weighted_mean(values: pd.Series, weights: np.ndarray, fallback: float) -> float:
    """Weighted average with fallback when empty/all-NaN."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return float(fallback)
    mask = np.isfinite(arr)
    if not mask.any():
        return float(fallback)
    w = weights[mask]
    x = arr[mask]
    total_w = float(w.sum())
    if total_w <= 0:
        return float(np.mean(x))
    return float(np.sum(x * w) / total_w)


def _team_profiles(train: pd.DataFrame, anchor: Optional[dt.date] = None) -> pd.DataFrame:
    """Per-team home/away SOT scored and conceded averages.

    When `anchor` is provided and CFG['sot']['recency']['enabled'] is true, matches
    closer to `anchor` receive higher weight (exponential decay). This keeps recent
    form from being drowned out by historical samples."""
    if train.empty:
        return pd.DataFrame(columns=[
            "league", "team", "h_for", "h_against", "a_for", "a_against", "n_home", "n_away"
        ])

    anchor = anchor or train["match_date"].max().date()
    rows: list[dict] = []
    for league, lg in train.groupby("league"):
        teams = sorted(set(lg["home_team"]).union(set(lg["away_team"])))
        w_lg = _recency_weights(lg["match_date"], anchor)
        lg_h_for = _weighted_mean(lg["home_sot"], w_lg, _SC["default_mu_team"])
        lg_h_against = _weighted_mean(lg["away_sot"], w_lg, _SC["default_mu_team"])
        lg_a_for = _weighted_mean(lg["away_sot"], w_lg, _SC["default_mu_team"])
        lg_a_against = _weighted_mean(lg["home_sot"], w_lg, _SC["default_mu_team"])

        for team in teams:
            h = lg[lg["home_team"] == team]
            a = lg[lg["away_team"] == team]
            w_h = _recency_weights(h["match_date"], anchor) if len(h) else np.ones(0)
            w_a = _recency_weights(a["match_date"], anchor) if len(a) else np.ones(0)
            rows.append({
                "league": league,
                "team": team,
                "h_for": _weighted_mean(h["home_sot"], w_h, lg_h_for),
                "h_against": _weighted_mean(h["away_sot"], w_h, lg_h_against),
                "a_for": _weighted_mean(a["away_sot"], w_a, lg_a_for),
                "a_against": _weighted_mean(a["home_sot"], w_a, lg_a_against),
                "n_home": int(len(h)),
                "n_away": int(len(a)),
            })
    return pd.DataFrame(rows)


def _league_params(train: pd.DataFrame, global_mu_home: float, global_mu_away: float) -> pd.DataFrame:
    rows: list[dict] = []
    for league, lg in train.groupby("league"):
        h_vals = lg["home_sot"].astype(float).to_numpy()
        a_vals = lg["away_sot"].astype(float).to_numpy()
        mu_h = float(np.mean(h_vals)) if len(h_vals) else global_mu_home
        mu_a = float(np.mean(a_vals)) if len(a_vals) else global_mu_away
        k_h = _estimate_nb_k(h_vals, default_k=_SC["default_k"])
        k_a = _estimate_nb_k(a_vals, default_k=_SC["default_k"])
        tempo_h = (mu_h / global_mu_home) if global_mu_home > 0 else 1.0
        tempo_a = (mu_a / global_mu_away) if global_mu_away > 0 else 1.0
        rows.append({
            "league": league,
            "mu_home": mu_h,
            "mu_away": mu_a,
            "k_home": k_h,
            "k_away": k_a,
            "tempo_home": float(np.clip(tempo_h, _SC["tempo_min"], _SC["tempo_max"])),
            "tempo_away": float(np.clip(tempo_a, _SC["tempo_min"], _SC["tempo_max"])),
            "n_train": int(len(lg)),
        })
    return pd.DataFrame(rows)


def _predict_lambdas(
    profiles: pd.DataFrame,
    league_params: pd.DataFrame,
    league: str,
    home: str,
    away: str,
    min_team_home: int,
    min_team_away: int,
    elo_ratings: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Tuple[float, float, bool]:
    """Returns (lambda_home_sot, lambda_away_sot, ok) in Flashscore scale (pre-scaling).
    Applies Elo/class strength multiplier from team_ratings.pkl when enabled."""
    lp = league_params[league_params["league"] == league]
    if lp.empty:
        return np.nan, np.nan, False
    mu_h = float(lp.iloc[0]["mu_home"])
    mu_a = float(lp.iloc[0]["mu_away"])
    tempo_h = float(lp.iloc[0]["tempo_home"])
    tempo_a = float(lp.iloc[0]["tempo_away"])

    league_profiles = profiles[profiles["league"] == league]
    h = league_profiles[league_profiles["team"] == home]
    a = league_profiles[league_profiles["team"] == away]
    if h.empty or a.empty:
        return np.nan, np.nan, False

    hrow = h.iloc[0]
    arow = a.iloc[0]
    if int(hrow["n_home"]) < min_team_home or int(arow["n_away"]) < min_team_away:
        return np.nan, np.nan, False

    # Industry form: home SOT = (home_for@home + away_against@away) / 2
    lam_h = (float(hrow["h_for"]) + float(arow["a_against"])) / 2.0
    lam_a = (float(arow["a_for"]) + float(hrow["h_against"])) / 2.0
    lam_h = lam_h * tempo_h
    lam_a = lam_a * tempo_a

    # Fix #3: Elo/class strength adjustment (Dixon-Coles attack vs opponent defence).
    if elo_ratings is not None:
        lam_h *= _elo_multiplier(elo_ratings, league, home, away)
        lam_a *= _elo_multiplier(elo_ratings, league, away, home)

    lam_h = float(np.clip(lam_h, _SC["lambda_min"], _SC["lambda_max"]))
    lam_a = float(np.clip(lam_a, _SC["lambda_min"], _SC["lambda_max"]))
    # Blend with league mean
    lam_h = _SC["blend_empirical"] * lam_h + _SC["blend_league_mean"] * mu_h
    lam_a = _SC["blend_empirical"] * lam_a + _SC["blend_league_mean"] * mu_a
    return lam_h, lam_a, True


def _compute_predictions(
    pred: pd.DataFrame,
    profiles: pd.DataFrame,
    league_params: pd.DataFrame,
    args: argparse.Namespace,
    elo_ratings: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> list[dict]:
    """For each match in `pred`, compute raw over probabilities at each line x side."""
    k_map_h = {str(x["league"]): float(x["k_home"]) for _, x in league_params.iterrows()}
    k_map_a = {str(x["league"]): float(x["k_away"]) for _, x in league_params.iterrows()}
    out: list[dict] = []
    for _, m in pred.iterrows():
        league = str(m["league"])
        scale = _scale_for(league)
        lam_h, lam_a, ok = _predict_lambdas(
            profiles=profiles,
            league_params=league_params,
            league=league,
            home=str(m["home_team"]),
            away=str(m["away_team"]),
            min_team_home=args.min_team_home,
            min_team_away=args.min_team_away,
            elo_ratings=elo_ratings,
        )
        if not ok or not np.isfinite(lam_h) or not np.isfinite(lam_a):
            continue
        k_h = float(k_map_h.get(league, _SC["default_k"]))
        k_a = float(k_map_a.get(league, _SC["default_k"]))
        row = {
            "match_date": m["match_date"].date().isoformat(),
            "league": league,
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "lambda_home_sot": lam_h,
            "lambda_away_sot": lam_a,
            "scaling_factor": scale,
            "k_home": k_h,
            "k_away": k_a,
            "home_sot_actual": float(m["home_sot"]),
            "away_sot_actual": float(m["away_sot"]),
        }
        for line_bk in _LINES:
            line_our = line_bk / scale
            p_h = _prob_over_bk(line_bk, lam_h, args.model, k_h, league)
            p_a = _prob_over_bk(line_bk, lam_a, args.model, k_a, league)
            row[f"p_over_home_{line_bk}"] = p_h
            row[f"p_over_away_{line_bk}"] = p_a
            row[f"y_over_home_{line_bk}"] = float(float(m["home_sot"]) > line_our)
            row[f"y_over_away_{line_bk}"] = float(float(m["away_sot"]) > line_our)
        out.append(row)
    return out


def _fit_platt(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Fit Platt calibration per (league, side, line). Returns params DataFrame."""
    rows: list[dict] = []
    global_by_sideline: dict[tuple[str, float], tuple[list, list]] = {}

    for side in ("home", "away"):
        for line_bk in _LINES:
            p_col = f"p_over_{side}_{line_bk}"
            y_col = f"y_over_{side}_{line_bk}"
            global_by_sideline[(side, float(line_bk))] = ([], [])
            for lg in pred_df["league"].unique():
                sub = pred_df[pred_df["league"] == lg]
                ps = sub[p_col].to_numpy(dtype=float)
                ys = sub[y_col].to_numpy(dtype=float)
                if len(ps) < 30:
                    continue
                a, b = fit_platt_logit(ps, ys)
                rows.append({
                    "league": str(lg).upper(),
                    "side": side,
                    "line": float(line_bk),
                    "method": "platt",
                    "a": a,
                    "b": b,
                    "n_train": int(len(ps)),
                })
                global_by_sideline[(side, float(line_bk))][0].extend(ps.tolist())
                global_by_sideline[(side, float(line_bk))][1].extend(ys.tolist())

    for (side, line_bk), (ps, ys) in global_by_sideline.items():
        if len(ps) < 30:
            continue
        a, b = fit_platt_logit(np.array(ps), np.array(ys))
        rows.append({
            "league": "__GLOBAL__",
            "side": side,
            "line": float(line_bk),
            "method": "platt",
            "a": a,
            "b": b,
            "n_train": int(len(ps)),
        })
    return pd.DataFrame(rows)


def _apply_calibration(pred_df: pd.DataFrame, cal_df: pd.DataFrame) -> pd.DataFrame:
    """Add calibrated columns to pred_df in-place (returns copy)."""
    df = pred_df.copy()
    for side in ("home", "away"):
        for line_bk in _LINES:
            p_col = f"p_over_{side}_{line_bk}"
            cal_col = f"p_over_{side}_{line_bk}_cal"
            df[cal_col] = df[p_col].copy()
            for lg in df["league"].unique():
                mask = df["league"] == lg
                params = cal_df[
                    (cal_df["league"] == str(lg).upper())
                    & (cal_df["side"] == side)
                    & (cal_df["line"] == float(line_bk))
                ]
                if params.empty:
                    params = cal_df[
                        (cal_df["league"] == "__GLOBAL__")
                        & (cal_df["side"] == side)
                        & (cal_df["line"] == float(line_bk))
                    ]
                if params.empty:
                    continue
                a = float(params.iloc[0]["a"])
                b = float(params.iloc[0]["b"])
                df.loc[mask, cal_col] = apply_platt_logit(
                    df.loc[mask, p_col].to_numpy(dtype=float), a, b
                )
    return df


def _metrics_side_line(df: pd.DataFrame, side: str, line_bk: float, p_col_suffix: str = "") -> dict:
    p_col = f"p_over_{side}_{line_bk}{p_col_suffix}"
    y_col = f"y_over_{side}_{line_bk}"
    p = np.clip(df[p_col].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    y = df[y_col].to_numpy(dtype=float)
    if len(p) == 0:
        return {"n": 0}
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    br = float(np.mean((p - y) ** 2))
    hr = float(np.mean((p >= 0.5) == (y >= 0.5)))
    return {
        "side": side,
        "line_bookmaker": line_bk,
        "line_our_scale": round(line_bk / _SCALE, 3),
        "n": int(len(p)),
        "hit_rate": hr,
        "log_loss": ll,
        "brier": br,
        "p_mean": float(np.mean(p)),
        "y_mean": float(np.mean(y)),
    }


def main() -> int:
    args = parse_args()
    hist_path = Path(args.history_csv)
    if not hist_path.exists():
        raise RuntimeError(f"SOT history not found: {hist_path}")

    df = pd.read_csv(hist_path)
    required = {"league", "match_date", "home_team", "away_team", "home_sot", "away_sot"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    for c in ("league", "home_team", "away_team"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "home_sot", "away_sot", "league", "home_team", "away_team"]).copy()
    df["home_sot"] = pd.to_numeric(df["home_sot"], errors="coerce")
    df["away_sot"] = pd.to_numeric(df["away_sot"], errors="coerce")
    df = df.dropna(subset=["home_sot", "away_sot"]).copy()
    df = df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    elo_ratings = _load_elo_ratings()
    if elo_ratings:
        print(f"Loaded Elo ratings for {len(elo_ratings)} leagues")

    min_date = df["match_date"].min().date()
    max_date = df["match_date"].max().date()
    anchor = min_date + dt.timedelta(days=args.lookback_days)
    all_rows: list[dict] = []

    while anchor <= max_date:
        train_start = anchor - dt.timedelta(days=args.lookback_days)
        pred_end = anchor + dt.timedelta(days=args.retrain_days)

        train = df[(df["match_date"].dt.date >= train_start) & (df["match_date"].dt.date < anchor)].copy()
        pred = df[(df["match_date"].dt.date >= anchor) & (df["match_date"].dt.date < pred_end)].copy()
        if train.empty or pred.empty:
            anchor = pred_end
            continue

        global_mu_h = float(np.mean(train["home_sot"]))
        global_mu_a = float(np.mean(train["away_sot"]))
        profiles = _team_profiles(train, anchor=anchor)
        league_params = _league_params(train, global_mu_home=global_mu_h, global_mu_away=global_mu_a)
        all_rows.extend(_compute_predictions(pred, profiles, league_params, args, elo_ratings))
        anchor = pred_end

    pred_df = pd.DataFrame(all_rows)
    if pred_df.empty:
        raise RuntimeError("No predictions. Check history / min-team filters.")

    # Fit + apply Platt calibration
    cal_params_df = _fit_platt(pred_df)
    Path(args.out_calibration_params).parent.mkdir(parents=True, exist_ok=True)
    cal_params_df.to_csv(args.out_calibration_params, index=False)

    pred_df = _apply_calibration(pred_df, cal_params_df)
    Path(args.out_predictions).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.out_predictions, index=False)

    # Summary: raw + calibrated, per (side, line)
    summary_rows: list[dict] = []
    for side in ("home", "away"):
        for line_bk in _LINES:
            raw = _metrics_side_line(pred_df, side, float(line_bk))
            raw["scope"] = "raw"
            summary_rows.append(raw)
            cal = _metrics_side_line(pred_df, side, float(line_bk), p_col_suffix="_cal")
            cal["scope"] = "calibrated"
            summary_rows.append(cal)
    summary = pd.DataFrame(summary_rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    # Train latest artifacts for daily use
    latest_end = max_date
    latest_start = latest_end - dt.timedelta(days=args.lookback_days)
    latest_train = df[
        (df["match_date"].dt.date >= latest_start) & (df["match_date"].dt.date <= latest_end)
    ].copy()
    global_mu_h_latest = float(np.mean(latest_train["home_sot"])) if not latest_train.empty else _SC["default_mu_team"]
    global_mu_a_latest = float(np.mean(latest_train["away_sot"])) if not latest_train.empty else _SC["default_mu_team"]
    prof_latest = _team_profiles(latest_train, anchor=latest_end)
    lp_latest = _league_params(latest_train, global_mu_home=global_mu_h_latest, global_mu_away=global_mu_a_latest)
    Path(args.out_team_profiles).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_league_params).parent.mkdir(parents=True, exist_ok=True)
    prof_latest.to_csv(args.out_team_profiles, index=False)
    lp_latest.to_csv(args.out_league_params, index=False)

    print("SOT_PER_TEAM_SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nGlobal scaling: {_SCALE:.2f}x | Per-league overrides: {len(_SCALE_PER_LEAGUE)}")
    print(f"Recency: {'ON' if _RECENCY.get('enabled') else 'OFF'} (half-life {_RECENCY.get('half_life_days', '-')}d)")
    print(f"Elo: {'ON' if _ELO_CFG.get('enabled') else 'OFF'} (elasticity {_ELO_CFG.get('sot_elasticity', '-')})")
    print(f"Lines (bookmaker scale): {_LINES}")
    print(f"Saved predictions:   {args.out_predictions}")
    print(f"Saved summary:       {args.out_summary}")
    print(f"Saved cal params:    {args.out_calibration_params}")
    print(f"Saved profiles:      {args.out_team_profiles}")
    print(f"Saved league prm:    {args.out_league_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())