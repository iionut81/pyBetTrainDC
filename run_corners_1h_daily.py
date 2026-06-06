from __future__ import annotations

"""Daily First-Half Corners Under 7.5 evaluations and recommendations.

Analog to run_corners_daily.py but operates on 1H corner model artifacts.

Usage:
    PYTHONIOENCODING=utf-8 python run_corners_1h_daily.py --api-key dummy --insecure
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from config import CFG
from data_loader import clean_output_dir, fetch_fixtures_from_api
from fhg_calibration import apply_calibration, calibration_from_row
from team_registry import resolve_team_or_warn

_CC = CFG["corners_1h"]

LINE = 7   # Under 7.5 means 1H total corners <= 7


def _norm_team(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prob_under(lam: float, model: str, k: float) -> float:
    lam = max(1e-6, float(lam))
    if model == "poisson":
        return float(poisson.cdf(LINE, lam))
    p = k / (k + lam)
    return float(nbinom.cdf(LINE, k, p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily Corners Under 7.5 1H evaluations and recommendations."
    )
    p.add_argument("--api-key", required=True)
    p.add_argument("--target-date", default=dt.date.today().isoformat())
    p.add_argument("--profiles-csv",    default="simulations/Corners 1H/data/corners_1h_team_profiles.csv")
    p.add_argument("--league-params-csv", default="simulations/Corners 1H/data/corners_1h_league_params.csv")
    p.add_argument("--calibration-csv",  default="simulations/Corners 1H/data/corners_1h_calibration.csv")
    p.add_argument("--odds-csv", default="", help="Optional CSV: league,home_team,away_team,odds_under_7_5_1h")
    p.add_argument("--model", choices=["poisson", "nb"], default="nb")
    p.add_argument("--min-prob",       type=float, default=_CC["min_probability"])
    p.add_argument("--min-odds",       type=float, default=_CC["min_odds"])
    p.add_argument("--max-odds",       type=float, default=_CC["max_odds"])
    p.add_argument("--max-fair-odds",  type=float, default=_CC["max_fair_odds"])
    p.add_argument("--series", default="1")
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def _load_odds_map(path: str) -> Dict[Tuple[str, str, str], float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    required = {"league", "home_team", "away_team", "odds_under_7_5_1h"}
    if not required.issubset(df.columns):
        return {}
    out: Dict[Tuple[str, str, str], float] = {}
    for _, r in df.iterrows():
        odd = _to_float(r.get("odds_under_7_5_1h"))
        if odd is None or odd <= 0:
            continue
        key = (str(r["league"]).strip().upper(),
               _norm_team(r["home_team"]),
               _norm_team(r["away_team"]))
        out[key] = odd
    return out


def main() -> int:
    args = parse_args()
    profiles_path    = Path(args.profiles_csv)
    league_params_path = Path(args.league_params_csv)

    if not profiles_path.exists() or not league_params_path.exists():
        raise RuntimeError(
            "Missing trained 1H corners artifacts. Run train_corners_1h.py first."
        )

    profiles     = pd.read_csv(profiles_path)
    league_params = pd.read_csv(league_params_path)
    if profiles.empty or league_params.empty:
        raise RuntimeError("Empty 1H corners profiles or league params.")

    for c in ("league", "team"):
        profiles[c] = profiles[c].astype(str).str.strip().str.lower()
    league_params["league"] = league_params["league"].astype(str).str.strip().str.upper()

    profiles["team"] = profiles.apply(
        lambda row: resolve_team_or_warn(row["team"], row["league"].upper()), axis=1
    )

    cal_path = Path(args.calibration_csv)
    cal_df   = pd.read_csv(cal_path) if cal_path.exists() else pd.DataFrame(
        columns=["league", "method", "a", "b"]
    )
    cal_map = {
        str(r["league"]).strip().upper(): calibration_from_row(dict(r))
        for _, r in cal_df.iterrows()
    }
    global_cal = cal_map.get("__GLOBAL__", {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0})

    odds_map = _load_odds_map(args.odds_csv)

    api_url  = f"https://v3.football.api-sports.io/fixtures?date={args.target_date}"
    fixtures = fetch_fixtures_from_api(api_url=api_url, api_key=args.api_key,
                                       verify_ssl=not args.insecure)
    if not fixtures:
        print("No fixtures loaded.")
        return 0

    allowed = [l.strip().upper() for l in _CC.get("allowed_leagues", [])]

    rows: list[dict] = []
    for fx in fixtures:
        league   = str(fx.league).strip().upper()
        home_raw = _norm_team(fx.home_team)
        away_raw = _norm_team(fx.away_team)
        home     = resolve_team_or_warn(home_raw, league)
        away     = resolve_team_or_warn(away_raw, league)

        if allowed and league not in allowed:
            continue

        lp = league_params[league_params["league"] == league]
        if lp.empty:
            continue
        mu    = float(lp.iloc[0].get("mu_total_1h", _CC["default_mu"]))
        k     = float(lp.iloc[0].get("k_dispersion", _CC["default_k"]))
        tempo = float(lp.iloc[0].get("tempo_factor", 1.0))

        pp = profiles[profiles["league"] == league.lower()]
        h  = pp[pp["team"] == home]
        a  = pp[pp["team"] == away]
        if h.empty or a.empty:
            missing = []
            if h.empty: missing.append(f"home={home!r}")
            if a.empty: missing.append(f"away={away!r}")
            print(f"  [SKIP] {league} {home_raw} vs {away_raw} — no profile for {', '.join(missing)}")
            continue

        hrow, arow = h.iloc[0], a.iloc[0]
        lam_base = (
            float(hrow["h_for"]) + float(arow["a_against"]) +
            float(arow["a_for"]) + float(hrow["h_against"])
        ) / 2.0
        lam = float(_CC["blend_empirical"] * (lam_base * tempo) +
                    _CC["blend_league_mean"] * mu)
        lam = float(np.clip(lam, _CC["lambda_min"], _CC["lambda_max"]))

        p_raw = _prob_under(lam=lam, model=args.model, k=k)
        calib = cal_map.get(league, global_cal)
        p_cal = float(apply_calibration(np.array([p_raw], dtype=float), calib)[0])
        fair_odds = (1.0 / p_cal) if p_cal > 0 else None

        offered = odds_map.get((league, home, away))
        implied = (1.0 / offered) if offered is not None and offered > 0 else None
        edge    = (p_cal - implied) if implied is not None else None

        if offered is not None:
            recommended = bool(
                args.min_odds <= offered <= args.max_odds
                and p_cal >= args.min_prob
                and edge is not None and edge > 0
            )
            odds_source = "market"
        else:
            recommended = bool(
                p_cal >= args.min_prob
                and fair_odds is not None
                and fair_odds <= args.max_fair_odds
            )
            odds_source = "missing"

        rows.append({
            "run_date":             dt.date.today().isoformat(),
            "match_date":           fx.match_date,
            "league":               league,
            "home_team":            home,
            "away_team":            away,
            "model":                args.model,
            "lambda_1h":            lam,
            "k_dispersion":         k,
            "p_under_7_5_raw":      p_raw,
            "p_under_7_5_cal":      p_cal,
            "p_under_7_5":          p_cal,
            "fair_odds_under_7_5":  fair_odds,
            "offered_odds":         offered,
            "implied_probability":  implied,
            "edge":                 edge,
            "odds_source":          odds_source,
            "recommended":          recommended,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No 1H corners rows evaluated.")
        return 0

    out = out.sort_values(
        ["p_under_7_5", "edge"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    rec = out[out["recommended"]].copy()

    eval_dir = Path(f"simulations/Corners 1H/evaluations")
    rec_dir  = Path(f"simulations/Corners 1H/recommendations")
    eval_path = eval_dir / f"{args.series}.1_Corners_1H_Evaluations.csv"
    rec_path  = rec_dir  / f"{args.series}.2_Corners_1H_Recommendations.csv"

    for d in (eval_dir, rec_dir):
        d.mkdir(parents=True, exist_ok=True)
    clean_output_dir(eval_dir)
    clean_output_dir(rec_dir)

    out.to_csv(eval_path, index=False)
    rec.to_csv(rec_path,  index=False)

    print(f"Saved 1H corners evaluations:    {eval_path}  rows={len(out)}")
    print(f"Saved 1H corners recommendations: {rec_path}  rows={len(rec)}")

    if not rec.empty:
        print("\nRecommendations:")
        for _, r in rec.iterrows():
            print(
                f"  {r['league']:4s}  {r['home_team']} vs {r['away_team']}"
                f"  lam={r['lambda_1h']:.2f}  p={r['p_under_7_5']:.1%}"
                f"  fair={r['fair_odds_under_7_5']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
