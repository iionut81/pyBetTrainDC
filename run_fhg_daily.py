from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import fetch_fixtures_from_api, load_team_ratings
from dixon_coles import expected_goals, resolve_team_strength
from fhg_calibration import apply_platt_logit
from fhg_weibull import p_goal_before_45


def implied_prob(odds: float | None) -> float | None:
    if odds is None or odds <= 0:
        return None
    return 1.0 / odds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily isolated FHG (first-half goal) evaluations and recommendations.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--target-date", default=dt.date.today().isoformat())
    p.add_argument("--ratings-pkl", default="data/historical/team_ratings.pkl")
    p.add_argument("--ratios-csv", default="simulations/FHG/data/fhg_league_ratios.csv")
    p.add_argument("--calibration-csv", default="simulations/FHG/data/fhg_calibration.csv")
    p.add_argument("--league-bias-csv", default="simulations/FHG/data/fhg_league_bias.csv")
    p.add_argument("--min-prob", type=float, default=0.78)
    p.add_argument("--min-odds", type=float, default=1.10)
    p.add_argument("--max-odds", type=float, default=1.25)
    p.add_argument("--series", default="1")
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_url = f"https://v3.football.api-sports.io/fixtures?date={args.target_date}"
    ratings = load_team_ratings(args.ratings_pkl)
    fixtures = fetch_fixtures_from_api(api_url=api_url, api_key=args.api_key, verify_ssl=not args.insecure)
    if not fixtures:
        print("No fixtures loaded.")
        return 0

    ratios = pd.read_csv(args.ratios_csv) if Path(args.ratios_csv).exists() else pd.DataFrame(columns=["league", "k_estimate"])
    k_map = {str(r["league"]).strip().upper(): float(r.get("k_estimate", 1.2)) for _, r in ratios.iterrows()}
    cal = (
        pd.read_csv(args.calibration_csv)
        if Path(args.calibration_csv).exists()
        else pd.DataFrame(columns=["league", "a", "b"])
    )
    cal_map = {
        str(r["league"]).strip().upper(): (float(r.get("a", 0.0)), float(r.get("b", 1.0)))
        for _, r in cal.iterrows()
    }
    global_ab = cal_map.get("__GLOBAL__", (0.0, 1.0))
    bias_df = (
        pd.read_csv(args.league_bias_csv)
        if Path(args.league_bias_csv).exists()
        else pd.DataFrame(columns=["league", "bias"])
    )
    bias_map = {str(x["league"]).upper(): float(x["bias"]) for _, x in bias_df.iterrows()}

    rows = []
    for fx in fixtures:
        resolved = resolve_team_strength(
            ratings=ratings,
            league=fx.league,
            home_team=fx.home_team,
            away_team=fx.away_team,
            default_home_advantage=0.0,
            default_rho=-0.05,
        )
        if resolved is None:
            continue
        home_s, away_s, league_params = resolved
        lam_h, lam_a = expected_goals(home=home_s, away=away_s, home_advantage=league_params.home_advantage)
        xg_total = float(lam_h + lam_a)
        k = float(k_map.get(fx.league.upper(), 1.2))
        p_fhg_raw = p_goal_before_45(expected_total_goals=xg_total, k=k)
        a, b = cal_map.get(fx.league.upper(), global_ab)
        p_fhg_cal = float(apply_platt_logit(p_fhg_raw, a=a, b=b))
        # Keep calibrated probability untouched; use league bias only as edge/ranking amplifier.
        p_fhg = p_fhg_cal
        bias = float(bias_map.get(fx.league.upper(), 1.0))

        # We reuse available market odds column (1X) as a placeholder only if true FHG odds are unavailable.
        offered_odds = fx.odds_1x
        imp = implied_prob(offered_odds)
        edge = None if imp is None else (p_fhg - imp)
        edge_weighted = None if edge is None else (edge * bias)

        recommended = bool(
            offered_odds is not None
            and args.min_odds <= offered_odds <= args.max_odds
            and p_fhg >= args.min_prob
            and edge_weighted is not None
            and edge_weighted > 0
        )
        rows.append(
            {
                "run_date": dt.date.today().isoformat(),
                "match_date": fx.match_date,
                "league": fx.league,
                "home_team": fx.home_team,
                "away_team": fx.away_team,
                "xg_total": xg_total,
                "k_estimate": k,
                "p_goal_before_45_raw": p_fhg_raw,
                "p_goal_before_45_cal": p_fhg_cal,
                "league_bias": bias,
                "p_goal_before_45": p_fhg,
                "offered_odds": offered_odds,
                "implied_probability": imp,
                "edge": edge,
                "edge_weighted": edge_weighted,
                "recommended": recommended,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No FHG rows evaluated.")
        return 0
    out = out.sort_values(["p_goal_before_45", "edge"], ascending=[False, False]).reset_index(drop=True)
    rec = out[out["recommended"]].copy()

    eval_path = Path(f"simulations/FHG/evaluations/{args.series}.1_FHG_Evaluations.csv")
    rec_path = Path(f"simulations/FHG/recommendations/{args.series}.2_FHG_Recommendations.csv")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(eval_path, index=False)
    rec.to_csv(rec_path, index=False)
    print(f"Saved FHG evaluations: {eval_path} rows={len(out)}")
    print(f"Saved FHG recommendations: {rec_path} rows={len(rec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
