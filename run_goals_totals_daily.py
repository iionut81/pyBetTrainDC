from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from data_loader import fetch_fixtures_from_api, load_team_ratings
from dixon_coles import expected_goals, resolve_team_strength, score_matrix
from fhg_calibration import apply_calibration, calibration_from_row

MARKETS = ["over_2_5", "under_3_5", "under_4_5"]

# Default recommendation thresholds per market
THRESHOLDS: Dict[str, Dict[str, float]] = {
    "over_2_5": {
        "min_prob": 0.63,
        "max_prob": 1.00,
        "min_odds": 1.40,
        "max_odds": 1.95,
        "max_fair_odds": 1.59,
    },
    "under_3_5": {
        "min_prob": 0.65,
        "max_prob": 0.85,  # tail above 0.85 is over-confident (+0.138 gap at 0.9-1.0)
        "min_odds": 1.25,
        "max_odds": 1.65,
        "max_fair_odds": 1.43,
    },
    "under_4_5": {
        "min_prob": 0.82,
        "max_prob": 0.93,  # avoid extreme tail (+0.043 gap at 0.9-1.0)
        "min_odds": 1.10,
        "max_odds": 1.35,
        "max_fair_odds": 1.22,
    },
}


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_goals_odds(bookmakers: list) -> Dict[str, Optional[float]]:
    """Parse Goals Over/Under odds (bet ID 5) from API-Football bookmakers list."""
    result: Dict[str, Optional[float]] = {m: None for m in MARKETS}
    for b in bookmakers:
        bets = b.get("bets") if isinstance(b, dict) else None
        if not isinstance(bets, list):
            continue
        for bet in bets:
            if not isinstance(bet, dict):
                continue
            bid = bet.get("id")
            bname = str(bet.get("name", "")).strip().lower()
            if bid != 5 and "goals over/under" not in bname and bname != "over/under":
                continue
            vals = bet.get("values", [])
            if not isinstance(vals, list):
                continue
            for v in vals:
                if not isinstance(v, dict):
                    continue
                label = str(v.get("value", "")).strip().lower()
                odd = _to_float(v.get("odd"))
                if odd is None or odd <= 0:
                    continue
                if label == "over 2.5":
                    result["over_2_5"] = odd
                elif label == "under 3.5":
                    result["under_3_5"] = odd
                elif label == "under 4.5":
                    result["under_4_5"] = odd
    return result


def _fetch_goals_odds(
    target_date: str,
    api_key: str,
    timeout: int = 20,
    verify_ssl: bool = True,
) -> Dict[int, Dict[str, Optional[float]]]:
    """Fetch Goals Over/Under odds for all fixtures on a given date."""
    headers = {"x-apisports-key": api_key}
    url = f"https://v3.football.api-sports.io/odds?date={target_date}"
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
        if resp.status_code != 200:
            print(f"[WARN] Goals odds endpoint returned HTTP {resp.status_code}")
            return {}
        po = resp.json()
        odds_raw = po.get("response", []) if isinstance(po, dict) else []
        if not isinstance(odds_raw, list):
            return {}
        out: Dict[int, Dict[str, Optional[float]]] = {}
        for item in odds_raw:
            if not isinstance(item, dict):
                continue
            fx = item.get("fixture", {})
            fid = fx.get("id") if isinstance(fx, dict) else None
            if fid is None:
                continue
            out[int(fid)] = _extract_goals_odds(item.get("bookmakers", []))
        print(f"Goals odds loaded: {len(out)} fixtures")
        return out
    except Exception as exc:
        print(f"[WARN] Could not fetch goals odds: {exc}")
        return {}


def _ou_probs(mat: np.ndarray) -> Dict[str, float]:
    n = mat.shape[0]
    idx = np.arange(n)
    ig, jg = np.meshgrid(idx, idx, indexing="ij")
    total = ig + jg
    return {
        "over_2_5": float(mat[total >= 3].sum()),
        "under_3_5": float(mat[total <= 3].sum()),
        "under_4_5": float(mat[total <= 4].sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily Goals Totals evaluations and recommendations.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--target-date", default=dt.date.today().isoformat())
    p.add_argument("--ratings-pkl", default="data/historical/team_ratings.pkl")
    p.add_argument("--calibration-csv", default="simulations/Goals/data/goals_calibration.csv")
    p.add_argument("--series", default="1")
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ratings = load_team_ratings(args.ratings_pkl)

    cal_path = Path(args.calibration_csv)
    cal_df = (
        pd.read_csv(cal_path)
        if cal_path.exists()
        else pd.DataFrame(columns=["league", "market", "method", "a", "b", "temperature"])
    )
    cal_map: Dict[Tuple[str, str], dict] = {}
    for _, r in cal_df.iterrows():
        lg = str(r["league"]).strip().upper()
        mk = str(r["market"]).strip()
        cal_map[(lg, mk)] = calibration_from_row(dict(r))

    global_cals = {
        market: cal_map.get(
            ("__GLOBAL__", market), {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0}
        )
        for market in MARKETS
    }

    # Fetch fixtures (also fetches DC odds internally, which we ignore here)
    api_url = f"https://v3.football.api-sports.io/fixtures?date={args.target_date}"
    fixtures = fetch_fixtures_from_api(api_url=api_url, api_key=args.api_key, verify_ssl=not args.insecure)
    if not fixtures:
        print("No fixtures loaded.")
        return 0

    # Fetch goals over/under odds separately
    goals_odds = _fetch_goals_odds(
        target_date=args.target_date,
        api_key=args.api_key,
        verify_ssl=not args.insecure,
    )

    rows: list[dict] = []
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
        mat = score_matrix(lambda_home=lam_h, lambda_away=lam_a, rho=league_params.rho, max_goals=10)
        probs = _ou_probs(mat)

        offered = goals_odds.get(fx.fixture_id, {}) if fx.fixture_id is not None else {}

        for market in MARKETS:
            p_raw = probs[market]
            calib = cal_map.get((fx.league.upper(), market), global_cals[market])
            p_cal = float(apply_calibration(np.array([p_raw], dtype=float), calib)[0])
            fair_odds = (1.0 / p_cal) if p_cal > 0 else None

            offered_odd = offered.get(market) if offered else None
            implied = (1.0 / offered_odd) if offered_odd is not None and offered_odd > 0 else None
            edge = (p_cal - implied) if implied is not None else None

            thresh = THRESHOLDS[market]
            in_prob_band = thresh["min_prob"] <= p_cal <= thresh["max_prob"]
            if offered_odd is not None:
                recommended = bool(
                    in_prob_band
                    and thresh["min_odds"] <= offered_odd <= thresh["max_odds"]
                    and edge is not None
                    and edge > 0
                )
                odds_source = "market"
            else:
                recommended = bool(
                    in_prob_band
                    and fair_odds is not None
                    and fair_odds <= thresh["max_fair_odds"]
                )
                odds_source = "missing"

            rows.append(
                {
                    "run_date": dt.date.today().isoformat(),
                    "match_date": fx.match_date,
                    "league": fx.league,
                    "home_team": fx.home_team,
                    "away_team": fx.away_team,
                    "market": market,
                    "lam_home": round(lam_h, 4),
                    "lam_away": round(lam_a, 4),
                    "p_raw": round(p_raw, 4),
                    "p_cal": round(p_cal, 4),
                    "fair_odds": round(fair_odds, 4) if fair_odds else None,
                    "offered_odds": offered_odd,
                    "implied_probability": round(implied, 4) if implied else None,
                    "edge": round(edge, 4) if edge is not None else None,
                    "odds_source": odds_source,
                    "recommended": recommended,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No goals rows evaluated.")
        return 0

    out = out.sort_values(
        ["market", "p_cal", "edge"], ascending=[True, False, False], na_position="last"
    ).reset_index(drop=True)
    rec = out[out["recommended"]].copy()

    eval_path = Path(f"simulations/Goals/evaluations/{args.series}.1_Goals_Evaluations.csv")
    rec_path = Path(f"simulations/Goals/recommendations/{args.series}.2_Goals_Recommendations.csv")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(eval_path, index=False)
    rec.to_csv(rec_path, index=False)

    print(f"Saved goals evaluations: {eval_path} rows={len(out)}")
    print(f"Saved goals recommendations: {rec_path} rows={len(rec)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
