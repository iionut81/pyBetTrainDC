from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from data_loader import fetch_fixtures_from_api, load_team_ratings
from decision_engine import evaluate_market
from dixon_coles import expected_goals, market_probabilities, resolve_team_strength, score_matrix
from simulation import run_monte_carlo

STRICT_LOW_VARIANCE_LEAGUES = {"E1", "RO1"}

# Maps API-Football team names (lowercased) to Transfermarkt training names.
# Add entries whenever a new league causes team name mismatches.
TEAM_ALIASES: dict[str, str] = {
    # SP2 – Spain Segunda División
    "cultural leonesa": "cyd leonesa",
    "ud almeria": "ud almería",
    "malaga": "málaga cf",
    "cadiz": "cádiz cf",
    "cordoba": "córdoba cf",
    "castellon": "castellón",
    "sporting de gijon": "sporting gijón",
    "huesca": "sd huesca",
    # D2 – Germany 2. Bundesliga
    "karlsruher sc": "karlsruhe",
    "1. fc kaiserslautern": "kaiserslautern",
    "fc schalke 04": "schalke 04",
    "ssv jahn regensburg": "jahn regensburg",
    # I2 – Italy Serie B
    "spezia calcio": "spezia",
    "cosenza calcio": "cosenza",
    "us cremonese": "cremonese",
    "venezia fc": "venezia",
    # DK1 – Denmark Superliga
    "agf aarhus": "agf",
    "fc midtjylland": "midtjylland",
    "fc nordsjaelland": "nordsjaelland",
    "silkeborg if": "silkeborg",
    # SW1 – Switzerland Super League
    "fc zürich": "fc zurich",
    "bsc young boys": "young boys",
    "fc lugano": "lugano",
    "fc lausanne-sport": "lausanne-sport",
}


def _apply_team_alias(name: str) -> str:
    return TEAM_ALIASES.get(name.lower(), name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily Dixon-Coles Double Chance evaluations and recommendations.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--target-date", default=dt.date.today().isoformat())
    p.add_argument("--ratings-pkl", default="data/historical/team_ratings.pkl")
    p.add_argument("--series", default="1")
    p.add_argument("--min-dc-probability", type=float, default=0.78)
    p.add_argument("--min-odds", type=float, default=1.25)
    p.add_argument("--max-odds", type=float, default=1.35)
    p.add_argument("--max-goals", type=int, default=6)
    p.add_argument("--iterations", type=int, default=50000)
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ratings = load_team_ratings(args.ratings_pkl)

    api_url = f"https://v3.football.api-sports.io/fixtures?date={args.target_date}"
    fixtures = fetch_fixtures_from_api(api_url=api_url, api_key=args.api_key, verify_ssl=not args.insecure)
    if not fixtures:
        print("No fixtures loaded.")
        return 0

    rows = []
    for fx in fixtures:
        allowed_variance_classes = (
            {"LOW"} if fx.league.upper() in STRICT_LOW_VARIANCE_LEAGUES else {"LOW", "MEDIUM"}
        )
        resolved = resolve_team_strength(
            ratings=ratings,
            league=fx.league,
            home_team=_apply_team_alias(fx.home_team),
            away_team=_apply_team_alias(fx.away_team),
            default_home_advantage=0.0,
            default_rho=-0.05,
        )
        if resolved is None:
            continue

        home_strength, away_strength, league_params = resolved
        lam_home, lam_away = expected_goals(
            home=home_strength,
            away=away_strength,
            home_advantage=league_params.home_advantage,
        )
        mat = score_matrix(lam_home, lam_away, rho=league_params.rho, max_goals=args.max_goals)
        probs = market_probabilities(mat)
        mc = run_monte_carlo(lam_home, lam_away, iterations=args.iterations)

        eval_1x = evaluate_market(
            market="1X",
            model_probability=probs["1X"],
            variance_value=mc["variance_1X"],
            upset_frequency=mc["upset_1X"],
            offered_odds=fx.odds_1x,
            min_dc_probability=args.min_dc_probability,
            min_odds=args.min_odds,
            max_odds=args.max_odds,
            allowed_variance_classes=allowed_variance_classes,
        )
        eval_x2 = evaluate_market(
            market="X2",
            model_probability=probs["X2"],
            variance_value=mc["variance_X2"],
            upset_frequency=mc["upset_X2"],
            offered_odds=fx.odds_x2,
            min_dc_probability=args.min_dc_probability,
            min_odds=args.min_odds,
            max_odds=args.max_odds,
            allowed_variance_classes=allowed_variance_classes,
        )

        for market_eval in (eval_1x, eval_x2):
            rows.append(
                {
                    "run_date": dt.date.today().isoformat(),
                    "match_date": fx.match_date,
                    "league": fx.league,
                    "home_team": fx.home_team,
                    "away_team": fx.away_team,
                    "lambda_home": lam_home,
                    "lambda_away": lam_away,
                    "p_home_win": probs["home_win"],
                    "p_draw": probs["draw"],
                    "p_away_win": probs["away_win"],
                    **market_eval,
                }
            )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        print("No matches could be evaluated.")
        return 0

    all_df = all_df.sort_values(["model_probability", "edge"], ascending=[False, False]).reset_index(drop=True)
    rec_df = all_df[all_df["recommended"]].copy()
    rec_df = rec_df.sort_values(["model_probability", "edge"], ascending=[False, False]).reset_index(drop=True)

    eval_path = Path(f"simulations/evaluations/{args.series}.1_Today_Evaluations.csv")
    rec_path = Path(f"simulations/recommendations/{args.series}.2_Today_Recommendations.csv")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(eval_path, index=False)
    rec_df.to_csv(rec_path, index=False)

    print(f"Evaluated markets:  {len(all_df)}")
    print(f"Recommended picks:  {len(rec_df)}")
    print(f"Saved evaluations:  {eval_path}")
    print(f"Saved recommendations: {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
