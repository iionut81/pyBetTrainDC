from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd

from data_loader import (
    Fixture,
    fetch_fixtures_from_api,
    fetch_fixtures_from_flashscore,
    load_fixtures_from_json,
    load_team_ratings,
)
from decision_engine import evaluate_market
from dixon_coles import expected_goals, market_probabilities, resolve_team_strength, score_matrix
from simulation import run_monte_carlo

STRICT_LOW_VARIANCE_LEAGUES = {"E1", "RO1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Dixon-Coles probability engine with stability filter.")
    parser.add_argument(
        "--provider",
        choices=["api", "json", "flashscore", "auto"],
        default="auto",
        help="Fixture source provider.",
    )
    parser.add_argument(
        "--ratings-pkl", default="data/historical/team_ratings.pkl", help="Path to pre-trained team ratings pickle."
    )
    parser.add_argument(
        "--fixtures-api-url",
        default="",
        help="Fixture API endpoint returning JSON list (or {'fixtures':[...]})",
    )
    parser.add_argument(
        "--fixtures-json",
        default="",
        help="Fallback local fixtures JSON file for testing if API is not used.",
    )
    parser.add_argument("--api-key", default="", help="Optional bearer token for fixtures API.")
    parser.add_argument(
        "--target-date",
        default=dt.date.today().isoformat(),
        help="Target fixture date in YYYY-MM-DD (used by flashscore fallback and run metadata).",
    )
    parser.add_argument(
        "--output-csv",
        default="simulations/recommendations/1.2_Today_Recommendations.csv",
        help="Recommended picks output.",
    )
    parser.add_argument(
        "--all-matches-csv",
        default="simulations/evaluations/1.1_Today_Evaluations.csv",
        help="Full evaluated markets output.",
    )
    parser.add_argument("--max-goals", type=int, default=6, help="Max goals in Poisson score matrix.")
    parser.add_argument("--iterations", type=int, default=50000, help="Monte Carlo iterations per match.")
    parser.add_argument("--min-dc-probability", type=float, default=0.78)
    parser.add_argument("--min-odds", type=float, default=1.25)
    parser.add_argument("--max-odds", type=float, default=1.35)
    parser.add_argument("--default-home-advantage", type=float, default=0.0)
    parser.add_argument("--default-rho", type=float, default=-0.05)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for API requests.")
    return parser.parse_args()


def load_fixtures(args: argparse.Namespace) -> List[Fixture]:
    if args.provider == "flashscore":
        return fetch_fixtures_from_flashscore(target_date_iso=args.target_date, verify_ssl=not args.insecure)
    if args.provider == "json":
        if not args.fixtures_json:
            raise ValueError("--fixtures-json is required when --provider json.")
        return load_fixtures_from_json(args.fixtures_json)
    if args.provider == "api":
        if not args.fixtures_api_url:
            raise ValueError("--fixtures-api-url is required when --provider api.")
        return fetch_fixtures_from_api(
            api_url=args.fixtures_api_url,
            api_key=args.api_key or None,
            verify_ssl=not args.insecure,
        )
    # auto mode: API -> JSON -> Flashscore
    if args.fixtures_api_url:
        try:
            return fetch_fixtures_from_api(
                api_url=args.fixtures_api_url,
                api_key=args.api_key or None,
                verify_ssl=not args.insecure,
            )
        except Exception:
            pass
    if args.fixtures_json:
        try:
            return load_fixtures_from_json(args.fixtures_json)
        except Exception:
            pass
    return fetch_fixtures_from_flashscore(target_date_iso=args.target_date, verify_ssl=not args.insecure)


def main() -> None:
    args = parse_args()
    ratings = load_team_ratings(args.ratings_pkl)
    fixtures = load_fixtures(args)
    if not fixtures:
        print("No fixtures loaded.")
        return

    rows = []
    for fx in fixtures:
        allowed_variance_classes = {"LOW"} if fx.league.upper() in STRICT_LOW_VARIANCE_LEAGUES else {"LOW", "MEDIUM"}
        resolved = resolve_team_strength(
            ratings=ratings,
            league=fx.league,
            home_team=fx.home_team,
            away_team=fx.away_team,
            default_home_advantage=args.default_home_advantage,
            default_rho=args.default_rho,
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
        print("No matches could be evaluated (likely missing team ratings coverage).")
        return

    Path(args.all_matches_csv).parent.mkdir(parents=True, exist_ok=True)
    all_df = all_df.sort_values(["model_probability", "edge"], ascending=[False, False]).reset_index(drop=True)
    all_df.to_csv(args.all_matches_csv, index=False)

    rec_df = all_df[all_df["recommended"]].copy()
    rec_df = rec_df.sort_values(["model_probability", "edge"], ascending=[False, False]).reset_index(drop=True)
    rec_df.to_csv(args.output_csv, index=False)

    print(f"Evaluated markets: {len(all_df)}")
    print(f"Recommended picks: {len(rec_df)}")
    print(f"Saved all evaluations to: {args.all_matches_csv}")
    print(f"Saved recommendations to: {args.output_csv}")


if __name__ == "__main__":
    main()
