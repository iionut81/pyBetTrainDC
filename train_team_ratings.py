from __future__ import annotations

import argparse
import datetime as dt
import pickle
from typing import Dict

import pandas as pd

from dc_double_chance import DixonColesModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Train league-wise Dixon-Coles ratings and export team_ratings.pkl")
    parser.add_argument("--history-csv", default="data/historical/historical_matches_transfermarkt.csv")
    parser.add_argument("--output-pkl", default="data/historical/team_ratings.pkl")
    parser.add_argument("--lookback-days", type=int, default=270)
    parser.add_argument("--decay-xi", type=float, default=0.0025)
    parser.add_argument(
        "--reference-date",
        default="",
        help="Optional YYYY-MM-DD. If empty, uses max match_date in history.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.history_csv)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    for c in ("league", "home_team", "away_team"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["match_date", "home_goals", "away_goals", "league", "home_team", "away_team"]).copy()
    if df.empty:
        raise ValueError("No usable rows in history CSV.")

    ref_date = dt.date.fromisoformat(args.reference_date) if args.reference_date else df["match_date"].max()
    min_date = ref_date - dt.timedelta(days=args.lookback_days)
    train = df[(df["match_date"] >= min_date) & (df["match_date"] <= ref_date)].copy()

    out: Dict[str, object] = {"leagues": {}}
    for league in sorted(train["league"].unique()):
        ldf = train[train["league"] == league].copy()
        n_teams = len(set(ldf["home_team"]).union(set(ldf["away_team"])))
        if n_teams < 6 or len(ldf) < 35:
            continue
        model = DixonColesModel(max_goals=10)
        try:
            model.fit(
                ldf[["home_team", "away_team", "home_goals", "away_goals", "match_date"]],
                decay_xi=args.decay_xi,
                reference_date=ref_date,
            )
        except Exception:
            continue

        teams = {}
        for i, t in enumerate(model.teams):
            teams[t] = {"attack": float(model.attack[i]), "defence": float(model.defense[i])}
        out["leagues"][league] = {
            "home_advantage": float(model.home_adv),
            "rho": float(model.rho),
            "teams": teams,
        }

    with open(args.output_pkl, "wb") as fh:
        pickle.dump(out, fh)

    n_leagues = len(out["leagues"])
    n_teams = sum(len(v["teams"]) for v in out["leagues"].values())
    print(f"Saved {args.output_pkl} with {n_leagues} leagues and {n_teams} teams (reference date {ref_date}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
