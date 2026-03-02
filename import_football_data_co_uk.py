from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


LEAGUES: Dict[str, str] = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "SP1": "Spain La Liga",
    "D1": "Germany Bundesliga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
    "N1": "Netherlands Eredivisie",
    "P1": "Portugal Primeira Liga",
}


def season_codes(n_seasons: int, end_season_code: str = "2526") -> List[str]:
    if len(end_season_code) != 4 or not end_season_code.isdigit():
        raise ValueError("end_season_code must look like 2526.")
    start_two = int(end_season_code[:2])
    return [f"{start_two - i:02d}{start_two - i + 1:02d}" for i in reversed(range(n_seasons))]


def parse_date(series: pd.Series) -> pd.Series:
    # football-data uses mixed date formats between files; dayfirst works for most historical rows.
    d1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    d2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return d1.fillna(d2).dt.date


def to_model_schema(df: pd.DataFrame, league_code: str, season_code: str) -> pd.DataFrame:
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "source": "football-data.co.uk",
            "league": league_code,
            "match_date": parse_date(df["Date"]),
            "home_team": (
                df["HomeTeam"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
            ),
            "away_team": (
                df["AwayTeam"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
            ),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            "odds_1x": pd.NA,
            "odds_x2": pd.NA,
            "season_code": season_code,
        }
    )
    out = out.dropna(subset=["match_date", "home_team", "away_team", "home_goals", "away_goals"]).copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    return out


def download_csv(session: requests.Session, season_code: str, league_code: str, timeout: int) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def load_local_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import football-data.co.uk CSVs (8 selected leagues, last 5 seasons by default) "
            "to Dixon-Coles training schema."
        )
    )
    parser.add_argument("--n-seasons", type=int, default=5, help="How many seasons per league.")
    parser.add_argument(
        "--end-season-code",
        default="2526",
        help="Season code for the latest season included, e.g. 2526 for 2025-26.",
    )
    parser.add_argument(
        "--input-dir",
        default="",
        help=(
            "Optional folder with manually downloaded CSV files named like 2526_E0.csv. "
            "If empty, script downloads directly from football-data.co.uk."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="historical_matches_football_data.csv",
        help="Path to write consolidated output CSV.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    seasons = season_codes(args.n_seasons, args.end_season_code)
    session = requests.Session()
    rows: List[pd.DataFrame] = []
    failures: List[str] = []
    input_dir = Path(args.input_dir) if args.input_dir else None

    for season in seasons:
        for league_code in LEAGUES:
            try:
                if input_dir is None:
                    df_raw = download_csv(session, season, league_code, timeout=args.timeout)
                else:
                    file_path = input_dir / f"{season}_{league_code}.csv"
                    df_raw = load_local_csv(file_path)
                shaped = to_model_schema(df_raw, league_code=league_code, season_code=season)
                rows.append(shaped)
                print(f"[OK] {season} {league_code}: {len(shaped)} matches")
            except Exception as exc:
                failures.append(f"{season} {league_code}: {exc}")
                print(f"[FAIL] {season} {league_code}: {exc}")

    if not rows:
        print("No data imported. Check network access or input file names.")
        if failures:
            print("Failures:")
            for f in failures:
                print(f"  - {f}")
        return 1

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"]).copy()
    all_df = all_df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
    all_df.to_csv(args.output_csv, index=False)

    print(f"\nWrote {len(all_df)} rows to {args.output_csv}")
    print(f"Leagues: {', '.join(LEAGUES.keys())}")
    print(f"Seasons: {', '.join(seasons)}")
    if failures:
        print("\nCompleted with partial failures:")
        for f in failures:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
