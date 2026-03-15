from __future__ import annotations

"""
import_wta_history.py
Download WTA match CSVs from Jeff Sackmann's GitHub and build a combined
historical dataset with derived serve/return statistics.

Usage:
  python import_wta_history.py
  python import_wta_history.py --start-year 2015 --end-year 2024
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

REQUIRED_STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def download_season(year: int, timeout: int = 30) -> pd.DataFrame:
    url = f"{BASE_URL}/wta_matches_{year}.csv"
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        print(f"    [WARN] HTTP {resp.status_code} for {year}")
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), low_memory=False)
    df["source_year"] = year
    return df


def derive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived percentage columns for winner and loser."""
    eps = 1e-9

    # Winner serve stats
    df["w_1stServeIn_pct"] = df["w_1stIn"] / df["w_svpt"].clip(lower=eps)
    df["w_1stServeWon_pct"] = df["w_1stWon"] / df["w_1stIn"].clip(lower=eps)
    w_2nd_attempts = (df["w_svpt"] - df["w_1stIn"]).clip(lower=eps)
    df["w_2ndServeWon_pct"] = df["w_2ndWon"] / w_2nd_attempts
    df["w_aceRate"] = df["w_ace"] / df["w_SvGms"].clip(lower=eps)
    df["w_bpSaved_pct"] = df["w_bpSaved"] / df["w_bpFaced"].clip(lower=eps)
    # Winner return stats (derived from loser's serve)
    l_return_pts = df["l_svpt"] - df["l_1stWon"] - df["l_2ndWon"]
    df["w_returnPtsWon_pct"] = l_return_pts / df["l_svpt"].clip(lower=eps)
    df["w_bpConverted_pct"] = (df["l_bpFaced"] - df["l_bpSaved"]) / df["l_bpFaced"].clip(lower=eps)

    # Loser serve stats
    df["l_1stServeIn_pct"] = df["l_1stIn"] / df["l_svpt"].clip(lower=eps)
    df["l_1stServeWon_pct"] = df["l_1stWon"] / df["l_1stIn"].clip(lower=eps)
    l_2nd_attempts = (df["l_svpt"] - df["l_1stIn"]).clip(lower=eps)
    df["l_2ndServeWon_pct"] = df["l_2ndWon"] / l_2nd_attempts
    df["l_aceRate"] = df["l_ace"] / df["l_SvGms"].clip(lower=eps)
    df["l_bpSaved_pct"] = df["l_bpSaved"] / df["l_bpFaced"].clip(lower=eps)
    # Loser return stats (derived from winner's serve)
    w_return_pts = df["w_svpt"] - df["w_1stWon"] - df["w_2ndWon"]
    df["l_returnPtsWon_pct"] = w_return_pts / df["w_svpt"].clip(lower=eps)
    df["l_bpConverted_pct"] = (df["w_bpFaced"] - df["w_bpSaved"]) / df["w_bpFaced"].clip(lower=eps)

    return df


def parse_score_games(score: str) -> int:
    """Parse total games from a score string like '6-3 7-5'."""
    if not isinstance(score, str):
        return 0
    total = 0
    for part in score.replace("[", "").replace("]", "").split():
        nums = part.split("-")
        if len(nums) == 2:
            try:
                total += int(nums[0]) + int(nums[1])
            except ValueError:
                continue
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import WTA match history from Sackmann GitHub.")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--output-csv", default="data/historical/wta_matches_combined.csv")
    p.add_argument("--include-qual", action="store_true",
                    help="Also download qualifying/ITF files (mostly missing stats)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Importing WTA matches {args.start_year}-{args.end_year}")
    frames: List[pd.DataFrame] = []
    for year in range(args.start_year, args.end_year + 1):
        df = download_season(year)
        if not df.empty:
            frames.append(df)
            print(f"    {year}: {len(df)} matches")

    if not frames:
        print("No data downloaded.")
        return 1

    raw = pd.concat(frames, ignore_index=True)
    print(f"\nTotal raw matches: {len(raw)}")

    # Parse date
    raw["match_date"] = pd.to_datetime(raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    # Filter to matches with complete serve stats
    before = len(raw)
    raw = raw.dropna(subset=REQUIRED_STAT_COLS).copy()
    after = len(raw)
    print(f"Matches with complete serve stats: {after} (dropped {before - after})")

    # Derive percentage stats
    raw = derive_stats(raw)

    # Parse total games from score
    raw["total_games"] = raw["score"].apply(parse_score_games)

    # Parse sets won
    def sets_won(score, side="winner"):
        if not isinstance(score, str):
            return 0
        w, l = 0, 0
        for part in score.replace("[", "").replace("]", "").split():
            nums = part.split("-")
            if len(nums) == 2:
                try:
                    a, b = int(nums[0]), int(nums[1])
                    if a > b:
                        w += 1
                    elif b > a:
                        l += 1
                except ValueError:
                    continue
        return w if side == "winner" else l

    raw["w_sets"] = raw["score"].apply(lambda s: sets_won(s, "winner"))
    raw["l_sets"] = raw["score"].apply(lambda s: sets_won(s, "loser"))

    # Dedup on tourney_id + match_num
    raw = raw.drop_duplicates(subset=["tourney_id", "match_num"], keep="last")

    # Select and rename key columns for clarity
    out = raw[[
        "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date",
        "match_date", "round", "best_of", "minutes",
        "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc",
        "winner_age", "winner_rank", "winner_rank_points",
        "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc",
        "loser_age", "loser_rank", "loser_rank_points",
        "score", "total_games", "w_sets", "l_sets",
        # Raw serve counts
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        # Derived percentages
        "w_1stServeIn_pct", "w_1stServeWon_pct", "w_2ndServeWon_pct",
        "w_aceRate", "w_bpSaved_pct", "w_returnPtsWon_pct", "w_bpConverted_pct",
        "l_1stServeIn_pct", "l_1stServeWon_pct", "l_2ndServeWon_pct",
        "l_aceRate", "l_bpSaved_pct", "l_returnPtsWon_pct", "l_bpConverted_pct",
    ]].copy()

    out = out.sort_values("match_date").reset_index(drop=True)

    # Save
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Summary
    surfaces = out["surface"].value_counts()
    years = out["match_date"].dt.year.value_counts().sort_index()
    print(f"\nSaved: {out_path}")
    print(f"Total matches: {len(out)}")
    print(f"\nBy surface:\n{surfaces.to_string()}")
    print(f"\nBy year:\n{years.to_string()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
