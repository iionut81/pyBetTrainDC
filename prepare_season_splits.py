from __future__ import annotations

import argparse
import os
import re
from typing import Tuple

import pandas as pd


def season_start_year(season_code: str) -> int:
    m = re.match(r"^\s*(\d{4})\s*/\s*(\d{2,4})\s*$", str(season_code))
    if m:
        return int(m.group(1))
    m2 = re.match(r"^\s*(\d{2})(\d{2})\s*$", str(season_code))
    if m2:
        return 2000 + int(m2.group(1))
    raise ValueError(f"Unsupported season_code format: {season_code}")


def split_5_seasons(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    seasons = sorted(df["season_code"].dropna().astype(str).unique(), key=season_start_year)
    if len(seasons) < 5:
        raise ValueError(f"Need at least 5 seasons, found {len(seasons)}: {seasons}")
    last5 = seasons[-5:]
    train_seasons = set(last5[:3])
    val_seasons = {last5[3]}
    test_seasons = {last5[4]}

    train = df[df["season_code"].astype(str).isin(train_seasons)].copy()
    val = df[df["season_code"].astype(str).isin(val_seasons)].copy()
    test = df[df["season_code"].astype(str).isin(test_seasons)].copy()

    for part in (train, val, test):
        part.sort_values(["match_date", "league", "home_team"], inplace=True)
        part.reset_index(drop=True, inplace=True)
    return train, val, test, last5


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create strict train/validation/backtest splits from latest 5 seasons."
    )
    parser.add_argument(
        "--input-csv",
        default="data/historical/historical_matches_transfermarkt.csv",
        help="Input historical matches CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="simulations/splits",
        help="Output folder for split CSV files.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"season_code", "match_date", "league", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.date
    df = df.dropna(subset=["match_date", "season_code", "league", "home_team", "away_team"]).copy()

    train, val, test, last5 = split_5_seasons(df)
    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "validation.csv")
    test_path = os.path.join(args.out_dir, "backtest.csv")
    meta_path = os.path.join(args.out_dir, "split_meta.csv")

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    meta = pd.DataFrame(
        [
            {"role": "train", "seasons": ",".join(last5[:3]), "rows": len(train)},
            {"role": "validation", "seasons": last5[3], "rows": len(val)},
            {"role": "backtest", "seasons": last5[4], "rows": len(test)},
        ]
    )
    meta.to_csv(meta_path, index=False)

    print(f"Latest 5 seasons: {last5}")
    print(f"Train seasons: {last5[:3]} rows={len(train)}")
    print(f"Validation season: {last5[3]} rows={len(val)}")
    print(f"Backtest season: {last5[4]} rows={len(test)}")
    print(f"Wrote: {train_path}, {val_path}, {test_path}, {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
