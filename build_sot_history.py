"""
build_sot_history.py

Extract Shots on Target (SOT) history from data/flashscore/match_stats.csv
into simulations/SOT/data/sot_history.csv — dedicated training-ready file
for the SOT Over/Under model.

Run after import_flashscore_stats.py refreshes match_stats.csv.
Idempotent — safe to run daily.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("sot_history")

SOURCE_CSV = Path("data/flashscore/match_stats.csv")
OUTPUT_CSV = Path("simulations/SOT/data/sot_history.csv")

REQUIRED_COLS = [
    "event_id",
    "league",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "home_shots_on_target",
    "away_shots_on_target",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SOT history for model training.")
    parser.add_argument("--source", default=str(SOURCE_CSV))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    source = Path(args.source)
    if not source.exists():
        raise RuntimeError(f"Source not found: {source}")

    df = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {source}: {missing}")

    df = df[REQUIRED_COLS].copy()
    df = df.rename(
        columns={
            "home_shots_on_target": "home_sot",
            "away_shots_on_target": "away_sot",
        }
    )

    # Clean types
    for c in ("league", "home_team", "away_team"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["home_sot"] = pd.to_numeric(df["home_sot"], errors="coerce")
    df["away_sot"] = pd.to_numeric(df["away_sot"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["match_date", "home_sot", "away_sot", "league", "home_team", "away_team"])
    LOG.info(f"Dropped {before - len(df)} rows with missing fields")

    # Dedupe on event_id, fallback on (league, match_date, home, away)
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    df = df.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"], keep="last")

    df["home_sot"] = df["home_sot"].astype(int)
    df["away_sot"] = df["away_sot"].astype(int)
    df = df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    LOG.info(f"Saved SOT history: {out_path}")
    LOG.info(f"  Rows: {len(df):,}")
    LOG.info(f"  Leagues: {df['league'].nunique()}")
    LOG.info(f"  Date range: {df['match_date'].min().date()} to {df['match_date'].max().date()}")
    totals = df["home_sot"] + df["away_sot"]
    LOG.info(f"  Total SOT mean: {totals.mean():.2f} (std: {totals.std():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())