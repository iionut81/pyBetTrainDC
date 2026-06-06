"""
analyze_backtest.py
Analyzes BAcktest.csv football match data and computes key statistics.
"""

import sys
import os
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BAcktest.csv")


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded {len(df):,} rows from {path}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at '{path}'")
        print("Please save BAcktest.csv to the same folder as this script and re-run.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        sys.exit(1)


def safe_numeric(series: pd.Series, invalid: float = -1) -> pd.Series:
    """Convert to numeric and mask invalid sentinel values."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s != invalid, other=np.nan)
    return s


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{100.0 * numerator / denominator:.1f}%"


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    raw = load_data(CSV_PATH)

    # ------------------------------------------------------------------ #
    # 1. Filter: complete matches with valid goal columns
    # ------------------------------------------------------------------ #
    section("1. DATA FILTERING")

    # Normalise column names (strip whitespace)
    raw.columns = [c.strip() for c in raw.columns]

    # Match Status filter
    if "Match Status" in raw.columns:
        df = raw[raw["Match Status"].astype(str).str.strip().str.lower() == "complete"].copy()
        print(f"Rows with Match Status == 'complete': {len(df):,}")
    else:
        df = raw.copy()
        print("WARNING: 'Match Status' column not found — using all rows.")

    # Goal columns
    home_goals_col = "Result - Home Team Goals"
    away_goals_col = "Result - Away Team Goals"

    for col in [home_goals_col, away_goals_col]:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' not found in CSV.")
            sys.exit(1)

    df["hg"] = safe_numeric(df[home_goals_col])
    df["ag"] = safe_numeric(df[away_goals_col])
    df = df.dropna(subset=["hg", "ag"])
    print(f"Rows after removing invalid goals (-1 / NaN): {len(df):,}")

    total = len(df)
    if total == 0:
        print("No valid matches found. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # 2. Total matches
    # ------------------------------------------------------------------ #
    section("2. TOTAL MATCHES")
    print(f"Total valid matches: {total:,}")

    # ------------------------------------------------------------------ #
    # 3. Goals distribution
    # ------------------------------------------------------------------ #
    section("3. GOALS DISTRIBUTION")

    df["total_goals"] = df["hg"] + df["ag"]
    avg_goals = df["total_goals"].mean()
    print(f"Average goals per match : {avg_goals:.3f}")
    print()

    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over_n = (df["total_goals"] > threshold).sum()
        print(f"  Over {threshold:.1f}  : {over_n:,} / {total:,}  ({pct(over_n, total)})")

    # Compare with model averages if columns exist
    print()
    avg_cols = {
        "Average Goals": "Avg Goals (pre-match model)",
        "Over05 Average": "Model Over 0.5",
        "Over15 Average": "Model Over 1.5",
        "Over25 Average": "Model Over 2.5",
        "Over35 Average": "Model Over 3.5",
        "Over45 Average": "Model Over 4.5",
    }
    for col, label in avg_cols.items():
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").mean()
            print(f"  {label:<30} : {val:.3f}")

    # ------------------------------------------------------------------ #
    # 4. BTTS
    # ------------------------------------------------------------------ #
    section("4. BTTS (Both Teams Scored)")

    btts_actual = ((df["hg"] >= 1) & (df["ag"] >= 1)).sum()
    print(f"BTTS actual hit rate    : {btts_actual:,} / {total:,}  ({pct(btts_actual, total)})")

    if "BTTS Average" in df.columns:
        btts_model_avg = pd.to_numeric(df["BTTS Average"], errors="coerce").mean()
        print(f"BTTS model average      : {btts_model_avg:.3f} ({btts_model_avg*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # 5. Corners
    # ------------------------------------------------------------------ #
    section("5. CORNERS")

    hc_col = "Home Team Corners"
    ac_col = "Away Team Corners"

    if hc_col in df.columns and ac_col in df.columns:
        df["hc"] = safe_numeric(df[hc_col])
        df["ac"] = safe_numeric(df[ac_col])
        corners_df = df.dropna(subset=["hc", "ac"]).copy()
        corners_df["total_corners"] = corners_df["hc"] + corners_df["ac"]
        n_corners = len(corners_df)

        print(f"Matches with valid corner data : {n_corners:,}")
        if n_corners > 0:
            avg_corners = corners_df["total_corners"].mean()
            print(f"Average total corners          : {avg_corners:.3f}")
            print()

            print("  Under thresholds:")
            for t in [10.5, 11.5, 12.5]:
                under_n = (corners_df["total_corners"] < t).sum()
                print(f"    Under {t:.1f}: {under_n:,} / {n_corners:,}  ({pct(under_n, n_corners)})")

            print("  Over thresholds:")
            for t in [6.5, 7.5, 8.5, 9.5]:
                over_n = (corners_df["total_corners"] > t).sum()
                print(f"    Over  {t:.1f}: {over_n:,} / {n_corners:,}  ({pct(over_n, n_corners)})")

            # Model corner averages
            print()
            corner_model_cols = {
                "Average Over 8.5 Corners": "Model Over 8.5",
                "Average Over 9.5 Corners": "Model Over 9.5",
                "Average Over 10.5 Corners": "Model Over 10.5",
            }
            for col, label in corner_model_cols.items():
                if col in corners_df.columns:
                    val = pd.to_numeric(corners_df[col], errors="coerce").mean()
                    print(f"  {label:<28} : {val:.3f} ({val*100:.1f}%)")
    else:
        print("Corner columns not found — skipping corners analysis.")

    # ------------------------------------------------------------------ #
    # 6. Match outcomes
    # ------------------------------------------------------------------ #
    section("6. MATCH OUTCOMES")

    home_wins = (df["hg"] > df["ag"]).sum()
    draws      = (df["hg"] == df["ag"]).sum()
    away_wins  = (df["hg"] < df["ag"]).sum()

    print(f"Home win  : {home_wins:,}  ({pct(home_wins, total)})")
    print(f"Draw      : {draws:,}  ({pct(draws, total)})")
    print(f"Away win  : {away_wins:,}  ({pct(away_wins, total)})")

    # ------------------------------------------------------------------ #
    # 7. Country distribution (top 10)
    # ------------------------------------------------------------------ #
    section("7. COUNTRY DISTRIBUTION (Top 10)")

    country_col = "Country"
    if country_col in df.columns:
        top_countries = (
            df[country_col]
            .astype(str)
            .str.strip()
            .value_counts()
            .head(10)
        )
        max_c = top_countries.max()
        for country, count in top_countries.items():
            bar = "#" * int(30 * count / max_c)
            print(f"  {country:<25} {count:>6,}  ({pct(count, total)})  {bar}")
    else:
        print("'Country' column not found — skipping.")

    # ------------------------------------------------------------------ #
    # 8. xG vs Actual Goals
    # ------------------------------------------------------------------ #
    section("8. xG vs ACTUAL GOALS")

    home_xg_col = "Home Team Pre-Match xG"
    away_xg_col = "Away Team Pre-Match xG"

    if home_xg_col in df.columns and away_xg_col in df.columns:
        df["home_xg"] = safe_numeric(df[home_xg_col])
        df["away_xg"] = safe_numeric(df[away_xg_col])
        xg_df = df.dropna(subset=["home_xg", "away_xg"]).copy()
        xg_df["total_xg"] = xg_df["home_xg"] + xg_df["away_xg"]
        n_xg = len(xg_df)

        print(f"Matches with valid xG data     : {n_xg:,}")
        if n_xg > 0:
            print(f"Average home xG                : {xg_df['home_xg'].mean():.3f}")
            print(f"Average away xG                : {xg_df['away_xg'].mean():.3f}")
            print(f"Average total xG               : {xg_df['total_xg'].mean():.3f}")
            print()
            avg_actual = xg_df["total_goals"].mean()
            avg_xg_total = xg_df["total_xg"].mean()
            diff = avg_actual - avg_xg_total
            print(f"Average actual goals (xG rows) : {avg_actual:.3f}")
            print(f"Average total xG               : {avg_xg_total:.3f}")
            print(f"Actual - xG delta              : {diff:+.3f}  ({'over-performing' if diff > 0 else 'under-performing'})")
    else:
        print("xG columns not found — skipping.")

    # ------------------------------------------------------------------ #
    # 9. League breakdown (top 15)
    # ------------------------------------------------------------------ #
    section("9. LEAGUE BREAKDOWN (Top 15)")

    league_col = "League"
    if league_col in df.columns:
        top_leagues = (
            df[league_col]
            .astype(str)
            .str.strip()
            .value_counts()
            .head(15)
        )
        for league, count in top_leagues.items():
            print(f"  {league:<40} {count:>6,}  ({pct(count, total)})")
    else:
        print("'League' column not found — skipping.")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    section("DONE")
    print(f"Analysis complete. {total:,} valid matches processed.")
    print()


if __name__ == "__main__":
    main()
