from __future__ import annotations

"""
build_corners_history_fdco.py
Download latest Football-Data.co.uk CSVs and update corners history.

Default behavior: fetch current season for the 8 major leagues with match stats
and merge into simulations/Corners U12.5/data/corners_history.csv.
"""

import argparse
import datetime as dt
import io
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Leagues with reliable match stats (corners) on football-data.co.uk
FDCO_LEAGUES: Dict[str, str] = {
    "E0": "E0",
    "E1": "E1",
    "SP1": "SP1",
    "D1": "D1",
    "I1": "I1",
    "F1": "F1",
    "N1": "N1",
    "P1": "P1",
}


def _season_code(start_year: int) -> str:
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"


def _default_start_year(today: dt.date) -> int:
    # Season typically starts in July; before July use previous year.
    return today.year if today.month >= 7 else today.year - 1


def _norm_team(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _parse_date(raw: object) -> Optional[str]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(str(raw).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fetch_csv(session: requests.Session, league_code: str, start_year: int, verify_ssl: bool) -> pd.DataFrame:
    season = _season_code(start_year)
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    r = session.get(url, timeout=30, verify=verify_ssl)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), on_bad_lines="skip", low_memory=False)


def parse_season(raw: pd.DataFrame, league_code: str, start_year: int) -> pd.DataFrame:
    # Require corners columns
    if "HC" not in raw.columns or "AC" not in raw.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, r in raw.iterrows():
        date_raw = r.get("Date")
        home = r.get("HomeTeam")
        away = r.get("AwayTeam")
        hc = r.get("HC")
        ac = r.get("AC")

        if pd.isna(date_raw) or pd.isna(home) or pd.isna(away):
            continue
        try:
            hc_int = int(float(hc))
            ac_int = int(float(ac))
        except (TypeError, ValueError):
            continue
        md = _parse_date(date_raw)
        if md is None:
            continue

        rows.append(
            {
                "source": "football-data.co.uk",
                "league": league_code,
                "season": _season_code(start_year),
                "match_date": md,
                "home_team": _norm_team(home),
                "away_team": _norm_team(away),
                "home_corners": hc_int,
                "away_corners": ac_int,
            }
        )
    return pd.DataFrame(rows)


def _merge_history(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        out = new_rows.copy()
    else:
        out = pd.concat([existing, new_rows], ignore_index=True)
    natural_cols = ["league", "season", "match_date", "home_team", "away_team"]
    out = out.drop_duplicates(subset=natural_cols, keep="last").reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update corners history from football-data.co.uk.")
    p.add_argument("--start-season", type=int, default=_default_start_year(dt.date.today()))
    p.add_argument("--end-season", type=int, default=None)
    p.add_argument("--out-history", default="simulations/Corners U12.5/data/corners_history.csv")
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    end_season = args.end_season if args.end_season is not None else args.start_season
    seasons = list(range(args.start_season, end_season + 1))

    out_path = Path(args.out_history)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.football-data.co.uk/data.php",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    all_new: list[pd.DataFrame] = []
    for season in seasons:
        code = _season_code(season)
        print(f"Season {season} ({code})")
        for league_code in FDCO_LEAGUES:
            try:
                raw = fetch_csv(session, league_code, season, verify_ssl=not args.insecure)
            except Exception as exc:
                print(f"  [WARN] {league_code} {season}: {exc}")
                continue
            parsed = parse_season(raw, league_code, season)
            if parsed.empty:
                print(f"  [SKIP] {league_code} {season}: no corners columns")
                continue
            print(f"  [OK] {league_code} {season}: {len(parsed)} rows")
            all_new.append(parsed)

    if all_new:
        new_rows = pd.concat(all_new, ignore_index=True)
        merged = _merge_history(existing, new_rows)
        merged.to_csv(out_path, index=False)
        print(f"\nUpdated history: {len(merged)} rows total")
    else:
        print("No new rows parsed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
