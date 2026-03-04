from __future__ import annotations

"""Backfill historical match data from football-data.co.uk.

football-data.co.uk provides free CSVs per league per season going back to the
1990s. Columns include Date, HomeTeam, AwayTeam, FTHG, FTAG, HTHG, HTAG.
This script downloads the requested seasons, normalises them to match the
fhg_history.csv schema, and appends/merges into the existing history file.

Leagues available (RS1/SA1 not on this source):
  E0, E1, SP1, D1, I1, F1, N1, P1, RO1
"""

import argparse
import datetime as dt
import io
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# League codes that exist on football-data.co.uk
FDCO_LEAGUES: Dict[str, str] = {
    "E0": "E0",
    "E1": "E1",
    "SP1": "SP1",
    "D1": "D1",
    "I1": "I1",
    "F1": "F1",
    "N1": "N1",
    "P1": "P1",
    "RO1": "RO1",
}


def _season_code(start_year: int) -> str:
    """Map season start year to football-data.co.uk URL code.

    Examples:
      2020 -> '2021'  (2020-21 season)
      2021 -> '2122'  (2021-22 season)
      2019 -> '1920'  (2019-20 season)
    """
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"


def _norm_team(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _parse_date(raw: str) -> Optional[str]:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return dt.datetime.strptime(str(raw).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def fetch_season(
    league_code: str,
    start_year: int,
    session: requests.Session,
    timeout: int = 30,
    verify_ssl: bool = True,
) -> pd.DataFrame:
    season_code = _season_code(start_year)
    url = f"{BASE_URL}/{season_code}/{league_code}.csv"
    try:
        r = session.get(url, timeout=timeout, verify=verify_ssl)
        if r.status_code == 404:
            print(f"  [SKIP] {league_code} {start_year}: 404 not found")
            return pd.DataFrame()
        r.raise_for_status()
        # football-data CSVs sometimes have trailing empty columns — low_memory=False avoids warnings
        raw = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip", low_memory=False)
        return raw
    except Exception as exc:
        print(f"  [WARN] {league_code} {start_year}: {exc}")
        return pd.DataFrame()


def parse_season(raw: pd.DataFrame, league_code: str, start_year: int) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in raw.iterrows():
        date_raw = r.get("Date")
        home = r.get("HomeTeam")
        away = r.get("AwayTeam")
        hg = r.get("FTHG")
        ag = r.get("FTAG")
        hth = r.get("HTHG")
        hta = r.get("HTAG")

        if pd.isna(date_raw) or pd.isna(home) or pd.isna(away):
            continue
        try:
            hg_int = int(float(hg))
            ag_int = int(float(ag))
        except (TypeError, ValueError):
            continue
        md = _parse_date(str(date_raw))
        if md is None:
            continue
        try:
            hth_val = int(float(hth)) if pd.notna(hth) else pd.NA
            hta_val = int(float(hta)) if pd.notna(hta) else pd.NA
        except (TypeError, ValueError):
            hth_val = pd.NA
            hta_val = pd.NA

        rows.append(
            {
                "source": "football_data_co_uk",
                "league": league_code,
                "season_start_year": start_year,
                "match_date": md,
                "home_team": _norm_team(home),
                "away_team": _norm_team(away),
                "home_goals": hg_int,
                "away_goals": ag_int,
                "ht_home_goals": hth_val,
                "ht_away_goals": hta_val,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill FHG/Goals history from football-data.co.uk."
    )
    p.add_argument("--start-season", type=int, default=2020,
                   help="First season start year (e.g. 2020 = 2020-21 season)")
    p.add_argument("--end-season", type=int, default=2021,
                   help="Last season start year inclusive")
    p.add_argument("--out-history", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--request-delay", type=float, default=1.5,
                   help="Seconds to wait between requests (be polite)")
    p.add_argument("--insecure", action="store_true")
    p.add_argument(
        "--local-dir", default="",
        help=(
            "Path to a folder containing pre-downloaded CSVs named as "
            "<LEAGUE_CODE>_<START_YEAR>.csv  e.g. E0_2020.csv, D1_2021.csv. "
            "When provided, skips HTTP fetching entirely."
        ),
    )
    return p.parse_args()


def load_local(local_dir: str, seasons: list[int]) -> list[pd.DataFrame]:
    """Load pre-downloaded CSVs from a local folder.

    Accepts two filename conventions:
      <LEAGUE>_<START_YEAR>.csv   e.g. E0_2020.csv
      <SEASON_CODE><LEAGUE>.csv   e.g. 2021E0.csv  (raw football-data filename)
    """
    frames = []
    base = Path(local_dir)
    for f in sorted(base.glob("*.csv")):
        stem = f.stem  # e.g. "E0_2020" or "2021E0"

        # Convention 1: LEAGUE_YEAR.csv
        if "_" in stem:
            parts = stem.split("_", 1)
            league_code = parts[0].upper()
            try:
                start_year = int(parts[1])
            except ValueError:
                continue
        else:
            # Convention 2: raw football-data name like "2021E0" or "E02021"
            # Try stripping the 4-digit season code from either end
            if stem[:4].isdigit():
                season_code = stem[:4]
                league_code = stem[4:].upper()
                start_year = int("20" + season_code[:2])
            elif stem[-4:].isdigit():
                season_code = stem[-4:]
                league_code = stem[:-4].upper()
                start_year = int("20" + season_code[:2])
            else:
                continue

        if league_code not in FDCO_LEAGUES:
            continue
        if start_year not in seasons:
            continue
        try:
            raw = pd.read_csv(f, on_bad_lines="skip", low_memory=False)
        except Exception as exc:
            print(f"  [WARN] {f.name}: {exc}")
            continue
        parsed = parse_season(raw, league_code, start_year)
        if parsed.empty:
            continue
        ht_coverage = parsed["ht_home_goals"].notna().mean()
        print(f"  [OK] {league_code} {start_year} from {f.name}: {len(parsed)} rows  HT={ht_coverage:.0%}")
        frames.append(parsed)
    return frames


def main() -> int:
    args = parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))

    new_frames: list[pd.DataFrame] = []

    if args.local_dir:
        print(f"Loading from local directory: {args.local_dir}")
        new_frames = load_local(args.local_dir, seasons)
    else:
        print(f"Fetching seasons {seasons} for leagues: {list(FDCO_LEAGUES)}")
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.football-data.co.uk/data.php",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        for league_code in FDCO_LEAGUES:
            for season in seasons:
                print(f"Fetching {league_code} {season} ({_season_code(season)})...")
                raw = fetch_season(league_code, season, session, verify_ssl=not args.insecure)
                if raw.empty:
                    continue
                parsed = parse_season(raw, league_code, season)
                if parsed.empty:
                    print(f"  [SKIP] {league_code} {season}: no parseable rows")
                    continue
                ht_coverage = parsed["ht_home_goals"].notna().mean()
                print(f"  [OK] {league_code} {season}: {len(parsed)} rows  HT={ht_coverage:.0%}")
                new_frames.append(parsed)
                time.sleep(args.request_delay)

    if not new_frames:
        print("No new rows fetched.")
        return 0

    new_df = pd.concat(new_frames, ignore_index=True)

    # Merge with existing history
    hist_path = Path(args.out_history)
    if hist_path.exists():
        existing = pd.read_csv(hist_path)
        print(f"\nExisting history: {len(existing)} rows")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = (
        combined
        .drop_duplicates(subset=["league", "match_date", "home_team", "away_team"])
        .sort_values(["match_date", "league", "home_team"])
        .reset_index(drop=True)
    )

    hist_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(hist_path, index=False)
    print(f"Saved: {hist_path}  total={len(combined)}  added={len(new_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
