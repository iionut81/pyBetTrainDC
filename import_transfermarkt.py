from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup


LEAGUES: Dict[str, str] = {
    "GB1": "E0",
    "GB2": "E1",
    "ES1": "SP1",
    "L1": "D1",
    "IT1": "I1",
    "FR1": "F1",
    "NL1": "N1",
    "PO1": "P1",
    "RO1": "RO1",
    "SER1": "RS1",
    "SA1": "SA1",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
BASE_COLUMNS = [
    "source",
    "league",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "odds_1x",
    "odds_x2",
    "season_code",
]


def clean_team_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def extract_date(raw: str) -> Optional[dt.date]:
    m = re.search(r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})", raw)
    if not m:
        return None
    token = m.group(1).replace(".", "/")
    parts = token.split("/")
    if len(parts[2]) == 2:
        fmt = "%d/%m/%y"
    else:
        fmt = "%d/%m/%Y"
    try:
        return dt.datetime.strptime(token, fmt).date()
    except ValueError:
        return None


def fetch_schedule_page(
    session: requests.Session, tm_comp_code: str, season_start_year: int, verify_ssl: bool
) -> str:
    # The slug segment is ignored for this endpoint; comp code drives content.
    url = f"https://www.transfermarkt.com/x/gesamtspielplan/wettbewerb/{tm_comp_code}/saison_id/{season_start_year}"
    resp = session.get(url, timeout=25, verify=verify_ssl)
    resp.raise_for_status()
    return resp.text


def parse_schedule_html(html: str, dc_league_code: str, season_label: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr")
    out: List[dict] = []
    current_date: Optional[dt.date] = None

    for tr in rows:
        score_link = tr.select_one("a.ergebnis-link")
        if score_link is None:
            continue

        score_text = score_link.get_text(" ", strip=True)
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", score_text)
        if not m:
            continue
        home_goals = int(m.group(1))
        away_goals = int(m.group(2))

        first_td = tr.select_one("td")
        if first_td is not None:
            maybe_date = extract_date(first_td.get_text(" ", strip=True))
            if maybe_date is not None:
                current_date = maybe_date
        if current_date is None:
            continue

        home_anchor = tr.select_one("td.text-right.hauptlink a")
        away_anchor = tr.select_one("td.no-border-links.hauptlink a")
        home_team = home_anchor.get_text(" ", strip=True) if home_anchor else ""
        away_team = away_anchor.get_text(" ", strip=True) if away_anchor else ""

        if not home_team or not away_team:
            tds = tr.find_all("td")
            if len(tds) >= 7:
                if not home_team:
                    home_team = tds[2].get_text(" ", strip=True)
                if not away_team:
                    away_team = tds[6].get_text(" ", strip=True)

        if not home_team or not away_team:
            continue

        out.append(
            {
                "source": "transfermarkt",
                "league": dc_league_code,
                "match_date": current_date,
                "home_team": clean_team_name(home_team),
                "away_team": clean_team_name(away_team),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "odds_1x": pd.NA,
                "odds_x2": pd.NA,
                "season_code": season_label,
            }
        )

    if not out:
        return pd.DataFrame(
            columns=BASE_COLUMNS
        )
    return pd.DataFrame(out)


def merge_into_store(new_df: pd.DataFrame, store_path: str) -> pd.DataFrame:
    if os.path.exists(store_path):
        existing = pd.read_csv(store_path)
    else:
        existing = pd.DataFrame(columns=BASE_COLUMNS)

    for col in BASE_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
        if col not in new_df.columns:
            new_df[col] = pd.NA

    merged = pd.concat([existing[BASE_COLUMNS], new_df[BASE_COLUMNS]], ignore_index=True)
    merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce").dt.date
    merged["home_team"] = merged["home_team"].astype(str).str.strip().str.lower()
    merged["away_team"] = merged["away_team"].astype(str).str.strip().str.lower()
    merged["league"] = merged["league"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"]).copy()
    merged = merged.dropna(subset=["match_date", "home_team", "away_team", "league"]).copy()
    merged = merged.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
    merged.to_csv(store_path, index=False)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Transfermarkt fixture/results pages and output Dixon-Coles training rows."
    )
    parser.add_argument(
        "--start-season-year",
        type=int,
        default=2021,
        help="Starting year of oldest season, e.g. 2021 for 2021/22.",
    )
    parser.add_argument("--n-seasons", type=int, default=5, help="Number of consecutive seasons to scrape.")
    parser.add_argument(
        "--output-csv",
        default="historical_matches_transfermarkt.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--merge-into",
        default="",
        help=(
            "Optional master history CSV to update in place (append + deduplicate). "
            "Example: historical_matches_transfermarkt.csv"
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.2,
        help="Delay between requests to avoid aggressive scraping.",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable TLS certificate verification (disabled by default).",
    )
    args = parser.parse_args()

    if not args.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

    all_rows: List[pd.DataFrame] = []
    failures: List[str] = []
    seasons = [args.start_season_year + i for i in range(args.n_seasons)]

    for season_start in seasons:
        season_label = f"{season_start}/{(season_start + 1) % 100:02d}"
        for tm_code, dc_code in LEAGUES.items():
            try:
                html = fetch_schedule_page(session, tm_code, season_start, args.verify_ssl)
                df = parse_schedule_html(html, dc_code, season_label=season_label)
                if df.empty:
                    failures.append(f"{season_label} {tm_code}: no parseable matches")
                    print(f"[WARN] {season_label} {tm_code}: no parseable matches")
                else:
                    all_rows.append(df)
                    print(f"[OK] {season_label} {tm_code}->{dc_code}: {len(df)} matches")
            except Exception as exc:
                failures.append(f"{season_label} {tm_code}: {exc}")
                print(f"[FAIL] {season_label} {tm_code}: {exc}")
            time.sleep(max(0.0, args.sleep_seconds))

    if not all_rows:
        print("No data collected from Transfermarkt.")
        if failures:
            print("Failures:")
            for f in failures:
                print(f"  - {f}")
        return 1

    out = pd.concat(all_rows, ignore_index=True)
    out = out.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"]).copy()
    out = out.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
    out.to_csv(args.output_csv, index=False)

    print(f"\nWrote {len(out)} rows to {args.output_csv}")
    if args.merge_into:
        merged = merge_into_store(out, args.merge_into)
        print(f"Merged into {args.merge_into}: {len(merged)} rows total")
    print(f"Leagues: {', '.join(sorted(set(out['league'])))}")
    print(f"Seasons: {', '.join(f'{y}/{(y+1)%100:02d}' for y in seasons)}")
    if failures:
        print(f"\nCompleted with {len(failures)} warnings/failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
