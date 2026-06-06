from __future__ import annotations

"""Scrape FBref match pages for first-half corner kick data.

Usage:
    python scrape_fbref_1h_corners.py --leagues E0 D1 SP1 I1 F1 --seasons 2022 2023 2024

Output:
    data/fbref/corners_1h_raw.csv   — one row per match with home/away 1H corners

Runs incrementally: already-scraped match URLs are skipped on re-run.
Rate-limited to avoid FBref bans (~2-4s between requests, back-off on 429).
"""

import argparse
import csv
import datetime as dt
import sys
import time
import random
import re
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows to avoid charmap encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# FBref league config
# ---------------------------------------------------------------------------

LEAGUE_CONFIG = {
    "E0":  {"comp_id": 9,  "name_url": "Premier-League",  "name_display": "Premier League"},
    "D1":  {"comp_id": 20, "name_url": "Bundesliga",       "name_display": "Bundesliga"},
    "SP1": {"comp_id": 12, "name_url": "La-Liga",          "name_display": "La Liga"},
    "I1":  {"comp_id": 11, "name_url": "Serie-A",          "name_display": "Serie A"},
    "F1":  {"comp_id": 13, "name_url": "Ligue-1",          "name_display": "Ligue 1"},
}

FBREF_BASE = "https://fbref.com"
OUT_CSV = Path("data/fbref/corners_1h_raw.csv")
CACHE_DIR = Path("data/fbref/cache")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Referer": "https://fbref.com/",
}

CSV_COLUMNS = [
    "source", "league", "season", "match_date",
    "home_team", "away_team",
    "home_corners_1h", "away_corners_1h",
    "home_corners_total", "away_corners_total",
    "match_url", "scraped_at",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, session: requests.Session, verify: bool = True,
         max_retries: int = 4) -> Optional[str]:
    """Fetch URL with exponential back-off on 429/503."""
    delay = 3.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30, verify=verify)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                wait = delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"    [rate-limit {r.status_code}] sleeping {wait:.1f}s …")
                time.sleep(wait)
                continue
            print(f"    [HTTP {r.status_code}] {url}")
            return None
        except requests.RequestException as exc:
            print(f"    [err] {exc} — attempt {attempt+1}/{max_retries}")
            time.sleep(delay * (attempt + 1))
    return None


def _cached_get(url: str, session: requests.Session, verify: bool = True) -> Optional[str]:
    """Disk-cached GET — avoids re-fetching on re-runs."""
    cache_key = re.sub(r"[^\w]", "_", url)[:200]
    cache_file = CACHE_DIR / f"{cache_key}.html"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")
    html = _get(url, session, verify=verify)
    if html:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(html, encoding="utf-8")
    return html


# ---------------------------------------------------------------------------
# Season string helpers
# ---------------------------------------------------------------------------

def _season_str(year: int) -> str:
    """2024 → '2024-2025'"""
    return f"{year}-{year + 1}"


# ---------------------------------------------------------------------------
# Schedule scraping — extract all match URLs for one league-season
# ---------------------------------------------------------------------------

def _schedule_url(comp_id: int, name_url: str, season: str) -> str:
    return (
        f"{FBREF_BASE}/en/comps/{comp_id}/{season}/schedule/"
        f"{season}-{name_url}-Scores-and-Fixtures"
    )


def _extract_match_urls(html: str) -> list[str]:
    """Return relative /en/matches/… URLs from the schedule table."""
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    # The schedule table has links to match reports in a 'Match Report' column
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if href.startswith("/en/matches/") and len(href) > len("/en/matches/"):
            # Exclude team/player sub-pages; match pages have an 8-char hash
            parts = href.split("/")
            # /en/matches/{hash}/{slug}  → len >= 4 parts after split
            if len(parts) >= 4 and re.match(r"^[0-9a-f]{8}$", parts[3]):
                full = FBREF_BASE + href
                if full not in urls:
                    urls.append(full)
    return urls


# ---------------------------------------------------------------------------
# Match page parsing — extract 1H and total corners
# ---------------------------------------------------------------------------

def _clean_int(val: str) -> Optional[int]:
    val = val.strip()
    if val in ("", "—", "-", "N/A"):
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _parse_corners(html: str) -> Optional[dict]:
    """
    Parse FBref match page for corner kicks.

    FBref's #team_stats table has rows like:
        <td>{home_val}</td>  <th scope="row">Corners</th>  <td>{away_val}</td>

    Some pages have TWO "Corners" rows — one for 1st half, one for 2nd half —
    separated by a header row that contains "1st Half" / "2nd Half".

    Returns dict with keys:
        home_1h, away_1h, home_total, away_total
    Returns None if corners not found at all.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Locate the team_stats table (may be wrapped in a div)
    table = soup.find("table", {"id": "team_stats"})
    if table is None:
        # Sometimes it is inside a div with id="div_team_stats"
        div = soup.find("div", {"id": "div_team_stats"})
        if div:
            table = div.find("table")
    if table is None:
        return None

    rows = table.find_all("tr")

    # Walk rows, tracking which half we are in
    current_half = "full"   # "full" means no half header seen yet
    corners_by_half: dict[str, tuple[Optional[int], Optional[int]]] = {}

    for row in rows:
        # Detect half-header rows (colspan row that says "1st Half" etc.)
        cells = row.find_all(["td", "th"])
        text_joined = " ".join(c.get_text(strip=True) for c in cells).lower()

        if "1st half" in text_joined or "first half" in text_joined:
            current_half = "1h"
            continue
        if "2nd half" in text_joined or "second half" in text_joined:
            current_half = "2h"
            continue

        # Look for a data row where the middle cell is "Corners"
        th = row.find("th", {"scope": "row"})
        if th and "corners" in th.get_text(strip=True).lower():
            tds = row.find_all("td")
            if len(tds) >= 2:
                home_val = _clean_int(tds[0].get_text(strip=True))
                away_val = _clean_int(tds[-1].get_text(strip=True))
                corners_by_half[current_half] = (home_val, away_val)

    if not corners_by_half:
        return None

    if "1h" in corners_by_half and "2h" in corners_by_half:
        h1h, a1h = corners_by_half["1h"]
        h2h, a2h = corners_by_half["2h"]
        h_total = _add(h1h, h2h)
        a_total = _add(a1h, a2h)
        return {"home_1h": h1h, "away_1h": a1h, "home_total": h_total, "away_total": a_total}

    if "full" in corners_by_half:
        h_total, a_total = corners_by_half["full"]
        return {"home_1h": None, "away_1h": None, "home_total": h_total, "away_total": a_total}

    return None


def _add(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    return a + b


# ---------------------------------------------------------------------------
# Match metadata — date, teams — from the match page itself
# ---------------------------------------------------------------------------

def _parse_match_meta(html: str) -> dict:
    """Extract match date, home team, away team from a FBref match page."""
    soup = BeautifulSoup(html, "html.parser")
    meta: dict = {}

    # Date: <span class="venuetime" data-venue-date="2024-08-17"> or <span itemprop="startDate">
    date_span = (
        soup.find("span", {"itemprop": "startDate"})
        or soup.find("span", class_=re.compile(r"venuetime"))
    )
    if date_span:
        raw_date = date_span.get("data-venue-date") or date_span.get_text(strip=True)
        # Normalize: take first 10 chars if it looks like ISO date
        if raw_date and re.match(r"\d{4}-\d{2}-\d{2}", raw_date):
            meta["match_date"] = raw_date[:10]

    # Teams: the scorebox has divs with class "scorebox"
    scorebox = soup.find("div", class_="scorebox")
    if scorebox:
        team_divs = scorebox.find_all("div", itemprop="performer")
        if len(team_divs) >= 2:
            meta["home_team"] = team_divs[0].get_text(separator=" ", strip=True)
            meta["away_team"] = team_divs[1].get_text(separator=" ", strip=True)
        else:
            # Fallback: strong tags in scorebox
            strongs = scorebox.find_all("strong")
            if len(strongs) >= 2:
                meta["home_team"] = strongs[0].get_text(strip=True)
                meta["away_team"] = strongs[1].get_text(strip=True)

    return meta


# ---------------------------------------------------------------------------
# Already-scraped URL index
# ---------------------------------------------------------------------------

def _load_scraped_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    scraped: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("match_url", "").strip()
            if url:
                scraped.add(url)
    return scraped


# ---------------------------------------------------------------------------
# CSV writer helpers
# ---------------------------------------------------------------------------

def _open_csv(path: Path) -> tuple:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if not exists:
        writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape FBref for 1H corners data.")
    p.add_argument(
        "--leagues", nargs="+",
        default=list(LEAGUE_CONFIG.keys()),
        choices=list(LEAGUE_CONFIG.keys()),
        help="League codes to scrape (default: all 5)",
    )
    p.add_argument(
        "--seasons", nargs="+", type=int,
        default=[2022, 2023, 2024],
        help="Start years of seasons to scrape (e.g. 2022 → 2022-2023)",
    )
    p.add_argument("--out-csv", default=str(OUT_CSV))
    p.add_argument("--delay", type=float, default=2.5,
                   help="Base delay in seconds between match requests")
    p.add_argument("--insecure", action="store_true",
                   help="Disable SSL verification")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch schedules only, do not fetch match pages")
    p.add_argument("--max-matches", type=int, default=0,
                   help="Stop after N matches (0=no limit, for testing)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out_csv)
    verify_ssl = not args.insecure

    scraped_urls = _load_scraped_urls(out_path)
    print(f"Already scraped: {len(scraped_urls)} matches")

    session = requests.Session()
    fh, writer = _open_csv(out_path)
    total_written = 0
    total_no_1h = 0

    try:
        for league in args.leagues:
            cfg = LEAGUE_CONFIG[league]
            comp_id = cfg["comp_id"]
            name_url = cfg["name_url"]

            for year in args.seasons:
                season = _season_str(year)
                sched_url = _schedule_url(comp_id, name_url, season)
                print(f"\n[{league} {season}] Fetching schedule: {sched_url}")

                sched_html = _cached_get(sched_url, session, verify=verify_ssl)
                if not sched_html:
                    print(f"  [SKIP] Failed to fetch schedule.")
                    continue

                match_urls = _extract_match_urls(sched_html)
                new_urls = [u for u in match_urls if u not in scraped_urls]
                print(f"  Found {len(match_urls)} matches, {len(new_urls)} new to scrape")

                if args.dry_run:
                    for u in match_urls[:5]:
                        print(f"    {u}")
                    continue

                for i, match_url in enumerate(new_urls):
                    if args.max_matches and total_written >= args.max_matches:
                        print(f"  [max-matches={args.max_matches}] stopping early")
                        break

                    # Politeness delay
                    jitter = random.uniform(0.5, 1.5)
                    time.sleep(args.delay + jitter)

                    match_html = _cached_get(match_url, session, verify=verify_ssl)
                    if not match_html:
                        print(f"  [{i+1}/{len(new_urls)}] FAIL {match_url}")
                        continue

                    meta = _parse_match_meta(match_html)
                    corners = _parse_corners(match_html)

                    if corners is None:
                        print(f"  [{i+1}/{len(new_urls)}] NO-CORNERS {match_url}")
                        scraped_urls.add(match_url)
                        continue

                    has_1h = corners["home_1h"] is not None
                    if not has_1h:
                        total_no_1h += 1

                    row = {
                        "source": "fbref",
                        "league": league,
                        "season": year,
                        "match_date": meta.get("match_date", ""),
                        "home_team": meta.get("home_team", ""),
                        "away_team": meta.get("away_team", ""),
                        "home_corners_1h": corners["home_1h"],
                        "away_corners_1h": corners["away_1h"],
                        "home_corners_total": corners["home_total"],
                        "away_corners_total": corners["away_total"],
                        "match_url": match_url,
                        "scraped_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                    writer.writerow(row)
                    fh.flush()
                    scraped_urls.add(match_url)
                    total_written += 1

                    status = "1H+TOT" if has_1h else "TOT-only"
                    print(
                        f"  [{i+1}/{len(new_urls)}] {status} "
                        f"{meta.get('match_date','')} "
                        f"{meta.get('home_team','')} vs {meta.get('away_team','')} "
                        f"→ 1H: {corners['home_1h']}-{corners['away_1h']} "
                        f"TOT: {corners['home_total']}-{corners['away_total']}"
                    )
    finally:
        fh.close()

    print(f"\nDone. Written {total_written} rows -> {out_path}")
    print(f"Matches without 1H split: {total_no_1h} (saved with total only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
