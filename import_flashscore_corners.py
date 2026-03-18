from __future__ import annotations

"""
import_flashscore_corners.py

Scrape per-match corner statistics from Flashscore to train a Dixon-Coles
style corners model.  Fills the gap for leagues/seasons that football-data.co.uk
and API-Football don't cover (RO1, RS1, SA1, D2, SP2, I2, SW1, DK1).

Phase 1 — Collect match event IDs from league results pages  (requests)
Phase 2 — Fetch corner counts via Flashscore internal API    (requests, ~0.3s/match)

Dependencies:  pip install pandas requests

Usage:
    python import_flashscore_corners.py --insecure                         # all leagues
    python import_flashscore_corners.py --insecure --leagues RO1 D2 SW1    # specific
    python import_flashscore_corners.py --insecure --resume                # continue
    python import_flashscore_corners.py --insecure --season-start 2024     # archive
"""

import argparse
import datetime as dt
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests as req

LOG = logging.getLogger("flashscore_corners")

# ---------------------------------------------------------------------------
# League configuration
# ---------------------------------------------------------------------------

FLASH_LEAGUES: Dict[str, dict] = {
    "E0":  {"country": "england",      "slug": "premier-league"},
    "E1":  {"country": "england",      "slug": "championship"},
    "D1":  {"country": "germany",      "slug": "bundesliga"},
    "D2":  {"country": "germany",      "slug": "2-bundesliga"},
    "SP1": {"country": "spain",        "slug": "laliga"},
    "SP2": {"country": "spain",        "slug": "laliga2"},
    "I1":  {"country": "italy",        "slug": "serie-a"},
    "I2":  {"country": "italy",        "slug": "serie-b"},
    "F1":  {"country": "france",       "slug": "ligue-1"},
    "N1":  {"country": "netherlands",  "slug": "eredivisie"},
    "P1":  {"country": "portugal",     "slug": "liga-portugal"},
    "RO1": {"country": "romania",      "slug": "superliga"},
    "RS1": {"country": "serbia",       "slug": "mozzart-bet-super-liga",
            "alt_slugs": ["super-liga", "superliga"]},
    "SA1": {"country": "saudi-arabia", "slug": "saudi-professional-league"},
    "SW1": {"country": "switzerland",  "slug": "super-league"},
    "DK1": {"country": "denmark",      "slug": "superliga"},
}

# Flashscore internal stats API
STATS_API = "https://2.flashscore.ninja/2/x/feed/df_st_1_{eid}"
STATS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.flashscore.com/",
    "x-fsign": "SW9D1eZo",
}

# Paths
QUEUE_CSV  = Path("simulations/Corners/data/flashscore_corners_queue.csv")
OUTPUT_CSV = Path("simulations/Corners/data/corners_flashscore.csv")
MERGE_INTO = Path("simulations/Corners U12.5/data/corners_history.csv")

# Timing
DELAY_RESULTS = 3        # sec between results page loads
DELAY_STATS   = 0.4      # sec between stats API calls (rate limit)
SAVE_EVERY    = 50        # incremental save interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _season_code(start: int) -> str:
    """2025 → '2526'"""
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _season_start_from_date(d: dt.date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _safe_int(v: object) -> Optional[int]:
    try:
        return int(str(v))
    except Exception:
        return None


def _results_urls(info: dict, season_start: Optional[int] = None) -> List[str]:
    """Build results page URL(s).  None → current season."""
    country = info["country"]
    slugs = [info["slug"]] + info.get("alt_slugs", [])
    urls = []
    for s in slugs:
        if season_start is None:
            urls.append(f"https://www.flashscore.com/football/{country}/{s}/results/")
        else:
            urls.append(
                f"https://www.flashscore.com/football/{country}/"
                f"{s}-{season_start}-{season_start + 1}/results/"
            )
    return urls


# ---------------------------------------------------------------------------
# Phase 1 — Collect match IDs from results feeds (requests)
# ---------------------------------------------------------------------------

def _extract_feed(html: str) -> str:
    m = re.search(
        r"cjs\.initialFeeds\['results'\]\s*=\s*\{\s*data:\s*`(.*?)`,\s*allEventsCount:",
        html, re.DOTALL,
    )
    return m.group(1) if m else ""


def _parse_fields(chunk: str) -> dict:
    fields: dict = {}
    for token in chunk.split(chr(172)):          # ¬
        if chr(247) in token:                     # ÷
            k, v = token.split(chr(247), 1)
            if k:
                fields[k] = v
    return fields


def _event_id_from_chunk(chunk: str) -> str:
    """Extract the Flashscore event ID (first value after ~AA÷)."""
    first = chunk.split(chr(172))[0]             # ¬
    return first.lstrip(chr(247)).strip()         # ÷


def parse_results_feed(feed: str, league: str, season_start: int) -> List[dict]:
    out: List[dict] = []
    if not feed:
        return out
    for chunk in feed.split("~AA")[1:]:
        eid = _event_id_from_chunk(chunk)
        f = _parse_fields(chunk)
        ts = _safe_int(f.get("AD"))
        home = f.get("AE")
        away = f.get("AF")
        if not eid or not home or not away:
            continue
        md = ""
        if ts is not None:
            md = dt.datetime.fromtimestamp(ts, dt.UTC).date().isoformat()
            s = _season_start_from_date(dt.datetime.fromtimestamp(ts, dt.UTC).date())
            if s != season_start:
                continue
        out.append({
            "event_id": eid,
            "league": league,
            "home_team": _norm(home),
            "away_team": _norm(away),
            "match_date": md,
        })
    return out


def phase1_collect(
    leagues: List[str], season_starts: List[int], verify_ssl: bool
) -> pd.DataFrame:
    """Collect match IDs using requests for one or more seasons per league."""
    headers = {"User-Agent": "Mozilla/5.0"}
    current_start = _season_start_from_date(dt.date.today())
    all_matches: List[dict] = []

    for league in leagues:
        info = FLASH_LEAGUES[league]
        league_total = 0
        for ss in season_starts:
            # Current season → no suffix; past seasons → archive URL
            urls = _results_urls(
                info, season_start=None if ss == current_start else ss
            )
            parsed: List[dict] = []
            for url in urls:
                try:
                    r = req.get(url, headers=headers, timeout=35, verify=verify_ssl)
                    r.raise_for_status()
                    feed = _extract_feed(r.text)
                    parsed = parse_results_feed(feed, league, ss)
                    if parsed:
                        break
                except Exception as exc:
                    LOG.warning(f"  {league} {ss}: {exc}")
            all_matches.extend(parsed)
            league_total += len(parsed)
            time.sleep(DELAY_RESULTS)
        LOG.info(f"  {league}: {league_total} matches ({len(season_starts)} seasons)")

    return pd.DataFrame(all_matches)


# ---------------------------------------------------------------------------
# Phase 2 — Fetch corner stats via Flashscore internal API (requests)
# ---------------------------------------------------------------------------

def _parse_corners_from_stats(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse the ¬÷ encoded stats feed for 'Corner kicks'.

    Format: ...~SD÷16¬SG÷Corner kicks¬SH÷{home}¬SI÷{away}¬...
    """
    fields = {}
    current_stat = ""
    for token in text.split(chr(172)):           # ¬
        if chr(247) in token:                     # ÷
            k, v = token.split(chr(247), 1)
            if k == "SG":
                current_stat = v
            elif k == "SH" and current_stat.lower() == "corner kicks":
                fields["home"] = _safe_int(v)
            elif k == "SI" and current_stat.lower() == "corner kicks":
                fields["away"] = _safe_int(v)
                break  # first occurrence is enough
    return fields.get("home"), fields.get("away")


def fetch_corners_api(
    session: req.Session, event_id: str
) -> Tuple[Optional[int], Optional[int]]:
    """Fetch corner counts via Flashscore stats API (~0.3s per call)."""
    url = STATS_API.format(eid=event_id)
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return _parse_corners_from_stats(r.text)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_incremental(out_csv: Path, new_rows: List[dict]) -> None:
    new_df = pd.DataFrame(new_rows)
    if out_csv.exists():
        existing = pd.read_csv(out_csv, dtype={"event_id": str})
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    combined = combined.sort_values(["match_date", "league"]).reset_index(drop=True)
    combined.to_csv(out_csv, index=False)


def _merge_into_history(out_csv: Path) -> None:
    if not out_csv.exists():
        LOG.info("  No output file yet — nothing to merge")
        return
    fs = pd.read_csv(out_csv, dtype={"event_id": str})
    fs = fs.dropna(subset=["home_corners", "away_corners"]).copy()
    if fs.empty:
        LOG.info("  Nothing to merge (no valid corner rows)")
        return
    cols = [
        "source", "league", "season", "match_date",
        "home_team", "away_team", "home_corners", "away_corners",
    ]
    fs = fs[[c for c in cols if c in fs.columns]]

    base = pd.read_csv(MERGE_INTO) if MERGE_INTO.exists() else pd.DataFrame()
    merged = pd.concat([base, fs], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["league", "match_date", "home_team", "away_team"], keep="last"
    )
    merged = merged.sort_values(
        ["match_date", "league", "home_team"]
    ).reset_index(drop=True)
    merged.to_csv(MERGE_INTO, index=False)
    LOG.info(f"  Merged into {MERGE_INTO}: {len(merged)} total rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape Flashscore corner statistics for corners DC model."
    )
    p.add_argument(
        "--season-start", type=int, default=2025,
        help="Season start year (default: 2025 for 2025/26)",
    )
    p.add_argument(
        "--seasons", default=None,
        help="Comma-separated season starts, e.g. 2021,2022,2023,2024,2025",
    )
    p.add_argument(
        "--leagues", nargs="*", default=None,
        help="League codes to scrape (default: all configured)",
    )
    p.add_argument(
        "--batch-size", type=int, default=5000,
        help="Max matches to process in Phase 2 per run",
    )
    p.add_argument("--resume", action="store_true",
                   help="Skip Phase 1 and resume Phase 2 from checkpoint")
    p.add_argument("--no-merge", action="store_true",
                   help="Don't merge into corners_history.csv")
    p.add_argument("--insecure", action="store_true",
                   help="Disable SSL verification")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    # Build season list: --seasons overrides --season-start
    if args.seasons:
        season_starts = [int(x.strip()) for x in args.seasons.split(",")]
    else:
        season_starts = [args.season_start]
    leagues = args.leagues or list(FLASH_LEAGUES.keys())
    out_csv = Path(args.out_csv) if args.out_csv else OUTPUT_CSV
    verify_ssl = not args.insecure

    for lg in leagues:
        if lg not in FLASH_LEAGUES:
            LOG.error(f"Unknown league '{lg}'. Available: {list(FLASH_LEAGUES.keys())}")
            return 1

    season_label = ",".join(str(s) for s in season_starts)
    LOG.info("=" * 60)
    LOG.info("FLASHSCORE CORNERS SCRAPER  (API mode)")
    LOG.info(f"  Seasons: {season_label}")
    LOG.info(f"  Leagues: {leagues}")
    LOG.info("=" * 60)

    QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1 — collect match IDs
    # ------------------------------------------------------------------
    if not args.resume:
        LOG.info("\n[PHASE 1] Collecting match IDs ...")
        queue_df = phase1_collect(leagues, season_starts, verify_ssl)

        if queue_df.empty:
            LOG.error("No matches found!")
            return 1

        queue_df = queue_df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
        queue_df.to_csv(QUEUE_CSV, index=False)
        LOG.info(
            f"\n  Queue saved: {QUEUE_CSV}  "
            f"({len(queue_df)} matches across {queue_df['league'].nunique()} leagues)"
        )
    else:
        if not QUEUE_CSV.exists():
            LOG.error(f"Queue not found at {QUEUE_CSV}. Run without --resume first.")
            return 1
        queue_df = pd.read_csv(QUEUE_CSV, dtype={"event_id": str})
        LOG.info(f"  Resumed queue: {len(queue_df)} matches")

    # ------------------------------------------------------------------
    # Phase 2 — fetch corner statistics via API
    # ------------------------------------------------------------------
    LOG.info(f"\n[PHASE 2] Fetching corner statistics via API ...")

    session = req.Session()
    session.headers.update(STATS_HEADERS)
    session.verify = verify_ssl

    # Already-processed event IDs
    done_ids: set[str] = set()
    if out_csv.exists():
        tmp = pd.read_csv(out_csv, dtype={"event_id": str})
        if "event_id" in tmp.columns:
            done_ids = set(tmp["event_id"].astype(str))
    LOG.info(f"  Already processed: {len(done_ids)} matches")

    todo = queue_df[~queue_df["event_id"].astype(str).isin(done_ids)].copy()
    if len(todo) > args.batch_size:
        todo = todo.head(args.batch_size)

    LOG.info(f"  To process: {len(todo)}")

    new_rows: List[dict] = []
    ok = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for idx, (_, row) in enumerate(todo.iterrows(), 1):
        eid = str(row["event_id"])
        try:
            hc, ac = fetch_corners_api(session, eid)

            # Derive season code from match date
            md_str = row.get("match_date", "")
            if md_str:
                md_date = dt.date.fromisoformat(md_str)
                ss = _season_start_from_date(md_date)
                season_code = _season_code(ss)
            else:
                season_code = _season_code(season_starts[0])

            rec = {
                "event_id": eid,
                "source": "flashscore",
                "league": row["league"],
                "season": season_code,
                "match_date": row.get("match_date", ""),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_corners": hc if hc is not None else pd.NA,
                "away_corners": ac if ac is not None else pd.NA,
            }
            new_rows.append(rec)

            if hc is not None and ac is not None:
                ok += 1
                if ok % 20 == 0 or ok <= 5:
                    elapsed = time.time() - t_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - idx) / rate if rate > 0 else 0
                    LOG.info(
                        f"    [{ok:4d}/{len(todo)}] {row['league']} "
                        f"{row['home_team']} v {row['away_team']}  "
                        f"{hc}-{ac}  ({rate:.1f} match/s, ETA {eta:.0f}s)"
                    )
            else:
                skipped += 1

            if len(new_rows) % SAVE_EVERY == 0:
                _save_incremental(out_csv, new_rows)

            time.sleep(DELAY_STATS)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                LOG.warning(f"    [{eid}] {exc}")
            elif errors == 6:
                LOG.warning("    (suppressing further errors...)")

    if new_rows:
        _save_incremental(out_csv, new_rows)

    elapsed = time.time() - t_start
    LOG.info(f"\n  Corners fetched : {ok}")
    LOG.info(f"  Skipped (no data): {skipped}")
    LOG.info(f"  Errors           : {errors}")
    LOG.info(f"  Time             : {elapsed:.0f}s ({elapsed/max(ok,1):.2f}s/match)")
    LOG.info(f"  Output           : {out_csv}")

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    if not args.no_merge:
        LOG.info("\n[MERGE] Merging into corners_history.csv ...")
        _merge_into_history(out_csv)

    LOG.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())