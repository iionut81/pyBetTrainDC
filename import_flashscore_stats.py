from __future__ import annotations

"""
import_flashscore_stats.py

Scrape comprehensive match statistics from Flashscore for all 16 leagues.
Collects: xG, shots, corners, fouls, cards, possession, passes, xA, and more.

Phase 1 — Collect match event IDs from league results pages  (requests)
Phase 2 — Fetch all stats via Flashscore internal API         (requests, ~0.5s/match)

Dependencies:  pip install pandas requests

Usage:
    python import_flashscore_stats.py --insecure                          # all leagues, current season
    python import_flashscore_stats.py --insecure --seasons 2021,2022,2023,2024,2025
    python import_flashscore_stats.py --insecure --leagues E0 D1 SP1      # specific leagues
    python import_flashscore_stats.py --insecure --resume                 # continue Phase 2
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

LOG = logging.getLogger("flashscore_stats")

# ---------------------------------------------------------------------------
# League configuration (same as corners scraper)
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
QUEUE_CSV  = Path("data/flashscore/stats_queue.csv")
OUTPUT_CSV = Path("data/flashscore/match_stats.csv")

# Timing
DELAY_RESULTS = 3
DELAY_STATS   = 0.4
SAVE_EVERY    = 50

# Stats to extract (Flashscore SG field name → our column name)
STAT_MAP = {
    "expected goals (xg)":          "xg",
    "ball possession":              "possession",
    "total shots":                  "shots",
    "shots on target":              "shots_on_target",
    "shots off target":             "shots_off_target",
    "blocked shots":                "blocked_shots",
    "shots inside the box":         "shots_inside_box",
    "shots outside the box":        "shots_outside_box",
    "corner kicks":                 "corners",
    "fouls":                        "fouls",
    "yellow cards":                 "yellow_cards",
    "red cards":                    "red_cards",
    "big chances":                  "big_chances",
    "passes":                       "passes",
    "crosses":                      "crosses",
    "long passes":                  "long_passes",
    "passes in final third":        "passes_final_third",
    "offsides":                     "offsides",
    "free kicks":                   "free_kicks",
    "throw ins":                    "throw_ins",
    "tackles":                      "tackles",
    "duels won":                    "duels_won",
    "clearances":                   "clearances",
    "interceptions":                "interceptions",
    "goalkeeper saves":             "gk_saves",
    "hit the woodwork":             "woodwork",
    "expected assists (xa)":        "xa",
    "xg on target (xgot)":         "xgot",
    "accurate through passes":      "through_passes",
    "errors leading to shot":       "errors_to_shot",
    "errors leading to goal":       "errors_to_goal",
    "goals prevented":              "goals_prevented",
    "xgot faced":                   "xgot_faced",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _season_code(start: int) -> str:
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


def _safe_float(v: str) -> Optional[float]:
    """Parse float from strings like '0.45' or '88% (269/307)'."""
    try:
        v = v.strip()
        if v.endswith("%"):
            return float(v.rstrip("%")) / 100.0
        # Handle "88% (269/307)" format — take percentage
        m = re.match(r"([\d.]+)%", v)
        if m:
            return float(m.group(1)) / 100.0
        return float(v)
    except Exception:
        return None


def _results_urls(info: dict, season_start: Optional[int] = None) -> List[str]:
    country = info["country"]
    slugs = [info["slug"]] + info.get("alt_slugs", [])
    current_start = _season_start_from_date(dt.date.today())
    urls = []
    for s in slugs:
        if season_start is None or season_start == current_start:
            urls.append(f"https://www.flashscore.com/football/{country}/{s}/results/")
        else:
            urls.append(
                f"https://www.flashscore.com/football/{country}/"
                f"{s}-{season_start}-{season_start + 1}/results/"
            )
    return urls


# ---------------------------------------------------------------------------
# Phase 1 — Collect match IDs from results feeds
# ---------------------------------------------------------------------------

def _extract_feed(html: str) -> str:
    m = re.search(
        r"cjs\.initialFeeds\['results'\]\s*=\s*\{\s*data:\s*`(.*?)`,\s*allEventsCount:",
        html, re.DOTALL,
    )
    return m.group(1) if m else ""


def _parse_fields(chunk: str) -> dict:
    fields: dict = {}
    for token in chunk.split(chr(172)):
        if chr(247) in token:
            k, v = token.split(chr(247), 1)
            if k:
                fields[k] = v
    return fields


def _event_id_from_chunk(chunk: str) -> str:
    first = chunk.split(chr(172))[0]
    return first.lstrip(chr(247)).strip()


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
        hg = _safe_int(f.get("AG"))
        ag = _safe_int(f.get("AH"))
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
            "home_goals": hg,
            "away_goals": ag,
        })
    return out


def phase1_collect(
    leagues: List[str], season_starts: List[int], verify_ssl: bool
) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    current_start = _season_start_from_date(dt.date.today())
    all_matches: List[dict] = []

    for league in leagues:
        info = FLASH_LEAGUES[league]
        league_total = 0
        for ss in season_starts:
            urls = _results_urls(info, season_start=ss)
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
# Phase 2 — Fetch ALL stats via Flashscore internal API
# ---------------------------------------------------------------------------

def parse_all_stats(text: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Parse the ¬÷ encoded stats feed into {stat_name: (home_val, away_val)}."""
    stats: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    current_stat = ""
    home_val: Optional[str] = None
    for token in text.split(chr(172)):
        if chr(247) in token:
            k, v = token.split(chr(247), 1)
            if k == "SG":
                current_stat = v.lower().strip()
                home_val = None
            elif k == "SH":
                home_val = v
            elif k == "SI" and current_stat:
                stats[current_stat] = (home_val, v)
                # Only keep first occurrence of each stat
                current_stat = ""
    return stats


def fetch_match_stats(session: req.Session, event_id: str) -> Dict[str, Optional[float]]:
    """Fetch all stats for a match. Returns flat dict with home_* and away_* columns."""
    url = STATS_API.format(eid=event_id)
    r = session.get(url, timeout=15)
    r.raise_for_status()

    raw_stats = parse_all_stats(r.text)
    result: Dict[str, Optional[float]] = {}

    for fs_name, col_name in STAT_MAP.items():
        if fs_name in raw_stats:
            h_str, a_str = raw_stats[fs_name]
            result[f"home_{col_name}"] = _safe_float(h_str) if h_str else None
            result[f"away_{col_name}"] = _safe_float(a_str) if a_str else None
        else:
            result[f"home_{col_name}"] = None
            result[f"away_{col_name}"] = None

    return result


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape comprehensive Flashscore match statistics."
    )
    p.add_argument("--season-start", type=int, default=2025)
    p.add_argument("--seasons", default=None,
                   help="Comma-separated season starts, e.g. 2021,2022,2023,2024,2025")
    p.add_argument("--leagues", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
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
    LOG.info("FLASHSCORE COMPREHENSIVE STATS SCRAPER")
    LOG.info(f"  Seasons: {season_label}")
    LOG.info(f"  Leagues: {leagues}")
    LOG.info("=" * 60)

    QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1
    if not args.resume:
        LOG.info("\n[PHASE 1] Collecting match IDs ...")
        queue_df = phase1_collect(leagues, season_starts, verify_ssl)
        if queue_df.empty:
            LOG.error("No matches found!")
            return 1
        queue_df = queue_df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
        queue_df.to_csv(QUEUE_CSV, index=False)
        LOG.info(f"\n  Queue: {QUEUE_CSV} ({len(queue_df)} matches, {queue_df['league'].nunique()} leagues)")
    else:
        if not QUEUE_CSV.exists():
            LOG.error(f"No queue at {QUEUE_CSV}. Run without --resume first.")
            return 1
        queue_df = pd.read_csv(QUEUE_CSV, dtype={"event_id": str})
        LOG.info(f"  Resumed queue: {len(queue_df)} matches")

    # Phase 2
    LOG.info(f"\n[PHASE 2] Fetching match statistics via API ...")

    session = req.Session()
    session.headers.update(STATS_HEADERS)
    session.verify = verify_ssl

    done_ids: set[str] = set()
    if out_csv.exists():
        tmp = pd.read_csv(out_csv, dtype={"event_id": str})
        if "event_id" in tmp.columns:
            done_ids = set(tmp["event_id"].astype(str))
    LOG.info(f"  Already processed: {len(done_ids)}")

    todo = queue_df[~queue_df["event_id"].astype(str).isin(done_ids)].copy()
    if len(todo) > args.batch_size:
        todo = todo.head(args.batch_size)
    LOG.info(f"  To process: {len(todo)}")

    new_rows: List[dict] = []
    ok = 0
    errors = 0
    t_start = time.time()

    for idx, (_, row) in enumerate(todo.iterrows(), 1):
        eid = str(row["event_id"])
        try:
            stats = fetch_match_stats(session, eid)

            # Derive season from match date
            md_str = row.get("match_date", "")
            if md_str:
                ss = _season_start_from_date(dt.date.fromisoformat(md_str))
                season_code = _season_code(ss)
            else:
                season_code = _season_code(season_starts[0])

            rec = {
                "event_id": eid,
                "source": "flashscore",
                "league": row["league"],
                "season": season_code,
                "match_date": md_str,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row.get("home_goals"),
                "away_goals": row.get("away_goals"),
                **stats,
            }
            new_rows.append(rec)
            ok += 1

            if ok % 20 == 0 or ok <= 3:
                elapsed = time.time() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(todo) - idx) / rate if rate > 0 else 0
                LOG.info(
                    f"    [{ok:5d}/{len(todo)}] {row['league']} "
                    f"{row['home_team']} v {row['away_team']}  "
                    f"({rate:.1f}/s, ETA {eta:.0f}s)"
                )

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
    LOG.info(f"\n  Stats fetched: {ok}")
    LOG.info(f"  Errors       : {errors}")
    LOG.info(f"  Time         : {elapsed:.0f}s ({elapsed/max(ok,1):.2f}s/match)")
    LOG.info(f"  Output       : {out_csv}")

    # Summary
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        LOG.info(f"\n  Total rows: {len(df)}")
        LOG.info(f"  Leagues: {sorted(df['league'].unique())}")
        non_null = df.notna().sum()
        stat_cols = [c for c in df.columns if c.startswith("home_xg") or c.startswith("home_shots")]
        for c in stat_cols:
            LOG.info(f"  {c}: {non_null.get(c, 0)} non-null of {len(df)}")

    LOG.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())