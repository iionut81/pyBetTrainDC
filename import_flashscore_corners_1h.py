from __future__ import annotations

"""
import_flashscore_corners_1h.py

Scrape first-half corner statistics from Flashscore for all 20 leagues.
Reuses the Phase-1 match-ID collection logic from import_flashscore_corners.py
and the _parse_corners_all_periods() function added to that module.

Output:
    simulations/Corners 1H/data/corners_1h_flashscore.csv   — raw per-match
    simulations/Corners 1H/data/corners_1h_history.csv      — deduplicated, merged

Usage:
    python import_flashscore_corners_1h.py --insecure
    python import_flashscore_corners_1h.py --insecure --seasons 2021,2022,2023,2024,2025
    python import_flashscore_corners_1h.py --insecure --leagues E0 D1 --resume
"""

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests as req

from import_flashscore_corners import (
    FLASH_LEAGUES,
    STATS_API,
    STATS_HEADERS,
    _season_code,
    _season_start_from_date,
    _parse_corners_all_periods,
    phase1_collect,
)

LOG = logging.getLogger("flashscore_corners_1h")

QUEUE_CSV   = Path("simulations/Corners 1H/data/flashscore_1h_queue.csv")
OUTPUT_CSV  = Path("simulations/Corners 1H/data/corners_1h_flashscore.csv")
HISTORY_CSV = Path("simulations/Corners 1H/data/corners_1h_history.csv")

DELAY_STATS = 0.4
SAVE_EVERY  = 50

OUT_COLUMNS = [
    "event_id", "source", "league", "season", "match_date",
    "home_team", "away_team",
    "home_corners_1h", "away_corners_1h",
    "home_corners_2h", "away_corners_2h",
    "home_corners_total", "away_corners_total",
]


# ---------------------------------------------------------------------------
# Incremental save
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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)


def _merge_history(out_csv: Path) -> None:
    if not out_csv.exists():
        return
    fs = pd.read_csv(out_csv, dtype={"event_id": str})
    fs = fs.dropna(subset=["home_corners_1h", "away_corners_1h"]).copy()
    if fs.empty:
        LOG.info("  Nothing to merge (no valid 1H rows)")
        return
    base = pd.read_csv(HISTORY_CSV) if HISTORY_CSV.exists() else pd.DataFrame()
    merged = pd.concat([base, fs], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["league", "match_date", "home_team", "away_team"], keep="last"
    )
    merged = merged.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(HISTORY_CSV, index=False)
    LOG.info(f"  Merged into {HISTORY_CSV}: {len(merged)} total rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape Flashscore 1st-half corner statistics."
    )
    p.add_argument("--season-start", type=int, default=2025)
    p.add_argument("--seasons", default=None,
                   help="Comma-separated season starts, e.g. 2021,2022,2023,2024,2025")
    p.add_argument("--leagues", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--insecure", action="store_true")
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
    verify_ssl = not args.insecure

    for lg in leagues:
        if lg not in FLASH_LEAGUES:
            LOG.error(f"Unknown league '{lg}'.")
            return 1

    season_label = ",".join(str(s) for s in season_starts)
    LOG.info("=" * 60)
    LOG.info("FLASHSCORE 1H CORNERS SCRAPER")
    LOG.info(f"  Seasons: {season_label}")
    LOG.info(f"  Leagues: {leagues}")
    LOG.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1 — collect match IDs
    # ------------------------------------------------------------------
    if not args.resume:
        LOG.info("\n[PHASE 1] Collecting match IDs ...")
        queue_df = phase1_collect(leagues, season_starts, verify_ssl)
        if queue_df.empty:
            LOG.error("No matches found in Phase 1!")
            return 1
        queue_df = queue_df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
        QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
        queue_df.to_csv(QUEUE_CSV, index=False)
        LOG.info(f"  Queue: {len(queue_df)} matches → {QUEUE_CSV}")
    else:
        if not QUEUE_CSV.exists():
            LOG.error(f"Queue not found at {QUEUE_CSV}. Run without --resume first.")
            return 1
        queue_df = pd.read_csv(QUEUE_CSV, dtype={"event_id": str})
        LOG.info(f"  Resumed queue: {len(queue_df)} matches")

    # ------------------------------------------------------------------
    # Phase 2 — fetch 1H corner stats
    # ------------------------------------------------------------------
    LOG.info("\n[PHASE 2] Fetching 1H corner statistics ...")

    session = req.Session()
    session.headers.update(STATS_HEADERS)
    session.verify = verify_ssl

    done_ids: set[str] = set()
    if OUTPUT_CSV.exists():
        tmp = pd.read_csv(OUTPUT_CSV, dtype={"event_id": str})
        if "event_id" in tmp.columns:
            done_ids = set(tmp["event_id"].astype(str))
    LOG.info(f"  Already processed: {len(done_ids)}")

    todo = queue_df[~queue_df["event_id"].astype(str).isin(done_ids)].copy()
    if len(todo) > args.batch_size:
        todo = todo.head(args.batch_size)
    LOG.info(f"  To process: {len(todo)}")

    new_rows: List[dict] = []
    ok = 0
    no_1h = 0
    errors = 0
    t_start = time.time()

    for idx, (_, row) in enumerate(todo.iterrows(), 1):
        eid = str(row["event_id"])
        try:
            url = STATS_API.format(eid=eid)
            r = session.get(url, timeout=15)
            r.raise_for_status()
            d = _parse_corners_all_periods(r.text)

            md_str = row.get("match_date", "")
            if md_str:
                md_date = dt.date.fromisoformat(str(md_str))
                ss = _season_start_from_date(md_date)
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
                "home_corners_1h": d["h1_h"],
                "away_corners_1h": d["h1_a"],
                "home_corners_2h": d["h2_h"],
                "away_corners_2h": d["h2_a"],
                "home_corners_total": d["match_h"],
                "away_corners_total": d["match_a"],
            }
            new_rows.append(rec)

            has_1h = d["h1_h"] is not None
            if has_1h:
                ok += 1
                if ok % 50 == 0 or ok <= 5:
                    elapsed = time.time() - t_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - idx) / rate if rate > 0 else 0
                    LOG.info(
                        f"    [{idx:5d}/{len(todo)}] {row['league']} "
                        f"{row['home_team']} v {row['away_team']}  "
                        f"1H={d['h1_h']}-{d['h1_a']} TOT={d['match_h']}-{d['match_a']}"
                        f"  ({rate:.1f}/s ETA {eta:.0f}s)"
                    )
            else:
                no_1h += 1

            if len(new_rows) % SAVE_EVERY == 0:
                _save_incremental(OUTPUT_CSV, new_rows)

            time.sleep(DELAY_STATS)

        except Exception as exc:
            errors += 1
            if errors <= 5:
                LOG.warning(f"    [{eid}] {exc}")
            elif errors == 6:
                LOG.warning("    (suppressing further errors...)")

    if new_rows:
        _save_incremental(OUTPUT_CSV, new_rows)

    elapsed = time.time() - t_start
    LOG.info(f"\n  With 1H data  : {ok}")
    LOG.info(f"  No 1H data    : {no_1h}")
    LOG.info(f"  Errors        : {errors}")
    LOG.info(f"  Time          : {elapsed:.0f}s ({elapsed / max(ok + no_1h, 1):.2f}s/match)")
    LOG.info(f"  Output        : {OUTPUT_CSV}")

    if not args.no_merge:
        LOG.info("\n[MERGE] Merging into corners_1h_history.csv ...")
        _merge_history(OUTPUT_CSV)

    LOG.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
