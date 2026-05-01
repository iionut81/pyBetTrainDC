"""Rebuild Flashscore match_stats.csv with fixed parser.

Reads existing match_stats.csv, re-fetches stats for each event_id
using the corrected parser (full match values, not 2nd half).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd
import requests as req
import urllib3

from import_flashscore_stats import (
    fetch_match_stats,
    STATS_API,
    STATS_HEADERS,
)

OUTPUT_CSV = Path("data/flashscore/match_stats.csv")
DELAY = 0.4

logging.basicConfig(format="%(asctime)s  %(levelname)s   %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild Flashscore stats with fixed parser")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--input", default=str(OUTPUT_CSV))
    p.add_argument("--output", default=str(OUTPUT_CSV) + ".new")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="Limit matches (0=all)")
    args = p.parse_args()

    if args.insecure:
        urllib3.disable_warnings()

    src = Path(args.input)
    dst = Path(args.output)

    log.info("=" * 60)
    log.info("REBUILD FLASHSCORE STATS (fixed parser)")
    log.info(f"  Input:  {src}")
    log.info(f"  Output: {dst}")
    log.info("=" * 60)

    df_in = pd.read_csv(src, dtype={"event_id": str})
    log.info(f"  Loaded {len(df_in)} rows from {src}")

    # Skip already-rebuilt rows if output exists
    done_ids: set = set()
    if dst.exists():
        df_done = pd.read_csv(dst, dtype={"event_id": str})
        done_ids = set(df_done["event_id"].astype(str).tolist())
        log.info(f"  Resuming: {len(done_ids)} already rebuilt")

    todo = df_in[~df_in["event_id"].astype(str).isin(done_ids)].copy()
    if args.limit > 0:
        todo = todo.head(args.limit)
    log.info(f"  To process: {len(todo)}")

    session = req.Session()
    session.headers.update(STATS_HEADERS)
    session.verify = not args.insecure

    new_rows: list[dict] = []
    n_done = 0
    n_err = 0
    t0 = time.time()

    for _, row in todo.iterrows():
        event_id = str(row["event_id"])
        try:
            stats = fetch_match_stats(session, event_id)
            time.sleep(DELAY)

            # Build full row: keep metadata from input, replace stats
            new_row = {
                "event_id": event_id,
                "source": row["source"],
                "league": row["league"],
                "season": row["season"],
                "match_date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
            }
            new_row.update(stats)
            new_rows.append(new_row)
            n_done += 1
        except Exception as e:
            n_err += 1
            log.warning(f"  Error on {event_id}: {e}")

        if (n_done + n_err) % args.save_every == 0:
            elapsed = time.time() - t0
            rate = (n_done + n_err) / elapsed if elapsed > 0 else 0
            eta = (len(todo) - n_done - n_err) / rate if rate > 0 else 0
            log.info(f"  [{n_done + n_err}/{len(todo)}] "
                     f"OK={n_done} ERR={n_err} "
                     f"({rate:.2f}/s, ETA {eta/60:.1f}min)")
            _flush(new_rows, dst)
            new_rows = []

    if new_rows:
        _flush(new_rows, dst)

    log.info("=" * 60)
    log.info(f"  DONE — OK={n_done} ERR={n_err} in {(time.time()-t0)/60:.1f}min")
    log.info(f"  Output: {dst}")
    log.info("=" * 60)
    return 0


def _flush(rows: list[dict], dst: Path) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if dst.exists():
        existing = pd.read_csv(dst, dtype={"event_id": str})
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    combined.to_csv(dst, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
