from __future__ import annotations

"""
import_sofascore_stats.py

Scrape comprehensive match statistics from Sofascore for all 20 leagues.
Sofascore replacement for import_flashscore_stats.py — writes to the SAME
data/flashscore/match_stats.csv (same 75-column schema) so train_team_ratings.py,
train_goals_totals.py and train_corners_under_12_5.py need no changes.

Phase 1 — Collect finished matches (event id, teams, goals, date) via
          sofascore_loader.fetch_full_season_history() (round-by-round backfill)
Phase 2 — Fetch full-match statistics per event via
          sofascore_loader.fetch_match_statistics_flat()

Rows are tagged source="sofascore" and deduplicated on (source, event_id) —
NOT on event_id alone, since Flashscore and Sofascore event ids are
independent numbering spaces and could collide by coincidence.

Usage:
    python import_sofascore_stats.py --insecure                          # all leagues, current season
    python import_sofascore_stats.py --insecure --seasons 2024,2025
    python import_sofascore_stats.py --insecure --leagues E0 D1 SP1
    python import_sofascore_stats.py --insecure --resume                 # continue Phase 2
"""

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from sofascore_loader import (
    LEAGUE_TOURNAMENT_IDS,
    fetch_full_season_history,
    fetch_match_statistics_flat,
    find_season_id_for_start_year,
    get_current_season_id,
)

LOG = logging.getLogger("sofascore_stats")

QUEUE_CSV = Path("data/flashscore/sofascore_stats_queue.csv")
OUTPUT_CSV = Path("data/flashscore/match_stats.csv")

DELAY_ROUNDS = 0.4
DELAY_STATS = 0.4
SAVE_EVERY = 50

# our_column -> (candidate Sofascore item names, parse mode)
# parse modes: "number" (plain float), "percent" ("52%" -> 0.52),
#              "fraction_total" ("18/37 (49%)" -> 37, the attempted count)
STAT_MAP: Dict[str, Tuple[List[str], str]] = {
    "xg":                   (["expected goals"], "number"),
    "possession":            (["ball possession"], "percent"),
    "shots":                 (["total shots"], "number"),
    "shots_on_target":       (["shots on target"], "number"),
    "shots_off_target":      (["shots off target"], "number"),
    "blocked_shots":         (["blocked shots"], "number"),
    "shots_inside_box":      (["shots inside box"], "number"),
    "shots_outside_box":     (["shots outside box"], "number"),
    "corners":                (["corner kicks"], "number"),
    "fouls":                 (["fouls"], "number"),
    "yellow_cards":          (["yellow cards"], "number"),
    "red_cards":              (["red cards"], "number"),
    "big_chances":            (["big chances"], "number"),
    "passes":                 (["passes"], "number"),
    "crosses":                (["crosses"], "fraction_total"),
    "long_passes":            (["long balls"], "fraction_total"),
    "passes_final_third":     (["final third entries"], "number"),
    "offsides":                (["offsides"], "number"),
    "free_kicks":              (["free kicks"], "number"),
    "throw_ins":                (["throw-ins", "throw ins"], "number"),
    "tackles":                   (["tackles"], "number"),
    "duels_won":                  (["duels"], "percent"),  # NOTE: share of duels won, not a raw count
    "clearances":                  (["clearances"], "number"),
    "interceptions":                (["interceptions"], "number"),
    "gk_saves":                      (["goalkeeper saves"], "number"),
    "woodwork":                       (["hit woodwork"], "number"),
    "xa":                              (["expected assists"], "number"),
    "xgot":                             (["xgot", "expected goals on target"], "number"),
    "through_passes":                   (["through balls"], "number"),
    "errors_to_shot":                    (["errors leading to shot"], "number"),
    "errors_to_goal":                     (["errors leading to goal"], "number"),
    "goals_prevented":                     (["goals prevented"], "number"),
    "xgot_faced":                           (["xgot faced"], "number"),
}


def _season_code(start: int) -> str:
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _season_start_from_date(d: dt.date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def _parse_stat_value(raw: Optional[str], mode: str) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if mode == "percent":
            return float(s.rstrip("%")) / 100.0
        if mode == "fraction_total":
            # "18/37 (49%)" -> 37 (attempted count)
            left = s.split("(")[0].strip()
            if "/" in left:
                return float(left.split("/")[1].strip())
            return float(left)
        return float(s)
    except Exception:
        return None


def _extract_stats_row(flat: Dict[str, tuple]) -> Dict[str, Optional[float]]:
    result: Dict[str, Optional[float]] = {}
    for col_name, (candidates, mode) in STAT_MAP.items():
        home_val: Optional[float] = None
        away_val: Optional[float] = None
        for cand in candidates:
            if cand in flat:
                h_raw, a_raw = flat[cand]
                home_val = _parse_stat_value(h_raw, mode)
                away_val = _parse_stat_value(a_raw, mode)
                break
        result[f"home_{col_name}"] = home_val
        result[f"away_{col_name}"] = away_val
    return result


def phase1_collect(
    leagues: List[str], season_starts: List[int], verify_ssl: bool
) -> pd.DataFrame:
    current_start = _season_start_from_date(dt.date.today())
    all_matches: List[dict] = []

    for league in leagues:
        tournament_id = LEAGUE_TOURNAMENT_IDS[league]
        league_total = 0
        for ss in season_starts:
            if ss == current_start:
                season_id = get_current_season_id(tournament_id, verify_ssl=verify_ssl)
            else:
                season_id = find_season_id_for_start_year(tournament_id, ss, verify_ssl=verify_ssl)
            if season_id is None:
                LOG.warning(f"  {league} {ss}: no season id found, skipping")
                continue

            matches = fetch_full_season_history(
                league, season_id=season_id, sleep_s=DELAY_ROUNDS, verify_ssl=verify_ssl
            )
            for m in matches:
                all_matches.append({
                    "event_id": str(m.event_id),
                    "league": league,
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "match_date": m.date,
                    "home_goals": m.home_score,
                    "away_goals": m.away_score,
                })
            league_total += len(matches)
        LOG.info(f"  {league}: {league_total} matches ({len(season_starts)} seasons)")

    return pd.DataFrame(all_matches)


def _save_incremental(out_csv: Path, new_rows: List[dict]) -> None:
    new_df = pd.DataFrame(new_rows)
    if out_csv.exists():
        existing = pd.read_csv(out_csv, dtype={"event_id": str})
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    if "source" not in combined.columns:
        combined["source"] = "flashscore"  # pre-existing rows predate the source column
    combined = combined.drop_duplicates(subset=["source", "event_id"], keep="last")
    combined = combined.sort_values(["match_date", "league"]).reset_index(drop=True)
    combined.to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape comprehensive Sofascore match statistics.")
    p.add_argument("--season-start", type=int, default=None)
    p.add_argument("--seasons", default=None,
                   help="Comma-separated season starts, e.g. 2023,2024,2025")
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
    current_start = _season_start_from_date(dt.date.today())
    if args.seasons:
        season_starts = [int(x.strip()) for x in args.seasons.split(",")]
    else:
        season_starts = [args.season_start or current_start]
    leagues = args.leagues or list(LEAGUE_TOURNAMENT_IDS.keys())
    out_csv = Path(args.out_csv) if args.out_csv else OUTPUT_CSV
    verify_ssl = not args.insecure

    for lg in leagues:
        if lg not in LEAGUE_TOURNAMENT_IDS:
            LOG.error(f"Unknown league '{lg}'. Available: {list(LEAGUE_TOURNAMENT_IDS.keys())}")
            return 1

    season_label = ",".join(str(s) for s in season_starts)
    LOG.info("=" * 60)
    LOG.info("SOFASCORE COMPREHENSIVE STATS SCRAPER")
    LOG.info(f"  Seasons: {season_label}")
    LOG.info(f"  Leagues: {leagues}")
    LOG.info("=" * 60)

    QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1
    if not args.resume:
        LOG.info("\n[PHASE 1] Collecting matches ...")
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

    done_ids: set[str] = set()
    if out_csv.exists():
        tmp = pd.read_csv(out_csv, dtype={"event_id": str})
        if "event_id" in tmp.columns and "source" in tmp.columns:
            done_ids = set(tmp.loc[tmp["source"] == "sofascore", "event_id"].astype(str))
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
            flat = fetch_match_statistics_flat(int(eid), verify_ssl=verify_ssl)
            stats = _extract_stats_row(flat)

            md_str = row.get("match_date", "")
            if md_str:
                ss = _season_start_from_date(dt.date.fromisoformat(md_str))
                season_code = _season_code(ss)
            else:
                season_code = _season_code(season_starts[0])

            rec = {
                "event_id": eid,
                "source": "sofascore",
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

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        LOG.info(f"\n  Total rows: {len(df)}")
        LOG.info(f"  By source: {df['source'].value_counts().to_dict() if 'source' in df.columns else 'n/a'}")

    LOG.info("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
