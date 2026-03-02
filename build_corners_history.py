from __future__ import annotations

"""
build_corners_history.py
Incrementally fetches corner data from API-Football /fixtures/statistics.
Resumable: already-processed fixture IDs are skipped on subsequent runs.

Run daily until corner history is complete:
    python build_corners_history.py --api-key YOUR_KEY
"""

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

API_BASE = "https://v3.football.api-sports.io"

LEAGUE_IDS: dict[str, int] = {
    "E0": 39,
    "D1": 78,
    "SP1": 140,
    "I1": 135,
    "F1": 61,
    "N1": 88,
    "P1": 94,
    "E1": 40,
    "RO1": 283,
    "RS1": 286,
    "SA1": 307,
}

QUEUE_PATH = Path("simulations/Corners/data/fixture_queue.csv")
HISTORY_PATH = Path("simulations/Corners/data/corners_history.csv")
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get(session: requests.Session, path: str, params: dict, timeout: int = 30) -> dict:
    r = session.get(f"{API_BASE}{path}", params=params, timeout=timeout, verify=False)
    r.raise_for_status()
    payload = r.json()
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API error: {errors}")
    return payload


def remaining_calls(session: requests.Session) -> int:
    r = session.get(f"{API_BASE}/status", timeout=15, verify=False)
    data = r.json()
    resp = data.get("response", {})
    if isinstance(resp, list):
        resp = resp[0] if resp else {}
    req = resp.get("requests", {})
    used = req.get("current", 0)
    limit = req.get("limit_day", 100)
    left = limit - used
    print(f"  API calls: {used} / {limit} -> {left} remaining")
    return left


def build_queue(session: requests.Session, seasons: list[int], budget: int) -> tuple[pd.DataFrame, int]:
    """Fetch fixture IDs for all leagues/seasons not yet in the queue. Returns updated queue and calls used."""
    existing = pd.read_csv(QUEUE_PATH) if QUEUE_PATH.exists() else pd.DataFrame()
    already_done = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            already_done.add((row["league"], int(row["season"])))

    rows: list[dict] = []
    calls_used = 0
    for league, league_id in LEAGUE_IDS.items():
        for season in seasons:
            if (league, season) in already_done:
                continue
            if calls_used >= budget:
                print(f"  [BUDGET] Stopping fixture ID fetch at {league} {season}")
                break
            print(f"  [QUEUE] Fetching fixture IDs: {league} {season} ...", end=" ")
            try:
                time.sleep(7)
                payload = _get(session, "/fixtures", {"league": league_id, "season": season, "status": "FT"})
                calls_used += 1
                items = payload.get("response", [])
                for it in items:
                    fix = it.get("fixture", {})
                    teams = it.get("teams", {})
                    fid = fix.get("id")
                    date = fix.get("date", "")[:10]
                    home = (teams.get("home", {}).get("name") or "").strip().lower()
                    away = (teams.get("away", {}).get("name") or "").strip().lower()
                    if fid and home and away:
                        rows.append({
                            "fixture_id": fid,
                            "league": league,
                            "season": season,
                            "match_date": date,
                            "home_team": home,
                            "away_team": away,
                            "processed": False,
                        })
                print(f"{len(items)} fixtures")
            except Exception as e:
                print(f"ERROR: {e}")
        else:
            continue
        break

    if rows:
        new_df = pd.DataFrame(rows)
        queue = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        queue = queue.drop_duplicates(subset=["fixture_id"]).reset_index(drop=True)
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        queue.to_csv(QUEUE_PATH, index=False)
        print(f"  Queue saved: {len(queue)} total fixtures ({len(rows)} new)")
    else:
        queue = existing

    return queue, calls_used


def fetch_statistics(session: requests.Session, queue: pd.DataFrame, budget: int) -> tuple[int, int]:
    """Call /fixtures/statistics for unprocessed fixtures. Returns (stats_fetched, calls_used)."""
    existing_history = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()
    processed_ids = set(existing_history["fixture_id"].astype(int).tolist()) if not existing_history.empty else set()

    unprocessed = queue[~queue["fixture_id"].astype(int).isin(processed_ids)].copy()
    print(f"  Fixtures to process: {len(unprocessed)} unprocessed of {len(queue)} total")

    new_rows: list[dict] = []
    calls_used = 0
    skipped_null = 0

    for _, row in unprocessed.iterrows():
        if calls_used >= budget:
            break
        fid = int(row["fixture_id"])
        try:
            time.sleep(7)
            payload = _get(session, "/fixtures/statistics", {"fixture": fid})
            calls_used += 1
            stats_list = payload.get("response", [])

            home_corners = None
            away_corners = None
            for i, team_block in enumerate(stats_list[:2]):
                for stat in team_block.get("statistics", []):
                    if stat.get("type") == "Corner Kicks":
                        val = stat.get("value")
                        if val is not None:
                            if i == 0:
                                home_corners = int(val)
                            else:
                                away_corners = int(val)

            if home_corners is None and away_corners is None:
                skipped_null += 1
                continue

            new_rows.append({
                "fixture_id": fid,
                "league": row["league"],
                "season": int(row["season"]),
                "match_date": row["match_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_corners": home_corners,
                "away_corners": away_corners,
            })

            # Save incrementally every 10 records
            if len(new_rows) % 10 == 0:
                _append_and_save(existing_history, new_rows)

        except Exception as e:
            print(f"  [WARN] fixture {fid}: {e}")

    if new_rows:
        _append_and_save(existing_history, new_rows)

    return len(new_rows), calls_used, skipped_null


def _append_and_save(existing: pd.DataFrame, new_rows: list[dict]) -> None:
    new_df = pd.DataFrame(new_rows)
    out = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    out = out.drop_duplicates(subset=["fixture_id"]).reset_index(drop=True)
    out.to_csv(HISTORY_PATH, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incrementally build corner history from API-Football.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--seasons", default="2023,2024", help="Comma-separated season start years")
    p.add_argument("--reserve", type=int, default=2, help="Keep N calls in reserve (safety buffer)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]

    session = requests.Session()
    session.headers.update({"x-apisports-key": args.api_key})

    print("=" * 60)
    print("CORNERS HISTORY BUILDER — API-Football")
    print("=" * 60)

    left = remaining_calls(session)
    usable = left - args.reserve

    if usable <= 0:
        print(f"[LIMIT] Daily limit reached. 0 usable calls remaining.")
        return 0

    print(f"\nUsable budget today: {usable} calls\n")

    # Phase 1: build fixture queue
    print("[PHASE 1] Building fixture ID queue...")
    queue_budget = min(len(LEAGUE_IDS) * len(seasons), usable)
    queue, queue_calls = build_queue(session, seasons, budget=queue_budget)
    usable -= queue_calls
    print(f"  Calls used for queue: {queue_calls} | Remaining budget: {usable}\n")

    if queue.empty or usable <= 0:
        print("[LIMIT] No budget left after building queue.")
        _print_status()
        return 0

    # Phase 2: fetch statistics
    print("[PHASE 2] Fetching corner statistics...")
    stats_fetched, stats_calls, skipped = fetch_statistics(session, queue, budget=usable)
    usable -= stats_calls

    print(f"\n  Corner records fetched today: {stats_fetched}")
    print(f"  Fixtures skipped (null stats): {skipped}")
    print(f"  Calls used for stats: {stats_calls}")

    print("\n" + "=" * 60)
    _print_status()
    remaining_calls(session)
    print("=" * 60)
    return 0


def _print_status() -> None:
    if HISTORY_PATH.exists():
        h = pd.read_csv(HISTORY_PATH)
        print(f"CORNERS HISTORY: {len(h)} records saved")
        if not h.empty:
            print(f"  Leagues: {sorted(h['league'].unique().tolist())}")
            print(f"  Seasons: {sorted(h['season'].unique().tolist())}")
            by_league = h.groupby("league").size().sort_values(ascending=False)
            print(f"  By league:\n{by_league.to_string()}")
    else:
        print("No corner history yet.")
    if QUEUE_PATH.exists():
        q = pd.read_csv(QUEUE_PATH)
        processed = pd.read_csv(HISTORY_PATH)["fixture_id"].nunique() if HISTORY_PATH.exists() else 0
        print(f"\nQUEUE: {len(q)} total fixtures | {processed} processed | {len(q) - processed} remaining")


if __name__ == "__main__":
    raise SystemExit(main())
