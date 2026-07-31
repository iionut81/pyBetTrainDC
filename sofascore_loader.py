"""Sofascore data loader — PRIMARY data source for fixtures, historical results
and goal-minute data (replaces Flashscore as primary; Flashscore stays as backup
in data_loader.py).

Public, unauthenticated JSON API (api.sofascore.com). No API key required.
Endpoints used:
  - /api/v1/search/all?q=...                                  -> resolve team/tournament IDs
  - /api/v1/unique-tournament/{ut}/seasons                    -> current season id
  - /api/v1/unique-tournament/{ut}/season/{s}/events/next/0   -> upcoming fixtures (daily refresh)
  - /api/v1/unique-tournament/{ut}/season/{s}/events/round/{r}-> full round (historical backfill)
  - /api/v1/team/{id}/events/last/0                           -> team recent results (FT + HT score)
  - /api/v1/event/{id}/incidents                              -> exact goal minutes
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, List, Optional
from urllib.parse import quote

from data_loader import Fixture, _norm_team

BASE_URL = "https://api.sofascore.com/api/v1"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# NOTE: api.sofascore.com fingerprints the TLS/HTTP client. Python's own
# ssl stack (requests/urllib) gets a 403 Forbidden after a handful of calls,
# while curl's handshake keeps working indefinitely. We therefore shell out
# to curl instead of using `requests` for this loader. Confirmed 2026-07-31.

# code -> Sofascore unique-tournament id (stable across seasons)
LEAGUE_TOURNAMENT_IDS: Dict[str, int] = {
    "E0": 17, "E1": 18,
    "D1": 35, "D2": 44,
    "SP1": 8, "SP2": 54,
    "I1": 23, "I2": 53,
    "F1": 34,
    "N1": 37,
    "P1": 238,
    "RO1": 152,
    "RS1": 210,
    "SA1": 955,
    "SW1": 215,
    "DK1": 39,
    "B1": 38, "B2": 9,
    "TR1": 52, "TR2": 98,
}

_season_cache: Dict[int, int] = {}


def _get(path: str, timeout: int = 20, verify_ssl: bool = True) -> Optional[dict]:
    url = f"{BASE_URL}{path}"
    args = [
        "curl", "-s", "--max-time", str(timeout),
        "-w", "\n%{http_code}",
        "-H", f"User-Agent: {USER_AGENT}",
        "-H", "Accept: application/json",
    ]
    if not verify_ssl:
        args.append("-k")
    args.append(url)
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout + 5,
            encoding="utf-8", errors="replace",
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    body, _, code = proc.stdout.rpartition("\n")
    if code.strip() != "200":
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


def get_current_season_id(tournament_id: int, verify_ssl: bool = True) -> Optional[int]:
    """Return the most recent season id for a tournament (cached per run)."""
    if tournament_id in _season_cache:
        return _season_cache[tournament_id]
    data = _get(f"/unique-tournament/{tournament_id}/seasons", verify_ssl=verify_ssl)
    seasons = (data or {}).get("seasons", [])
    if not seasons:
        return None
    season_id = seasons[0].get("id")
    _season_cache[tournament_id] = season_id
    return season_id


def find_season_id_for_start_year(
    tournament_id: int, season_start: int, verify_ssl: bool = True
) -> Optional[int]:
    """Resolve the season id whose 'year' label (e.g. '24/25') matches a given
    season-start year (2024). Used for multi-season historical backfill."""
    data = _get(f"/unique-tournament/{tournament_id}/seasons", verify_ssl=verify_ssl)
    seasons = (data or {}).get("seasons", [])
    target = f"{season_start % 100:02d}"
    for s in seasons:
        year = str(s.get("year", ""))
        if year.startswith(target) or year == str(season_start):
            return s.get("id")
    return None


def search_team_id(name: str, verify_ssl: bool = True) -> Optional[int]:
    """Resolve a team name to its Sofascore team id via the search endpoint."""
    data = _get(f"/search/all?q={quote(name)}", verify_ssl=verify_ssl)
    for r in (data or {}).get("results", []):
        if r.get("type") == "team":
            ent = r.get("entity", {})
            sport = (ent.get("sport") or {}).get("name")
            if sport == "Football":
                return ent.get("id")
    return None


def _event_date_iso(event: dict) -> Optional[str]:
    ts = event.get("startTimestamp")
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).date().isoformat()


def fetch_fixtures_from_sofascore(target_date_iso: str, verify_ssl: bool = True) -> List[Fixture]:
    """Daily fixture refresh across the 20 football leagues. Drop-in replacement
    for fetch_fixtures_from_flashscore() — returns the same Fixture type."""
    out: List[Fixture] = []
    for league_code, tournament_id in LEAGUE_TOURNAMENT_IDS.items():
        season_id = get_current_season_id(tournament_id, verify_ssl=verify_ssl)
        if season_id is None:
            continue
        for feed in ("events/next/0", "events/last/0"):
            data = _get(
                f"/unique-tournament/{tournament_id}/season/{season_id}/{feed}",
                verify_ssl=verify_ssl,
            )
            for event in (data or {}).get("events", []):
                if _event_date_iso(event) != target_date_iso:
                    continue
                home = event.get("homeTeam", {}).get("name")
                away = event.get("awayTeam", {}).get("name")
                if not home or not away:
                    continue
                out.append(
                    Fixture(
                        league=league_code,
                        match_date=target_date_iso,
                        home_team=_norm_team(home),
                        away_team=_norm_team(away),
                        odds_1x=None,
                        odds_x2=None,
                        fixture_id=event.get("id"),
                    )
                )
        time.sleep(0.4)  # be polite to the unofficial API, avoid rate-limit blocks

    uniq: Dict[tuple, Fixture] = {}
    for f in out:
        uniq[(f.league, f.match_date, f.home_team, f.away_team)] = f
    return list(uniq.values())


@dataclass
class RecentMatch:
    event_id: int
    league: str
    date: str
    home_team: str
    away_team: str
    home_score: Optional[int]
    away_score: Optional[int]
    home_score_ht: Optional[int]
    away_score_ht: Optional[int]


def fetch_recent_team_matches(team_id: int, n: int = 20, verify_ssl: bool = True) -> List[RecentMatch]:
    """Last N finished matches for a team, with full-time and half-time scores."""
    data = _get(f"/team/{team_id}/events/last/0", verify_ssl=verify_ssl)
    out: List[RecentMatch] = []
    for event in (data or {}).get("events", [])[:n]:
        home = event.get("homeTeam", {}) or {}
        away = event.get("awayTeam", {}) or {}
        hs = event.get("homeScore", {}) or {}
        as_ = event.get("awayScore", {}) or {}
        out.append(
            RecentMatch(
                event_id=event.get("id"),
                league=(event.get("tournament", {}) or {}).get("name", ""),
                date=_event_date_iso(event) or "",
                home_team=_norm_team(home.get("name", "")),
                away_team=_norm_team(away.get("name", "")),
                home_score=hs.get("current"),
                away_score=as_.get("current"),
                home_score_ht=hs.get("period1"),
                away_score_ht=as_.get("period1"),
            )
        )
    return out


@dataclass
class GoalEvent:
    minute: int
    added_time: Optional[int]
    is_home: bool
    home_score_after: int
    away_score_after: int


def fetch_goal_minutes(event_id: int, verify_ssl: bool = True) -> List[GoalEvent]:
    """Exact minute of every goal in a finished match."""
    data = _get(f"/event/{event_id}/incidents", verify_ssl=verify_ssl)
    out: List[GoalEvent] = []
    for inc in (data or {}).get("incidents", []):
        if inc.get("incidentType") != "goal":
            continue
        out.append(
            GoalEvent(
                minute=inc.get("time"),
                added_time=inc.get("addedTime"),
                is_home=bool(inc.get("isHome")),
                home_score_after=inc.get("homeScore"),
                away_score_after=inc.get("awayScore"),
            )
        )
    return out


def fetch_match_statistics_flat(event_id: int, verify_ssl: bool = True) -> Dict[str, tuple]:
    """Full-match statistics for one event, flattened across all groups into
    {stat_name_lower: (home_value_str, away_value_str)}. Uses the 'ALL' period
    (full 90 min), not the 1st/2nd-half breakdowns."""
    data = _get(f"/event/{event_id}/statistics", verify_ssl=verify_ssl)
    out: Dict[str, tuple] = {}
    for period_block in (data or {}).get("statistics", []):
        if period_block.get("period") != "ALL":
            continue
        for group in period_block.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = str(item.get("name", "")).lower().strip()
                if name:
                    out[name] = (item.get("home"), item.get("away"))
    return out


def fetch_full_season_history(
    league_code: str, season_id: Optional[int] = None, max_rounds: int = 60,
    sleep_s: float = 0.4, verify_ssl: bool = True,
) -> List[RecentMatch]:
    """Full season backfill (all rounds) for retrain — replaces/augments
    API-Football historical fetch. Stops at the first round that 404s."""
    tournament_id = LEAGUE_TOURNAMENT_IDS.get(league_code)
    if tournament_id is None:
        return []
    if season_id is None:
        season_id = get_current_season_id(tournament_id, verify_ssl=verify_ssl)
    if season_id is None:
        return []

    out: List[RecentMatch] = []
    for round_num in range(1, max_rounds + 1):
        data = _get(
            f"/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_num}",
            verify_ssl=verify_ssl,
        )
        if data is None:
            break
        events = data.get("events", [])
        if not events:
            break
        for event in events:
            if event.get("status", {}).get("type") != "finished":
                continue
            home = event.get("homeTeam", {}) or {}
            away = event.get("awayTeam", {}) or {}
            hs = event.get("homeScore", {}) or {}
            as_ = event.get("awayScore", {}) or {}
            out.append(
                RecentMatch(
                    event_id=event.get("id"),
                    league=league_code,
                    date=_event_date_iso(event) or "",
                    home_team=_norm_team(home.get("name", "")),
                    away_team=_norm_team(away.get("name", "")),
                    home_score=hs.get("current"),
                    away_score=as_.get("current"),
                    home_score_ht=hs.get("period1"),
                    away_score_ht=as_.get("period1"),
                )
            )
        time.sleep(sleep_s)
    return out


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else datetime.now(UTC).date().isoformat()
    fixtures = fetch_fixtures_from_sofascore(target)
    print(f"Found {len(fixtures)} fixtures across 20 leagues for {target}")
    for f in fixtures[:30]:
        print(f.league, f.match_date, f.home_team, "vs", f.away_team)
