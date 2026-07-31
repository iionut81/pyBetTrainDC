from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests


def clean_output_dir(directory: str | Path) -> None:
    """Remove all CSV files from an output directory before writing new evaluations."""
    d = Path(directory)
    if d.exists():
        for f in d.glob("*.csv"):
            try:
                f.unlink()
            except PermissionError:
                pass  # file open in Excel, skip


@dataclass
class Fixture:
    league: str
    match_date: str
    home_team: str
    away_team: str
    odds_1x: Optional[float] = None
    odds_x2: Optional[float] = None
    fixture_id: Optional[int] = None


FLASHSCORE_LEAGUE_URLS: Dict[str, str] = {
    "E0": "https://www.flashscore.com/football/england/premier-league/fixtures/",
    "E1": "https://www.flashscore.com/football/england/championship/fixtures/",
    "D1": "https://www.flashscore.com/football/germany/bundesliga/fixtures/",
    "D2": "https://www.flashscore.com/football/germany/2-bundesliga/fixtures/",
    "SP1": "https://www.flashscore.com/football/spain/laliga/fixtures/",
    "SP2": "https://www.flashscore.com/football/spain/laliga2/fixtures/",
    "I1": "https://www.flashscore.com/football/italy/serie-a/fixtures/",
    "I2": "https://www.flashscore.com/football/italy/serie-b/fixtures/",
    "F1": "https://www.flashscore.com/football/france/ligue-1/fixtures/",
    "N1": "https://www.flashscore.com/football/netherlands/eredivisie/fixtures/",
    "P1": "https://www.flashscore.com/football/portugal/liga-portugal/fixtures/",
    "RO1": "https://www.flashscore.com/football/romania/superliga/fixtures/",
    "RS1": "https://www.flashscore.com/football/serbia/mozzart-bet-super-liga/fixtures/",
    "SA1": "https://www.flashscore.com/football/saudi-arabia/saudi-professional-league/fixtures/",
    "SW1": "https://www.flashscore.com/football/switzerland/super-league/fixtures/",
    "DK1": "https://www.flashscore.com/football/denmark/superliga/fixtures/",
    "B1": "https://www.flashscore.com/football/belgium/jupiler-pro-league/fixtures/",
    "B2": "https://www.flashscore.com/football/belgium/challenger-pro-league/fixtures/",
    "TR1": "https://www.flashscore.com/football/turkey/super-lig/fixtures/",
    "TR2": "https://www.flashscore.com/football/turkey/1-lig/fixtures/",
}

API_LEAGUE_MAP: Dict[tuple[str, str], str] = {
    ("england", "premier league"): "E0",
    ("england", "championship"): "E1",
    ("spain", "la liga"): "SP1",
    ("germany", "bundesliga"): "D1",
    ("italy", "serie a"): "I1",
    ("france", "ligue 1"): "F1",
    ("netherlands", "eredivisie"): "N1",
    ("portugal", "primeira liga"): "P1",
    ("portugal", "liga portugal"): "P1",
    ("romania", "liga i"): "RO1",
    ("romania", "superliga"): "RO1",
    ("serbia", "super liga"): "RS1",
    ("saudi arabia", "pro league"): "SA1",
    ("saudi-arabia", "pro league"): "SA1",
    ("saudi arabia", "saudi professional league"): "SA1",
    ("saudi-arabia", "saudi professional league"): "SA1",
    ("spain", "segunda division"): "SP2",
    ("spain", "laliga2"): "SP2",
    ("spain", "segunda división"): "SP2",
    ("germany", "2. bundesliga"): "D2",
    ("germany", "2 bundesliga"): "D2",
    ("italy", "serie b"): "I2",
    ("switzerland", "super league"): "SW1",
    ("switzerland", "super league 1"): "SW1",
    ("denmark", "superliga"): "DK1",
    ("belgium", "jupiler pro league"): "B1",
    ("belgium", "pro league"): "B1",
    ("belgium", "challenger pro league"): "B2",
    ("turkey", "super lig"): "TR1",
    ("turkey", "süper lig"): "TR1",
    ("turkey", "1. lig"): "TR2",
}


def _norm_team(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def load_team_ratings(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("Ratings file must contain a dictionary.")
    return raw


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_fixture_item(item: dict) -> Optional[Fixture]:
    # API-Football nested payload
    if isinstance(item.get("teams"), dict):
        league_name = _norm_text((item.get("league") or {}).get("name"))
        country_name = _norm_text((item.get("league") or {}).get("country"))
        league_code = API_LEAGUE_MAP.get((country_name, league_name))
        if not league_code:
            return None
        item = {
            "league": league_code,
            "date": (item.get("fixture") or {}).get("date"),
            "homeTeam": ((item.get("teams") or {}).get("home") or {}).get("name"),
            "awayTeam": ((item.get("teams") or {}).get("away") or {}).get("name"),
            "odds": {},
        }

    # football-data.org nested payload
    if isinstance(item.get("homeTeam"), dict):
        item = {
            "league": (item.get("competition") or {}).get("code") or (item.get("competition") or {}).get("name"),
            "date": item.get("utcDate") or item.get("date"),
            "homeTeam": (item.get("homeTeam") or {}).get("name"),
            "awayTeam": (item.get("awayTeam") or {}).get("name"),
            "odds": item.get("odds") or {},
        }

    league = (
        item.get("league")
        or item.get("competition")
        or item.get("Div")
        or item.get("strLeague")
        or item.get("league_name")
    )
    date = item.get("date") or item.get("match_date") or item.get("dateEvent") or item.get("strTimestamp")
    home = (
        item.get("homeTeam")
        or item.get("home_team")
        or item.get("home")
        or item.get("strHomeTeam")
        or item.get("home_name")
    )
    away = (
        item.get("awayTeam")
        or item.get("away_team")
        or item.get("away")
        or item.get("strAwayTeam")
        or item.get("away_name")
    )
    if not (league and date and home and away):
        return None

    odds_1x = item.get("odds_1x")
    odds_x2 = item.get("odds_x2")
    if odds_1x is None and isinstance(item.get("odds"), dict):
        odds_1x = item["odds"].get("1X")
    if odds_x2 is None and isinstance(item.get("odds"), dict):
        odds_x2 = item["odds"].get("X2")

    return Fixture(
        league=str(league).strip(),
        match_date=str(date).strip(),
        home_team=_norm_team(str(home)),
        away_team=_norm_team(str(away)),
        odds_1x=_to_float(odds_1x),
        odds_x2=_to_float(odds_x2),
    )


def fetch_fixtures_from_api(
    api_url: str,
    api_key: Optional[str] = None,
    timeout: int = 20,
    verify_ssl: bool = True,
) -> List[Fixture]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-Auth-Token"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-apisports-key"] = api_key

    # Primary: Sofascore (free, all 20 leagues, no quota, exact goal minutes).
    # Fallback: Flashscore, used only if Sofascore returns nothing (outage/block).
    # Secondary: API-Football odds enrichment where available.
    if "v3.football.api-sports.io/fixtures" in api_url:
        parsed_url = urlparse(api_url)
        qs = parse_qs(parsed_url.query)
        target_date = (qs.get("date") or [""])[0]

        # 1. Sofascore fixtures (primary source)
        fs_fixtures: List[Fixture] = []
        source_name = "Sofascore"
        if target_date:
            try:
                from sofascore_loader import fetch_fixtures_from_sofascore

                fs_fixtures = fetch_fixtures_from_sofascore(
                    target_date_iso=target_date, verify_ssl=verify_ssl
                )
            except Exception:
                fs_fixtures = []

        if not fs_fixtures and target_date:
            source_name = "Flashscore (fallback)"
            try:
                fs_fixtures = fetch_fixtures_from_flashscore(
                    target_date_iso=target_date, verify_ssl=verify_ssl
                )
            except Exception:
                pass

        # 2. API-Football fixtures + odds (secondary, for odds enrichment)
        try:
            api_fixtures = _fetch_api_football_fixtures_with_odds(
                api_url=api_url, headers=headers, timeout=timeout, verify_ssl=verify_ssl
            )
        except Exception:
            api_fixtures = []

        # Build odds lookup from API-Football
        api_odds: Dict[tuple[str, str, str], Fixture] = {}
        for fx in api_fixtures:
            api_odds[(fx.league, fx.home_team, fx.away_team)] = fx

        # Enrich Flashscore fixtures with API-Football odds where available
        for fx in fs_fixtures:
            api_fx = api_odds.pop((fx.league, fx.home_team, fx.away_team), None)
            if api_fx:
                fx.odds_1x = api_fx.odds_1x
                fx.odds_x2 = api_fx.odds_x2
                fx.fixture_id = api_fx.fixture_id

        # Add any API-Football fixtures not found in Flashscore
        for api_fx in api_odds.values():
            fs_fixtures.append(api_fx)

        n_api = len(api_fixtures)
        print(f"Fixtures: {len(fs_fixtures)} ({source_name}, API enrichment: {n_api})")
        return fs_fixtures

    resp = requests.get(api_url, headers=headers, timeout=timeout, verify=verify_ssl)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):
        if "fixtures" in payload:
            items = payload["fixtures"]
        elif "response" in payload:
            items = payload["response"]
        elif "matches" in payload:
            items = payload["matches"]
        elif "events" in payload:
            items = payload["events"]
        else:
            items = payload
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("Fixture API response must be a list or {'fixtures': [...]} structure.")

    out: List[Fixture] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fx = _parse_fixture_item(item)
        if fx is not None:
            out.append(fx)
    return out


def _extract_api_football_dc_odds(bookmakers: list) -> tuple[Optional[float], Optional[float]]:
    for b in bookmakers:
        bets = b.get("bets") if isinstance(b, dict) else None
        if not isinstance(bets, list):
            continue
        for bet in bets:
            if not isinstance(bet, dict):
                continue
            bid = bet.get("id")
            bname = str(bet.get("name", "")).strip().lower()
            if bid != 12 and bname != "double chance":
                continue
            vals = bet.get("values", [])
            if not isinstance(vals, list):
                continue
            o1x = None
            ox2 = None
            for v in vals:
                if not isinstance(v, dict):
                    continue
                label = str(v.get("value", "")).strip().lower()
                odd = _to_float(v.get("odd"))
                if odd is None:
                    continue
                if label in {"home/draw", "1x"}:
                    o1x = odd
                elif label in {"draw/away", "x2"}:
                    ox2 = odd
            if o1x is not None or ox2 is not None:
                return o1x, ox2
    return None, None


def _fetch_api_football_fixtures_with_odds(
    api_url: str, headers: dict, timeout: int, verify_ssl: bool
) -> List[Fixture]:
    # Fetch fixtures
    rf = requests.get(api_url, headers=headers, timeout=timeout, verify=verify_ssl)
    rf.raise_for_status()
    pf = rf.json()
    fixtures_raw = pf.get("response", []) if isinstance(pf, dict) else []
    if not isinstance(fixtures_raw, list):
        raise ValueError("API-Football fixtures response malformed.")

    # Resolve date for odds endpoint from query string (?date=YYYY-MM-DD)
    parsed = urlparse(api_url)
    qs = parse_qs(parsed.query)
    target_date = (qs.get("date") or [""])[0]
    odds_by_fixture_id: Dict[int, tuple[Optional[float], Optional[float]]] = {}

    if target_date:
        odds_url = f"https://v3.football.api-sports.io/odds?date={target_date}"
        ro = requests.get(odds_url, headers=headers, timeout=timeout, verify=verify_ssl)
        if ro.status_code == 200:
            po = ro.json()
            odds_raw = po.get("response", []) if isinstance(po, dict) else []
            if isinstance(odds_raw, list):
                for item in odds_raw:
                    if not isinstance(item, dict):
                        continue
                    fx = item.get("fixture", {})
                    fid = fx.get("id") if isinstance(fx, dict) else None
                    if fid is None:
                        continue
                    o1x, ox2 = _extract_api_football_dc_odds(item.get("bookmakers", []))
                    odds_by_fixture_id[int(fid)] = (o1x, ox2)

    out: List[Fixture] = []
    for item in fixtures_raw:
        if not isinstance(item, dict):
            continue
        league_name = _norm_text((item.get("league") or {}).get("name"))
        country_name = _norm_text((item.get("league") or {}).get("country"))
        league_code = API_LEAGUE_MAP.get((country_name, league_name))
        if not league_code:
            continue

        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        date = fixture.get("date")
        fid = fixture.get("id")
        if not (home and away and date):
            continue
        o1x, ox2 = odds_by_fixture_id.get(int(fid), (None, None)) if fid is not None else (None, None)

        out.append(
            Fixture(
                league=league_code,
                match_date=str(date).strip(),
                home_team=_norm_team(str(home)),
                away_team=_norm_team(str(away)),
                odds_1x=o1x,
                odds_x2=ox2,
                fixture_id=int(fid) if fid is not None else None,
            )
        )
    return out


def load_fixtures_from_json(path: str) -> List[Fixture]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("fixtures") if isinstance(raw, dict) and "fixtures" in raw else raw
    if not isinstance(items, list):
        raise ValueError("JSON fixture file must be a list or {'fixtures': [...]} structure.")

    out: List[Fixture] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fx = _parse_fixture_item(item)
        if fx is not None:
            out.append(fx)
    return out


def _extract_flashscore_feed(html: str) -> str:
    m = re.search(r"cjs\.initialFeeds\['fixtures'\]\s*=\s*\{\s*data:\s*`(.*?)`,\s*allEventsCount:", html, re.DOTALL)
    if not m:
        return ""
    return m.group(1)


def _parse_flashscore_feed_to_fixtures(feed: str, league_code: str, target_date_iso: str) -> List[Fixture]:
    out: List[Fixture] = []
    if not feed:
        return out

    # Each event starts at "~AA÷...". We only need date (AD), home (AE), away (AF).
    parts = feed.split("~AA÷")
    if len(parts) < 2:
        return out

    for chunk in parts[1:]:
        ad = re.search(r"(?:^|¬)AD÷(\d+)", chunk)
        ae = re.search(r"(?:^|¬)AE÷([^¬]+)", chunk)
        af = re.search(r"(?:^|¬)AF÷([^¬]+)", chunk)
        if not (ad and ae and af):
            continue
        try:
            ts = int(ad.group(1))
            match_date = datetime.fromtimestamp(ts, UTC).date().isoformat()
        except ValueError:
            continue
        if match_date != target_date_iso:
            continue

        out.append(
            Fixture(
                league=league_code,
                match_date=match_date,
                home_team=_norm_team(ae.group(1)),
                away_team=_norm_team(af.group(1)),
                odds_1x=None,
                odds_x2=None,
            )
        )
    return out


def fetch_fixtures_from_flashscore(target_date_iso: str, verify_ssl: bool = True) -> List[Fixture]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    out: List[Fixture] = []
    for league_code, url in FLASHSCORE_LEAGUE_URLS.items():
        try:
            resp = requests.get(url, headers=headers, timeout=25, verify=verify_ssl)
            resp.raise_for_status()
            feed = _extract_flashscore_feed(resp.text)
            out.extend(_parse_flashscore_feed_to_fixtures(feed, league_code, target_date_iso))
        except Exception:
            continue
    # Deduplicate by key.
    uniq: Dict[tuple[str, str, str, str], Fixture] = {}
    for f in out:
        uniq[(f.league, f.match_date, f.home_team, f.away_team)] = f
    return list(uniq.values())
