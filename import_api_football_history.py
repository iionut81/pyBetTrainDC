from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


API_BASE = "https://v3.football.api-sports.io"


@dataclass(frozen=True)
class LeagueTarget:
    country_queries: List[str]
    preferred_names: List[str]
    code: str
    league_id: int


TARGETS: Dict[str, LeagueTarget] = {
    "romania": LeagueTarget(
        country_queries=["Romania"],
        preferred_names=["superliga", "liga i"],
        code="RO1",
        league_id=283,
    ),
    "serbia": LeagueTarget(
        country_queries=["Serbia"],
        preferred_names=["super liga"],
        code="RS1",
        league_id=286,
    ),
    "saudi": LeagueTarget(
        country_queries=["Saudi-Arabia", "Saudi Arabia"],
        preferred_names=["pro league", "saudi professional league"],
        code="SA1",
        league_id=307,
    ),
}


def _norm(txt: object) -> str:
    return " ".join(str(txt or "").strip().lower().split())


def _season_label(season_start_year: int) -> str:
    return f"{season_start_year}/{(season_start_year + 1) % 100:02d}"


def _api_get(session: requests.Session, path: str, params: dict, timeout: int, verify_ssl: bool) -> dict:
    resp = session.get(f"{API_BASE}{path}", params=params, timeout=timeout, verify=verify_ssl)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected API payload type at {path}.")
    return payload


def resolve_league_id(
    session: requests.Session,
    target: LeagueTarget,
    season: int,
    timeout: int,
    verify_ssl: bool,
) -> Optional[int]:
    candidates: List[dict] = []
    for country in target.country_queries:
        payload = _api_get(
            session=session,
            path="/leagues",
            params={"country": country, "season": season},
            timeout=timeout,
            verify_ssl=verify_ssl,
        )
        items = payload.get("response", [])
        if isinstance(items, list):
            candidates.extend(x for x in items if isinstance(x, dict))

    scored: List[tuple[int, int]] = []
    for idx, row in enumerate(candidates):
        league = row.get("league") or {}
        seasons = row.get("seasons") or []
        if not isinstance(league, dict):
            continue
        if str(league.get("type", "")).strip().lower() != "league":
            continue
        lid = league.get("id")
        if lid is None:
            continue
        name = _norm(league.get("name"))
        is_current = any(bool(s.get("current")) for s in seasons if isinstance(s, dict))
        score = 0
        if any(pref in name for pref in target.preferred_names):
            score += 100
        if is_current:
            score += 20
        if "women" in name or "u19" in name or "u21" in name:
            score -= 200
        scored.append((score, int(lid)))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def fetch_fixtures_for_league_season(
    session: requests.Session,
    league_id: int,
    season: int,
    timeout: int,
    verify_ssl: bool,
) -> List[dict]:
    payload = _api_get(
        session=session,
        path="/fixtures",
        params={
            "league": league_id,
            "season": season,
            "status": "FT",
        },
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    items = payload.get("response", [])
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def to_rows(fixtures: List[dict], league_code: str, season: int) -> List[dict]:
    rows: List[dict] = []
    for item in fixtures:
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        goals = item.get("goals") or {}
        home_team = ((teams.get("home") or {}).get("name") or "").strip().lower()
        away_team = ((teams.get("away") or {}).get("name") or "").strip().lower()
        date_raw = fixture.get("date")
        hg = goals.get("home")
        ag = goals.get("away")
        if not (home_team and away_team and date_raw is not None):
            continue
        try:
            match_date = dt.datetime.fromisoformat(str(date_raw).replace("Z", "+00:00")).date()
        except ValueError:
            continue
        if hg is None or ag is None:
            continue
        rows.append(
            {
                "match_date": match_date.isoformat(),
                "league": league_code,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": int(hg),
                "away_goals": int(ag),
                "season_code": _season_label(season),
                "source": "api_football",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import historical finished fixtures from API-Football for extra leagues."
    )
    parser.add_argument("--api-key", required=True, help="API-Football key (x-apisports-key).")
    parser.add_argument(
        "--targets",
        default="romania,serbia,saudi",
        help="Comma-separated targets from: romania, serbia, saudi",
    )
    parser.add_argument("--n-seasons", type=int, default=5, help="Number of seasons back from end-season.")
    parser.add_argument(
        "--end-season",
        type=int,
        default=dt.date.today().year,
        help="Latest season start year to include (e.g. 2025 for 2025/26).",
    )
    parser.add_argument(
        "--output-csv",
        default="data/historical/historical_matches_api_football_extra.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--merge-into",
        default="",
        help="Optional path to merge imported rows into an existing master history CSV.",
    )
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = [x.strip().lower() for x in args.targets.split(",") if x.strip()]
    unknown = [x for x in requested if x not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Supported: {sorted(TARGETS.keys())}")

    max_season = min(args.end_season, 2024)
    if args.end_season > 2024:
        print(f"[WARN] API plan limit detected: capping end season to 2024 (requested {args.end_season}).")
    seasons = [max_season - i for i in reversed(range(args.n_seasons))]
    headers = {"x-apisports-key": args.api_key}
    session = requests.Session()
    session.headers.update(headers)

    all_rows: List[dict] = []
    failures: List[str] = []
    for tgt_name in requested:
        target = TARGETS[tgt_name]
        for season in seasons:
            try:
                league_id = target.league_id
                fixtures = fetch_fixtures_for_league_season(
                    session=session,
                    league_id=league_id,
                    season=season,
                    timeout=args.timeout,
                    verify_ssl=not args.insecure,
                )
                rows = to_rows(fixtures, league_code=target.code, season=season)
                all_rows.extend(rows)
                print(f"[OK] {tgt_name} {season} league_id={league_id}: {len(rows)} matches")
            except Exception as exc:
                failures.append(f"{tgt_name} {season}: {exc}")
                print(f"[FAIL] {tgt_name} {season}: {exc}")

    if not all_rows:
        raise RuntimeError("No rows imported from API-Football.")

    out_df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["match_date", "league", "home_team", "away_team"], keep="last"
    )
    out_df = out_df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")

    if args.merge_into:
        merge_path = Path(args.merge_into)
        if merge_path.exists():
            base = pd.read_csv(merge_path)
            merged = pd.concat([base, out_df], ignore_index=True)
        else:
            merged = out_df.copy()
        merged = merged.drop_duplicates(subset=["match_date", "league", "home_team", "away_team"], keep="last")
        merged = merged.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
        merge_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merge_path, index=False)
        print(f"Merged history saved: {merge_path} rows={len(merged)}")

    if failures:
        print("Completed with warnings:")
        for f in failures:
            print(f" - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
