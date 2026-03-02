from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

from fhg_weibull import k_from_fh_share


LEAGUE_IDS: Dict[str, int] = {
    "E0": 39,
    "E1": 40,
    "SP1": 140,
    "D1": 78,
    "I1": 135,
    "F1": 61,
    "N1": 88,
    "P1": 94,
    "RO1": 283,
    "RS1": 286,
    "SA1": 307,
}


def _norm_team(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _api_get(session: requests.Session, url: str, params: dict, verify_ssl: bool, timeout: int = 35) -> dict:
    r = session.get(url, params=params, timeout=timeout, verify=verify_ssl)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected non-dict API payload.")
    return payload


def fetch_api_history(
    api_key: str,
    seasons: List[int],
    verify_ssl: bool,
    requests_per_minute: int,
    max_retries: int,
    retry_sleep_seconds: int,
) -> pd.DataFrame:
    s = requests.Session()
    s.headers.update({"x-apisports-key": api_key})

    rows: List[dict] = []
    rpm = max(1, requests_per_minute)
    min_interval = 60.0 / float(rpm)
    last_call_ts = 0.0
    for league_code, league_id in LEAGUE_IDS.items():
        for season in seasons:
            payload = None
            for attempt in range(max_retries + 1):
                try:
                    wait = min_interval - (time.time() - last_call_ts)
                    if wait > 0:
                        time.sleep(wait)
                    payload = _api_get(
                        s,
                        "https://v3.football.api-sports.io/fixtures",
                        {"league": league_id, "season": season, "status": "FT"},
                        verify_ssl=verify_ssl,
                    )
                    last_call_ts = time.time()
                    errors = payload.get("errors")
                    if errors and "rateLimit" in str(errors):
                        if attempt < max_retries:
                            print(
                                f"[WARN] {league_code} {season}: rate limit, retry in {retry_sleep_seconds}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(max(1, retry_sleep_seconds))
                            continue
                    break
                except Exception as exc:
                    if attempt < max_retries:
                        print(
                            f"[WARN] {league_code} {season}: {exc} "
                            f"(retry {attempt + 1}/{max_retries} in {retry_sleep_seconds}s)"
                        )
                        time.sleep(max(1, retry_sleep_seconds))
                        continue
                    print(f"[WARN] {league_code} {season}: {exc}")
                    payload = None

            if not isinstance(payload, dict):
                continue
            errors = payload.get("errors")
            if errors:
                print(f"[WARN] {league_code} {season}: {errors}")
                continue

            items = payload.get("response", [])
            if not isinstance(items, list):
                continue
            added = 0
            for it in items:
                if not isinstance(it, dict):
                    continue
                fixture = it.get("fixture") or {}
                teams = it.get("teams") or {}
                goals = it.get("goals") or {}
                score = it.get("score") or {}
                htf = score.get("halftime") or {}
                date_raw = fixture.get("date")
                home = ((teams.get("home") or {}).get("name") or "").strip()
                away = ((teams.get("away") or {}).get("name") or "").strip()
                hg = goals.get("home")
                ag = goals.get("away")
                hth = htf.get("home")
                hta = htf.get("away")
                if not (date_raw and home and away and hg is not None and ag is not None):
                    continue
                try:
                    md = dt.datetime.fromisoformat(str(date_raw).replace("Z", "+00:00")).date().isoformat()
                except ValueError:
                    continue
                rows.append(
                    {
                        "source": "api_football",
                        "league": league_code,
                        "season_start_year": season,
                        "match_date": md,
                        "home_team": _norm_team(home),
                        "away_team": _norm_team(away),
                        "home_goals": int(hg),
                        "away_goals": int(ag),
                        "ht_home_goals": int(hth) if hth is not None else pd.NA,
                        "ht_away_goals": int(hta) if hta is not None else pd.NA,
                    }
                )
                added += 1
            print(f"[OK] {league_code} {season}: {added} fixtures")
    return pd.DataFrame(rows)


def fallback_fill_from_transfermarkt(df: pd.DataFrame, transfermarkt_csv: str, seasons: List[int]) -> pd.DataFrame:
    # Transfermarkt store lacks HT goals. We still add rows for traceability and leave HT as NA.
    if not Path(transfermarkt_csv).exists():
        return df
    t = pd.read_csv(transfermarkt_csv)
    if t.empty:
        return df
    t["match_date"] = pd.to_datetime(t["match_date"], errors="coerce").dt.date
    t = t.dropna(subset=["match_date", "league", "home_team", "away_team", "home_goals", "away_goals"]).copy()
    t["season_start_year"] = t["match_date"].apply(lambda d: d.year if d.month >= 7 else d.year - 1)
    t = t[t["season_start_year"].isin(seasons)].copy()
    t["source"] = "transfermarkt_fallback"
    t["ht_home_goals"] = pd.NA
    t["ht_away_goals"] = pd.NA
    cols = [
        "source",
        "league",
        "season_start_year",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "ht_home_goals",
        "ht_away_goals",
    ]

    # Add fallback only where API rows are missing for that league-season.
    present = set(zip(df["league"], df["season_start_year"])) if not df.empty else set()
    missing_pairs = set((lg, s) for lg in LEAGUE_IDS for s in seasons if (lg, s) not in present)
    if not missing_pairs:
        return df
    add = t[t.apply(lambda r: (r["league"], int(r["season_start_year"])) in missing_pairs, axis=1)][cols].copy()
    if add.empty:
        return df
    out = pd.concat([df, add], ignore_index=True)
    return out


def build_ratios(history_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if history_df.empty:
        return pd.DataFrame(
            columns=["league", "matches", "fh_goal_share", "p_fhg_empirical", "k_estimate", "data_source_quality"]
        )
    for league, g in history_df.groupby("league"):
        gh = g.copy()
        has_ht = gh["ht_home_goals"].notna() & gh["ht_away_goals"].notna()
        ght = gh[has_ht].copy()
        if ght.empty:
            rows.append(
                {
                    "league": league,
                    "matches": len(gh),
                    "fh_goal_share": 0.45,
                    "p_fhg_empirical": pd.NA,
                    "k_estimate": 1.2,
                    "data_source_quality": "fallback_no_ht",
                }
            )
            continue
        fh_goals = (ght["ht_home_goals"].astype(float) + ght["ht_away_goals"].astype(float)).sum()
        ft_goals = (ght["home_goals"].astype(float) + ght["away_goals"].astype(float)).sum()
        fh_share = 0.45 if ft_goals <= 0 else float(fh_goals / ft_goals)
        p_fhg = float(((ght["ht_home_goals"].astype(float) + ght["ht_away_goals"].astype(float)) > 0).mean())
        rows.append(
            {
                "league": league,
                "matches": int(len(ght)),
                "fh_goal_share": fh_share,
                "p_fhg_empirical": p_fhg,
                "k_estimate": k_from_fh_share(fh_share),
                "data_source_quality": "api_ht",
            }
        )
    return pd.DataFrame(rows).sort_values("league").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FHG historical dataset and league Weibull ratios.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--start-season", type=int, default=2022)
    p.add_argument("--end-season", type=int, default=2024)
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--requests-per-minute", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--retry-sleep-seconds", type=int, default=65)
    p.add_argument("--transfermarkt-csv", default="data/historical/historical_matches_transfermarkt.csv")
    p.add_argument("--out-history", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--out-ratios", default="simulations/FHG/data/fhg_league_ratios.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))
    hist = fetch_api_history(
        args.api_key,
        seasons=seasons,
        verify_ssl=not args.insecure,
        requests_per_minute=args.requests_per_minute,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    hist = fallback_fill_from_transfermarkt(hist, args.transfermarkt_csv, seasons=seasons)
    if hist.empty:
        raise RuntimeError("No FHG history rows collected from API or fallback.")
    hist = hist.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"]).copy()
    hist = hist.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    ratios = build_ratios(hist)

    out_h = Path(args.out_history)
    out_r = Path(args.out_ratios)
    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_r.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out_h, index=False)
    ratios.to_csv(out_r, index=False)
    print(f"Saved history: {out_h} rows={len(hist)}")
    print(f"Saved ratios:  {out_r} rows={len(ratios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
