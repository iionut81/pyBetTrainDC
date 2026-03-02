from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


FLASH_LEAGUES: Dict[str, List[str]] = {
    "E0": ["https://www.flashscore.com/football/england/premier-league/results/"],
    "E1": ["https://www.flashscore.com/football/england/championship/results/"],
    "SP1": ["https://www.flashscore.com/football/spain/laliga/results/"],
    "D1": ["https://www.flashscore.com/football/germany/bundesliga/results/"],
    "I1": ["https://www.flashscore.com/football/italy/serie-a/results/"],
    "F1": ["https://www.flashscore.com/football/france/ligue-1/results/"],
    "N1": ["https://www.flashscore.com/football/netherlands/eredivisie/results/"],
    "P1": ["https://www.flashscore.com/football/portugal/liga-portugal/results/"],
    "RO1": ["https://www.flashscore.com/football/romania/superliga/results/"],
    "RS1": [
        "https://www.flashscore.com/football/serbia/super-liga/results/",
        "https://www.flashscore.com/football/serbia/superliga/results/",
    ],
    "SA1": ["https://www.flashscore.com/football/saudi-arabia/saudi-professional-league/results/"],
}


def _norm_team(name: object) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _extract_results_feed(html: str) -> str:
    m = re.search(r"cjs\.initialFeeds\['results'\]\s*=\s*\{\s*data:\s*`(.*?)`,\s*allEventsCount:", html, re.DOTALL)
    return m.group(1) if m else ""


def _parse_event_fields(chunk: str) -> dict:
    fields = {}
    for token in chunk.split(chr(172)):  # '¬'
        if chr(247) in token:  # '÷'
            k, v = token.split(chr(247), 1)
            if k:
                fields[k] = v
    return fields


def _safe_int(v: object) -> int | None:
    try:
        return int(str(v))
    except Exception:
        return None


def parse_feed(feed: str, league: str, target_season_start: int) -> List[dict]:
    out: List[dict] = []
    if not feed:
        return out
    for chunk in feed.split("~AA")[1:]:
        f = _parse_event_fields(chunk)
        ts = _safe_int(f.get("AD"))
        hg = _safe_int(f.get("AG"))
        ag = _safe_int(f.get("AH"))
        hth = _safe_int(f.get("AS"))  # 1st half home
        hta = _safe_int(f.get("AU"))  # 1st half away
        home = f.get("AE")
        away = f.get("AF")
        if ts is None or hg is None or ag is None or not home or not away:
            continue
        md = dt.datetime.fromtimestamp(ts, dt.UTC).date()
        season_start = md.year if md.month >= 7 else md.year - 1
        if season_start != target_season_start:
            continue
        out.append(
            {
                "source": "flashscore_fhg",
                "league": league,
                "season_start_year": season_start,
                "match_date": md.isoformat(),
                "home_team": _norm_team(home),
                "away_team": _norm_team(away),
                "home_goals": hg,
                "away_goals": ag,
                "ht_home_goals": hth if hth is not None else pd.NA,
                "ht_away_goals": hta if hta is not None else pd.NA,
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import current-season FHG history from Flashscore results feeds.")
    p.add_argument("--season-start", type=int, default=2025, help="Season start year, e.g. 2025 for 2025/26.")
    p.add_argument("--out-csv", default="simulations/FHG/data/fhg_history_2025_26_flashscore.csv")
    p.add_argument(
        "--merge-into",
        default="simulations/FHG/data/fhg_history.csv",
        help="Optional FHG history file to merge this snapshot into.",
    )
    p.add_argument("--insecure", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    headers = {"User-Agent": "Mozilla/5.0"}
    verify_ssl = not args.insecure

    rows: List[dict] = []
    for league, urls in FLASH_LEAGUES.items():
        try:
            parsed: List[dict] = []
            last_exc: Exception | None = None
            for url in urls:
                try:
                    r = requests.get(url, headers=headers, timeout=35, verify=verify_ssl)
                    r.raise_for_status()
                    feed = _extract_results_feed(r.text)
                    parsed = parse_feed(feed, league=league, target_season_start=args.season_start)
                    if parsed:
                        break
                except Exception as exc:
                    last_exc = exc
            rows.extend(parsed)
            if parsed:
                print(f"[OK] {league}: {len(parsed)} rows")
            elif last_exc is not None:
                raise last_exc
            else:
                print(f"[WARN] {league}: no rows parsed")
        except Exception as exc:
            print(f"[WARN] {league}: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows parsed from Flashscore results feeds.")

    df = df.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"]).copy()
    df = df.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved snapshot: {out} rows={len(df)}")

    if args.merge_into:
        merge_path = Path(args.merge_into)
        if merge_path.exists():
            base = pd.read_csv(merge_path)
            merged = pd.concat([base, df], ignore_index=True)
        else:
            merged = df.copy()
        merged = merged.drop_duplicates(subset=["league", "match_date", "home_team", "away_team"], keep="last")
        merged = merged.sort_values(["match_date", "league", "home_team"]).reset_index(drop=True)
        merged.to_csv(merge_path, index=False)
        print(f"Merged into: {merge_path} rows={len(merged)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
