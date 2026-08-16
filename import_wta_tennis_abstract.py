"""
Import recent WTA matches from Tennis Abstract into wta_matches_combined.csv.

Scrapes the Tennis Abstract JS match files for active players,
extracts matches from a given start date, and merges them into
the Sackmann-format history CSV.

Usage:
  python import_wta_tennis_abstract.py --insecure
  python import_wta_tennis_abstract.py --insecure --min-date 2025-01-01
  python import_wta_tennis_abstract.py --insecure --top-n 250
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
TA_BASE = "https://www.tennisabstract.com"

# Tennis Abstract matchmx column indices (44 columns per row)
_COL = {
    "date": 0,        # '20260304'
    "tourney": 1,      # 'Indian Wells'
    "surface": 2,      # 'Hard'
    "level": 3,        # 'PM' / 'G' / 'P' etc
    "wl": 4,           # 'W' or 'L'
    "sets_won": 5,
    "sets_lost": 6,
    "round": 8,        # 'R64', 'SF', 'F' etc
    "score": 9,        # '6-4 6-2'
    "best_of": 10,     # '3'
    "opp_name": 11,    # 'Himeno Sakatsume'
    "opp_rank": 12,    # '136'
    "opp_seed": 13,
    "opp_entry": 14,   # 'Q', 'WC', 'LL'
    "opp_hand": 15,    # 'R' or 'L'
    "opp_dob": 16,     # '20010803'
    "opp_ht": 17,
    "opp_ioc": 18,     # 'JPN'
    "minutes": 19,
    "match_id_num": 20,
    # Serve stats (player)
    "aces": 21,
    "dfs": 22,
    "svpt": 23,
    "first_in": 24,
    "first_won": 25,
    "second_won": 26,
    "bp_saved": 27,
    "bp_faced": 28,
    # Opponent serve stats
    "o_aces": 29,
    "o_dfs": 30,
    "o_sv_gms": 31,    # extra column shifting opponent stats
    "o_svpt": 32,
    "o_first_in": 33,
    "o_first_won": 34,
    "o_second_won": 35,
    "o_bp_saved": 36,
    "o_bp_faced": 37,
}


def _safe_int(row: list, col: str) -> Optional[int]:
    idx = _COL.get(col, -1)
    if idx < 0 or idx >= len(row):
        return None
    v = row[idx]
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return None


def _safe_str(row: list, col: str) -> str:
    idx = _COL.get(col, -1)
    if idx < 0 or idx >= len(row):
        return ""
    return str(row[idx]).strip()


def _parse_date(date_str: str) -> Optional[dt.date]:
    try:
        return dt.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except (ValueError, IndexError):
        return None


def _parse_score_games(score: str) -> Tuple[int, int, int]:
    """Parse score into (total_games, w_sets, l_sets)."""
    total = 0
    w_sets = 0
    l_sets = 0
    for s in score.split():
        s_clean = s.replace("(", "-").replace(")", "").split("-")
        try:
            a, b = int(s_clean[0]), int(s_clean[1])
            total += a + b
            if a > b:
                w_sets += 1
            else:
                l_sets += 1
        except (ValueError, IndexError):
            continue
    return total, w_sets, l_sets


def fetch_player_js(name_key: str, verify_ssl: bool = True) -> Optional[List[list]]:
    """Fetch and parse a Tennis Abstract JS match file."""
    url = f"{TA_BASE}/jsmatches/{name_key}.js"
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT},
            timeout=20, verify=verify_ssl,
        )
        if resp.status_code != 200:
            return None
        mm = re.search(r"var matchmx\s*=\s*(\[.*?\]);", resp.text, re.DOTALL)
        if not mm:
            return None
        return json.loads(mm.group(1))
    except Exception:
        return None


def get_active_player_keys(
    existing_csv: str,
    verify_ssl: bool = True,
    top_n: int = 200,
) -> List[Tuple[str, str, int]]:
    """Build list of (ta_key, player_name, player_id) for active WTA players.

    Uses existing CSV for name→id mapping, plus scrapes TA directory listing
    for recently modified JS files.
    """
    # Load existing name map
    hist = pd.read_csv(existing_csv)
    name_to_id: Dict[str, int] = {}
    name_to_hand: Dict[str, str] = {}
    name_to_ht: Dict[str, str] = {}
    name_to_ioc: Dict[str, str] = {}

    for col_id, col_name, col_hand, col_ht, col_ioc in [
        ("winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc"),
        ("loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc"),
    ]:
        for _, r in hist[[col_id, col_name, col_hand, col_ht, col_ioc]].drop_duplicates(col_name).iterrows():
            if pd.isna(r[col_id]) or pd.isna(r[col_name]):
                continue
            name = str(r[col_name]).strip()
            name_to_id[name.lower()] = int(r[col_id])
            name_to_hand[name.lower()] = str(r[col_hand]) if pd.notna(r[col_hand]) else ""
            name_to_ht[name.lower()] = str(r[col_ht]) if pd.notna(r[col_ht]) else ""
            name_to_ioc[name.lower()] = str(r[col_ioc]) if pd.notna(r[col_ioc]) else ""

    # Scrape TA directory listing for recently modified JS files
    print("  Fetching Tennis Abstract player directory...")
    try:
        resp = requests.get(
            f"{TA_BASE}/jsmatches/?C=M;O=D",
            headers={"User-Agent": USER_AGENT},
            timeout=30, verify=verify_ssl,
        )
        # Parse filenames from HTML directory listing
        js_files = re.findall(r'href="([A-Z][a-zA-Z]+\.js)"', resp.text)
    except Exception as exc:
        print(f"  [WARN] Could not fetch TA directory: {exc}")
        js_files = []

    # Take most recently modified files (directory sorted by mod date desc)
    js_files = js_files[:top_n]
    print(f"  Found {len(js_files)} recently updated player files")

    players: List[Tuple[str, str, int]] = []
    seen_keys: Set[str] = set()
    next_synthetic_id = 900000

    for js_name in js_files:
        key = js_name.replace(".js", "")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Convert TA key to name: "ArynaSabalenka" → "Aryna Sabalenka"
        # Split on uppercase letters
        parts = re.findall(r"[A-Z][a-z]*", key)
        name = " ".join(parts)

        pid = name_to_id.get(name.lower())
        if pid is None:
            # Try variations
            for known_name, known_id in name_to_id.items():
                if known_name.replace(" ", "") == key.lower():
                    pid = known_id
                    name = known_name.title()
                    break

        if pid is None:
            pid = next_synthetic_id
            next_synthetic_id += 1

        players.append((key, name, pid))

    return players


def extract_matches(
    matchmx: List[list],
    player_name: str,
    player_id: int,
    min_date: dt.date,
    existing_keys: Set[str],
    name_to_id: Dict[str, int],
) -> List[dict]:
    """Extract matches from a player's matchmx, returning Sackmann-format rows.

    Only returns matches where the player WON, to avoid duplicates
    (we'll get the loser's stats from the opponent columns).
    """
    rows = []
    for row in matchmx:
        if len(row) < 38:
            continue

        wl = _safe_str(row, "wl")
        if wl != "W":
            continue  # Only winner perspective to avoid duplicates

        date_str = _safe_str(row, "date")
        match_date = _parse_date(date_str)
        if match_date is None or match_date < min_date:
            continue

        opp_name = _safe_str(row, "opp_name")
        if not opp_name:
            continue

        # Dedup key
        tourney = _safe_str(row, "tourney")
        round_str = _safe_str(row, "round")
        dedup_key = f"{match_date}|{player_name.lower()}|{opp_name.lower()}|{tourney}"
        if dedup_key in existing_keys:
            continue
        existing_keys.add(dedup_key)

        score = _safe_str(row, "score")
        total_games, w_sets, l_sets = _parse_score_games(score)
        surface = _safe_str(row, "surface")
        level = _safe_str(row, "level")
        minutes = _safe_int(row, "minutes")

        # Winner (player) serve stats
        w_ace = _safe_int(row, "aces")
        w_df = _safe_int(row, "dfs")
        w_svpt = _safe_int(row, "svpt")
        w_1stIn = _safe_int(row, "first_in")
        w_1stWon = _safe_int(row, "first_won")
        w_2ndWon = _safe_int(row, "second_won")
        w_bpSaved = _safe_int(row, "bp_saved")
        w_bpFaced = _safe_int(row, "bp_faced")

        # Loser (opponent) serve stats
        l_ace = _safe_int(row, "o_aces")
        l_df = _safe_int(row, "o_dfs")
        l_svpt = _safe_int(row, "o_svpt")
        l_1stIn = _safe_int(row, "o_first_in")
        l_1stWon = _safe_int(row, "o_first_won")
        l_2ndWon = _safe_int(row, "o_second_won")
        l_bpSaved = _safe_int(row, "o_bp_saved")
        l_bpFaced = _safe_int(row, "o_bp_faced")

        # Service games (approximate if not available)
        w_SvGms = (w_svpt + 3) // 4 if w_svpt else None
        l_SvGms = _safe_int(row, "o_sv_gms")

        # Opponent metadata
        opp_rank = _safe_int(row, "opp_rank")
        opp_hand = _safe_str(row, "opp_hand")
        opp_dob = _safe_str(row, "opp_dob")
        opp_ht = _safe_str(row, "opp_ht")
        opp_ioc = _safe_str(row, "opp_ioc")
        opp_age = None
        if opp_dob:
            opp_bd = _parse_date(opp_dob)
            if opp_bd:
                opp_age = round((match_date - opp_bd).days / 365.25, 1)

        # Resolve opponent ID
        loser_id = name_to_id.get(opp_name.lower(), 0)

        # Map TA level codes to Sackmann tourney_level
        level_map = {"G": "G", "PM": "P", "P": "P", "I": "I", "F": "F", "D": "D"}
        tourney_level = level_map.get(level, "")

        # Compute derived columns
        eps = 1e-9

        def _pct(num, den):
            if num is None or den is None or den == 0:
                return None
            return round(num / den, 4)

        def _return_won(svpt, fwon, swon):
            if svpt is None or fwon is None or swon is None:
                return None
            return round((svpt - fwon - swon) / max(svpt, 1), 4)

        def _bp_conv(faced, saved):
            if faced is None or saved is None or faced == 0:
                return None
            return round((faced - saved) / faced, 4)

        rows.append({
            "tourney_id": f"TA-{match_date.year}-{tourney.replace(' ', '_')}",
            "tourney_name": tourney,
            "surface": surface,
            "tourney_level": tourney_level,
            "tourney_date": match_date.strftime("%Y%m%d"),
            "match_date": match_date.isoformat(),
            "round": round_str,
            "best_of": 3,
            "minutes": minutes,
            "winner_id": player_id,
            "winner_name": player_name,
            "winner_hand": "",
            "winner_ht": "",
            "winner_ioc": "",
            "winner_age": None,
            "winner_rank": None,
            "winner_rank_points": None,
            "loser_id": loser_id,
            "loser_name": opp_name,
            "loser_hand": opp_hand,
            "loser_ht": opp_ht if opp_ht else None,
            "loser_ioc": opp_ioc,
            "loser_age": opp_age,
            "loser_rank": opp_rank,
            "loser_rank_points": None,
            "score": score,
            "total_games": total_games,
            "w_sets": w_sets,
            "l_sets": l_sets,
            "w_ace": w_ace,
            "w_df": w_df,
            "w_svpt": w_svpt,
            "w_1stIn": w_1stIn,
            "w_1stWon": w_1stWon,
            "w_2ndWon": w_2ndWon,
            "w_SvGms": w_SvGms,
            "w_bpSaved": w_bpSaved,
            "w_bpFaced": w_bpFaced,
            "l_ace": l_ace,
            "l_df": l_df,
            "l_svpt": l_svpt,
            "l_1stIn": l_1stIn,
            "l_1stWon": l_1stWon,
            "l_2ndWon": l_2ndWon,
            "l_SvGms": l_SvGms,
            "l_bpSaved": l_bpSaved,
            "l_bpFaced": l_bpFaced,
            # Derived percentage columns
            "w_1stServeIn_pct": _pct(w_1stIn, w_svpt),
            "w_1stServeWon_pct": _pct(w_1stWon, w_1stIn),
            "w_2ndServeWon_pct": _pct(w_2ndWon, (w_svpt - w_1stIn) if w_svpt and w_1stIn else None),
            "w_aceRate": _pct(w_ace, w_SvGms) if w_ace is not None and w_SvGms else None,
            "w_bpSaved_pct": _pct(w_bpSaved, w_bpFaced),
            "w_returnPtsWon_pct": _return_won(l_svpt, l_1stWon, l_2ndWon),
            "w_bpConverted_pct": _bp_conv(l_bpFaced, l_bpSaved),
            "l_1stServeIn_pct": _pct(l_1stIn, l_svpt),
            "l_1stServeWon_pct": _pct(l_1stWon, l_1stIn),
            "l_2ndServeWon_pct": _pct(l_2ndWon, (l_svpt - l_1stIn) if l_svpt and l_1stIn else None),
            "l_aceRate": _pct(l_ace, l_SvGms) if l_ace is not None and l_SvGms else None,
            "l_bpSaved_pct": _pct(l_bpSaved, l_bpFaced),
            "l_returnPtsWon_pct": _return_won(w_svpt, w_1stWon, w_2ndWon),
            "l_bpConverted_pct": _bp_conv(w_bpFaced, w_bpSaved),
        })

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Import recent WTA matches from Tennis Abstract.")
    p.add_argument("--history-csv", default="data/historical/wta_matches_combined.csv")
    p.add_argument("--output-csv", default="data/historical/wta_matches_combined.csv",
                   help="Output CSV (default: update in place)")
    p.add_argument("--min-date", default="2024-12-01",
                   help="Only import matches from this date onward")
    p.add_argument("--top-n", type=int, default=200,
                   help="Number of most recently active players to scrape")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between requests to be polite")
    p.add_argument("--insecure", action="store_true")
    args = p.parse_args()

    verify_ssl = not args.insecure
    if args.insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    min_date = dt.date.fromisoformat(args.min_date)
    print(f"Importing WTA matches from {min_date} onward...")

    # Load existing data
    hist = pd.read_csv(args.history_csv)
    print(f"  Existing history: {len(hist)} rows (up to {hist['match_date'].max()})")

    # Build name→id map from existing data
    name_to_id: Dict[str, int] = {}
    for col_id, col_name in [("winner_id", "winner_name"), ("loser_id", "loser_name")]:
        for _, r in hist[[col_id, col_name]].drop_duplicates(col_name).iterrows():
            if pd.isna(r[col_id]) or pd.isna(r[col_name]):
                continue
            name_to_id[str(r[col_name]).strip().lower()] = int(r[col_id])

    # Build existing match keys for dedup
    existing_keys: Set[str] = set()
    for _, r in hist.iterrows():
        key = f"{r['match_date']}|{str(r['winner_name']).lower()}|{str(r['loser_name']).lower()}|"
        existing_keys.add(key)

    # Get active players
    players = get_active_player_keys(
        args.history_csv, verify_ssl=verify_ssl, top_n=args.top_n,
    )
    print(f"  Will scrape {len(players)} players")

    # Scrape each player
    all_new: List[dict] = []
    fetched = 0
    skipped = 0
    for i, (ta_key, name, pid) in enumerate(players):
        matchmx = fetch_player_js(ta_key, verify_ssl=verify_ssl)
        if matchmx is None:
            skipped += 1
            continue
        fetched += 1

        new_rows = extract_matches(
            matchmx, name, pid, min_date, existing_keys, name_to_id,
        )
        if new_rows:
            all_new.extend(new_rows)

        if (i + 1) % 25 == 0:
            print(f"    Progress: {i+1}/{len(players)} players, {len(all_new)} new matches")

        time.sleep(args.sleep)

    print(f"\n  Fetched: {fetched}, Skipped: {skipped}, New matches: {len(all_new)}")

    if not all_new:
        print("  No new matches to add.")
        return 0

    # Merge with existing
    new_df = pd.DataFrame(all_new)

    # Ensure column order matches existing
    for col in hist.columns:
        if col not in new_df.columns:
            new_df[col] = None

    new_df = new_df[hist.columns]
    merged = pd.concat([hist, new_df], ignore_index=True)

    # Deduplicate on (match_date, winner_name, loser_name)
    merged["_dedup"] = (
        merged["match_date"].astype(str) + "|"
        + merged["winner_name"].str.lower() + "|"
        + merged["loser_name"].str.lower()
    )
    merged = merged.drop_duplicates(subset="_dedup", keep="first").drop(columns="_dedup")
    merged = merged.sort_values("match_date").reset_index(drop=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"\n  Saved: {out_path} ({len(merged)} rows, was {len(hist)})")
    print(f"  New date range: {merged['match_date'].min()} — {merged['match_date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())