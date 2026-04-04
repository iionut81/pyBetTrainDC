from __future__ import annotations

"""
run_wta_daily.py
Daily WTA match evaluations and recommendations.
Fetches fixtures from the official WTA API (api.wtatennis.com).

Usage:
  python -X utf8 run_wta_daily.py
  python -X utf8 run_wta_daily.py --series 1
  python -X utf8 run_wta_daily.py --target-date 2026-03-15
"""

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_loader import clean_output_dir

import numpy as np
import pandas as pd
import requests

from config import CFG
from fhg_calibration import apply_calibration, calibration_from_row
from wta_api import build_json_session, fetch_json
from wta_elo import SurfaceElo
from wta_markov import (
    predict_match,
    simulate_match,
)
from wta_markov import PlayerServeStats
from wta_ratings import build_player_match_stats, compute_player_stats_fast
from wta_scoring import parse_set1_games
from wta_set1_filters import eval_set1_o75_gates, merge_set1_o75_config
from wta_tiebreak import build_tiebreak_features, load_tiebreak_model, predict_tiebreak

_WTA = CFG["wta"]
MARKETS_CFG = _WTA["markets"]
STABILITY = _WTA["stability"]
_ELO_CFG = _WTA.get("elo", {})
BLEND_W = _ELO_CFG.get("blend_weight", 0.60)
_S175_RAW = _WTA.get("set1_o75")
S175: Dict = _S175_RAW if isinstance(_S175_RAW, dict) else {}
_GRASS_POLICY_RAW = _WTA.get("grass_policy")
GRASS_POLICY: Dict = _GRASS_POLICY_RAW if isinstance(_GRASS_POLICY_RAW, dict) else {}

WTA_API_BASE = "https://api.wtatennis.com/tennis"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ── WTA API helpers ───────────────────────────────────────────────────────────

_VERIFY_SSL = True
_WTA_HTTP: Dict[str, Any] = {}


def _configure_wta_http() -> None:
    api = _WTA.get("api") or {}
    _WTA_HTTP["session"] = build_json_session(
        USER_AGENT,
        max_retries=int(api.get("max_retries", 3)),
        backoff_factor=float(api.get("backoff_factor", 0.5)),
    )
    cd = str(api.get("cache_dir") or "").strip()
    _WTA_HTTP["cache_dir"] = Path(cd) if cd else None
    _WTA_HTTP["cache_ttl"] = float(api.get("cache_ttl_seconds", 0) or 0)
    _WTA_HTTP["timeout"] = float(api.get("timeout_seconds", 20))


def _fetch_json(url: str) -> dict:
    return fetch_json(
        _WTA_HTTP["session"],
        url,
        timeout=_WTA_HTTP["timeout"],
        verify=_VERIFY_SSL,
        cache_dir=_WTA_HTTP["cache_dir"],
        cache_ttl_seconds=_WTA_HTTP["cache_ttl"],
    )


def fetch_active_tournaments(target_date: str) -> List[dict]:
    """Find WTA tournaments that are in progress or starting on target_date.

    The WTA API lists tournaments chronologically (1960-2026). We scan backward
    from the last page to find the current year quickly.
    """
    td = dt.date.fromisoformat(target_date)
    year = td.year

    # First request: discover total entries to compute last page
    # Note: WTA API caps pageSize at 100
    page_size = 100
    url0 = f"{WTA_API_BASE}/tournaments?page=0&pageSize={page_size}"
    data0 = _fetch_json(url0)
    total = data0.get("pageInfo", {}).get("numEntries", 0)
    if total == 0:
        return []

    last_page = max(0, (total - 1) // page_size)

    tournaments: List[dict] = []
    # Scan backward — 2026 tournaments are on the last few pages
    for page in range(last_page, max(last_page - 5, -1), -1):
        url = f"{WTA_API_BASE}/tournaments?page={page}&pageSize={page_size}"
        data = _fetch_json(url)
        content = data.get("content", [])
        if not content:
            continue

        found_year = False
        for t in content:
            if t.get("year") != year:
                continue
            found_year = True
            start = t.get("startDate", "")
            end = t.get("endDate", "")
            try:
                start_d = dt.date.fromisoformat(start)
                end_d = dt.date.fromisoformat(end)
            except (ValueError, TypeError):
                continue
            # Qualifying rounds can start ~4 days before official startDate
            if start_d - dt.timedelta(days=4) <= td <= end_d + dt.timedelta(days=1):
                t["_start_date"] = start_d
                t["_end_date"] = end_d
                tournaments.append(t)

        # Stop scanning if we've gone past our year
        if content and not found_year and content[0].get("year", 9999) < year:
            break

    return tournaments


def fetch_upcoming_matches(tournament_group_id: int, year: int) -> List[dict]:
    """Fetch reliable upcoming main-draw singles matches for a tournament.

    The WTA `/matches` feed can leave stale `MatchState == "U"` rows from an
    already-completed round while the real next-round pairings are either later
    in the payload or not published yet. When that happens, prefer the deepest
    active round that exists in the feed; if the feed still lags behind, rebuild
    the next round from the winners of the deepest fully completed round.
    """
    url = f"{WTA_API_BASE}/tournaments/{tournament_group_id}/{year}/matches"
    data = _fetch_json(url)
    matches = data.get("matches", [])
    main_singles = [
        m for m in matches
        if m.get("DrawLevelType") == "M"
        and m.get("DrawMatchType") == "S"
        and m.get("PlayerIDA")
        and m.get("PlayerIDB")
    ]

    if not main_singles:
        return []

    def _round_token(value: object) -> str:
        return str(value).strip().upper()

    def _round_rank(value: object) -> int:
        token = _round_token(value)
        if token.isdigit():
            return int(token)
        special = {"Q": 100, "S": 101, "F": 102}
        return special.get(token, -1)

    def _next_round_token(current_token: str, current_count: int) -> Optional[str]:
        if current_count <= 1:
            return None
        if current_count == 2:
            return "F"
        if current_count == 4:
            return "S"
        if current_count == 8:
            return "Q"
        if current_token.isdigit():
            return str(int(current_token) + 1)
        return None

    def _winner_side(match: dict) -> Optional[str]:
        result = str(match.get("ResultString", "") or "")
        pre = result.split(" d ", 1)[0]
        last_a = str(match.get("PlayerNameLastA", "") or "").strip()
        last_b = str(match.get("PlayerNameLastB", "") or "").strip()
        if last_a and last_a in pre and (not last_b or last_b not in pre):
            return "A"
        if last_b and last_b in pre and (not last_a or last_a not in pre):
            return "B"

        winner_code = str(match.get("Winner", "") or "").strip()
        if winner_code in {"2", "4", "6"}:
            return "A"
        if winner_code in {"3", "5", "7"}:
            return "B"
        return None

    def _winner_payload(match: dict) -> Optional[dict]:
        side = _winner_side(match)
        if side is None:
            return None
        suffix = side
        return {
            "PlayerID": match.get(f"PlayerID{suffix}"),
            "PlayerNameFirst": match.get(f"PlayerNameFirst{suffix}"),
            "PlayerNameLast": match.get(f"PlayerNameLast{suffix}"),
            "PlayerCountry": match.get(f"PlayerCountry{suffix}"),
            "Seed": match.get(f"Seed{suffix}", ""),
            "EntryType": match.get(f"EntryType{suffix}", ""),
        }

    def _rebuild_next_round(round_matches: List[dict], next_round: str) -> List[dict]:
        rebuilt: List[dict] = []
        ordered = sorted(
            round_matches,
            key=lambda m: (_to_float(m.get("DateSeq")) or 0.0, str(m.get("MatchID", ""))),
        )
        for i in range(0, len(ordered), 2):
            if i + 1 >= len(ordered):
                break
            wa = _winner_payload(ordered[i])
            wb = _winner_payload(ordered[i + 1])
            if wa is None or wb is None:
                continue
            rebuilt.append({
                "MatchState": "U",
                "DrawLevelType": "M",
                "DrawMatchType": "S",
                "RoundID": next_round,
                "MatchID": f"SYNTH-{next_round}-{(i // 2) + 1}",
                "MatchTimeStamp": "",
                "EventID": ordered[i].get("EventID"),
                "EventYear": ordered[i].get("EventYear"),
                "PlayerIDA": wa["PlayerID"],
                "PlayerIDB": wb["PlayerID"],
                "PlayerNameFirstA": wa["PlayerNameFirst"],
                "PlayerNameLastA": wa["PlayerNameLast"],
                "PlayerNameFirstB": wb["PlayerNameFirst"],
                "PlayerNameLastB": wb["PlayerNameLast"],
                "PlayerCountryA": wa["PlayerCountry"],
                "PlayerCountryB": wb["PlayerCountry"],
                "SeedA": wa["Seed"],
                "SeedB": wb["Seed"],
                "EntryTypeA": wa["EntryType"],
                "EntryTypeB": wb["EntryType"],
            })
        return rebuilt

    open_matches = [m for m in main_singles if str(m.get("MatchState", "")).strip().upper() != "F"]
    if open_matches:
        highest_open_rank = max(_round_rank(m.get("RoundID")) for m in open_matches)
        open_matches = [m for m in open_matches if _round_rank(m.get("RoundID")) == highest_open_rank]
    else:
        highest_open_rank = -1

    finished_by_round: Dict[str, List[dict]] = {}
    for match in main_singles:
        if str(match.get("MatchState", "")).strip().upper() != "F":
            continue
        token = _round_token(match.get("RoundID"))
        finished_by_round.setdefault(token, []).append(match)

    if not finished_by_round:
        return [m for m in open_matches if str(m.get("MatchState", "")).strip().upper() == "U"]

    deepest_finished_token = max(finished_by_round, key=_round_rank)
    deepest_finished_rank = _round_rank(deepest_finished_token)
    deepest_finished_matches = finished_by_round[deepest_finished_token]
    next_round_token = _next_round_token(deepest_finished_token, len(deepest_finished_matches))
    next_round_rank = _round_rank(next_round_token) if next_round_token else -1

    # If the feed only exposes unfinished matches from an earlier round, rebuild
    # the next-round bracket from actual winners in the deepest completed round.
    if (
        next_round_token
        and highest_open_rank < next_round_rank
        and not any(_round_rank(m.get("RoundID")) == deepest_finished_rank for m in open_matches)
    ):
        rebuilt = _rebuild_next_round(deepest_finished_matches, next_round_token)
        if rebuilt:
            return rebuilt

    return [m for m in open_matches if str(m.get("MatchState", "")).strip().upper() == "U"]


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_round_id(value: object) -> int:
    """Map WTA round tokens to the numeric scale used by downstream filters."""
    token = str(value).strip().upper()
    if not token:
        return 0
    if token.isdigit():
        return int(token)
    special = {"Q": 3, "S": 4, "F": 5}
    return special.get(token, 0)


# ── Player ID mapping (WTA API ID → Sackmann ID) ────────────────────────────

def build_name_to_sackmann_id(hist: pd.DataFrame) -> Dict[str, int]:
    """Build lookup from normalized player name → Sackmann player_id."""
    name_map: Dict[str, int] = {}
    for col_id, col_name in [("winner_id", "winner_name"), ("loser_id", "loser_name")]:
        for _, r in hist[[col_id, col_name]].drop_duplicates(col_name).iterrows():
            name = str(r[col_name]).strip().lower()
            name_map[name] = int(r[col_id])
    return name_map


def resolve_player_id(
    wta_first: str, wta_last: str, name_map: Dict[str, int]
) -> Optional[int]:
    """Try to find Sackmann player ID from WTA API name."""
    # Sackmann format: "Firstname Lastname"
    full = f"{wta_first} {wta_last}".strip().lower()
    if full in name_map:
        return name_map[full]

    # Try last name only (common for unique last names)
    last = wta_last.strip().lower()
    matches = [(k, v) for k, v in name_map.items() if k.endswith(f" {last}")]
    if len(matches) == 1:
        return matches[0][1]

    return None


# ── Tennis Abstract fallback ──────────────────────────────────────────────────

TA_BASE = "https://www.tennisabstract.com"

# Column indices in Tennis Abstract matchmx
# Note: index 31 = oSvGms (extra column not in matchhead), shifting opp stats by 1
_TA_COLS = {
    "date": 0, "surf": 2, "wl": 4, "aces": 21, "dfs": 22, "pts": 23,
    "firsts": 24, "fwon": 25, "swon": 26, "saved": 27, "chances": 28,
    "oaces": 29, "opts": 32, "ofirsts": 33, "ofwon": 34, "oswon": 35,
    "osaved": 36, "ochances": 37,
}

_ta_cache: Dict[str, Optional[List[list]]] = {}


def _ta_js_name(first: str, last: str) -> str:
    """Convert WTA API name to Tennis Abstract JS filename format: FirstnameLastname."""
    f = first.strip().replace(" ", "").replace("-", "")
    l = last.strip().replace(" ", "").replace("-", "")
    return f"{f}{l}"


def _fetch_ta_matches(first: str, last: str) -> Optional[List[list]]:
    """Fetch match-level data from Tennis Abstract for a player."""
    key = _ta_js_name(first, last)
    if key in _ta_cache:
        return _ta_cache[key]

    url = f"{TA_BASE}/jsmatches/{key}.js"
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=15, verify=_VERIFY_SSL,
        )
        if resp.status_code != 200:
            _ta_cache[key] = None
            return None
        mm = re.search(r"var matchmx\s*=\s*(\[.*\]);", resp.text, re.DOTALL)
        if not mm:
            _ta_cache[key] = None
            return None
        matchmx = json.loads(mm.group(1))
        _ta_cache[key] = matchmx
        return matchmx
    except Exception:
        _ta_cache[key] = None
        return None


def _ta_int(row: list, col: str) -> int:
    """Safely extract an integer from a Tennis Abstract match row."""
    idx = _TA_COLS.get(col, -1)
    if idx < 0 or idx >= len(row):
        return 0
    v = row[idx]
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return 0


def compute_stats_from_ta(
    first: str, last: str, surface: str,
    window: int = 20, decay: float = 0.05,
) -> Optional[PlayerServeStats]:
    """Compute PlayerServeStats from Tennis Abstract data as fallback."""
    matchmx = _fetch_ta_matches(first, last)
    if matchmx is None:
        return None

    # Filter to surface and matches with serve stats
    surf_matches = []
    for row in matchmx:
        if len(row) < 38:
            continue
        if row[_TA_COLS["surf"]] != surface:
            continue
        pts = _ta_int(row, "pts")
        opts = _ta_int(row, "opts")
        if pts < 10 or opts < 10:
            continue
        surf_matches.append(row)

    if len(surf_matches) < 3:
        return None

    # Take last `window` matches (matchmx is chronological)
    recent = surf_matches[-window:]
    n = len(recent)

    # Exponential weights (most recent first)
    w = np.exp(-decay * np.arange(n))
    w = w / w.sum()

    eps = 1e-9
    stats_arrays = []
    for row in reversed(recent):  # reverse so most recent = index 0
        svpt = max(_ta_int(row, "pts"), 1)
        first_in = _ta_int(row, "firsts")
        fwon = _ta_int(row, "fwon")
        swon = _ta_int(row, "swon")
        aces = _ta_int(row, "aces")
        bp_saved = _ta_int(row, "saved")
        bp_faced = _ta_int(row, "chances")
        # Opponent serve stats (for return calculation)
        opts = max(_ta_int(row, "opts"), 1)
        ofwon = _ta_int(row, "ofwon")
        oswon = _ta_int(row, "oswon")
        obp_saved = _ta_int(row, "osaved")
        obp_faced = _ta_int(row, "ochances")

        second_attempts = max(svpt - first_in, 1)
        sv_games = max((svpt + 3) // 4, 1)  # approximate service games

        stats_arrays.append([
            first_in / svpt,                             # 1stServeIn_pct
            fwon / max(first_in, 1),                     # 1stServeWon_pct
            swon / second_attempts,                      # 2ndServeWon_pct
            aces / sv_games,                             # aceRate
            bp_saved / max(bp_faced, eps),                # bpSaved_pct
            (opts - ofwon - oswon) / opts,                # returnPtsWon_pct
            (obp_faced - obp_saved) / max(obp_faced, eps), # bpConverted_pct
        ])

    vals = np.array(stats_arrays, dtype=float)

    def wmean(col_idx: int) -> float:
        v = vals[:, col_idx]
        valid = ~np.isnan(v)
        if valid.sum() == 0:
            return 0.5
        ww = w[valid]
        return float(np.average(v[valid], weights=ww / ww.sum()))

    return PlayerServeStats(
        first_serve_in_pct=wmean(0),
        first_serve_won_pct=wmean(1),
        second_serve_won_pct=wmean(2),
        ace_rate=wmean(3),
        bp_saved_pct=wmean(4),
        return_pts_won_pct=wmean(5),
        bp_converted_pct=wmean(6),
        n_matches=n,
    )


# ── Stability Filter (Step 11) ────────────────────────────────────────────────

def count_recent_matches(
    pms: pd.DataFrame, player_id: int, reference_date: pd.Timestamp, days: int,
) -> int:
    """Count how many matches a player played in the last `days` days."""
    if pms.empty:
        return 0
    cutoff = reference_date - pd.Timedelta(days=days)
    mask = (
        (pms["player_id"] == player_id)
        & (pms["match_date"] >= cutoff)
        & (pms["match_date"] < reference_date)
    )
    return int(mask.sum())


def stability_check(
    result: dict,
    mc: dict,
    pms: pd.DataFrame,
    sid_a: int,
    sid_b: int,
    reference_date: pd.Timestamp,
    player_a: str,
    player_b: str,
) -> Optional[str]:
    """Return rejection reason string, or None if match passes all filters."""
    cfg = STABILITY

    # 1. Fatigue: reject if either player played >N matches in last 5 days
    window = cfg["fatigue_window_days"]
    max_m = cfg["max_matches_last_5d"]
    recent_a = count_recent_matches(pms, sid_a, reference_date, window)
    recent_b = count_recent_matches(pms, sid_b, reference_date, window)
    if recent_a > max_m:
        return f"fatigue {player_a} ({recent_a} matches last {window}d)"
    if recent_b > max_m:
        return f"fatigue {player_b} ({recent_b} matches last {window}d)"

    # 2. Injury news — manual override via injuries.txt (one name per line)
    injuries = _load_injury_list()
    for name in [player_a, player_b]:
        if name.strip().lower() in injuries:
            return f"injury flag: {name}"

    # 3. Serve hold difference too small → match too unpredictable
    hold_diff = abs(result["p_hold_a"] - result["p_hold_b"])
    if hold_diff < cfg["min_hold_diff"]:
        return f"hold_diff {hold_diff:.4f} < {cfg['min_hold_diff']}"

    # 4. Variance HIGH — MC std of total games above threshold
    if mc["std_total_games"] > cfg["variance_std_threshold"]:
        return f"variance HIGH (std={mc['std_total_games']:.2f})"

    # 5. Match prob too extreme (one-sided blowout — poor odds value)
    p = result["p_match_a"]
    if p > cfg["max_match_prob"] or p < cfg["min_match_prob"]:
        return f"extreme p_match={p:.4f}"

    return None


def _load_injury_list() -> set:
    """Load player names from injuries.txt (one per line). Returns lowercase set."""
    p = Path("injuries.txt")
    if not p.exists():
        return set()
    return {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily WTA evaluations and recommendations.")
    p.add_argument("--target-date", default=dt.date.today().isoformat())
    p.add_argument("--history-csv", default="data/historical/wta_matches_combined.csv")
    p.add_argument("--calibration-csv", default="simulations/WTA/data/wta_calibration.csv")
    p.add_argument("--series", default="1")
    p.add_argument("--mc-iterations", type=int, default=_WTA["mc_iterations"])
    p.add_argument("--rolling-window", type=int, default=_WTA["rolling_window"])
    p.add_argument("--recency-decay", type=float, default=_WTA["recency_decay"])
    p.add_argument("--min-matches-for-rating", type=int, default=_WTA["min_matches_for_rating"])
    p.add_argument("--insecure", action="store_true", help="Disable SSL verification (for corporate proxies)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target_date

    if args.insecure:
        global _VERIFY_SSL
        _VERIFY_SSL = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    _configure_wta_http()

    # ── Fetch fixtures from WTA API ──────────────────────────────────────────
    print(f"Fetching active WTA tournaments for {target} ...")
    tournaments = fetch_active_tournaments(target)
    if not tournaments:
        print("No active WTA tournaments found.")
        return 0

    all_upcoming: List[Tuple[dict, dict]] = []  # (tournament, match)
    for t in tournaments:
        gid = t["tournamentGroup"]["id"]
        name = t["tournamentGroup"]["name"]
        level = t["tournamentGroup"].get("level", "")
        surface = t.get("surface", "Hard")
        year = t.get("year", dt.date.fromisoformat(target).year)
        print(f"  {name} ({level}, {surface}) ...")

        try:
            matches = fetch_upcoming_matches(gid, year)
        except Exception as exc:
            print(f"    [WARN] Could not fetch matches: {exc}")
            continue

        for m in matches:
            all_upcoming.append((t, m))
        if matches:
            print(f"    {len(matches)} upcoming singles matches")
        else:
            print(f"    No upcoming singles matches")

    if not all_upcoming:
        print("No upcoming WTA matches found.")
        return 0

    print(f"\nTotal upcoming fixtures: {len(all_upcoming)}")

    # ── Load historical data ─────────────────────────────────────────────────
    print(f"\nLoading {args.history_csv} for player ratings ...")
    hist = pd.read_csv(args.history_csv)
    hist["match_date"] = pd.to_datetime(hist["match_date"], errors="coerce")
    hist = hist.dropna(subset=["match_date"]).copy()
    hist["winner_id"] = hist["winner_id"].astype(int)
    hist["loser_id"] = hist["loser_id"].astype(int)

    tier_w = _WTA.get("tier_weights")
    pms = build_player_match_stats(hist, tier_weights=tier_w if tier_w else None)
    name_map = build_name_to_sackmann_id(hist)
    print(f"  Player-match stats: {len(pms)} rows, name map: {len(name_map)} players")

    # Load Elo snapshot (built by train_wta.py)
    elo_path = Path("simulations/WTA/data/wta_elo_snapshot.pkl")
    elo: Optional[SurfaceElo] = None
    if elo_path.exists():
        elo = SurfaceElo.load(str(elo_path))
        total_rated = sum(len(v) for v in elo.ratings.values())
        print(f"  Loaded Elo snapshot: {total_rated} player-surface ratings")
    else:
        print("  [WARN] No Elo snapshot found, using pure Markov")

    # ── Load calibration ─────────────────────────────────────────────────────
    cal_path = Path(args.calibration_csv)
    cal_df = (
        pd.read_csv(cal_path)
        if cal_path.exists()
        else pd.DataFrame(columns=["surface", "market", "method", "a", "b", "temperature"])
    )
    cal_map: Dict[Tuple[str, str], dict] = {}
    for _, r in cal_df.iterrows():
        sf = str(r["surface"]).strip()
        mk = str(r["market"]).strip()
        cal_map[(sf, mk)] = calibration_from_row(dict(r))

    global_cals = {
        market: cal_map.get(("__GLOBAL__", market), {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0})
        for market in ["match_winner", "tiebreak", "set1_over_7_5", "set1_over_9_5"]
    }

    # Load tiebreak model
    tb_path = Path("simulations/WTA/data/wta_tiebreak_model.pkl")
    tb_weights = load_tiebreak_model(str(tb_path))
    if tb_weights is not None:
        print(f"  Loaded tiebreak model ({len(tb_weights)} weights)")
    else:
        print("  [WARN] No tiebreak model found, tiebreak predictions disabled")

    # Compute per-player tiebreak rates, Set 1 avg games, and per-surface base rates
    player_tb_hits: Dict[int, int] = {}
    player_tb_total: Dict[int, int] = {}
    player_s1_sum: Dict[int, float] = {}
    player_s1_count: Dict[int, int] = {}
    surface_tb_hits: Dict[str, int] = {}
    surface_tb_total: Dict[str, int] = {}
    surface_s1_sum: Dict[str, float] = {}
    surface_s1_count: Dict[str, int] = {}
    for _, hrow in hist.iterrows():
        s1g = parse_set1_games(str(hrow.get("score", "")))
        if s1g <= 0:
            continue
        is_tb = int(s1g >= 13)
        surf = hrow.get("surface", "Hard")
        surface_tb_total[surf] = surface_tb_total.get(surf, 0) + 1
        surface_tb_hits[surf] = surface_tb_hits.get(surf, 0) + is_tb
        surface_s1_sum[surf] = surface_s1_sum.get(surf, 0.0) + s1g
        surface_s1_count[surf] = surface_s1_count.get(surf, 0) + 1
        for pid in (int(hrow["winner_id"]), int(hrow["loser_id"])):
            player_tb_total[pid] = player_tb_total.get(pid, 0) + 1
            player_tb_hits[pid] = player_tb_hits.get(pid, 0) + is_tb
            player_s1_sum[pid] = player_s1_sum.get(pid, 0.0) + s1g
            player_s1_count[pid] = player_s1_count.get(pid, 0) + 1

    def _get_surface_tb_rate(surf: str) -> float:
        total = surface_tb_total.get(surf, 0)
        return surface_tb_hits.get(surf, 0) / max(total, 1) if total > 0 else 0.12

    def _get_player_tb_rate(pid: Optional[int], surf: str) -> float:
        if pid is None:
            return _get_surface_tb_rate(surf)
        total = player_tb_total.get(pid, 0)
        if total < 5:
            return _get_surface_tb_rate(surf)
        return player_tb_hits.get(pid, 0) / total

    def _get_player_avg_s1(pid: Optional[int], surf: str) -> float:
        if pid is None:
            cnt = surface_s1_count.get(surf, 0)
            return surface_s1_sum.get(surf, 0.0) / max(cnt, 1) if cnt > 0 else 9.5
        cnt = player_s1_count.get(pid, 0)
        if cnt < 5:
            cnt_s = surface_s1_count.get(surf, 0)
            return surface_s1_sum.get(surf, 0.0) / max(cnt_s, 1) if cnt_s > 0 else 9.5
        return player_s1_sum.get(pid, 0.0) / cnt

    print(f"  Tiebreak rates: {', '.join(f'{s}={surface_tb_hits.get(s,0)}/{surface_tb_total.get(s,0)} ({_get_surface_tb_rate(s):.1%})' for s in ['Hard','Clay','Grass'])}")

    # ── Evaluate each fixture ────────────────────────────────────────────────
    rows_winner: list[dict] = []
    rows_s1_7: list[dict] = []
    rows_s1o: list[dict] = []
    rows_tb: list[dict] = []
    for t, m in all_upcoming:
        surface = t.get("surface", "Hard")
        tourney = t["tournamentGroup"]["name"]
        level = t["tournamentGroup"].get("level", "")

        first_a = str(m.get("PlayerNameFirstA", "")).strip()
        last_a = str(m.get("PlayerNameLastA", "")).strip()
        first_b = str(m.get("PlayerNameFirstB", "")).strip()
        last_b = str(m.get("PlayerNameLastB", "")).strip()
        player_a = f"{first_a} {last_a}"
        player_b = f"{first_b} {last_b}"

        # Resolve Sackmann IDs and compute stats (with Tennis Abstract fallback)
        sid_a = resolve_player_id(first_a, last_a, name_map)
        sid_b = resolve_player_id(first_b, last_b, name_map)

        stats_a = None
        stats_b = None
        src_a = "sackmann"
        src_b = "sackmann"

        # Try Sackmann CSV first
        if sid_a is not None:
            stats_a = compute_player_stats_fast(
                pms, sid_a, surface,
                window=args.rolling_window, decay=args.recency_decay,
            )
        if sid_b is not None:
            stats_b = compute_player_stats_fast(
                pms, sid_b, surface,
                window=args.rolling_window, decay=args.recency_decay,
            )

        # Check minimum matches threshold from Sackmann
        if stats_a is not None and stats_a.n_matches < args.min_matches_for_rating:
            stats_a = None
        if stats_b is not None and stats_b.n_matches < args.min_matches_for_rating:
            stats_b = None

        # Fallback to Tennis Abstract for missing stats
        if stats_a is None:
            stats_a = compute_stats_from_ta(
                first_a, last_a, surface,
                window=args.rolling_window, decay=args.recency_decay,
            )
            if stats_a is not None:
                src_a = "tennisabstract"
        if stats_b is None:
            stats_b = compute_stats_from_ta(
                first_b, last_b, surface,
                window=args.rolling_window, decay=args.recency_decay,
            )
            if stats_b is not None:
                src_b = "tennisabstract"

        if stats_a is None or stats_b is None:
            missing = []
            if stats_a is None:
                missing.append(f"A={player_a}")
            if stats_b is None:
                missing.append(f"B={player_b}")
            print(f"  [SKIP] {player_a} vs {player_b} ({surface}) — no stats for {', '.join(missing)}")
            continue

        if stats_a.n_matches < 3 or stats_b.n_matches < 3:
            print(f"  [SKIP] {player_a} vs {player_b} — too few matches (A={stats_a.n_matches}, B={stats_b.n_matches})")
            continue

        data_src = f"{src_a}/{src_b}"

        # Analytical prediction
        result = predict_match(stats_a, stats_b)

        # Monte Carlo for games distribution
        mc = simulate_match(
            result["p_serve_a"], result["p_serve_b"],
            n_simulations=args.mc_iterations,
        )

        # Step 11 — Stability Filter (skip fatigue check if no Sackmann ID)
        ref_date = pd.Timestamp(target)
        reject_reason = stability_check(
            result, mc, pms,
            sid_a if sid_a is not None else -1,
            sid_b if sid_b is not None else -1,
            ref_date, player_a, player_b,
        )
        if reject_reason:
            print(f"  [UNSTABLE] {player_a} vs {player_b} — {reject_reason}")
            continue

        # Blend Markov + Elo for match winner
        p_markov = result["p_match_a"]
        p_elo = None
        if elo is not None and sid_a is not None and sid_b is not None:
            p_elo = elo.predict(sid_a, sid_b, surface)
        if p_elo is not None:
            p_match_raw = BLEND_W * p_markov + (1.0 - BLEND_W) * p_elo
        else:
            p_match_raw = p_markov

        # Set 1 Over 7.5 (analytical from Markov chain + momentum)
        p_s1_7_analytical = result["p_set1_over_7_5"]
        avg_s1_a_7 = _get_player_avg_s1(sid_a, surface)
        avg_s1_b_7 = _get_player_avg_s1(sid_b, surface)
        avg_s1_pair_7 = (avg_s1_a_7 + avg_s1_b_7) / 2.0
        momentum_7 = 0.01 * (avg_s1_pair_7 - 7.5)
        p_s1_7_raw = max(0.05, min(0.99, p_s1_7_analytical + momentum_7))

        # Set 1 Over 9.5 (analytical from Markov chain + momentum)
        p_s1o_analytical = result["p_set1_over_9_5"]
        # Momentum: player historical Set 1 avg games
        avg_s1_a = _get_player_avg_s1(sid_a, surface)
        avg_s1_b = _get_player_avg_s1(sid_b, surface)
        avg_s1_pair = (avg_s1_a + avg_s1_b) / 2.0
        momentum = 0.02 * (avg_s1_pair - 9.5)
        p_s1o_raw = max(0.05, min(0.95, p_s1o_analytical + momentum))

        # Tiebreak prediction
        p_tb_raw = None
        if tb_weights is not None:
            tb_feat = build_tiebreak_features(
                stats_a, stats_b, surface,
                p_elo=p_elo,
                tb_rate_a=_get_player_tb_rate(sid_a, surface),
                tb_rate_b=_get_player_tb_rate(sid_b, surface),
                surface_tb_rate=_get_surface_tb_rate(surface),
            )
            p_tb_raw = float(predict_tiebreak(tb_feat, tb_weights)[0])

        # Calibrate match winner (Platt well-fitted)
        cal_mw = cal_map.get((surface, "match_winner"), global_cals["match_winner"])
        p_match_cal = float(apply_calibration(np.array([p_match_raw]), cal_mw)[0])

        # Calibrate set1 over 7.5
        cal_s1_7 = cal_map.get((surface, "set1_over_7_5"), global_cals.get("set1_over_7_5", {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0}))
        p_s1_7_cal = float(apply_calibration(np.array([p_s1_7_raw]), cal_s1_7)[0])

        # Calibrate set1 over 9.5
        cal_s1o = cal_map.get((surface, "set1_over_9_5"), global_cals.get("set1_over_9_5", {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0}))
        p_s1o_cal = float(apply_calibration(np.array([p_s1o_raw]), cal_s1o)[0])

        # Calibrate tiebreak if available
        if p_tb_raw is not None:
            cal_tb = cal_map.get((surface, "tiebreak"), global_cals["tiebreak"])
            p_tb_cal = float(apply_calibration(np.array([p_tb_raw]), cal_tb)[0])
        else:
            p_tb_cal = None

        # Winner-side probability and fair odds
        predicted_winner = player_a if p_match_cal >= 0.50 else player_b
        p_side = p_match_cal if predicted_winner == player_a else (1.0 - p_match_cal)
        fair_odds_side = (1.0 / p_side) if p_side > 0 else None

        fair_odds_s1_7 = (1.0 / p_s1_7_cal) if p_s1_7_cal > 0 else None
        fair_odds_s1o = (1.0 / p_s1o_cal) if p_s1o_cal > 0 else None
        fair_odds_tb = (1.0 / p_tb_cal) if p_tb_cal and p_tb_cal > 0 else None

        # Recommendations
        mw_cfg = MARKETS_CFG["match_winner"]
        rec_match = bool(
            mw_cfg["min_prob"] <= p_side <= mw_cfg["max_prob"]
            and fair_odds_side is not None
            and fair_odds_side <= mw_cfg["max_odds"]
        )

        s1o_cfg = MARKETS_CFG["set1_over_9_5"]
        rec_s1o = bool(
            s1o_cfg["min_prob"] <= p_s1o_cal <= s1o_cfg["max_prob"]
            and fair_odds_s1o is not None
            and fair_odds_s1o <= s1o_cfg["max_odds"]
        )

        tb_cfg = MARKETS_CFG["tiebreak"]
        rec_tb = bool(
            p_tb_cal is not None
            and tb_cfg["min_prob"] <= p_tb_cal <= tb_cfg["max_prob"]
            and fair_odds_tb is not None
            and fair_odds_tb <= tb_cfg["max_odds"]
        )

        # Set1 Over 7.5: wta_set1_filters (same logic as ablation script) + grass_policy overrides
        exp_games = mc["expected_total_games"]
        p_hold_a = result["p_hold_a"]
        p_hold_b = result["p_hold_b"]
        match_round = normalize_round_id(m.get("RoundID", 0))
        _go75 = GRASS_POLICY.get("set1_o75")
        o75_eff = merge_set1_o75_config(
            S175,
            _go75 if isinstance(_go75, dict) else None,
            surface=surface,
        )
        s1gates = eval_set1_o75_gates(
            p_hold_a,
            p_hold_b,
            exp_games,
            float(p_s1_7_cal),
            surface,
            level,
            match_round,
            o75_eff,
        )
        p_s1_7_adj = s1gates["p_s1_7_adj"]
        blowout_score = s1gates["blowout_score"]
        competitive_set = s1gates["competitive_set"]
        collapse_risk = s1gates["collapse_risk"]
        elite_pick = s1gates["elite_pick"]
        rec_s1_7 = s1gates["rec_s1_7"]

        if surface.lower() == "grass":
            dr = {str(x).strip() for x in (GRASS_POLICY.get("disable_recommendations") or [])}
            if "match_winner" in dr:
                rec_match = False
            if "set1_over_7_5" in dr:
                rec_s1_7 = False
            if "set1_over_9_5" in dr:
                rec_s1o = False
            if "tiebreak" in dr:
                rec_tb = False

        base = {
            "run_date": dt.date.today().isoformat(),
            "match_date": m.get("MatchTimeStamp", ""),
            "tournament": tourney,
            "level": level,
            "surface": surface,
            "round": m.get("RoundID", ""),
            "player_a": player_a,
            "player_b": player_b,
            "data_source": data_src,
            "p_hold_a": round(result["p_hold_a"], 4),
            "p_hold_b": round(result["p_hold_b"], 4),
            "p_markov": round(p_markov, 4),
            "p_elo": round(p_elo, 4) if p_elo is not None else None,
            "expected_games": round(mc["expected_total_games"], 2),
        }

        # Winner
        rows_winner.append({
            **base,
            "predicted_winner": predicted_winner,
            "p_raw": round(p_match_raw, 4),
            "p_cal": round(p_side, 4),
            "Chances": f"{p_side * 100:.1f}%",
            "fair_odds": round(fair_odds_side, 4) if fair_odds_side else None,
            "recommended": rec_match,
        })

        # Set 1 Over 7.5
        rows_s1_7.append({
            **base,
            "p_raw": round(p_s1_7_raw, 4),
            "p_cal": round(p_s1_7_cal, 4),
            "p_cal_adj": round(p_s1_7_adj, 4),
            "Chances": f"{p_s1_7_cal * 100:.1f}%",
            "fair_odds": round(fair_odds_s1_7, 4) if fair_odds_s1_7 else None,
            "blowout_score": blowout_score,
            "competitive_set": competitive_set,
            "collapse_risk": collapse_risk,
            "elite_pick": elite_pick,
            "recommended": rec_s1_7,
        })

        # Set 1 Over 9.5
        rows_s1o.append({
            **base,
            "p_raw": round(p_s1o_raw, 4),
            "p_cal": round(p_s1o_cal, 4),
            "Chances": f"{p_s1o_cal * 100:.1f}%",
            "fair_odds": round(fair_odds_s1o, 4) if fair_odds_s1o else None,
            "recommended": rec_s1o,
        })

        # Tiebreak
        if p_tb_cal is not None:
            rows_tb.append({
                **base,
                "p_raw": round(p_tb_raw, 4),
                "p_cal": round(p_tb_cal, 4),
                "Chances": f"{p_tb_cal * 100:.1f}%",
                "fair_odds": round(fair_odds_tb, 4) if fair_odds_tb else None,
                "recommended": rec_tb,
            })

    if not rows_winner:
        print("No WTA matches evaluated.")
        return 0

    base_dir = Path("simulations/WTA/evaluations")
    base_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(base_dir)
    rec_dir = Path("simulations/WTA/recommendations")
    rec_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dir(rec_dir)
    s = args.series

    files = [
        (f"{s}.1_WTA_Winner.csv", rows_winner, "Winner"),
        (f"{s}.2_WTA_Set1_Over_7_5.csv", rows_s1_7, "Set1 Over 7.5"),
        (f"{s}.3_WTA_Set1_Over_9_5.csv", rows_s1o, "Set1 Over 9.5"),
        (f"{s}.4_WTA_Tiebreak.csv", rows_tb, "Tiebreak"),
    ]

    for fname, row_list, label in files:
        df = pd.DataFrame(row_list)
        df = df.sort_values("p_cal", ascending=False).reset_index(drop=True)
        path = base_dir / fname
        df.to_csv(path, index=False)
        n_rec = int(df["recommended"].sum())
        print(f"  {label:20s} -> {path}  ({len(df)} matches, {n_rec} recommended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
