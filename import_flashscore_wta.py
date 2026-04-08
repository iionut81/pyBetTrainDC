"""
import_flashscore_wta.py

Scrape WTA match results and serve statistics from Flashscore.
Produces output compatible with wta_matches_combined.csv (Sackmann format).

Phase 1 — Collect finished match event IDs from tournament results pages
Phase 2 — Fetch serve stats via Flashscore internal API (~0.5s/match)

Usage:
    python import_flashscore_wta.py --insecure
    python import_flashscore_wta.py --insecure --tournaments linz charleston
    python import_flashscore_wta.py --insecure --min-date 2026-01-01
    python import_flashscore_wta.py --insecure --resume
    python import_flashscore_wta.py --insecure --merge-into data/historical/wta_matches_combined.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("flashscore_wta")

# ── Tournament config ────────────────────────────────────────────────────────

WTA_TOURNAMENTS: Dict[str, dict] = {
    "linz": {"surface": "Clay", "name": "Linz"},
    "charleston": {"surface": "Clay", "name": "Charleston"},
    "miami": {"surface": "Hard", "name": "Miami"},
    "indian-wells": {"surface": "Hard", "name": "Indian Wells"},
    "dubai": {"surface": "Hard", "name": "Dubai"},
    "doha": {"surface": "Hard", "name": "Doha"},
    "rome": {"surface": "Clay", "name": "Rome"},
    "madrid": {"surface": "Clay", "name": "Madrid"},
    "roland-garros": {"surface": "Clay", "name": "Roland Garros"},
    "wimbledon": {"surface": "Grass", "name": "Wimbledon"},
    "us-open": {"surface": "Hard", "name": "US Open"},
    "australian-open": {"surface": "Hard", "name": "Australian Open"},
    "beijing": {"surface": "Hard", "name": "Beijing"},
    "cincinnati": {"surface": "Hard", "name": "Cincinnati"},
    "stuttgart": {"surface": "Clay", "name": "Stuttgart"},
    "eastbourne": {"surface": "Grass", "name": "Eastbourne"},
    "montreal": {"surface": "Hard", "name": "Montreal"},
    "toronto": {"surface": "Hard", "name": "Toronto"},
    "adelaide": {"surface": "Hard", "name": "Adelaide"},
    "brisbane": {"surface": "Hard", "name": "Brisbane"},
    "ostrava": {"surface": "Hard", "name": "Ostrava"},
    "guadalajara": {"surface": "Hard", "name": "Guadalajara"},
    "san-diego": {"surface": "Hard", "name": "San Diego"},
    "tokyo": {"surface": "Hard", "name": "Tokyo"},
    "seoul": {"surface": "Hard", "name": "Seoul"},
    "bogota": {"surface": "Clay", "name": "Bogota"},
    "singapore": {"surface": "Hard", "name": "Singapore"},
    "tallinn": {"surface": "Hard", "name": "Tallinn"},
    "monastir": {"surface": "Hard", "name": "Monastir"},
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
STATS_API = "https://2.flashscore.ninja/2/x/feed/df_st_1_{eid}"
FS_HEADERS = {
    "User-Agent": USER_AGENT,
    "x-fsign": "SW9D1eZo",
    "Referer": "https://www.flashscore.com/",
}

QUEUE_CSV = Path("data/flashscore/wta_queue.csv")
OUTPUT_CSV = Path("data/flashscore/wta_match_stats.csv")

ROUND_MAP = {
    "Final": "F",
    "Semi-finals": "SF",
    "Quarter-finals": "QF",
    "1/8-finals": "R16",
    "1/16-finals": "R32",
    "1/32-finals": "R64",
    "1/64-finals": "R128",
}


# ── Name resolution ──────────────────────────────────────────────────────────

def slug_to_name(slug: str) -> Tuple[str, str]:
    """Convert Flashscore slug to (first_name, last_name).

    Flashscore uses lastname-firstname format in slugs.
    Examples:
        'cirstea-sorana' → ('Sorana', 'Cirstea')
        'bouzas-maneiro-jessica' → ('Jessica', 'Bouzas Maneiro')
        'ruse-elena-gabriela' → ('Elena Gabriela', 'Ruse')

    Heuristic: last part is usually the first name, everything before is last name.
    But some players have multi-part first names. We use the convention that
    the FIRST part of the slug is always the last name (single word).
    """
    parts = slug.strip().split("-")
    if not parts:
        return ("", "")
    if len(parts) == 1:
        return ("", parts[0].title())
    # First part = last name, rest = first name
    last = parts[0].title()
    first = " ".join(p.title() for p in parts[1:])
    return first, last


# ── Phase 1: Collect match IDs ───────────────────────────────────────────────

def _parse_flashscore_feed(text: str) -> List[dict]:
    """Parse Flashscore custom ¬÷ format into match dicts."""
    # Try both separators
    blocks = text.split("~AA\xf7")
    if len(blocks) < 2:
        blocks = text.split("\xacAA\xf7")

    matches = []
    for block in blocks[1:]:
        fields: Dict[str, str] = {}
        for pair in block.split("\xac"):
            if "\xf7" in pair:
                key, val = pair.split("\xf7", 1)
                fields[key] = val

        state = fields.get("AB", "")
        if state != "3":  # only finished matches
            continue

        slug_a = fields.get("WU", "")
        slug_b = fields.get("WV", "")
        if not slug_a or not slug_b:
            continue

        # Event ID
        event_id = block[:8]

        # Date
        timestamp_str = fields.get("AD", "0")
        try:
            ts = int(timestamp_str)
            match_date = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            continue

        # Set scores
        sets_a, sets_b = [], []
        for sa_key, sb_key in [("BA", "BB"), ("BC", "BD"), ("BE", "BF"), ("BG", "BH"), ("BI", "BJ")]:
            sa = fields.get(sa_key, "")
            sb = fields.get(sb_key, "")
            if sa and sb:
                try:
                    sets_a.append(int(sa))
                    sets_b.append(int(sb))
                except ValueError:
                    break

        if not sets_a:
            continue

        # Determine winner
        wins_a = sum(1 for a, b in zip(sets_a, sets_b) if a > b)
        wins_b = sum(1 for a, b in zip(sets_a, sets_b) if b > a)

        if wins_a == 0 and wins_b == 0:
            continue

        winner_side = "A" if wins_a > wins_b else "B"

        # Build score string
        score = " ".join(f"{a}-{b}" for a, b in zip(sets_a, sets_b))

        # Player names
        first_a, last_a = slug_to_name(slug_a)
        first_b, last_b = slug_to_name(slug_b)
        name_a = f"{first_a} {last_a}".strip()
        name_b = f"{first_b} {last_b}".strip()

        round_name = fields.get("ER", "")
        round_code = ROUND_MAP.get(round_name, round_name)

        # Short names for display
        short_a = fields.get("AE", name_a)
        short_b = fields.get("AF", name_b)

        matches.append({
            "event_id": event_id,
            "match_date": match_date,
            "player_a": name_a,
            "player_b": name_b,
            "short_a": short_a,
            "short_b": short_b,
            "winner_side": winner_side,
            "winner_name": name_a if winner_side == "A" else name_b,
            "loser_name": name_b if winner_side == "A" else name_a,
            "score": score,
            "round": round_code,
            "sets_a": sets_a,
            "sets_b": sets_b,
            "set1_games": (sets_a[0] + sets_b[0]) if sets_a else None,
        })

    return matches


def fetch_tournament_matches(
    slug: str, verify_ssl: bool = True
) -> List[dict]:
    """Fetch all finished matches for a WTA tournament from Flashscore."""
    urls = [
        f"https://www.flashscore.com/tennis/wta-singles/{slug}/results/",
        f"https://www.flashscore.com/tennis/wta-singles/{slug}/",
    ]
    all_matches: Dict[str, dict] = {}

    for url in urls:
        try:
            resp = requests.get(url, headers=FS_HEADERS, timeout=25, verify=verify_ssl)
            resp.raise_for_status()
            parsed = _parse_flashscore_feed(resp.text)
            for m in parsed:
                all_matches[m["event_id"]] = m
        except Exception:
            continue

    return list(all_matches.values())


# ── Phase 2: Fetch serve stats ───────────────────────────────────────────────

def _extract_fraction(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract numerator/denominator from '57% (20/35)' or '6/10'."""
    m = re.search(r"(\d+)/(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_int(text: str) -> Optional[int]:
    """Extract integer from '1' or '48%'."""
    m = re.match(r"(\d+)", text.strip())
    return int(m.group(1)) if m else None


def _extract_pct(text: str) -> Optional[float]:
    """Extract percentage as decimal from '48%'."""
    m = re.match(r"(\d+)%", text.strip())
    return int(m.group(1)) / 100.0 if m else None


def fetch_match_stats(event_id: str, verify_ssl: bool = True) -> Optional[dict]:
    """Fetch and parse serve stats for a single match."""
    url = STATS_API.format(eid=event_id)
    try:
        resp = requests.get(url, headers=FS_HEADERS, timeout=15, verify=verify_ssl)
        if resp.status_code != 200 or len(resp.text) < 10:
            return None
    except Exception:
        return None

    text = resp.text
    # We only care about the "Match" section (SE÷Match), not per-set
    in_match_section = False
    in_service = False
    stats: Dict[str, str] = {}  # {stat_name: "SH_value|SI_value"}

    current_section = ""
    for line in text.split("~"):
        line = line.strip()
        if not line:
            continue

        fields: Dict[str, str] = {}
        for pair in line.split("\xac"):
            if "\xf7" in pair:
                k, v = pair.split("\xf7", 1)
                fields[k] = v

        if "SE" in fields:
            current_section = fields["SE"]
        if "SG" in fields and current_section == "Match":
            stat_name = fields["SG"].lower().strip()
            sh = fields.get("SH", "")
            si = fields.get("SI", "")
            stats[stat_name] = (sh, si)

    if not stats:
        return None

    # Parse stats for player A (SH) and player B (SI)
    result: Dict[str, Optional[int]] = {}

    for side, idx in [("a", 0), ("b", 1)]:
        aces = stats.get("aces", ("", ""))
        result[f"{side}_ace"] = _extract_int(aces[idx]) if aces[idx] else None

        dfs = stats.get("double faults", ("", ""))
        result[f"{side}_df"] = _extract_int(dfs[idx]) if dfs[idx] else None

        fsp = stats.get("1st serve percentage", ("", ""))
        fsp_pct = _extract_pct(fsp[idx]) if fsp[idx] else None

        fwon = stats.get("1st serve points won", ("", ""))
        fwon_num, fwon_den = _extract_fraction(fwon[idx]) if fwon[idx] else (None, None)
        result[f"{side}_1stWon"] = fwon_num
        result[f"{side}_1stIn"] = fwon_den  # 1stIn = denominator of 1st serve points won

        swon = stats.get("2nd serve points won", ("", ""))
        swon_num, swon_den = _extract_fraction(swon[idx]) if swon[idx] else (None, None)
        result[f"{side}_2ndWon"] = swon_num

        # svpt = total serve points
        sp = stats.get("service points won", ("", ""))
        sp_num, sp_den = _extract_fraction(sp[idx]) if sp[idx] else (None, None)
        result[f"{side}_svpt"] = sp_den  # denominator = total serve points

        # If 1stIn not available from fraction, calculate from percentage
        if result[f"{side}_1stIn"] is None and fsp_pct is not None and result[f"{side}_svpt"] is not None:
            result[f"{side}_1stIn"] = round(fsp_pct * result[f"{side}_svpt"])

        bp = stats.get("break points saved", ("", ""))
        bp_saved, bp_faced = _extract_fraction(bp[idx]) if bp[idx] else (None, None)
        result[f"{side}_bpSaved"] = bp_saved
        result[f"{side}_bpFaced"] = bp_faced

        sg = stats.get("service games won", ("", ""))
        sg_won, sg_total = _extract_fraction(sg[idx]) if sg[idx] else (None, None)
        result[f"{side}_SvGms"] = sg_total

    return result


# ── Main pipeline ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape WTA match results and serve stats from Flashscore."
    )
    p.add_argument("--insecure", action="store_true", help="Disable SSL verification")
    p.add_argument("--tournaments", nargs="*", default=None, help="Tournament slugs to scrape (default: all)")
    p.add_argument("--min-date", default=None, help="Only matches after this date (YYYY-MM-DD)")
    p.add_argument("--resume", action="store_true", help="Resume Phase 2 from existing queue")
    p.add_argument("--merge-into", default=None, help="Path to merge results into (e.g. wta_matches_combined.csv)")
    p.add_argument("--sleep", type=float, default=0.5, help="Delay between API calls (seconds)")
    p.add_argument("--out-csv", default=str(OUTPUT_CSV), help="Output CSV path")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    verify = not args.insecure

    if args.insecure:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slugs = args.tournaments or list(WTA_TOURNAMENTS.keys())
    min_date = args.min_date

    # Load existing output for incremental processing
    existing_ids: set = set()
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        if "event_id" in existing_df.columns:
            existing_ids = set(existing_df["event_id"].dropna().astype(str))

    # ── Phase 1: Collect match IDs ──────────────────────────────────────────
    if not args.resume:
        LOG.info("=" * 60)
        LOG.info("FLASHSCORE WTA SCRAPER")
        LOG.info(f"  Tournaments: {len(slugs)}")
        LOG.info(f"  Min date: {min_date or 'none'}")
        LOG.info("=" * 60)
        LOG.info("")
        LOG.info("[PHASE 1] Collecting match IDs ...")

        all_matches: List[dict] = []
        for slug in slugs:
            info = WTA_TOURNAMENTS.get(slug, {"surface": "Hard", "name": slug.title()})
            try:
                matches = fetch_tournament_matches(slug, verify_ssl=verify)
                # Filter by date
                if min_date:
                    matches = [m for m in matches if m["match_date"] >= min_date]
                # Add tournament info
                for m in matches:
                    m["tournament"] = info["name"]
                    m["surface"] = info["surface"]
                    m["slug"] = slug
                all_matches.extend(matches)
                LOG.info(f"  {info['name']:20s}: {len(matches)} matches")
            except Exception as exc:
                LOG.warning(f"  {info['name']:20s}: ERROR {exc}")
                continue

        # Deduplicate by event_id
        seen: Dict[str, dict] = {}
        for m in all_matches:
            seen[m["event_id"]] = m
        all_matches = list(seen.values())

        # Save queue
        queue_df = pd.DataFrame(all_matches)
        QUEUE_CSV.parent.mkdir(parents=True, exist_ok=True)
        queue_df.to_csv(QUEUE_CSV, index=False)
        LOG.info(f"")
        LOG.info(f"  Queue: {QUEUE_CSV} ({len(all_matches)} matches)")
    else:
        LOG.info("[PHASE 1] Skipped (--resume). Loading existing queue ...")
        if not QUEUE_CSV.exists():
            LOG.error(f"Queue file not found: {QUEUE_CSV}")
            return 1
        queue_df = pd.read_csv(QUEUE_CSV)
        all_matches = queue_df.to_dict("records")
        LOG.info(f"  Queue: {len(all_matches)} matches")

    # ── Phase 2: Fetch serve stats ──────────────────────────────────────────
    LOG.info("")
    LOG.info("[PHASE 2] Fetching match statistics via API ...")

    to_process = [m for m in all_matches if str(m.get("event_id", "")) not in existing_ids]
    LOG.info(f"  Already processed: {len(existing_ids)}")
    LOG.info(f"  To process: {len(to_process)}")

    rows: List[dict] = []
    errors = 0
    t0 = time.time()

    for i, m in enumerate(to_process):
        eid = str(m["event_id"])

        stats = fetch_match_stats(eid, verify_ssl=verify)

        # Determine winner/loser stats mapping
        # Player A in Flashscore = SH, Player B = SI
        winner_side = m.get("winner_side", "A")
        w_prefix = "a" if winner_side == "A" else "b"
        l_prefix = "b" if winner_side == "A" else "a"

        row = {
            "event_id": eid,
            "source": "flashscore",
            "match_date": m.get("match_date", ""),
            "surface": m.get("surface", "Hard"),
            "tourney_name": m.get("tournament", ""),
            "round": m.get("round", ""),
            "winner_name": m.get("winner_name", ""),
            "loser_name": m.get("loser_name", ""),
            "score": m.get("score", ""),
        }

        if stats:
            # Map to winner/loser
            for sackmann_col, fs_col in [
                ("w_ace", f"{w_prefix}_ace"),
                ("w_df", f"{w_prefix}_df"),
                ("w_svpt", f"{w_prefix}_svpt"),
                ("w_1stIn", f"{w_prefix}_1stIn"),
                ("w_1stWon", f"{w_prefix}_1stWon"),
                ("w_2ndWon", f"{w_prefix}_2ndWon"),
                ("w_SvGms", f"{w_prefix}_SvGms"),
                ("w_bpSaved", f"{w_prefix}_bpSaved"),
                ("w_bpFaced", f"{w_prefix}_bpFaced"),
                ("l_ace", f"{l_prefix}_ace"),
                ("l_df", f"{l_prefix}_df"),
                ("l_svpt", f"{l_prefix}_svpt"),
                ("l_1stIn", f"{l_prefix}_1stIn"),
                ("l_1stWon", f"{l_prefix}_1stWon"),
                ("l_2ndWon", f"{l_prefix}_2ndWon"),
                ("l_SvGms", f"{l_prefix}_SvGms"),
                ("l_bpSaved", f"{l_prefix}_bpSaved"),
                ("l_bpFaced", f"{l_prefix}_bpFaced"),
            ]:
                row[sackmann_col] = stats.get(fs_col)
        else:
            errors += 1

        rows.append(row)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1) if i > 0 else 0
            LOG.info(f"    Progress: {i + 1}/{len(to_process)} matches, {errors} errors, {rate:.2f}s/match")

        # Save every 50 matches
        if (i + 1) % 50 == 0 and rows:
            _save_incremental(rows, out_path, existing_ids)

        time.sleep(args.sleep)

    # Final save
    if rows:
        _save_incremental(rows, out_path, existing_ids)

    elapsed = time.time() - t0
    rate = elapsed / len(to_process) if to_process else 0
    LOG.info("")
    LOG.info(f"  Stats fetched: {len(to_process) - errors}")
    LOG.info(f"  Errors       : {errors}")
    LOG.info(f"  Time         : {int(elapsed)}s ({rate:.2f}s/match)")
    LOG.info(f"  Output       : {out_path}")

    # Final stats
    if out_path.exists():
        final_df = pd.read_csv(out_path)
        LOG.info(f"")
        LOG.info(f"  Total rows: {len(final_df)}")
        has_stats = final_df["w_svpt"].notna().sum()
        LOG.info(f"  w_svpt: {has_stats} non-null of {len(final_df)}")

    # ── Optional merge ──────────────────────────────────────────────────────
    if args.merge_into and out_path.exists():
        _merge_into_history(out_path, Path(args.merge_into))

    LOG.info("")
    LOG.info("Done!")
    return 0


def _save_incremental(rows: List[dict], out_path: Path, existing_ids: set) -> None:
    """Append new rows to the output CSV."""
    new_df = pd.DataFrame(rows)
    if out_path.exists():
        old_df = pd.read_csv(out_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
    else:
        combined = new_df
    combined.to_csv(out_path, index=False)
    for r in rows:
        existing_ids.add(str(r.get("event_id", "")))
    rows.clear()


def _merge_into_history(fs_path: Path, hist_path: Path) -> None:
    """Merge Flashscore WTA stats into the main wta_matches_combined.csv."""
    LOG.info(f"  Merging into {hist_path} ...")
    fs = pd.read_csv(fs_path)
    hist = pd.read_csv(hist_path)

    # Only merge rows that have serve stats
    fs_with_stats = fs[fs["w_svpt"].notna()].copy()
    if fs_with_stats.empty:
        LOG.info("    No rows with stats to merge.")
        return

    # Create merge key
    fs_with_stats["_key"] = (
        fs_with_stats["match_date"].astype(str).str[:10] + "_"
        + fs_with_stats["winner_name"].str.strip().str.lower() + "_"
        + fs_with_stats["loser_name"].str.strip().str.lower()
    )
    hist["_key"] = (
        hist["match_date"].astype(str).str[:10] + "_"
        + hist["winner_name"].str.strip().str.lower() + "_"
        + hist["loser_name"].str.strip().str.lower()
    )

    existing_keys = set(hist["_key"].dropna())
    new_rows = fs_with_stats[~fs_with_stats["_key"].isin(existing_keys)]

    if new_rows.empty:
        LOG.info("    No new matches to add.")
        hist.drop(columns=["_key"], inplace=True)
        return

    # Map Flashscore columns to history columns
    mapped = new_rows.rename(columns={
        "tourney_name": "tourney_name",
    })

    # Keep only columns that exist in history
    hist_cols = set(hist.columns) - {"_key"}
    for col in hist_cols:
        if col not in mapped.columns:
            mapped[col] = np.nan

    mapped = mapped[[c for c in hist.columns if c != "_key" and c in mapped.columns]]

    combined = pd.concat([hist.drop(columns=["_key"]), mapped], ignore_index=True)
    combined = combined.sort_values("match_date").reset_index(drop=True)
    combined.to_csv(hist_path, index=False)

    LOG.info(f"    Added {len(new_rows)} new matches. Total: {len(combined)}")


if __name__ == "__main__":
    raise SystemExit(main())
