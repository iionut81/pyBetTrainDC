from __future__ import annotations

"""Weekly retrain pipeline.

Runs in order:
  1. build_fhg_history.py        — refresh FHG/Goals history from API-Football
  2. build_corners_history.py    — refresh Corners history from API-Football
  3. train_team_ratings.py       — rebuild DC team ratings pkl (FHG + Goals daily)
  4. train_fhg_calibration.py    — retrain FHG Platt calibration
  5. train_fhg_league_bias.py    — retrain FHG league bias factors
  6. train_goals_totals.py       — retrain Goals Totals calibration
  7. train_corners_under_12_5.py — retrain Corners NB model + Platt calibration

Usage:
  python run_weekly_retrain.py --api-key YOUR_KEY --insecure
  python run_weekly_retrain.py --api-key YOUR_KEY --insecure --skip-history  (retrain only)
"""

import argparse
import datetime as dt
import subprocess
import sys
from typing import List

from notify import notify


def _current_season() -> int:
    """Current season start year: switches on July 1."""
    today = dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def _run(label: str, cmd: List[str]) -> bool:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"  → {status}")
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weekly retrain: refresh history + retrain all pipelines.")
    p.add_argument("--api-key", required=True)
    p.add_argument("--insecure", action="store_true",
                   help="Disable SSL verification (required on some corporate networks)")
    p.add_argument("--season", type=int, default=None,
                   help="Season start year to refresh (default: auto-detected from today)")
    p.add_argument("--skip-history", action="store_true",
                   help="Skip API history fetching — retrain on existing data only")
    p.add_argument("--skip-fhg", action="store_true", help="Skip FHG pipeline")
    p.add_argument("--skip-goals", action="store_true", help="Skip Goals Totals pipeline")
    p.add_argument("--skip-corners", action="store_true", help="Skip Corners pipeline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable
    season = args.season or _current_season()
    insecure = ["--insecure"] if args.insecure else []

    print(f"\n{'═' * 60}")
    print(f"  Weekly Retrain  |  season={season}  |  {dt.date.today().isoformat()}")
    print(f"{'═' * 60}")

    results: list[tuple[str, bool]] = []

    def step(label: str, cmd: List[str]) -> bool:
        ok = _run(label, cmd)
        results.append((label, ok))
        return ok

    # ── 0. Run test suite ───────────────────────────────────────────────────
    if not step("Test suite (pytest)", [py, "-m", "pytest", "tests/", "-q"]):
        print("  ⚠ Tests failed — aborting retrain to avoid training on broken code.")
        return 1

    # ── 1. Refresh FHG / Goals history ──────────────────────────────────────
    if not args.skip_history and not args.skip_fhg:
        step(
            "FHG history refresh (API-Football)",
            [py, "build_fhg_history.py",
             "--api-key", args.api_key,
             "--start-season", str(season),
             "--end-season", str(season),
             ] + insecure,
        )

    # ── 2. Refresh Corners history ───────────────────────────────────────────
    if not args.skip_history and not args.skip_corners:
        step(
            "Corners history refresh (API-Football)",
            [py, "build_corners_history.py",
             "--api-key", args.api_key,
             "--seasons", str(season),
             ],
        )

    # ── 3. Rebuild team ratings (used by FHG + Goals daily runners) ──────────
    if not args.skip_fhg or not args.skip_goals:
        step(
            "Team ratings retrain (DC pkl)",
            [py, "train_team_ratings.py"],
        )

    # ── 4. Retrain FHG calibration ───────────────────────────────────────────
    if not args.skip_fhg:
        step("FHG calibration retrain", [py, "train_fhg_calibration.py"])
        step("FHG league bias retrain", [py, "train_fhg_league_bias.py"])

    # ── 5. Retrain Goals Totals ──────────────────────────────────────────────
    if not args.skip_goals:
        step("Goals Totals retrain", [py, "train_goals_totals.py"])

    # ── 6. Retrain Corners U12.5 ─────────────────────────────────────────────
    if not args.skip_corners:
        step("Corners U12.5 retrain", [py, "train_corners_under_12_5.py"])

    # ── Summary ──────────────────────────────────────────────────────────────
    n_ok = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)

    print(f"\n{'═' * 60}")
    print(f"  Weekly retrain complete — {n_ok} OK  {n_fail} FAILED")
    for label, ok in results:
        icon = "✓" if ok else "✗"
        print(f"    {icon}  {label}")
    print(f"{'═' * 60}\n")

    notify("Weekly Retrain", results)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
