"""Daily prediction orchestrator.

Runs all daily pipelines in sequence, logs results, and sends
failure notifications. Replaces the PowerShell run_daily_all.ps1
as the single entry point for daily automation.

Usage:
  python run_daily_all.py --api-key YOUR_KEY --insecure
  python run_daily_all.py --api-key YOUR_KEY --days-ahead 1 --series 18
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from typing import List

from notify import notify


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
    p = argparse.ArgumentParser(description="Daily pipeline: history update + all predictions.")
    p.add_argument("--api-key", default="", help="API-Football key (or set API_FOOTBALL_TOKEN env var)")
    p.add_argument("--days-ahead", type=int, default=0, help="0=today, 1=tomorrow")
    p.add_argument("--series", default="1")
    p.add_argument("--ratings-pkl", default="data/historical/team_ratings.pkl")
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--skip-history", action="store_true")
    p.add_argument("--skip-dc", action="store_true")
    p.add_argument("--skip-fhg", action="store_true")
    p.add_argument("--skip-goals", action="store_true")
    p.add_argument("--skip-corners", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable
    import os
    api_key = args.api_key or os.environ.get("API_FOOTBALL_TOKEN", "")
    target_date = (dt.date.today() + dt.timedelta(days=args.days_ahead)).isoformat()
    insecure = ["--insecure"] if args.insecure else []

    print(f"\n{'═' * 60}")
    print(f"  Daily Pipeline  |  target={target_date}  |  series={args.series}")
    print(f"{'═' * 60}")

    results: list[tuple[str, bool]] = []

    def step(label: str, cmd: List[str]) -> bool:
        ok = _run(label, cmd)
        results.append((label, ok))
        return ok

    # ── 1. History update (Transfermarkt) ──────────────────────────────────
    if not args.skip_history:
        step(
            "History update (Transfermarkt)",
            [py, "import_transfermarkt.py",
             "--start-season-year", "2025",
             "--n-seasons", "1",
             "--output-csv", "simulations/transfermarkt_daily_2025_26_snapshot.csv",
             "--merge-into", "data/historical/historical_matches_transfermarkt.csv",
             "--sleep-seconds", "0.3"],
        )

    # ── 2. DC Double Chance predictions ────────────────────────────────────
    if not args.skip_dc:
        if api_key:
            step(
                "DC predictions (API-Football)",
                [py, "run_dc_daily.py",
                 "--api-key", api_key,
                 "--target-date", target_date,
                 "--ratings-pkl", args.ratings_pkl,
                 "--series", args.series] + insecure,
            )
        else:
            step(
                "DC predictions (Flashscore fallback)",
                [py, "main.py",
                 "--provider", "flashscore",
                 "--target-date", target_date,
                 "--ratings-pkl", args.ratings_pkl] + insecure,
            )

    # ── 3. FHG predictions ─────────────────────────────────────────────────
    if not args.skip_fhg and api_key:
        step(
            "FHG predictions",
            [py, "run_fhg_daily.py",
             "--api-key", api_key,
             "--target-date", target_date,
             "--series", args.series] + insecure,
        )

    # ── 4. Goals Totals predictions ────────────────────────────────────────
    if not args.skip_goals and api_key:
        step(
            "Goals Totals predictions",
            [py, "run_goals_totals_daily.py",
             "--api-key", api_key,
             "--target-date", target_date,
             "--series", args.series] + insecure,
        )

    # ── 5. Corners U12.5 predictions ──────────────────────────────────────
    if not args.skip_corners and api_key:
        step(
            "Corners U12.5 predictions",
            [py, "run_corners_daily.py",
             "--api-key", api_key,
             "--target-date", target_date,
             "--series", args.series] + insecure,
        )

    # ── Summary & notification ─────────────────────────────────────────────
    n_ok = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)

    print(f"\n{'═' * 60}")
    print(f"  Daily pipeline complete — {n_ok} OK  {n_fail} FAILED")
    for label, ok in results:
        icon = "✓" if ok else "✗"
        print(f"    {icon}  {label}")
    print(f"{'═' * 60}\n")

    notify("Daily Pipeline", results)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())