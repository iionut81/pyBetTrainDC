from __future__ import annotations

"""
analyze_wta_ablation.py
Compare Set1 Over 7.5 outcomes vs model, and vs production gates (wta_set1_filters).

Requires ``wta_predictions.csv`` from train_wta.py. For full gate replication set in config.yaml:

  wta.backtest.store_expected_games: true

Then re-run: python train_wta.py

Usage:
  python analyze_wta_ablation.py
  python analyze_wta_ablation.py --predictions simulations/WTA/backtests/wta_predictions.csv
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import CFG
from wta_set1_filters import eval_set1_o75_gates, merge_set1_o75_config


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> int:
    p = argparse.ArgumentParser(description="WTA backtest ablation / surface splits for Set1 O7.5.")
    p.add_argument(
        "--predictions",
        default="simulations/WTA/backtests/wta_predictions.csv",
        help="Output of train_wta.py walk-forward merge",
    )
    args = p.parse_args()
    path = Path(args.predictions)
    if not path.is_file():
        print(f"Missing {path} — run train_wta.py first.")
        return 1

    df = pd.read_csv(path)
    req = {"surface", "p_set1_over_7_5", "y_set1_over_7_5"}
    if not req.issubset(df.columns):
        print(f"CSV must contain columns {req!r}")
        return 1

    wta: Dict[str, Any] = CFG.get("wta", {})
    s175 = wta.get("set1_o75") if isinstance(wta.get("set1_o75"), dict) else {}
    grass_o75 = (wta.get("grass_policy") or {}).get("set1_o75")
    grass_o75 = grass_o75 if isinstance(grass_o75, dict) else None

    sub = df.dropna(subset=["y_set1_over_7_5"]).copy()
    y = sub["y_set1_over_7_5"].to_numpy(dtype=float)
    praw = sub["p_set1_over_7_5"].to_numpy(dtype=float)

    print("=== Set 1 Over 7.5 — by surface (all rows, raw walk-forward p) ===")
    for surf in sorted(sub["surface"].astype(str).unique()):
        m = sub["surface"].astype(str) == surf
        if m.sum() < 20:
            continue
        print(
            f"  {surf:6s} n={int(m.sum()):5d}  "
            f"logloss={_log_loss(praw[m], y[m]):.4f}  brier={_brier(praw[m], y[m]):.4f}  "
            f"rate_hit={float(y[m].mean()):.3f}  p_mean={float(praw[m].mean()):.3f}"
        )

    has_exp = "expected_total_games" in sub.columns and sub["expected_total_games"].notna().any()
    has_hold = "p_hold_w" in sub.columns and "p_hold_l" in sub.columns
    if not has_exp or not has_hold:
        print()
        if not has_exp:
            print("Tip: set wta.backtest.store_expected_games: true and re-run train_wta.py for gate ablation.")
        if not has_hold:
            print("Missing p_hold_w / p_hold_l — cannot run Set1 O7.5 gates.")
        return 0

    print()
    print("=== Set1 O7.5 gates (same module as run_wta_daily) — calibrated p proxy = p_set1_over_7_5 ===")

    rec_flags = []
    for _, row in sub.iterrows():
        surf = str(row["surface"])
        o75 = merge_set1_o75_config(s175, grass_o75, surface=surf)
        rid = int(row["round_id"]) if "round_id" in row and pd.notna(row.get("round_id")) else 0
        eg = float(row["expected_total_games"])
        g = eval_set1_o75_gates(
            float(row["p_hold_w"]),
            float(row["p_hold_l"]),
            eg,
            float(row["p_set1_over_7_5"]),
            surf,
            "",  # tourney level unknown in Sackmann export → is_lower_tier True
            rid,
            o75,
        )
        rec_flags.append(bool(g["rec_s1_7"]))

    sub = sub.assign(_rec=np.array(rec_flags, dtype=bool))
    base_mask = np.ones(len(sub), dtype=bool)
    gated_mask = sub["_rec"].to_numpy()

    def report(label: str, mask: np.ndarray) -> None:
        if mask.sum() < 5:
            print(f"  {label}: too few samples ({int(mask.sum())})")
            return
        print(
            f"  {label:28s} n={int(mask.sum()):5d}  "
            f"logloss={_log_loss(praw[mask], y[mask]):.4f}  "
            f"hit_rate={float(y[mask].mean()):.3f}  "
            f"p_mean={float(praw[mask].mean()):.3f}"
        )

    report("all rows", base_mask)
    report("gates would recommend", gated_mask)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
