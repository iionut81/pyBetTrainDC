from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_fhg_history import build_ratios


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild FHG league ratios from existing FHG history CSV.")
    p.add_argument("--history-csv", default="simulations/FHG/data/fhg_history.csv")
    p.add_argument("--out-ratios", default="simulations/FHG/data/fhg_league_ratios.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    h = pd.read_csv(args.history_csv)
    if h.empty:
        raise RuntimeError("History CSV is empty; cannot build ratios.")
    ratios = build_ratios(h)
    out = Path(args.out_ratios)
    out.parent.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(out, index=False)
    print(f"Saved ratios: {out} rows={len(ratios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

