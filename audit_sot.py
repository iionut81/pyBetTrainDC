"""
audit_sot.py — Full audit for SOT per-team model (v2.0)

Produces a DC-style audit report:
  - Per-league hit rate (best/worst)
  - Sharpness (prediction distribution)
  - Calibration gaps (predicted vs observed)
  - OOS production backtest
  - ROI simulation at different odds levels

Run: python audit_sot.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

PRED = Path("simulations/SOT/backtests/sot_predictions.csv")
LINES = [2.5, 3.5, 4.5, 5.5]

# Probability threshold for "high confidence" recommendations.
MIN_CONFIDENCE = 0.70

# OOS window (last 8 months).
OOS_START = dt.date(2025, 9, 1)


def _load() -> pd.DataFrame:
    if not PRED.exists():
        raise RuntimeError(f"Missing predictions: {PRED}. Run train_sot_per_team.py first.")
    df = pd.read_csv(PRED)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).copy()
    df["league"] = df["league"].astype(str).str.upper()
    return df


def _long_format(df: pd.DataFrame, use_calibrated: bool = False) -> pd.DataFrame:
    """Melt predictions into (match, side, line, p_over, y_over) rows.
    If use_calibrated is True, reads p_over_{side}_{line}_cal columns."""
    rows: list[dict] = []
    suffix = "_cal" if use_calibrated else ""
    for _, r in df.iterrows():
        base = {
            "match_date": r["match_date"].date() if hasattr(r["match_date"], "date") else r["match_date"],
            "league": r["league"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
        }
        for side in ("home", "away"):
            team = r["home_team"] if side == "home" else r["away_team"]
            for line in LINES:
                p_col = f"p_over_{side}_{line}{suffix}"
                y_col = f"y_over_{side}_{line}"
                if p_col not in r or y_col not in r:
                    continue
                rows.append({
                    **base,
                    "side": side,
                    "team": team,
                    "line": line,
                    "p_over": float(r[p_col]),
                    "y_over": float(r[y_col]),
                })
    return pd.DataFrame(rows)


def _hit_recommender(x: pd.DataFrame, min_p: float) -> dict:
    """Treat probability >= min_p as OVER bet, < (1-min_p) as UNDER bet. Others = no bet."""
    over_bet = x[x["p_over"] >= min_p].copy()
    under_bet = x[x["p_over"] <= (1 - min_p)].copy()
    over_hits = int(over_bet["y_over"].sum())
    under_hits = int((under_bet["y_over"] == 0).sum())
    total_bets = len(over_bet) + len(under_bet)
    total_hits = over_hits + under_hits
    hit_rate = (total_hits / total_bets) if total_bets > 0 else np.nan
    return {
        "n_bets": total_bets,
        "over_bets": len(over_bet),
        "under_bets": len(under_bet),
        "hits": total_hits,
        "hit_rate": hit_rate,
    }


def _metrics_bucket(x: pd.DataFrame) -> dict:
    """Standard probabilistic metrics on full prediction set."""
    if x.empty:
        return {"n": 0, "log_loss": np.nan, "brier": np.nan, "p_mean": np.nan}
    p = np.clip(x["p_over"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    y = x["y_over"].to_numpy(dtype=float)
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    br = float(np.mean((p - y) ** 2))
    return {
        "n": int(len(x)),
        "log_loss": ll,
        "brier": br,
        "p_mean": float(np.mean(p)),
    }


def _per_league_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lg, g in long_df.groupby("league"):
        rec = _hit_recommender(g, MIN_CONFIDENCE)
        met = _metrics_bucket(g)
        rows.append({
            "league": lg,
            "n_preds": met["n"],
            "n_bets": rec["n_bets"],
            "hit_rate": rec["hit_rate"],
            "log_loss": met["log_loss"],
            "brier": met["brier"],
            "p_mean": met["p_mean"],
        })
    t = pd.DataFrame(rows)
    t = t.sort_values("hit_rate", ascending=False, na_position="last").reset_index(drop=True)
    return t


def _sharpness(long_df: pd.DataFrame) -> pd.DataFrame:
    buckets = [
        ("0.0-0.3", 0.0, 0.3),
        ("0.3-0.5", 0.3, 0.5),
        ("0.5-0.7", 0.5, 0.7),
        ("0.7-0.8", 0.7, 0.8),
        ("0.8-0.9", 0.8, 0.9),
        ("0.9-1.0", 0.9, 1.0001),
    ]
    rows = []
    for label, lo, hi in buckets:
        m = (long_df["p_over"] >= lo) & (long_df["p_over"] < hi)
        rows.append({"bucket": label, "count": int(m.sum())})
    return pd.DataFrame(rows)


def _calibration(long_df: pd.DataFrame) -> pd.DataFrame:
    buckets = [
        ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90),
        ("0.90-1.00", 0.90, 1.0001),
    ]
    rows = []
    for label, lo, hi in buckets:
        m = (long_df["p_over"] >= lo) & (long_df["p_over"] < hi)
        sub = long_df[m]
        if sub.empty:
            continue
        pred = float(sub["p_over"].mean())
        obs = float(sub["y_over"].mean())
        diff = pred - obs
        note = "well calibrated" if abs(diff) < 0.02 else (
            "slightly over-confident" if diff > 0 else "slightly under-confident"
        )
        rows.append({
            "bucket": label,
            "predicted": round(pred, 4),
            "observed": round(obs, 4),
            "diff": round(diff, 3),
            "note": note,
            "n": len(sub),
        })
    return pd.DataFrame(rows)


def _oos_backtest(long_df: pd.DataFrame, start: dt.date) -> dict:
    oos = long_df[long_df["match_date"] >= start]
    rec = _hit_recommender(oos, MIN_CONFIDENCE)
    # ROI simulation at different fair-odds assumptions.
    # For each bet (prob p), assume we get odds = target_odds.
    # Profit if win = (odds - 1), loss = -1.
    roi_rows = []
    for target_odds in [1.20, 1.25, 1.30, 1.35, 1.45, 1.60]:
        over_bet = oos[oos["p_over"] >= MIN_CONFIDENCE]
        under_bet = oos[oos["p_over"] <= (1 - MIN_CONFIDENCE)]
        # Only bets where fair > target (positive edge).
        stakes = 0
        pnl = 0.0
        for _, r in over_bet.iterrows():
            fair = 1.0 / r["p_over"]
            if target_odds >= fair:
                stakes += 1
                pnl += (target_odds - 1.0) if r["y_over"] == 1 else -1.0
        for _, r in under_bet.iterrows():
            p_u = 1.0 - r["p_over"]
            fair = 1.0 / p_u
            if target_odds >= fair:
                stakes += 1
                pnl += (target_odds - 1.0) if r["y_over"] == 0 else -1.0
        roi = (pnl / stakes * 100) if stakes > 0 else np.nan
        roi_rows.append({"target_odds": target_odds, "stakes": stakes, "pnl": round(pnl, 2), "roi_pct": round(roi, 2) if not np.isnan(roi) else None})
    return {
        "window": f"{start.isoformat()} → {oos['match_date'].max().isoformat() if not oos.empty else 'n/a'}",
        "n_bets": rec["n_bets"],
        "hits": rec["hits"],
        "hit_rate": rec["hit_rate"],
        "roi_table": pd.DataFrame(roi_rows),
    }


def _per_line_breakdown(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in LINES:
        sub = long_df[long_df["line"] == line]
        rec = _hit_recommender(sub, MIN_CONFIDENCE)
        met = _metrics_bucket(sub)
        rows.append({
            "line": line,
            "n_preds": met["n"],
            "n_bets": rec["n_bets"],
            "hit_rate": rec["hit_rate"],
            "log_loss": met["log_loss"],
            "brier": met["brier"],
        })
    return pd.DataFrame(rows)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", action="store_true", help="Audit Platt-calibrated probabilities instead of raw NB")
    args = parser.parse_args()

    df = _load()
    long_df = _long_format(df, use_calibrated=args.calibrated)
    calib_tag = "CALIBRATED" if args.calibrated else "RAW"
    print(f"[AUDIT MODE: {calib_tag}]")

    _print_section(f"SOT AUDIT v2.0 — {len(long_df):,} predictions across {df['league'].nunique()} leagues")
    print(f"Window: {long_df['match_date'].min()} → {long_df['match_date'].max()}")
    print(f"High-confidence threshold: p_over >= {MIN_CONFIDENCE} (OVER) or <= {1-MIN_CONFIDENCE:.2f} (UNDER)")

    # Per-league
    _print_section("Per-league table (sorted by hit_rate)")
    per_lg = _per_league_table(long_df)
    print(per_lg.to_string(index=False))

    _print_section("Best 6 leagues:")
    print(per_lg.head(6).to_string(index=False))

    _print_section("Worst 6 leagues:")
    print(per_lg.tail(6).to_string(index=False))

    # Per-line
    _print_section("Per-line breakdown (all leagues combined)")
    print(_per_line_breakdown(long_df).to_string(index=False))

    # Sharpness
    _print_section("Sharpness (prediction distribution)")
    print(_sharpness(long_df).to_string(index=False))

    # Calibration
    _print_section("Calibration gaps (high-confidence buckets)")
    print(_calibration(long_df).to_string(index=False))

    # OOS
    _print_section(f"Production backtest (OOS from {OOS_START.isoformat()})")
    oos = _oos_backtest(long_df, OOS_START)
    print(f"Window: {oos['window']}")
    print(f"Predictions (high-conf bets): {oos['n_bets']:,}")
    print(f"Wins: {oos['hits']:,}")
    if oos['hit_rate'] is not None:
        print(f"Hit rate: {oos['hit_rate']:.4f}")

    _print_section("ROI simulation (OOS, high-conf bets, flat 1 unit stake, only when bookie odds >= fair)")
    print(oos["roi_table"].to_string(index=False))

    # Overall summary
    all_rec = _hit_recommender(long_df, MIN_CONFIDENCE)
    _print_section("Overall Summary")
    print(f"Total predictions: {len(long_df):,}")
    print(f"High-confidence bets: {all_rec['n_bets']:,} ({all_rec['over_bets']:,} OVER, {all_rec['under_bets']:,} UNDER)")
    print(f"Hits: {all_rec['hits']:,}")
    if all_rec['hit_rate'] is not None:
        print(f"Overall hit rate: {all_rec['hit_rate']:.4f}")

    # Save CSVs
    out_dir = Path("simulations/SOT/backtests")
    per_lg.to_csv(out_dir / "sot_audit_per_league.csv", index=False)
    _per_line_breakdown(long_df).to_csv(out_dir / "sot_audit_per_line.csv", index=False)
    oos["roi_table"].to_csv(out_dir / "sot_audit_roi.csv", index=False)
    print(f"\nSaved CSVs to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())