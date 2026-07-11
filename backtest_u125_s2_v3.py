"""
backtest_u125_s2_v3.py
Comprehensive WTA U12.5 Set 2 backtest.

What this script tests:
  Model A — current: p_tiebreak < 0.098 (calibrated on S1 TB)
  Model B — S2 recalibration: same features, target = S2 TB
  Model C — B + per-player rolling S2 TB rates
  Model D — C + cascade risk (S1 TB as in-play signal)

Key finding it quantifies:
  The current tb_p_cal is calibrated on S1 tiebreaks and applied to S2.
  This script measures the calibration gap and tests per-player S2 TB rates
  as an additional pre-match signal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
HIST_PATH = ROOT / "data/historical/wta_matches_combined.csv"
PRED_PATH = ROOT / "simulations/WTA/backtests/wta_predictions.csv"
OUT_DIR   = ROOT / "simulations/WTA/backtests"

# ── Thresholds ─────────────────────────────────────────────────────────────────
CURRENT_THRESH   = 0.098   # current operational p_tiebreak threshold
OPTIMAL_THRESH   = 0.080   # tighter threshold to test
HOLD_ASYM_MIN    = 0.15    # minimum hold gap for "class gap" picks
MIN_HOLD_MAX     = 0.50    # max allowed for the weaker player's hold

# ── Score parsing ──────────────────────────────────────────────────────────────

def _clean_score(score: str) -> str:
    return re.sub(r'\s*(RET|W/O|DEF|Def\.?|Abd\.?).*', '', str(score), flags=re.I).strip()


def _parse_sets(score: str) -> list[tuple[int, int, bool]]:
    """Return list of (winner_games, loser_games, is_tiebreak) per set."""
    result = []
    for part in _clean_score(score).split():
        m = re.match(r'^(\d+)-(\d+)(\(\d+\))?$', part)
        if m:
            w, l, tb = int(m.group(1)), int(m.group(2)), m.group(3) is not None
            result.append((w, l, tb))
    return result


def extract_set_info(score: str) -> dict:
    sets = _parse_sets(score)
    out = {
        "s1_total": None, "s1_margin": None, "s1_tb": None,
        "s2_total": None, "s2_tb": None, "n_sets": len(sets),
    }
    if len(sets) >= 1:
        w, l, tb = sets[0]
        out["s1_total"]  = w + l
        out["s1_margin"] = w - l   # positive = winner dominated
        out["s1_tb"]     = int(tb)
    if len(sets) >= 2:
        w, l, tb = sets[1]
        out["s2_total"] = w + l
        out["s2_tb"]    = int(tb)
    return out


# ── Load and parse ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading historical data …")
    hist = pd.read_csv(HIST_PATH, low_memory=False)
    hist["match_date"] = pd.to_datetime(hist["match_date"])

    set_info = hist["score"].apply(extract_set_info)
    for col in ["s1_total", "s1_margin", "s1_tb", "s2_total", "s2_tb", "n_sets"]:
        hist[col] = set_info.apply(lambda d: d[col])

    # Under 12.5 S2 = no tiebreak (7-5 is 12 games = U12.5, 7-6 = 13 = over)
    hist["s2_u125"] = np.where(
        hist["s2_total"].notna(),
        (hist["s2_total"] <= 12).astype(float),
        np.nan,
    )

    print("Loading predictions …")
    preds = pd.read_csv(PRED_PATH, low_memory=False)
    preds["match_date"] = pd.to_datetime(preds["match_date"])

    return hist, preds


# ── Per-player rolling S2 TB rates ────────────────────────────────────────────

def build_player_s2_tb_rates(hist: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    For each match, compute the player's historical S2 TB rate from the
    preceding `window` matches on the same surface.

    Returns dataframe with columns:
      player_id, match_date, surface, s2_tb_rate_rolling
    """
    # Flatten to player perspective (winner + loser)
    w = hist[["match_date", "surface", "winner_id", "s2_tb"]].copy()
    w.columns = ["match_date", "surface", "player_id", "s2_tb"]

    l = hist[["match_date", "surface", "loser_id", "s2_tb"]].copy()
    l.columns = ["match_date", "surface", "player_id", "s2_tb"]

    flat = pd.concat([w, l], ignore_index=True)
    flat = flat.dropna(subset=["s2_tb"]).sort_values("match_date")

    rows = []
    for (pid, surf), grp in flat.groupby(["player_id", "surface"]):
        grp = grp.sort_values("match_date").reset_index(drop=True)
        # Rolling rate: for each row, use the preceding `window` matches
        for i, row in grp.iterrows():
            past = grp.loc[:i-1] if i > 0 else grp.iloc[:0]
            past = past.tail(window)
            if len(past) >= 5:
                rate = past["s2_tb"].mean()
            else:
                rate = np.nan  # not enough data
            rows.append({
                "player_id":         pid,
                "surface":           surf,
                "match_date":        row["match_date"],
                "s2_tb_rate_rolling": rate,
            })

    return pd.DataFrame(rows)


# ── Join predictions with historical outcomes ──────────────────────────────────

def join_preds_hist(preds: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Join the predictions CSV (has model features) with the historical CSV
    (has actual S2 outcomes).

    Match on date + surface + winner/loser names.
    """
    hist_2s = hist[hist["s2_tb"].notna()].copy()

    merged = pd.merge(
        preds[["match_date", "surface", "tourney_name", "round",
               "winner_name", "loser_name", "winner_id", "loser_id",
               "p_tiebreak", "p_hold_w", "p_hold_l", "p_elo",
               "p_markov", "y_tiebreak", "actual_set1_games"]],
        hist_2s[["match_date", "surface", "winner_name", "loser_name",
                  "s1_total", "s1_margin", "s1_tb", "s2_total", "s2_tb", "s2_u125"]],
        on=["match_date", "surface", "winner_name", "loser_name"],
        how="inner",
    )

    merged["hold_asym"]     = (merged["p_hold_w"] - merged["p_hold_l"]).abs()
    merged["min_hold"]      = merged[["p_hold_w", "p_hold_l"]].min(axis=1)
    merged["combined_hold"] = merged["p_hold_w"] + merged["p_hold_l"]
    merged["elo_closeness"] = 1.0 - 2.0 * (merged["p_elo"] - 0.5).abs()

    return merged


# ── Logistic regression helpers ────────────────────────────────────────────────

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    n, d = X.shape
    lam = 1.0 / max(C, 1e-9)

    def nll(w):
        z = X @ w[:d] + w[d]
        p = np.clip(_sigmoid(z), 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(loss + 0.5 * lam * np.sum(w[:d] ** 2) / n)

    w0 = np.zeros(d + 1)
    res = minimize(nll, w0, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    d = X.shape[1] if X.ndim == 2 else len(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _sigmoid(X @ w[:d] + w[d])


# ── Walk-forward cross-validation ─────────────────────────────────────────────

def walk_forward_eval(df: pd.DataFrame, feature_cols: list[str],
                      target: str = "s2_tb", train_months: int = 24,
                      min_train: int = 500) -> pd.DataFrame:
    """
    Month-by-month walk-forward: train on past N months, predict next month.
    Returns dataframe with predictions for every test sample.
    """
    df = df.sort_values("match_date").reset_index(drop=True)
    df = df.dropna(subset=feature_cols + [target])

    results = []
    dates = df["match_date"]
    min_date = dates.min()
    max_date = dates.max()

    test_months = pd.date_range(
        start=min_date + pd.DateOffset(months=train_months),
        end=max_date,
        freq="MS",
    )

    for month_start in test_months:
        month_end = month_start + pd.DateOffset(months=1)
        train_mask = dates < month_start
        test_mask  = (dates >= month_start) & (dates < month_end)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target].values
        X_test  = df.loc[test_mask, feature_cols].values

        if len(X_train) < min_train or len(X_test) == 0:
            continue

        w = fit_logistic(X_train, y_train)
        p_pred = predict_logistic(X_test, w).ravel()

        for idx, pred in zip(df.index[test_mask], p_pred):
            results.append({"idx": idx, "p_pred_wf": pred})

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results).set_index("idx")
    return df.join(res_df, how="inner")


# ── Reporting helpers ──────────────────────────────────────────────────────────

def hr_report(df: pd.DataFrame, label: str, threshold_col: str, threshold: float,
              min_n: int = 20) -> dict:
    sub = df[df[threshold_col] < threshold]
    if len(sub) < min_n:
        return {"label": label, "n": len(sub), "hr": None, "tb_losses": None}
    hr  = 1.0 - sub["s2_tb"].mean()
    return {
        "label":     label,
        "n":         len(sub),
        "hr":        round(hr * 100, 1),
        "tb_losses": int(sub["s2_tb"].sum()),
    }


def print_table(rows: list[dict]) -> None:
    hdr = f"  {'Segment':<52} {'N':>5}  {'HR%':>7}  {'TB losses':>10}"
    print(hdr)
    print("  " + "-" * 80)
    for r in rows:
        hr_s = f"{r['hr']:.1f}%" if r["hr"] is not None else "n/a"
        tb_s = str(r["tb_losses"]) if r["tb_losses"] is not None else "n/a"
        print(f"  {r['label']:<52} {r['n']:>5}  {hr_s:>7}  {tb_s:>10}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    hist, preds = load_data()

    print("\nJoining predictions with historical outcomes …")
    df = join_preds_hist(preds, hist)
    print(f"Matched {len(df)} rows (of {len(preds)} predictions × {len(hist)} historical)")

    # ── SECTION 1: Baseline S2 statistics ─────────────────────────────────────
    print("\n" + "=" * 85)
    print("SECTION 1 — BASELINE S2 STATISTICS (all matched rows)")
    print("=" * 85)

    for surf in ["All", "Hard", "Clay", "Grass"]:
        sub = df if surf == "All" else df[df["surface"] == surf]
        if len(sub) == 0:
            continue
        s2_tb_rate = sub["s2_tb"].mean()
        s1_tb_rate = sub["y_tiebreak"].mean()
        cascade    = sub[sub["s1_tb"] == 1]["s2_tb"].mean() if sub["s1_tb"].notna().any() else None
        no_cascade = sub[sub["s1_tb"] == 0]["s2_tb"].mean() if sub["s1_tb"].notna().any() else None
        cascade_s  = f"{cascade*100:.1f}%" if cascade is not None else "n/a"
        no_casc_s  = f"{no_cascade*100:.1f}%" if no_cascade is not None else "n/a"
        print(f"\n  {surf} (N={len(sub)}):")
        print(f"    Baseline S2 TB rate:          {s2_tb_rate*100:.1f}%  -> S2 U12.5 baseline: {(1-s2_tb_rate)*100:.1f}%")
        print(f"    Baseline S1 TB rate:          {s1_tb_rate*100:.1f}%")
        print(f"    S2 TB rate after S1 TB:       {cascade_s}  (cascade risk)")
        print(f"    S2 TB rate after S1 no-TB:    {no_casc_s}  (no cascade)")

    # ── SECTION 2: Current model (p_tiebreak threshold) ───────────────────────
    print("\n" + "=" * 85)
    print("SECTION 2 — CURRENT MODEL (p_tiebreak < threshold)")
    print("=" * 85)
    print()

    segments_base = []
    for surf in ["All", "Hard", "Clay", "Grass"]:
        sub = df if surf == "All" else df[df["surface"] == surf]
        for thresh, label in [(CURRENT_THRESH, f"current (<{CURRENT_THRESH})"),
                               (OPTIMAL_THRESH, f"tighter (<{OPTIMAL_THRESH})")]:
            r = hr_report(sub, f"{surf} — {label}", "p_tiebreak", thresh)
            segments_base.append(r)

    print_table(segments_base)

    # ── SECTION 3: Hold asymmetry filters ─────────────────────────────────────
    print("\n" + "=" * 85)
    print("SECTION 3 — HOLD ASYMMETRY FILTERS (on top of current threshold)")
    print("=" * 85)
    print()

    sub_curr = df[df["p_tiebreak"] < CURRENT_THRESH]
    asym_segments = []
    for surf in ["All", "Clay", "Grass"]:
        s = sub_curr if surf == "All" else sub_curr[sub_curr["surface"] == surf]
        for asym in [0.10, 0.15, 0.20]:
            filtered = s[s["hold_asym"] >= asym]
            r = hr_report(filtered, f"{surf} | hold_asym≥{asym}", "p_tiebreak", 999)
            r["label"] = f"{surf} | p_tb<{CURRENT_THRESH} + asym≥{asym}"
            asym_segments.append(r)

    print_table(asym_segments)

    # ── SECTION 4: Cascade analysis (S1 TB → S2 TB) ───────────────────────────
    print("\n" + "=" * 85)
    print("SECTION 4 — CASCADE ANALYSIS (S1 TB effect on S2)")
    print("=" * 85)
    print()

    sub_curr = df[df["p_tiebreak"] < CURRENT_THRESH].dropna(subset=["s1_tb"])
    cascade_rows = []
    for surf in ["All", "Hard", "Clay", "Grass"]:
        s = sub_curr if surf == "All" else sub_curr[sub_curr["surface"] == surf]
        if len(s) < 10:
            continue
        for s1_tb_val, s1_label in [(0, "S1 no-TB"), (1, "S1 was TB")]:
            fs = s[s["s1_tb"] == s1_tb_val]
            if len(fs) < 5:
                continue
            hr  = 1 - fs["s2_tb"].mean()
            cascade_rows.append({
                "label":     f"{surf} | {s1_label}",
                "n":         len(fs),
                "hr":        round(hr * 100, 1),
                "tb_losses": int(fs["s2_tb"].sum()),
            })

    print_table(cascade_rows)

    # ── SECTION 5: Per-player S2 TB rates ─────────────────────────────────────
    print("\n" + "=" * 85)
    print("SECTION 5 — PER-PLAYER S2 TB RATES (rolling 30 matches per surface)")
    print("Computing rolling rates … (may take ~60s)")
    print("=" * 85)

    s2_rates = build_player_s2_tb_rates(hist, window=30)

    # Join winner S2 TB rate
    df = df.merge(
        s2_rates.rename(columns={"player_id": "winner_id", "s2_tb_rate_rolling": "s2_tb_rate_w"}),
        on=["match_date", "surface", "winner_id"], how="left",
    )
    # Join loser S2 TB rate
    df = df.merge(
        s2_rates.rename(columns={"player_id": "loser_id", "s2_tb_rate_rolling": "s2_tb_rate_l"}),
        on=["match_date", "surface", "loser_id"], how="left",
    )

    df["s2_tb_rate_avg"] = (df["s2_tb_rate_w"].fillna(0.12) + df["s2_tb_rate_l"].fillna(0.12)) / 2.0
    df["s2_tb_rate_max"] = df[["s2_tb_rate_w", "s2_tb_rate_l"]].max(axis=1)

    print("\n  Distribution of s2_tb_rate_avg (for p_tb < current threshold):")
    sub_curr = df[df["p_tiebreak"] < CURRENT_THRESH].dropna(subset=["s2_tb_rate_avg"])
    print(sub_curr["s2_tb_rate_avg"].describe().to_string())

    s2_rate_rows = []
    for surf in ["All", "Clay", "Grass"]:
        s = sub_curr if surf == "All" else sub_curr[sub_curr["surface"] == surf]
        for rate_thresh in [0.08, 0.10, 0.12, 0.15]:
            fs = s[s["s2_tb_rate_avg"] <= rate_thresh]
            r = hr_report(fs, f"{surf} | s2_rate_avg≤{rate_thresh}", "p_tiebreak", 999)
            r["label"] = f"{surf} | p_tb<{CURRENT_THRESH} + s2_rate_avg≤{rate_thresh}"
            s2_rate_rows.append(r)

    print()
    print_table(s2_rate_rows)

    # ── SECTION 6: Walk-forward logistic regression comparison ─────────────────
    print("\n" + "=" * 85)
    print("SECTION 6 — WALK-FORWARD LOGISTIC REGRESSION (2-year train, 1-month test)")
    print("=" * 85)

    # Model A features: current model features only
    feats_A = ["p_tiebreak", "hold_asym", "min_hold", "elo_closeness"]
    # Model B features: + per-player S2 TB rates
    feats_B = feats_A + ["s2_tb_rate_avg", "s2_tb_rate_max"]
    # Model C features: + cascade (S1 TB, only valid in-play context)
    feats_C = feats_B + ["s1_tb"]

    for model_name, feats in [("A (current proxy)", feats_A),
                               ("B (+ S2 TB rates)", feats_B),
                               ("C (+ cascade S1 TB, in-play)", feats_C)]:
        print(f"\n  Training Model {model_name} …")
        wf = walk_forward_eval(df, feats, target="s2_tb", train_months=24)
        if len(wf) == 0:
            print(f"    Not enough data.")
            continue

        # Compute HR at different probability thresholds
        print(f"    Test samples: {len(wf)}")
        for thresh in [0.05, 0.08, 0.10, 0.12, 0.15]:
            sub = wf[wf["p_pred_wf"] < thresh]
            if len(sub) < 20:
                continue
            hr = 1 - sub["s2_tb"].mean()
            print(f"    p_pred < {thresh:.2f}: N={len(sub):4d} | HR={hr*100:.1f}% | TB losses={int(sub['s2_tb'].sum())}")

    # ── SECTION 7: Summary table by surface ────────────────────────────────────
    print("\n" + "=" * 85)
    print("SECTION 7 — CURRENT MODEL PERFORMANCE SUMMARY")
    print("=" * 85)
    print()
    print(f"  {'Surface':<10} {'Baseline HR':>12} {'p_tb<0.098':>11} {'p_tb<0.080':>11} {'+asym≥0.15':>11}")
    print("  " + "-" * 60)
    for surf in ["Hard", "Clay", "Grass"]:
        s = df[df["surface"] == surf]
        base   = (1 - s["s2_tb"].mean()) * 100
        curr   = (1 - s[s["p_tiebreak"] < CURRENT_THRESH]["s2_tb"].mean()) * 100 if len(s[s["p_tiebreak"] < CURRENT_THRESH]) >= 20 else None
        tight  = (1 - s[s["p_tiebreak"] < OPTIMAL_THRESH]["s2_tb"].mean()) * 100 if len(s[s["p_tiebreak"] < OPTIMAL_THRESH]) >= 20 else None
        combo  = (1 - s[(s["p_tiebreak"] < CURRENT_THRESH) & (s["hold_asym"] >= 0.15)]["s2_tb"].mean()) * 100 if len(s[(s["p_tiebreak"] < CURRENT_THRESH) & (s["hold_asym"] >= 0.15)]) >= 20 else None
        curr_s  = f"{curr:.1f}%" if curr is not None else "n/a"
        tight_s = f"{tight:.1f}%" if tight is not None else "n/a"
        combo_s = f"{combo:.1f}%" if combo is not None else "n/a"
        n_curr  = len(s[s["p_tiebreak"] < CURRENT_THRESH])
        print(f"  {surf:<10} {base:>11.1f}% {curr_s:>11}  {tight_s:>10}  {combo_s:>10}  (N curr={n_curr})")

    print()
    print("Done. Results above show diagnostic picture of current model on S2.")
    print()
    print("Key interpretation:")
    print("  - Section 4 shows CASCADE effect: if S1 TB → S2 TB rate jumps → in-play signal")
    print("  - Section 5 shows per-player S2 TB rate as PRE-MATCH signal")
    print("  - Section 6 walk-forward shows gain from each feature group")
    print("  - Section 7 summary: compare across surfaces")


if __name__ == "__main__":
    main()
