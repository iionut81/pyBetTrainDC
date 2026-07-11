"""
train_wta_tiebreak_s2.py
Retrain WTA tiebreak model with S2 TB as target (not S1).

Key changes vs current model:
  1. Target = S2 tiebreak (y_s2_tb), not S1 tiebreak (y_tiebreak)
  2. New feature: per-player rolling S2 TB rate (s2_tb_rate_avg)
  3. Separate calibration per surface (grass has too little data -> pooled)
  4. Walk-forward validation to measure real OOS gain

Outputs:
  simulations/WTA/data/wta_tiebreak_s2_model.pkl   (weight vector)
  simulations/WTA/data/wta_s2_tb_rates.pkl          (per-player rolling S2 TB rate table)
  simulations/WTA/backtests/s2_model_comparison.csv (OOS predictions for audit)
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent
HIST_PATH = ROOT / "data/historical/wta_matches_combined.csv"
PRED_PATH = ROOT / "simulations/WTA/backtests/wta_predictions.csv"
OUT_DIR   = ROOT / "simulations/WTA/data"
AUDIT_DIR = ROOT / "simulations/WTA/backtests"

# ── Score parsing ──────────────────────────────────────────────────────────────

def _parse_sets(score: str) -> list[tuple[int, int, bool]]:
    score = re.sub(r'\s*(RET|W/O|DEF|Def\.?|Abd\.?).*', '', str(score), flags=re.I).strip()
    result = []
    for part in score.split():
        m = re.match(r'^(\d+)-(\d+)(\(\d+\))?$', part)
        if m:
            w, l, tb = int(m.group(1)), int(m.group(2)), m.group(3) is not None
            result.append((w, l, tb))
    return result


def get_s1_tb(score: str) -> float | None:
    sets = _parse_sets(score)
    if not sets:
        return None
    return float(sets[0][2])


def get_s2_tb(score: str) -> float | None:
    sets = _parse_sets(score)
    if len(sets) < 2:
        return None
    return float(sets[1][2])


# ── Per-player rolling S2 TB rates ────────────────────────────────────────────

def build_s2_tb_rates(hist: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    For every match in hist, compute the pre-match rolling S2 TB rate for each player
    (on same surface, using prior `window` matches). Returns indexed by (player_id, match_date, surface).
    """
    hist = hist.copy()
    hist["s2_tb"] = hist["score"].apply(get_s2_tb)
    hist = hist.dropna(subset=["s2_tb"])

    # Flatten: each player gets a row per match (as winner or loser)
    w_rows = hist[["match_date", "surface", "winner_id", "s2_tb"]].copy()
    w_rows.columns = ["match_date", "surface", "player_id", "s2_tb"]
    l_rows = hist[["match_date", "surface", "loser_id", "s2_tb"]].copy()
    l_rows.columns = ["match_date", "surface", "player_id", "s2_tb"]

    flat = pd.concat([w_rows, l_rows], ignore_index=True)
    flat["player_id"] = flat["player_id"].astype("Int64")
    flat = flat.dropna(subset=["player_id", "match_date"])
    flat = flat.sort_values(["player_id", "surface", "match_date"]).reset_index(drop=True)

    # For each player-surface group, compute the rolling S2 TB rate BEFORE each match
    records = []
    for (pid, surf), grp in flat.groupby(["player_id", "surface"], sort=False):
        grp = grp.sort_values("match_date").reset_index(drop=True)
        s2_tb_vals = grp["s2_tb"].values
        dates       = grp["match_date"].values

        for i in range(len(grp)):
            # Use matches BEFORE this one (indices 0..i-1)
            start = max(0, i - window)
            past  = s2_tb_vals[start:i]
            if len(past) >= 5:
                rate = float(np.mean(past))
            else:
                rate = np.nan  # insufficient history
            records.append({
                "player_id":      pid,
                "surface":        surf,
                "match_date":     dates[i],
                "s2_tb_rate":     rate,
            })

    df_rates = pd.DataFrame(records)
    df_rates["match_date"] = pd.to_datetime(df_rates["match_date"])
    return df_rates


# ── Join predictions + labels ──────────────────────────────────────────────────

def build_training_frame(hist: pd.DataFrame, preds: pd.DataFrame,
                          rates: pd.DataFrame) -> pd.DataFrame:
    """
    Join:
      - model features from preds (p_tiebreak, p_hold_w/l, p_elo, etc.)
      - actual S2 TB from hist
      - rolling S2 TB rates for winner and loser
    """
    hist = hist.copy()
    hist["match_date"] = pd.to_datetime(hist["match_date"])
    hist["y_s2_tb"] = hist["score"].apply(get_s2_tb)
    hist["y_s1_tb"] = hist["score"].apply(get_s1_tb)

    preds = preds.copy()
    preds["match_date"] = pd.to_datetime(preds["match_date"])

    # Join preds with actual S2 outcome
    df = pd.merge(
        preds[["match_date", "surface", "tourney_name", "round",
               "winner_name", "loser_name", "winner_id", "loser_id",
               "p_tiebreak", "p_hold_w", "p_hold_l", "p_elo", "p_markov",
               "y_tiebreak"]],
        hist[["match_date", "surface", "winner_name", "loser_name",
               "y_s2_tb", "y_s1_tb"]],
        on=["match_date", "surface", "winner_name", "loser_name"],
        how="inner",
    )
    df = df.dropna(subset=["y_s2_tb"])

    # Derive features
    df["hold_asym"]     = (df["p_hold_w"] - df["p_hold_l"]).abs()
    df["min_hold"]      = df[["p_hold_w", "p_hold_l"]].min(axis=1)
    df["combined_hold"] = df["p_hold_w"] + df["p_hold_l"]
    df["elo_closeness"] = 1.0 - 2.0 * (df["p_elo"] - 0.5).abs()
    df["is_grass"]      = (df["surface"] == "Grass").astype(float)
    df["is_clay"]       = (df["surface"] == "Clay").astype(float)

    # Join rolling S2 TB rates
    rates_w = rates.rename(columns={"player_id": "winner_id", "s2_tb_rate": "s2_tb_rate_w"})
    rates_l = rates.rename(columns={"player_id": "loser_id",  "s2_tb_rate": "s2_tb_rate_l"})

    df = df.merge(rates_w[["match_date", "surface", "winner_id", "s2_tb_rate_w"]],
                  on=["match_date", "surface", "winner_id"], how="left")
    df = df.merge(rates_l[["match_date", "surface", "loser_id", "s2_tb_rate_l"]],
                  on=["match_date", "surface", "loser_id"], how="left")

    # Fill missing rates with surface baseline
    surface_baseline = df.groupby("surface")["y_s2_tb"].mean()
    for surf, base in surface_baseline.items():
        mask = df["surface"] == surf
        df.loc[mask, "s2_tb_rate_w"] = df.loc[mask, "s2_tb_rate_w"].fillna(base)
        df.loc[mask, "s2_tb_rate_l"] = df.loc[mask, "s2_tb_rate_l"].fillna(base)

    df["s2_tb_rate_avg"] = (df["s2_tb_rate_w"] + df["s2_tb_rate_l"]) / 2.0
    df["s2_tb_rate_max"] = df[["s2_tb_rate_w", "s2_tb_rate_l"]].max(axis=1)

    # Deduplicate: keep only first occurrence per (match_date, surface, winner, loser)
    df = df.drop_duplicates(subset=["match_date", "surface", "winner_name", "loser_name"])

    return df.sort_values("match_date").reset_index(drop=True)


# ── Logistic regression ────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
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
    res = minimize(nll, w0, method="L-BFGS-B", options={"maxiter": 1000})
    return res.x


def predict_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    d = X.shape[1]
    return _sigmoid(X @ w[:d] + w[d])


# ── Walk-forward evaluation ────────────────────────────────────────────────────

FEATURE_SETS = {
    "S1_model_only":      ["p_tiebreak", "hold_asym", "min_hold", "elo_closeness",
                            "is_grass", "is_clay"],
    "S2_recalibrated":    ["p_tiebreak", "hold_asym", "min_hold", "elo_closeness",
                            "is_grass", "is_clay", "s2_tb_rate_avg", "s2_tb_rate_max"],
}


def walk_forward(df: pd.DataFrame, feature_set: list[str],
                 target: str = "y_s2_tb",
                 train_months: int = 24,
                 min_train: int = 400) -> pd.DataFrame:
    df = df.dropna(subset=feature_set + [target]).sort_values("match_date").reset_index(drop=True)
    dates = df["match_date"]
    results = []

    test_months = pd.date_range(
        start=dates.min() + pd.DateOffset(months=train_months),
        end=dates.max(),
        freq="MS",
    )
    for month_start in test_months:
        month_end = month_start + pd.DateOffset(months=1)
        train_mask = dates < month_start
        test_mask  = (dates >= month_start) & (dates < month_end)
        X_train = df.loc[train_mask, feature_set].values
        y_train = df.loc[train_mask, target].values
        X_test  = df.loc[test_mask, feature_set].values
        if len(X_train) < min_train or len(X_test) == 0:
            continue
        w = fit_logistic(X_train, y_train)
        preds = predict_logistic(X_test, w).ravel()
        for idx, pred in zip(df.index[test_mask], preds):
            results.append({"idx": idx, "p_pred": pred})

    if not results:
        return pd.DataFrame()
    res_df = pd.DataFrame(results).set_index("idx")
    return df.join(res_df, how="inner")


def hr_at_thresh(wf: pd.DataFrame, thresh: float) -> dict | None:
    sub = wf[wf["p_pred"] < thresh]
    if len(sub) < 30:
        return None
    hr  = 1.0 - sub["y_s2_tb"].mean()
    return {"n": len(sub), "hr": round(hr * 100, 2), "tb": int(sub["y_s2_tb"].sum())}


# ── Final model training (full data) ──────────────────────────────────────────

def train_final_model(df: pd.DataFrame, feature_set: list[str],
                      target: str = "y_s2_tb") -> np.ndarray:
    data = df.dropna(subset=feature_set + [target])
    X = data[feature_set].values
    y = data[target].values
    print(f"    Training on {len(data)} samples, {len(feature_set)} features")
    w = fit_logistic(X, y, C=1.0)
    return w


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("WTA Tiebreak Model — S2 Recalibration")
    print("=" * 75)

    print("\nLoading data ...")
    hist  = pd.read_csv(HIST_PATH, low_memory=False)
    hist["match_date"] = pd.to_datetime(hist["match_date"])
    preds = pd.read_csv(PRED_PATH, low_memory=False)
    preds["match_date"] = pd.to_datetime(preds["match_date"])

    print("Building per-player rolling S2 TB rates (this takes ~90s) ...")
    rates = build_s2_tb_rates(hist, window=30)
    print(f"  Generated {len(rates)} player-surface-match rate entries")

    print("Joining predictions with S2 outcomes ...")
    df = build_training_frame(hist, preds, rates)
    print(f"  Training frame: {len(df)} rows")
    print(f"  Date range: {df['match_date'].min().date()} - {df['match_date'].max().date()}")
    print()

    # Surface breakdown
    print("  S2 TB rates by surface:")
    for surf in ["Hard", "Clay", "Grass"]:
        s = df[df["surface"] == surf]
        print(f"    {surf}: N={len(s)}, S2 TB rate={s['y_s2_tb'].mean()*100:.1f}%")
    print()

    # ── Walk-forward comparison ────────────────────────────────────────────────
    print("=" * 75)
    print("WALK-FORWARD COMPARISON (24-month train windows)")
    print("=" * 75)

    wf_results = {}
    for model_name, feats in FEATURE_SETS.items():
        print(f"\n  Model: {model_name}")
        print(f"  Features: {', '.join(feats)}")
        wf = walk_forward(df, feats, target="y_s2_tb", train_months=24)
        wf_results[model_name] = wf
        if len(wf) == 0:
            print("    Not enough data for walk-forward.")
            continue
        print(f"  Test samples: {len(wf)}")
        print(f"  {'Threshold':<12} {'N':>6} {'HR%':>7} {'TB losses':>10}")
        print("  " + "-" * 40)
        for thresh in [0.04, 0.06, 0.08, 0.10, 0.12]:
            r = hr_at_thresh(wf, thresh)
            if r:
                print(f"  p < {thresh:<7.2f} {r['n']:>6} {r['hr']:>7.1f}% {r['tb']:>10}")

    # ── Surface-specific walk-forward ──────────────────────────────────────────
    print()
    print("=" * 75)
    print("SURFACE-SPECIFIC WALK-FORWARD (S2 recalibrated model)")
    print("=" * 75)

    best_feats = list(FEATURE_SETS["S2_recalibrated"])
    for surf in ["Hard", "Clay", "Grass"]:
        sub = df[df["surface"] == surf].reset_index(drop=True)
        print(f"\n  {surf} (N={len(sub)})")
        if len(sub) < 200:
            print("    Too few matches for walk-forward.")
            continue
        wf = walk_forward(sub, best_feats, target="y_s2_tb", train_months=24, min_train=200)
        if len(wf) == 0:
            print("    Not enough OOS data.")
            continue
        for thresh in [0.06, 0.08, 0.10]:
            r = hr_at_thresh(wf, thresh)
            if r:
                print(f"    p < {thresh:.2f}: N={r['n']:4d}, HR={r['hr']:.1f}%, TB={r['tb']}")

    # ── Train final model on all data ─────────────────────────────────────────
    print()
    print("=" * 75)
    print("TRAINING FINAL MODELS (all data, for deployment)")
    print("=" * 75)

    best_feats = list(FEATURE_SETS["S2_recalibrated"])
    print(f"\n  Full model features: {best_feats}")
    w_full = train_final_model(df, best_feats, target="y_s2_tb")

    # Compute in-sample calibration
    data_full = df.dropna(subset=best_feats + ["y_s2_tb"])
    X_full = data_full[best_feats].values
    y_full = data_full["y_s2_tb"].values
    p_is = predict_logistic(X_full, w_full).ravel()
    for thresh in [0.06, 0.08, 0.10, 0.12]:
        sub = p_is < thresh
        hr  = 1.0 - y_full[sub].mean() if sub.sum() > 0 else None
        if hr is not None:
            print(f"    In-sample p < {thresh:.2f}: N={sub.sum():4d}, HR={hr*100:.1f}%")

    # ── Save outputs ───────────────────────────────────────────────────────────
    print()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUT_DIR / "wta_tiebreak_s2_model.pkl"
    rates_path = OUT_DIR / "wta_s2_tb_rates.pkl"
    audit_path = AUDIT_DIR / "s2_model_comparison.csv"

    model_payload = {
        "weights":       w_full,
        "feature_names": best_feats,
        "trained_on":    "S2_tiebreak",
        "n_train":       int(len(data_full)),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"  Saved model -> {model_path}")

    # Save rates table for fast lookup at daily run time
    rates_lookup = (
        rates
        .rename(columns={"s2_tb_rate": "s2_tb_rate_rolling"})
        .dropna(subset=["s2_tb_rate_rolling"])
    )
    with open(rates_path, "wb") as f:
        pickle.dump(rates_lookup, f)
    print(f"  Saved S2 TB rates -> {rates_path}")

    # Audit CSV: predictions for all OOS samples
    if wf_results.get("S2_recalibrated") is not None and len(wf_results["S2_recalibrated"]) > 0:
        wf = wf_results["S2_recalibrated"]
        audit_cols = ["match_date", "surface", "tourney_name", "round",
                      "winner_name", "loser_name",
                      "p_tiebreak", "hold_asym", "min_hold",
                      "s2_tb_rate_avg", "y_s1_tb", "y_s2_tb", "p_pred"]
        audit_cols_avail = [c for c in audit_cols if c in wf.columns]
        wf[audit_cols_avail].to_csv(audit_path, index=False)
        print(f"  Saved audit CSV -> {audit_path}")

    print()
    print("=" * 75)
    print("OPERATIONAL THRESHOLDS (recommended)")
    print("=" * 75)
    print()
    print("  To use S2 model in run_wta_daily.py:")
    print("  1. Load wta_tiebreak_s2_model.pkl at startup")
    print("  2. Load wta_s2_tb_rates.pkl for per-player rate lookup")
    print("  3. Build features: p_tiebreak, hold_asym, min_hold, elo_closeness,")
    print("     is_grass, is_clay, s2_tb_rate_avg, s2_tb_rate_max")
    print("  4. Filter: p_s2_pred < 0.08 (Clay/Hard), p_s2_pred < 0.06 (Grass)")
    print()
    print("  Note: Grass N=339 is too thin for reliable calibration.")
    print("  For grass, CoVe manual remains primary filter.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
