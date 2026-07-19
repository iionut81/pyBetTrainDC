from __future__ import annotations

"""
train_wta.py
Walk-forward backtesting and Platt logit calibration for the WTA tennis model.

Usage:
  python -X utf8 train_wta.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import CFG
from fhg_calibration import (
    apply_isotonic,
    apply_platt_logit,
    apply_temperature,
    fit_isotonic,
    fit_platt_logit,
    fit_temperature,
)
from wta_elo import SurfaceElo
from wta_glicko import SurfaceGlicko
from wta_markov import (
    PlayerServeStats,
    predict_match,
    simulate_match,
)
from wta_ratings import build_player_match_stats, compute_player_stats_fast
from wta_scoring import parse_set1_games
from wta_tiebreak import (
    build_tiebreak_features,
    fit_tiebreak_logistic,
    predict_tiebreak,
    save_tiebreak_model,
)

_TW = CFG["training"]["wta"]
_WTA = CFG["wta"]
_ELO_CFG = _WTA.get("elo", {})
STABILITY = _WTA["stability"]
BLEND_W = _ELO_CFG.get("blend_weight", 0.60)
_MW_TEMP = bool(_WTA.get("calibration", {}).get("match_winner_temperature", False))
_BT = _WTA.get("backtest") or {}
_STORE_EG = bool(_BT.get("store_expected_games", False))
_MC_EG = int(_BT.get("expected_games_mc", 800))

MARKETS = ["match_winner", "tiebreak", "set1_over_7_5", "set1_over_9_5"]


def _sackmann_round_id(round_val: object) -> int:
    """Map common Sackmann round labels to blowout layer IDs (approx. WTA RoundID scale)."""
    s = str(round_val or "").upper().strip()
    if s in ("SF", "BSF", "S"):
        return 4
    if s in ("F", "BR", "R"):
        return 5
    return 0


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _calibration_buckets(p: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_buckets):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        mask = (p >= lo) & (p < hi)
        if i == n_buckets - 1:
            mask = (p >= lo) & (p <= hi)
        if mask.sum() > 0:
            rows.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "p_mean": float(p[mask].mean()),
                "y_mean": float(y[mask].mean()),
                "count": int(mask.sum()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            })
    return pd.DataFrame(rows)



def precompute_elo_predictions(
    df: pd.DataFrame, elo_cfg: dict,
) -> Tuple[Dict[Tuple[int, int, str], float], SurfaceElo]:
    """One chronological pass: predict-then-update for every match.

    Returns (elo_preds, final_elo) where elo_preds maps
    (winner_id, loser_id, match_date_str) -> p_elo_winner.
    """
    elo = SurfaceElo(
        initial_rating=elo_cfg.get("initial_rating", 1500),
        k_initial=elo_cfg.get("k_initial", 40.0),
        k_floor=elo_cfg.get("k_floor", 10.0),
        k_decay_rate=elo_cfg.get("k_decay_rate", 0.05),
    )
    preds: Dict[Tuple[int, int, str], float] = {}

    for _, row in df.sort_values("match_date").iterrows():
        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])
        surface = row["surface"]
        date_key = str(row["match_date"])

        # Predict BEFORE update (no leakage)
        p = elo.predict(w_id, l_id, surface)
        if p is not None:
            preds[(w_id, l_id, date_key)] = p

        # Update AFTER prediction (with margin-of-victory)
        _w_sets = int(row.get("w_sets", 0) or 0)
        _l_sets = int(row.get("l_sets", 0) or 0)
        _tg = int(row.get("total_games", 0) or 0)
        _md = pd.Timestamp(row["match_date"])
        elo.update(w_id, l_id, surface,
                   w_sets=_w_sets, l_sets=_l_sets, total_games=_tg, match_date=_md)
        elo.update(w_id, l_id, "__ALL__",
                   w_sets=_w_sets, l_sets=_l_sets, total_games=_tg, match_date=_md)

    elo.last_processed_date = df["match_date"].max()
    return preds, elo


def precompute_glicko_predictions(
    df: pd.DataFrame,
) -> Tuple[Dict[Tuple[int, int, str], float], SurfaceGlicko]:
    """One chronological pass: predict-then-update for every match, Glicko-2.

    Validated (backtest_glicko_vs_elo_blended.py) to beat SurfaceElo by ~10.8%
    Brier / ~29% log-loss on 41,631 historical matches — replaces Elo as the
    rating signal blended with the Markov model (see BLEND_W usage below).
    """
    glicko = SurfaceGlicko()
    preds: Dict[Tuple[int, int, str], float] = {}

    for _, row in df.sort_values("match_date").iterrows():
        w_id = int(row["winner_id"])
        l_id = int(row["loser_id"])
        surface = row["surface"]
        date_key = str(row["match_date"])

        p = glicko.predict(w_id, l_id, surface)
        if p is not None:
            preds[(w_id, l_id, date_key)] = p

        glicko.update(w_id, l_id, surface, match_date=pd.Timestamp(row["match_date"]))

    glicko.last_processed_date = df["match_date"].max()
    return preds, glicko


def walk_forward(
    df: pd.DataFrame,
    pms: pd.DataFrame,
    surface: str,
    lookback_days: int,
    retrain_days: int,
    min_train_matches: int,
    min_players: int,
    rolling_window: int,
    recency_decay: float,
    min_matches_for_rating: int,
    elo_preds: Dict[Tuple[int, int, str], float] = None,
    glicko_preds: Dict[Tuple[int, int, str], float] = None,
    blend_weight: float = 0.60,
) -> pd.DataFrame:
    """Walk-forward prediction for a single surface. Blends Markov + Elo.
    Trains tiebreak logistic regression per window."""
    g = df[df["surface"] == surface].sort_values("match_date").reset_index(drop=True)
    if g.empty:
        return pd.DataFrame(), None

    # Pre-filter pms to this surface for speed
    pms_surf = pms[pms["surface"] == surface].copy()

    min_date = g["match_date"].min()
    max_date = g["match_date"].max()
    anchor = min_date + pd.Timedelta(days=lookback_days)

    rows: List[dict] = []
    last_tb_weights = None  # carry forward tiebreak model across windows

    while anchor <= max_date:
        train_start = anchor - pd.Timedelta(days=lookback_days)
        pred_end = anchor + pd.Timedelta(days=retrain_days)

        train = g[(g["match_date"] >= train_start) & (g["match_date"] < anchor)]
        pred = g[(g["match_date"] >= anchor) & (g["match_date"] < pred_end)]

        if pred.empty:
            anchor = pred_end
            continue

        players = set(train["winner_id"]).union(set(train["loser_id"]))
        if len(train) < min_train_matches or len(players) < min_players:
            anchor = pred_end
            continue

        # Pre-filter pms to training window
        pms_window = pms_surf[
            (pms_surf["match_date"] >= train_start) & (pms_surf["match_date"] < anchor)
        ]

        # ── Compute per-player tiebreak rates, Set 1 avg games, and surface base rate ──
        player_tb_hits: Dict[int, int] = {}
        player_tb_total: Dict[int, int] = {}
        player_s1_games_sum: Dict[int, float] = {}   # for momentum adjustment
        player_s1_games_count: Dict[int, int] = {}
        tb_total_count = 0
        tb_hit_count = 0
        s1_games_all: List[int] = []
        for _, trow in train.iterrows():
            s1g = parse_set1_games(trow.get("score", ""))
            if s1g <= 0:
                continue
            is_tb = int(s1g >= 13)
            tb_total_count += 1
            tb_hit_count += is_tb
            s1_games_all.append(s1g)
            for pid in (int(trow["winner_id"]), int(trow["loser_id"])):
                player_tb_total[pid] = player_tb_total.get(pid, 0) + 1
                player_tb_hits[pid] = player_tb_hits.get(pid, 0) + is_tb
                player_s1_games_sum[pid] = player_s1_games_sum.get(pid, 0.0) + s1g
                player_s1_games_count[pid] = player_s1_games_count.get(pid, 0) + 1

        surface_tb_rate = tb_hit_count / max(tb_total_count, 1)
        surface_avg_s1_games = np.mean(s1_games_all) if s1_games_all else 9.5

        def _player_tb_rate(pid: int) -> float:
            total = player_tb_total.get(pid, 0)
            if total < 5:
                return surface_tb_rate
            return player_tb_hits.get(pid, 0) / total

        def _player_avg_s1_games(pid: int) -> float:
            cnt = player_s1_games_count.get(pid, 0)
            if cnt < 5:
                return surface_avg_s1_games
            return player_s1_games_sum.get(pid, 0.0) / cnt

        # ── Fit tiebreak logistic on training window ──
        tb_X_train = []
        tb_y_train = []
        for _, trow in train.iterrows():
            s1g = parse_set1_games(trow.get("score", ""))
            if s1g <= 0:
                continue
            tw_id = int(trow["winner_id"])
            tl_id = int(trow["loser_id"])
            ts_w = compute_player_stats_fast(pms_window, tw_id, surface, window=rolling_window, decay=recency_decay)
            ts_l = compute_player_stats_fast(pms_window, tl_id, surface, window=rolling_window, decay=recency_decay)
            if ts_w is None or ts_l is None:
                continue
            if ts_w.n_matches < min_matches_for_rating or ts_l.n_matches < min_matches_for_rating:
                continue
            # Look up rating-model prediction for this training match.
            # Glicko-2 (validated better-calibrated, see precompute_glicko_predictions)
            # is the primary signal; fall back to Elo if unavailable.
            t_p_elo = None
            if elo_preds is not None:
                t_p_elo = elo_preds.get((tw_id, tl_id, str(trow["match_date"])))
            t_p_gli = None
            if glicko_preds is not None:
                t_p_gli = glicko_preds.get((tw_id, tl_id, str(trow["match_date"])))
            feat = build_tiebreak_features(
                ts_w, ts_l, surface,
                p_elo=t_p_gli if t_p_gli is not None else t_p_elo,
                tb_rate_a=_player_tb_rate(tw_id),
                tb_rate_b=_player_tb_rate(tl_id),
                surface_tb_rate=surface_tb_rate,
            )
            tb_X_train.append(feat)
            tb_y_train.append(float(s1g >= 13))

        if len(tb_X_train) >= 100:
            X_tb = np.array(tb_X_train)
            y_tb = np.array(tb_y_train)
            # Higher C = less regularization, allows weights to grow for discrimination
            last_tb_weights = fit_tiebreak_logistic(X_tb, y_tb, C=10.0)

        for _, row in pred.iterrows():
            w_id = int(row["winner_id"])
            l_id = int(row["loser_id"])

            stats_w = compute_player_stats_fast(
                pms_window, w_id, surface,
                window=rolling_window, decay=recency_decay,
            )
            stats_l = compute_player_stats_fast(
                pms_window, l_id, surface,
                window=rolling_window, decay=recency_decay,
            )

            if stats_w is None or stats_l is None:
                continue
            if stats_w.n_matches < min_matches_for_rating or stats_l.n_matches < min_matches_for_rating:
                continue

            result = predict_match(stats_w, stats_l)

            if _STORE_EG:
                mc_bt = simulate_match(
                    result["p_serve_a"], result["p_serve_b"], n_simulations=_MC_EG,
                )
                exp_total_games = float(mc_bt["expected_total_games"])
            else:
                exp_total_games = float("nan")

            # Step 11 — Stability Filter
            hold_diff = abs(result["p_hold_a"] - result["p_hold_b"])
            if hold_diff < STABILITY["min_hold_diff"]:
                continue
            if result["p_match_a"] > STABILITY["max_match_prob"] or result["p_match_a"] < STABILITY["min_match_prob"]:
                continue
            # Fatigue: check recent matches in pms_window
            match_date = row["match_date"]
            fatigue_cutoff = match_date - pd.Timedelta(days=STABILITY["fatigue_window_days"])
            fatigued = False
            for pid in (w_id, l_id):
                recent_cnt = int((
                    (pms_window["player_id"] == pid)
                    & (pms_window["match_date"] >= fatigue_cutoff)
                    & (pms_window["match_date"] < match_date)
                ).sum())
                if recent_cnt > STABILITY["max_matches_last_5d"]:
                    fatigued = True
                    break
            if fatigued:
                continue

            # Blend Markov + rating-model (Glicko-2 primary, Elo fallback/diagnostic)
            p_markov = result["p_match_a"]
            date_key = str(row["match_date"])
            p_elo = elo_preds.get((w_id, l_id, date_key)) if elo_preds is not None else None
            p_gli = glicko_preds.get((w_id, l_id, date_key)) if glicko_preds is not None else None
            p_rating = p_gli if p_gli is not None else p_elo
            if p_rating is not None:
                p_blended = blend_weight * p_markov + (1.0 - blend_weight) * p_rating
            else:
                p_blended = p_markov

            set1_games = parse_set1_games(row.get("score", ""))

            # Tiebreak prediction
            p_tiebreak = np.nan
            if last_tb_weights is not None:
                tb_feat = build_tiebreak_features(
                    stats_w, stats_l, surface,
                    p_elo=p_rating,
                    tb_rate_a=_player_tb_rate(w_id),
                    tb_rate_b=_player_tb_rate(l_id),
                    surface_tb_rate=surface_tb_rate,
                )
                p_tiebreak = float(predict_tiebreak(tb_feat, last_tb_weights)[0])

            # Set 1 Over 7.5: analytical + momentum adjustment from player history
            p_s1_7_raw = result["p_set1_over_7_5"]

            # Set 1 Over 9.5: analytical + momentum adjustment from player history
            p_s1o_raw = result["p_set1_over_9_5"]
            avg_s1_a = _player_avg_s1_games(w_id)
            avg_s1_b = _player_avg_s1_games(l_id)
            avg_s1_pair = (avg_s1_a + avg_s1_b) / 2.0
            momentum_7 = 0.01 * (avg_s1_pair - 7.5)
            p_s1_7_adj = max(0.05, min(0.99, p_s1_7_raw + momentum_7))

            momentum = 0.02 * (avg_s1_pair - 9.5)
            p_s1o_adj = max(0.05, min(0.95, p_s1o_raw + momentum))

            rows.append({
                "match_date": row["match_date"],
                "surface": surface,
                "tourney_name": row.get("tourney_name", ""),
                "round": row.get("round", ""),
                "winner_name": row["winner_name"],
                "loser_name": row["loser_name"],
                "winner_id": w_id,
                "loser_id": l_id,
                "p_match_winner": p_blended,
                "p_markov": p_markov,
                "p_elo": p_rating if p_rating is not None else np.nan,  # now Glicko-2 primary (validated better-calibrated); falls back to Elo
                "p_elo_classic": p_elo if p_elo is not None else np.nan,  # raw Sackmann-Elo, diagnostic only
                "p_tiebreak": p_tiebreak,
                "p_set1_over_7_5": p_s1_7_adj,
                "p_set1_over_7_5_raw": p_s1_7_raw,
                "p_set1_over_9_5": p_s1o_adj,
                "p_set1_over_9_5_raw": p_s1o_raw,
                "avg_s1_games_pair": round(avg_s1_pair, 2),
                "p_hold_w": result["p_hold_a"],
                "p_hold_l": result["p_hold_b"],
                "p_set_w": result["p_set_a"],
                "y_match_winner": 1.0,
                "y_tiebreak": float(set1_games >= 13) if set1_games > 0 else np.nan,
                "y_set1_over_7_5": float(set1_games >= 8) if set1_games > 0 else np.nan,
                "y_set1_over_9_5": float(set1_games >= 10) if set1_games > 0 else np.nan,
                "actual_set1_games": set1_games if set1_games > 0 else np.nan,
                "expected_total_games": exp_total_games,
                "round_id": _sackmann_round_id(row.get("round")),
            })

        anchor = pred_end
        if len(rows) % 1000 < 100:
            print(f"    {surface}: {len(rows)} samples (anchor={anchor.date()})")

    return pd.DataFrame(rows), last_tb_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WTA prediction model.")
    p.add_argument("--history-csv", default="data/historical/wta_matches_combined.csv")
    p.add_argument("--out-calibration-csv", default="simulations/WTA/data/wta_calibration.csv")
    p.add_argument("--out-predictions-csv", default="simulations/WTA/backtests/wta_predictions.csv")
    p.add_argument("--out-summary-csv", default="simulations/WTA/backtests/wta_backtest_summary.csv")
    p.add_argument("--out-buckets-csv", default="simulations/WTA/backtests/wta_calibration_buckets.csv")
    p.add_argument("--lookback-days", type=int, default=_TW["lookback_days"])
    p.add_argument("--retrain-days", type=int, default=_TW["retrain_days"])
    p.add_argument("--min-train-matches", type=int, default=_TW["min_train_matches"])
    p.add_argument("--min-players", type=int, default=_TW["min_players"])
    p.add_argument("--min-samples", type=int, default=_TW["min_samples"])
    p.add_argument("--rolling-window", type=int, default=_WTA["rolling_window"])
    p.add_argument("--recency-decay", type=float, default=_WTA["recency_decay"])
    p.add_argument("--min-matches-for-rating", type=int, default=_WTA["min_matches_for_rating"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading {args.history_csv} ...")
    df = pd.read_csv(args.history_csv)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).copy()
    # Drop rows without player IDs (e.g. Flashscore-only matches without Sackmann mapping)
    before = len(df)
    df = df.dropna(subset=["winner_id", "loser_id"]).copy()
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows without player IDs")
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    print(f"  Rows: {len(df)}")

    # Pre-build player match stats table ONCE
    print("Building player match stats index ...")
    tier_w = _WTA.get("tier_weights")
    pms = build_player_match_stats(df, tier_weights=tier_w if tier_w else None)
    print(f"  Player-match rows: {len(pms)}")

    # Pre-compute Elo predictions for all matches (single chronological pass)
    print("Pre-computing surface-specific Elo predictions ...")
    elo_preds, final_elo = precompute_elo_predictions(df, _ELO_CFG)
    total_rated = sum(len(v) for v in final_elo.ratings.values())
    print(f"  Elo predictions: {len(elo_preds):,}, players rated: {total_rated}")

    # Pre-compute Glicko-2 predictions (primary rating signal — validated ~10.8%
    # lower Brier / ~29% lower log-loss than Elo, backtest_glicko_vs_elo_blended.py)
    print("Pre-computing surface-specific Glicko-2 predictions ...")
    glicko_preds, final_glicko = precompute_glicko_predictions(df)
    total_rated_gli = sum(len(v) for v in final_glicko.ratings.values())
    print(f"  Glicko-2 predictions: {len(glicko_preds):,}, players rated: {total_rated_gli}")

    surfaces = _WTA["surfaces"]
    all_preds: List[pd.DataFrame] = []
    cal_rows: List[dict] = []
    summary_rows: List[dict] = []
    bucket_dfs: List[pd.DataFrame] = []
    all_cal_pairs: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        m: ([], []) for m in MARKETS
    }

    final_tb_weights = None
    for surface in surfaces:
        print(f"\nWalk-forward surface={surface} ...")
        wf, tb_weights = walk_forward(
            df=df, pms=pms, surface=surface,
            lookback_days=args.lookback_days,
            retrain_days=args.retrain_days,
            min_train_matches=args.min_train_matches,
            min_players=args.min_players,
            rolling_window=args.rolling_window,
            recency_decay=args.recency_decay,
            min_matches_for_rating=args.min_matches_for_rating,
            elo_preds=elo_preds, glicko_preds=glicko_preds, blend_weight=BLEND_W,
        )
        if tb_weights is not None:
            final_tb_weights = tb_weights
        if wf.empty or len(wf) < args.min_samples:
            print(f"  Skipped (samples={len(wf) if not wf.empty else 0})")
            continue

        all_preds.append(wf)
        print(f"  {surface}: {len(wf)} total samples")

        market_cols = {
            "match_winner": ("p_match_winner", "y_match_winner"),
            "tiebreak": ("p_tiebreak", "y_tiebreak"),
            "set1_over_7_5": ("p_set1_over_7_5", "y_set1_over_7_5"),
            "set1_over_9_5": ("p_set1_over_9_5", "y_set1_over_9_5"),
        }

        for market, (p_col, y_col) in market_cols.items():
            # Drop rows where y is NaN (e.g. unparseable scores for set1)
            valid = wf[[p_col, y_col]].dropna()
            if valid.empty:
                continue
            p_raw = valid[p_col].to_numpy(dtype=float)
            y = valid[y_col].to_numpy(dtype=float)

            if market == "match_winner":
                p_raw_sym = np.concatenate([p_raw, 1.0 - p_raw])
                y_sym = np.concatenate([y, np.zeros_like(y)])
                p_raw, y = p_raw_sym, y_sym

            if market == "tiebreak":
                # Isotonic regression preserves discrimination for rare-event tiebreak
                xb, yv = fit_isotonic(p_raw, y)
                p_cal = apply_isotonic(p_raw, xb, yv)
                cal_rows.append({
                    "surface": surface,
                    "market": market,
                    "method": "isotonic",
                    "a": 0.0,
                    "b": 0.0,
                    "temperature": 1.0,
                    "n_train": len(p_raw),
                    "x_breaks": json.dumps(xb.tolist()),
                    "y_values": json.dumps(yv.tolist()),
                })
            else:
                a_val, b_val = fit_platt_logit(p_raw, y)
                p_platt = apply_platt_logit(p_raw, a_val, b_val)
                if market == "match_winner" and _MW_TEMP:
                    t_val = fit_temperature(p_platt, y)
                else:
                    t_val = 1.0
                p_cal = apply_temperature(p_platt, t_val)
                cal_rows.append({
                    "surface": surface,
                    "market": market,
                    "method": "platt",
                    "a": a_val,
                    "b": b_val,
                    "temperature": t_val,
                    "n_train": len(p_raw),
                })

            all_cal_pairs[market][0].append(p_raw)
            all_cal_pairs[market][1].append(y)

            summary_rows.append({
                "surface": surface,
                "market": market,
                "n": len(p_raw),
                "p_mean": float(p_raw.mean()),
                "y_mean": float(y.mean()),
                "gap_raw": float(p_raw.mean() - y.mean()),
                "log_loss_raw": _log_loss(p_raw, y),
                "log_loss_cal": _log_loss(p_cal, y),
                "brier_raw": _brier(p_raw, y),
                "brier_cal": _brier(p_cal, y),
            })

            bdf = _calibration_buckets(p_cal, y)
            if not bdf.empty:
                bdf["surface"] = surface
                bdf["market"] = market
                bucket_dfs.append(bdf)

    # Global calibration
    for market in MARKETS:
        ps_list, ys_list = all_cal_pairs[market]
        if ps_list:
            p_all = np.concatenate(ps_list)
            y_all = np.concatenate(ys_list)
            n_total = len(p_all)
        else:
            p_all, y_all = np.array([]), np.array([])
            n_total = 0

        if market == "tiebreak" and n_total > 0:
            gxb, gyv = fit_isotonic(p_all, y_all)
            cal_rows.append({
                "surface": "__GLOBAL__",
                "market": market,
                "method": "isotonic",
                "a": 0.0,
                "b": 0.0,
                "temperature": 1.0,
                "n_train": n_total,
                "x_breaks": json.dumps(gxb.tolist()),
                "y_values": json.dumps(gyv.tolist()),
            })
        else:
            ga, gb = fit_platt_logit(p_all, y_all) if n_total > 0 else (0.0, 1.0)
            if n_total > 0:
                p_gpl = apply_platt_logit(p_all, ga, gb)
                g_temp = fit_temperature(p_gpl, y_all) if (market == "match_winner" and _MW_TEMP) else 1.0
            else:
                g_temp = 1.0
            cal_rows.append({
                "surface": "__GLOBAL__",
                "market": market,
                "method": "platt",
                "a": ga,
                "b": gb,
                "temperature": g_temp,
                "n_train": n_total,
            })

    # Save
    cal_df = pd.DataFrame(cal_rows).sort_values(["surface", "market"]).reset_index(drop=True)
    Path(args.out_calibration_csv).parent.mkdir(parents=True, exist_ok=True)
    cal_df.to_csv(args.out_calibration_csv, index=False)
    print(f"\nSaved calibration: {args.out_calibration_csv} rows={len(cal_df)}")

    if all_preds:
        pred_df = pd.concat(all_preds, ignore_index=True)
        Path(args.out_predictions_csv).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(args.out_predictions_csv, index=False)
        print(f"Saved predictions: {args.out_predictions_csv} rows={len(pred_df)}")

    if summary_rows:
        summ_df = pd.DataFrame(summary_rows).sort_values(["surface", "market"]).reset_index(drop=True)
        Path(args.out_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summ_df.to_csv(args.out_summary_csv, index=False)
        print(f"Saved summary: {args.out_summary_csv} rows={len(summ_df)}")

        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        print(f"\n  {'Surface':<10} {'Market':<20} {'n':>7} {'p_mean':>7} {'y_mean':>7} {'LL_raw':>7} {'LL_cal':>7} {'Brier_cal':>9}")
        for _, r in summ_df.iterrows():
            print(f"  {r['surface']:<10} {r['market']:<20} {int(r['n']):>7,} {r['p_mean']:>7.4f} {r['y_mean']:>7.4f} {r['log_loss_raw']:>7.4f} {r['log_loss_cal']:>7.4f} {r['brier_cal']:>9.4f}")

    if bucket_dfs:
        bkt_df = pd.concat(bucket_dfs, ignore_index=True)
        Path(args.out_buckets_csv).parent.mkdir(parents=True, exist_ok=True)
        bkt_df.to_csv(args.out_buckets_csv, index=False)
        print(f"\nSaved calibration buckets: {args.out_buckets_csv} rows={len(bkt_df)}")

    # Save Elo snapshot for daily pipeline
    elo_path = "simulations/WTA/data/wta_elo_snapshot.pkl"
    final_elo.save(elo_path)
    print(f"Saved Elo snapshot: {elo_path}")

    glicko_path = "simulations/WTA/data/wta_glicko_snapshot.pkl"
    final_glicko.save(glicko_path)
    print(f"Saved Glicko-2 snapshot: {glicko_path}")

    # Save tiebreak model for daily pipeline
    if final_tb_weights is not None:
        tb_path = "simulations/WTA/data/wta_tiebreak_model.pkl"
        save_tiebreak_model(final_tb_weights, tb_path)
        print(f"Saved tiebreak model: {tb_path}")
    else:
        print("[WARN] No tiebreak model trained (insufficient data)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
