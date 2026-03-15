from __future__ import annotations

"""
wta_tiebreak.py
Logistic regression model for predicting tiebreak probability in WTA set 1.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from wta_markov import PlayerServeStats, serve_point_win_prob, game_win_prob

FEATURE_NAMES = [
    "hold_avg", "hold_diff", "ace_sum", "bp_saved_avg",
    "min_hold", "return_weakness", "is_grass", "is_clay",
    "elo_closeness", "tb_rate_avg", "surface_tb_rate",
]
N_FEATURES = len(FEATURE_NAMES)


def build_tiebreak_features(
    stats_a: PlayerServeStats,
    stats_b: PlayerServeStats,
    surface: str,
    avg_return: float = 0.43,
    opp_adj: float = 0.50,
    p_elo: Optional[float] = None,
    tb_rate_a: float = 0.12,
    tb_rate_b: float = 0.12,
    surface_tb_rate: float = 0.12,
) -> np.ndarray:
    """Build feature vector for tiebreak prediction.

    Uses the same opponent-adjusted hold probabilities as predict_match.

    New features:
      - elo_closeness: 1 - 2*|p_elo - 0.5|  (1.0 = dead even, 0.0 = one-sided)
      - tb_rate_avg: average historical tiebreak rate of the two players
      - surface_tb_rate: base tiebreak rate for this surface in the training window
    """
    p_serve_a = serve_point_win_prob(stats_a)
    p_serve_b = serve_point_win_prob(stats_b)
    p_serve_a_adj = p_serve_a - opp_adj * (stats_b.return_pts_won_pct - avg_return)
    p_serve_b_adj = p_serve_b - opp_adj * (stats_a.return_pts_won_pct - avg_return)
    p_serve_a_adj = max(0.40, min(0.82, p_serve_a_adj))
    p_serve_b_adj = max(0.40, min(0.82, p_serve_b_adj))
    hold_a = game_win_prob(p_serve_a_adj)
    hold_b = game_win_prob(p_serve_b_adj)

    # Elo closeness: 1.0 when ratings identical, 0.0 when maximally lopsided
    if p_elo is not None:
        elo_closeness = 1.0 - 2.0 * abs(p_elo - 0.5)
    else:
        elo_closeness = 0.5  # neutral fallback

    return np.array([
        (hold_a + hold_b) / 2.0,                                                  # hold_avg
        abs(hold_a - hold_b),                                                      # hold_diff
        stats_a.ace_rate + stats_b.ace_rate,                                       # ace_sum
        (stats_a.bp_saved_pct + stats_b.bp_saved_pct) / 2.0,                      # bp_saved_avg
        min(hold_a, hold_b),                                                       # min_hold
        1.0 - (stats_a.return_pts_won_pct + stats_b.return_pts_won_pct) / 2.0,    # return_weakness
        float(surface == "Grass"),                                                 # is_grass
        float(surface == "Clay"),                                                  # is_clay
        elo_closeness,                                                             # elo_closeness
        (tb_rate_a + tb_rate_b) / 2.0,                                            # tb_rate_avg
        surface_tb_rate,                                                           # surface_tb_rate
    ])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_tiebreak_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Fit L2-regularized logistic regression. Returns weight vector (n_features + 1,)."""
    n, d = X.shape
    lam = 1.0 / max(C, 1e-9)

    def nll(w: np.ndarray) -> float:
        weights, bias = w[:d], w[d]
        z = X @ weights + bias
        p = np.clip(_sigmoid(z), 1e-9, 1.0 - 1e-9)
        loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = 0.5 * lam * np.sum(weights ** 2) / n
        return float(loss + reg)

    w0 = np.zeros(d + 1)
    res = minimize(nll, w0, method="L-BFGS-B", options={"maxiter": 500})
    return res.x


def predict_tiebreak(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict tiebreak probability from features and weight vector."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    d = X.shape[1]
    return _sigmoid(X @ w[:d] + w[d])


def save_tiebreak_model(w: np.ndarray, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(w, f)


def load_tiebreak_model(path: str) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)
