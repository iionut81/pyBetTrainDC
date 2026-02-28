from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.optimize import minimize


def clamp_prob(p: np.ndarray | float) -> np.ndarray | float:
    if isinstance(p, np.ndarray):
        return np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(max(1e-9, min(1.0 - 1e-9, p)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(clamp_prob(p), dtype=float)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_logit(p_raw: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    p_raw = np.asarray(clamp_prob(p_raw), dtype=float)
    y = np.asarray(y, dtype=float)
    x = _logit(p_raw)

    def nll(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        p = clamp_prob(_sigmoid(a + b * x))
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize(nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B", bounds=[(-10, 10), (-10, 10)])
    if not res.success:
        return 0.0, 1.0
    a, b = float(res.x[0]), float(res.x[1])
    return a, b


def apply_platt_logit(p_raw: np.ndarray | float, a: float, b: float) -> np.ndarray:
    arr = np.asarray(clamp_prob(p_raw), dtype=float)
    x = _logit(arr)
    return clamp_prob(_sigmoid(a + b * x))

