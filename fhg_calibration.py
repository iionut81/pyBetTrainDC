from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar


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


def fit_temperature(p_platt: np.ndarray, y: np.ndarray) -> float:
    """Fit temperature T by minimising NLL of sigmoid(logit(p_platt) / T) against y.

    T > 1 compresses over-confident high predictions back toward the centre.
    Returns 1.0 (no-op) if optimisation fails.
    """
    p = np.asarray(clamp_prob(p_platt), dtype=float)
    y = np.asarray(y, dtype=float)
    x = _logit(p)

    def nll(T: float) -> float:
        p_t = clamp_prob(_sigmoid(x / T))
        return float(-np.mean(y * np.log(p_t) + (1.0 - y) * np.log(1.0 - p_t)))

    res = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    if not res.success:
        return 1.0
    return float(np.clip(res.x, 0.5, 5.0))


def apply_temperature(p: np.ndarray | float, temperature: float) -> np.ndarray:
    """Apply temperature scaling: sigmoid(logit(p) / T)."""
    arr = np.asarray(clamp_prob(p), dtype=float)
    if abs(temperature - 1.0) < 1e-9:
        return arr
    return clamp_prob(_sigmoid(_logit(arr) / temperature))


def fit_isotonic(p_raw: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(clamp_prob(p_raw), dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p)
    p = p[order]
    y = y[order]

    blocks = []
    for i in range(len(p)):
        blocks.append({"sum_w": 1.0, "sum_y": float(y[i]), "left": float(p[i]), "right": float(p[i])})
        while len(blocks) >= 2:
            m1 = blocks[-2]["sum_y"] / blocks[-2]["sum_w"]
            m2 = blocks[-1]["sum_y"] / blocks[-1]["sum_w"]
            if m1 <= m2:
                break
            b2 = blocks.pop()
            b1 = blocks.pop()
            blocks.append(
                {
                    "sum_w": b1["sum_w"] + b2["sum_w"],
                    "sum_y": b1["sum_y"] + b2["sum_y"],
                    "left": b1["left"],
                    "right": b2["right"],
                }
            )

    x_breaks = np.array([b["right"] for b in blocks], dtype=float)
    y_values = np.array([b["sum_y"] / b["sum_w"] for b in blocks], dtype=float)
    return x_breaks, np.asarray(clamp_prob(y_values), dtype=float)


def apply_isotonic(p_raw: np.ndarray | float, x_breaks: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(clamp_prob(p_raw), dtype=float)
    if len(x_breaks) == 0 or len(y_values) == 0:
        return arr
    idx = np.searchsorted(x_breaks, arr, side="left")
    idx = np.clip(idx, 0, len(y_values) - 1)
    return np.asarray(clamp_prob(y_values[idx]), dtype=float)


def calibration_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    method = str(row.get("method", "platt")).strip().lower()
    out: Dict[str, Any] = {"method": method}
    # temperature=1.0 is a no-op; old CSVs without this column work unchanged
    raw_t = row.get("temperature", 1.0)
    out["temperature"] = float(raw_t) if raw_t not in (None, "", "nan") else 1.0
    if method == "isotonic":
        try:
            xb = json.loads(str(row.get("x_breaks", "[]")))
            yv = json.loads(str(row.get("y_values", "[]")))
            out["x_breaks"] = np.asarray(xb, dtype=float)
            out["y_values"] = np.asarray(yv, dtype=float)
        except Exception:
            out["x_breaks"] = np.array([], dtype=float)
            out["y_values"] = np.array([], dtype=float)
        return out

    out["a"] = float(row.get("a", 0.0))
    out["b"] = float(row.get("b", 1.0))
    return out


def apply_calibration(p_raw: np.ndarray | float, calib: Dict[str, Any]) -> np.ndarray:
    method = str(calib.get("method", "platt")).lower()
    temperature = float(calib.get("temperature", 1.0))
    if method == "isotonic":
        xb = np.asarray(calib.get("x_breaks", np.array([], dtype=float)), dtype=float)
        yv = np.asarray(calib.get("y_values", np.array([], dtype=float)), dtype=float)
        p_cal = apply_isotonic(p_raw, xb, yv)
    else:
        p_cal = apply_platt_logit(p_raw, a=float(calib.get("a", 0.0)), b=float(calib.get("b", 1.0)))
    return apply_temperature(p_cal, temperature)
