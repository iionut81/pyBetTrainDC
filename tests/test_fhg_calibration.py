"""Tests for fhg_calibration.py — Platt, isotonic, temperature calibration."""
from __future__ import annotations

import numpy as np
import pytest

from fhg_calibration import (
    clamp_prob,
    apply_platt_logit,
    fit_platt_logit,
    apply_temperature,
    fit_temperature,
    fit_isotonic,
    apply_isotonic,
    apply_calibration,
    calibration_from_row,
)


class TestClampProb:
    def test_array(self):
        arr = clamp_prob(np.array([0.0, 0.5, 1.0]))
        assert arr[0] > 0
        assert arr[2] < 1

    def test_scalar(self):
        assert clamp_prob(0.0) > 0
        assert clamp_prob(1.0) < 1


class TestPlattLogit:
    def test_identity_when_a0_b1(self):
        p = np.array([0.3, 0.5, 0.7])
        result = apply_platt_logit(p, a=0.0, b=1.0)
        np.testing.assert_allclose(result, p, atol=1e-6)

    def test_output_bounded(self):
        p = np.array([0.01, 0.5, 0.99])
        result = apply_platt_logit(p, a=0.5, b=1.2)
        assert (result > 0).all()
        assert (result < 1).all()

    def test_fit_recovers_identity(self):
        # If predictions are already well-calibrated, fit should give a≈0, b≈1
        rng = np.random.default_rng(42)
        p = rng.uniform(0.2, 0.8, size=500)
        y = (rng.random(500) < p).astype(float)
        a, b = fit_platt_logit(p, y)
        assert abs(a) < 0.5
        assert abs(b - 1.0) < 0.5


class TestTemperature:
    def test_temperature_1_is_identity(self):
        p = np.array([0.3, 0.5, 0.8])
        result = apply_temperature(p, 1.0)
        np.testing.assert_allclose(result, p, atol=1e-6)

    def test_high_temperature_compresses(self):
        p = np.array([0.1, 0.9])
        result = apply_temperature(p, 2.0)
        # High T compresses toward 0.5
        assert result[0] > 0.1
        assert result[1] < 0.9

    def test_fit_returns_positive(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(0.3, 0.7, 200)
        y = (rng.random(200) < p).astype(float)
        T = fit_temperature(p, y)
        assert T > 0


class TestIsotonic:
    def test_monotonicity(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        x_breaks, y_values = fit_isotonic(p, y)
        # y_values should be non-decreasing
        for i in range(len(y_values) - 1):
            assert y_values[i] <= y_values[i + 1] + 1e-9

    def test_apply_bounded(self):
        x_breaks = np.array([0.3, 0.6, 0.9])
        y_values = np.array([0.2, 0.5, 0.85])
        result = apply_isotonic(np.array([0.1, 0.5, 0.95]), x_breaks, y_values)
        assert (result > 0).all()
        assert (result < 1).all()

    def test_empty_breaks(self):
        p = np.array([0.5])
        result = apply_isotonic(p, np.array([]), np.array([]))
        np.testing.assert_allclose(result, p, atol=1e-6)


class TestApplyCalibration:
    def test_platt_method(self):
        calib = {"method": "platt", "a": 0.0, "b": 1.0, "temperature": 1.0}
        result = apply_calibration(0.7, calib)
        assert float(result) == pytest.approx(0.7, abs=1e-5)

    def test_isotonic_method(self):
        calib = {
            "method": "isotonic",
            "x_breaks": np.array([0.3, 0.7]),
            "y_values": np.array([0.25, 0.75]),
            "temperature": 1.0,
        }
        result = apply_calibration(np.array([0.5]), calib)
        assert 0 < float(result[0]) < 1


class TestCalibrationFromRow:
    def test_platt_row(self):
        row = {"method": "platt", "a": 0.1, "b": 0.9, "temperature": 1.2}
        out = calibration_from_row(row)
        assert out["method"] == "platt"
        assert out["a"] == pytest.approx(0.1)
        assert out["temperature"] == pytest.approx(1.2)

    def test_defaults(self):
        row = {"method": "platt"}
        out = calibration_from_row(row)
        assert out["a"] == pytest.approx(0.0)
        assert out["b"] == pytest.approx(1.0)
        assert out["temperature"] == pytest.approx(1.0)