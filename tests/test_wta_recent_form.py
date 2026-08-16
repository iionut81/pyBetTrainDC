"""Tests for wta_recent_form — normalized recent serve-form variance."""
from __future__ import annotations

import pandas as pd
import pytest

from wta_recent_form import (
    RAW_CEILING,
    RAW_FLOOR,
    build_player_index,
    recent_form_variance,
    recent_form_variance_indexed,
)


def _history(player: str, dates, pcts) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": [player] * len(dates),
            "match_date": pd.to_datetime(dates),
            "serve_pts_won_pct": pcts,
        }
    )


class TestRecentFormVariance:
    def test_none_when_insufficient_history(self):
        history = _history(
            "Alice",
            [f"2026-01-{d:02d}" for d in range(1, 6)],
            [0.60] * 5,
        )
        result = recent_form_variance(history, "Alice", pd.Timestamp("2026-08-01"), n_matches=12)
        assert result is None

    def test_perfectly_consistent_scores_zero(self):
        history = _history(
            "Alice",
            [f"2026-01-{d:02d}" for d in range(1, 13)],
            [0.60] * 12,
        )
        result = recent_form_variance(history, "Alice", pd.Timestamp("2026-08-01"), n_matches=12)
        assert result == pytest.approx(0.0)

    def test_at_or_above_ceiling_caps_at_one(self):
        # alternating far apart -> raw std well above RAW_CEILING
        pcts = [0.40, 0.90] * 6
        history = _history("Alice", [f"2026-01-{d:02d}" for d in range(1, 13)], pcts)
        result = recent_form_variance(history, "Alice", pd.Timestamp("2026-08-01"), n_matches=12)
        assert result == pytest.approx(1.0)

    def test_only_uses_matches_strictly_before_as_of(self):
        # 11 matches before the cutoff + 1 after -> still insufficient (needs 12 before)
        dates = [f"2026-01-{d:02d}" for d in range(1, 12)] + ["2026-02-01"]
        history = _history("Alice", dates, [0.60] * 12)
        result = recent_form_variance(history, "Alice", pd.Timestamp("2026-01-20"), n_matches=12)
        assert result is None

    def test_higher_raw_spread_gives_higher_or_equal_variance(self):
        stable = _history("Stable", [f"2026-01-{d:02d}" for d in range(1, 13)], [0.60] * 12)
        erratic_pcts = [0.50, 0.70] * 6  # raw std 0.10, comfortably between floor/ceiling
        erratic = _history("Erratic", [f"2026-01-{d:02d}" for d in range(1, 13)], erratic_pcts)
        as_of = pd.Timestamp("2026-08-01")
        stable_var = recent_form_variance(stable, "Stable", as_of, n_matches=12)
        erratic_var = recent_form_variance(erratic, "Erratic", as_of, n_matches=12)
        assert erratic_var > stable_var

    def test_floor_and_ceiling_are_configured_sane(self):
        assert 0.0 < RAW_FLOOR < RAW_CEILING < 1.0


class TestIndexedMatchesNaive:
    def test_indexed_matches_naive_implementation(self):
        history = pd.concat(
            [
                _history("Alice", [f"2026-01-{d:02d}" for d in range(1, 16)], [0.55, 0.65] * 7 + [0.60]),
                _history("Bea", [f"2026-02-{d:02d}" for d in range(1, 13)], [0.60] * 12),
            ],
            ignore_index=True,
        )
        index = build_player_index(history)
        as_of = pd.Timestamp("2026-08-01")

        for player in ("Alice", "Bea", "Unknown"):
            naive = recent_form_variance(history, player, as_of, n_matches=12)
            indexed = recent_form_variance_indexed(index, player, as_of, n_matches=12)
            assert naive == indexed
