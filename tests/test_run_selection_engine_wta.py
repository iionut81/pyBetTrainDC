"""Tests for run_selection_engine_wta.load_today_matches — date filtering,
T23:59 placeholder-time exclusion, and duplicate rows."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from run_selection_engine_wta import load_today_matches

COLUMNS = [
    "match_date", "player_a", "player_b", "surface", "p_hold_a", "p_hold_b",
    "p_cal_adj", "tb_p_cal", "tournament", "level", "round",
]


def _row(match_date, player_a="A", player_b="B", **overrides):
    row = {
        "match_date": match_date,
        "player_a": player_a,
        "player_b": player_b,
        "surface": "Hard",
        "p_hold_a": 0.70,
        "p_hold_b": 0.68,
        "p_cal_adj": 0.90,
        "tb_p_cal": 0.10,
        "tournament": "Test Open",
        "level": "WTA 500",
        "round": 1,
    }
    row.update(overrides)
    return row


def _write_csv(tmp_path, rows):
    path = tmp_path / "matches.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


class TestDateFiltering:
    def test_only_todays_matches_are_kept(self, tmp_path):
        rows = [
            _row("2026-08-17T11:00-04:00", player_a="Today"),
            _row("2026-08-16T11:00-04:00", player_a="Yesterday"),
            _row("2026-08-18T11:00-04:00", player_a="Tomorrow"),
        ]
        path = _write_csv(tmp_path, rows)
        matches = load_today_matches(path, date(2026, 8, 17))
        assert [m.competitors[0] for m in matches] == ["Today"]

    def test_late_local_time_does_not_roll_into_next_utc_day(self, tmp_path):
        # A naive UTC conversion of 23:59-04:00 rolls into the next calendar
        # day and would wrongly drop this match -- must not happen.
        rows = [_row("2026-08-17T20:30-04:00", player_a="LateLocal")]
        path = _write_csv(tmp_path, rows)
        matches = load_today_matches(path, date(2026, 8, 17))
        assert [m.competitors[0] for m in matches] == ["LateLocal"]


class TestPlaceholderTime:
    def test_t2359_placeholder_is_excluded(self, tmp_path):
        rows = [
            _row("2026-08-17T23:59-04:00", player_a="Unconfirmed"),
            _row("2026-08-17T14:00-04:00", player_a="Confirmed"),
        ]
        path = _write_csv(tmp_path, rows)
        matches = load_today_matches(path, date(2026, 8, 17))
        assert [m.competitors[0] for m in matches] == ["Confirmed"]

    def test_all_placeholder_gives_no_matches(self, tmp_path):
        rows = [_row("2026-08-17T23:59-04:00")]
        path = _write_csv(tmp_path, rows)
        matches = load_today_matches(path, date(2026, 8, 17))
        assert matches == []


class TestDuplicateRows:
    def test_duplicate_match_rows_both_pass_through(self, tmp_path):
        # No dedup is performed -- each CSV row becomes its own MatchInput
        # with a unique match_id derived from row position, so an accidental
        # duplicate row in the source CSV produces two independent matches
        # rather than crashing or silently merging.
        rows = [
            _row("2026-08-17T11:00-04:00", player_a="Dup", player_b="Licate"),
            _row("2026-08-17T11:00-04:00", player_a="Dup", player_b="Licate"),
        ]
        path = _write_csv(tmp_path, rows)
        matches = load_today_matches(path, date(2026, 8, 17))
        assert len(matches) == 2
        assert matches[0].match_id != matches[1].match_id
        assert all(m.competitors == ("Dup", "Licate") for m in matches)


class TestMissingData:
    def test_missing_optional_field_does_not_crash(self, tmp_path):
        row = _row("2026-08-17T11:00-04:00")
        row["p_cal_adj"] = None
        path = _write_csv(tmp_path, [row])
        matches = load_today_matches(path, date(2026, 8, 17))
        assert len(matches) == 1
        assert "p_cal_adj" not in matches[0].stats
