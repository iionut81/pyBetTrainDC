"""Tests for wta_scoring.parse_set1_games."""

from wta_scoring import parse_set1_games


def test_parse_straight_sets():
    assert parse_set1_games("6-2 6-4") == 8
    assert parse_set1_games("6-0 6-1") == 6


def test_parse_tiebreak_notation():
    assert parse_set1_games("7-6(3) 6-3") == 13
    assert parse_set1_games("7-6(8) 3-6 6-4") == 13


def test_parse_invalid():
    assert parse_set1_games("") == -1
    assert parse_set1_games("walkover") == -1
    assert parse_set1_games(None) == -1  # type: ignore[arg-type]
