"""Tests for team_registry.py — deterministic team name resolution."""
from __future__ import annotations

import pytest

from team_registry import resolve_team, resolve_team_or_warn, get_canonical_names, reload


@pytest.fixture(autouse=True)
def _fresh_registry():
    reload()
    yield


class TestResolveTeam:
    # E0
    def test_api_football_long_name(self):
        assert resolve_team("manchester city", "E0") == "man city"

    def test_canonical_passthrough(self):
        assert resolve_team("man city", "E0") == "man city"

    def test_man_united_corners_alias(self):
        assert resolve_team("man united", "E0") == "man utd"

    def test_nottingham_forest(self):
        assert resolve_team("nottingham forest", "E0") == "nott'm forest"

    # E1
    def test_hull_corners(self):
        assert resolve_team("hull", "E1") == "hull city"

    def test_sheffield_weds_corners(self):
        assert resolve_team("sheffield weds", "E1") == "sheff wed"

    def test_stoke_corners(self):
        assert resolve_team("stoke", "E1") == "stoke city"

    # D1
    def test_bayern_api_football(self):
        assert resolve_team("fc bayern münchen", "D1") == "bayern munich"

    def test_dortmund_api_football(self):
        assert resolve_team("borussia dortmund", "D1") == "dortmund"

    def test_gladbach_corners(self):
        assert resolve_team("m'gladbach", "D1") == "mönchengladbach"

    # SP1
    def test_atletico(self):
        assert resolve_team("atletico madrid", "SP1") == "atlético"

    def test_ath_bilbao_corners(self):
        assert resolve_team("ath bilbao", "SP1") == "athletic club"

    # F1
    def test_psg(self):
        assert resolve_team("paris saint germain", "F1") == "psg"

    def test_psg_corners(self):
        assert resolve_team("paris sg", "F1") == "psg"

    # I1
    def test_milan_corners(self):
        assert resolve_team("milan", "I1") == "ac milan"

    def test_verona_corners(self):
        assert resolve_team("verona", "I1") == "hellas verona"

    # N1
    def test_nijmegen_corners(self):
        assert resolve_team("nijmegen", "N1") == "nec nijmegen"

    def test_zwolle_corners(self):
        assert resolve_team("zwolle", "N1") == "pec zwolle"

    # P1
    def test_sp_lisbon_corners(self):
        assert resolve_team("sp lisbon", "P1") == "sporting"

    def test_sp_braga_corners(self):
        assert resolve_team("sp braga", "P1") == "braga"

    # Unknown
    def test_unknown_returns_none(self):
        assert resolve_team("zzz totally fake", "E0") is None

    def test_wrong_league_returns_none(self):
        # "manchester city" alias is only registered under E0
        assert resolve_team("manchester city", "SP1") is None

    # Case insensitivity
    def test_case_insensitive(self):
        assert resolve_team("Manchester City", "E0") == "man city"
        assert resolve_team("MANCHESTER CITY", "E0") == "man city"


class TestResolveTeamOrWarn:
    def test_known_team(self):
        assert resolve_team_or_warn("manchester city", "E0") == "man city"

    def test_unknown_returns_normed(self, capsys):
        result = resolve_team_or_warn("totally unknown fc", "E0")
        assert result == "totally unknown fc"
        assert "WARN" in capsys.readouterr().out


class TestGetCanonicalNames:
    def test_e0_has_names(self):
        names = get_canonical_names("E0")
        assert "man city" in names
        assert "man utd" in names

    def test_unknown_league_empty(self):
        names = get_canonical_names("ZZ99")
        assert len(names) == 0