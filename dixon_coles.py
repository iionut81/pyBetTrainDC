from __future__ import annotations

import re
import unicodedata
import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Optional

import numpy as np
from scipy.stats import poisson


@dataclass
class TeamStrength:
    attack: float
    defence: float


@dataclass
class LeagueParams:
    home_advantage: float = 0.0
    rho: float = -0.05


_TEAM_STOPWORDS = {
    "fc",
    "cf",
    "sc",
    "ac",
    "rc",
    "ud",
    "cd",
    "cp",
    "fk",
    "as",
    "sv",
    "bv",
    "the",
}


def _canonical_team_name(name: str) -> str:
    txt = unicodedata.normalize("NFKD", str(name))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9]+", " ", txt)
    tokens = [t for t in txt.split() if t and t not in _TEAM_STOPWORDS and not t.isdigit()]
    return " ".join(tokens)


def _resolve_team_name(known_teams: Dict[str, dict], incoming_name: str) -> Optional[str]:
    if incoming_name in known_teams:
        return incoming_name

    target = _canonical_team_name(incoming_name)
    if not target:
        return None

    # Exact canonical match first.
    canon_pairs = [(team, _canonical_team_name(team)) for team in known_teams.keys()]
    canon_exact = [team for team, canon in canon_pairs if canon == target]
    if len(canon_exact) == 1:
        return canon_exact[0]
    if len(canon_exact) > 1:
        return sorted(canon_exact, key=len)[0]

    # Substring containment (handles "hull" vs "hull city", "porto" vs "fc porto").
    contains = []
    for team, canon in canon_pairs:
        if canon and (target in canon or canon in target):
            contains.append((team, canon))
    if contains:
        contains.sort(key=lambda x: (abs(len(x[1]) - len(target)), len(x[1])))
        return contains[0][0]

    # Fuzzy fallback.
    best_team = None
    best_score = 0.0
    for team, canon in canon_pairs:
        if not canon:
            continue
        score = SequenceMatcher(None, target, canon).ratio()
        if score > best_score:
            best_score = score
            best_team = team
    if best_team is not None and best_score >= 0.72:
        return best_team
    return None


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - (lam * mu * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam * rho)
    if x == 1 and y == 0:
        return 1.0 + (mu * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def expected_goals(home: TeamStrength, away: TeamStrength, home_advantage: float) -> tuple[float, float]:
    # Keep the same parameter convention as training:
    # lambda_home = exp(attack_home + defence_away + home_advantage)
    # lambda_away = exp(attack_away + defence_home)
    # where more negative defence means stronger defensive team.
    lam_home = math.exp(home.attack + away.defence + home_advantage)
    lam_away = math.exp(away.attack + home.defence)
    return lam_home, lam_away


def score_matrix(lambda_home: float, lambda_away: float, rho: float, max_goals: int = 6) -> np.ndarray:
    h = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    a = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    mat = np.outer(h, a)
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= _tau(x, y, lambda_home, lambda_away, rho)
    total = mat.sum()
    if total > 0:
        mat = mat / total
    return mat


def market_probabilities(score_probs: np.ndarray) -> Dict[str, float]:
    p_home = float(np.tril(score_probs, k=-1).sum())
    p_draw = float(np.trace(score_probs))
    p_away = float(np.triu(score_probs, k=1).sum())
    return {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
        "1X": p_home + p_draw,
        "X2": p_draw + p_away,
    }


def resolve_team_strength(
    ratings: dict,
    league: str,
    home_team: str,
    away_team: str,
    default_home_advantage: float,
    default_rho: float,
) -> Optional[tuple[TeamStrength, TeamStrength, LeagueParams]]:
    # Supports two formats:
    # 1) {"team":{"attack":..,"defence":..}, ...}
    # 2) {"leagues":{"E0":{"home_advantage":..,"rho":..,"teams":{...}}}}
    if "leagues" in ratings and isinstance(ratings["leagues"], dict):
        leagues_map = ratings["leagues"]
        league_block = (
            leagues_map.get(league)
            or leagues_map.get(str(league).upper())
            or leagues_map.get(str(league).lower())
        )
        if not isinstance(league_block, dict):
            return None
        teams = league_block.get("teams", {})
        home_resolved = _resolve_team_name(teams, home_team)
        away_resolved = _resolve_team_name(teams, away_team)
        h = teams.get(home_resolved) if home_resolved else None
        a = teams.get(away_resolved) if away_resolved else None
        if not (isinstance(h, dict) and isinstance(a, dict)):
            return None
        params = LeagueParams(
            home_advantage=float(league_block.get("home_advantage", default_home_advantage)),
            rho=float(league_block.get("rho", default_rho)),
        )
        return (
            TeamStrength(float(h["attack"]), float(h["defence"])),
            TeamStrength(float(a["attack"]), float(a["defence"])),
            params,
        )

    h = ratings.get(home_team)
    a = ratings.get(away_team)
    if not (isinstance(h, dict) and isinstance(a, dict)):
        return None
    return (
        TeamStrength(float(h["attack"]), float(h["defence"])),
        TeamStrength(float(a["attack"]), float(a["defence"])),
        LeagueParams(default_home_advantage, default_rho),
    )
