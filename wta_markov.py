from __future__ import annotations

"""
wta_markov.py
Core tennis probability engine using Markov chains and Monte Carlo simulation.

Converts point-level win probabilities into game, set, and match probabilities
for WTA (best-of-3) matches.
"""

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


# ── Point → Game (exact Markov chain) ─────────────────────────────────────────

def game_win_prob(p: float) -> float:
    """Probability of holding serve given P(win point on serve) = p.

    Uses the exact closed-form solution for a tennis game:
      P(hold) = p^4 * (1 + 4q + 10q^2) + 20*p^3*q^3 * p^2 / (1 - 2pq)
    where q = 1 - p.
    """
    q = 1.0 - p
    pq2 = 2.0 * p * q
    if abs(1.0 - pq2) < 1e-12:
        return 0.5
    return (
        p ** 4 * (1.0 + 4.0 * q + 10.0 * q ** 2)
        + 20.0 * (p ** 3) * (q ** 3) * (p ** 2) / (1.0 - pq2)
    )


# ── Tiebreak (exact Markov chain) ────────────────────────────────────────────

def tiebreak_win_prob(p_a: float, p_b: float) -> float:
    """P(player A wins tiebreak) given serve point probabilities.

    A serves first. Serve pattern: A, B, B, A, A, B, B, ...
    At 6-6+ the serve pattern continues and we solve the deuce cycle analytically.
    """
    memo: Dict[tuple, float] = {}

    def _server_is_a(point_num: int) -> bool:
        if point_num == 0:
            return True
        return ((point_num - 1) // 2) % 2 == 0

    def _dp(a: int, b: int, point_num: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0

        # At 6-6+, solve the 2-point cycle analytically to avoid infinite recursion
        if a >= 6 and b >= 6 and a == b:
            s1_is_a = _server_is_a(point_num)
            s2_is_a = _server_is_a(point_num + 1)
            p1 = p_a if s1_is_a else (1.0 - p_b)
            p2 = p_a if s2_is_a else (1.0 - p_b)
            q1, q2 = 1.0 - p1, 1.0 - p2
            # P(A wins from deuce) = p1*p2 / (1 - p1*q2 - q1*p2)
            denom = p1 * p2 + q1 * q2
            if denom < 1e-12:
                return 0.5
            return p1 * p2 / denom

        key = (a, b, point_num)
        if key in memo:
            return memo[key]

        s_is_a = _server_is_a(point_num)
        p_win = p_a if s_is_a else (1.0 - p_b)
        result = p_win * _dp(a + 1, b, point_num + 1) + (1.0 - p_win) * _dp(a, b + 1, point_num + 1)
        memo[key] = result
        return result

    return _dp(0, 0, 0)


# ── Game-by-game → Set (exact Markov chain) ──────────────────────────────────

def set_win_prob(p_hold_a: float, p_hold_b: float, p_tb_a: float = None) -> float:
    """P(player A wins a set) given hold probabilities.

    Uses DP over states (games_a, games_b, who_serves_next).
    A serves first in the set. At 6-6 we go to tiebreak.
    """
    if p_tb_a is None:
        # Estimate tiebreak prob from serve point probs isn't available here,
        # so approximate: player who holds more has slight tiebreak edge
        p_tb_a = 0.5 + (p_hold_a - p_hold_b) * 0.8

    p_break_a = 1.0 - p_hold_b   # A breaks B's serve
    p_break_b = 1.0 - p_hold_a   # B breaks A's serve

    memo: Dict[tuple, float] = {}

    def _dp(ga: int, gb: int, a_serves: bool) -> float:
        # A won set
        if ga >= 6 and ga - gb >= 2:
            return 1.0
        # B won set
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        # Tiebreak at 6-6
        if ga == 6 and gb == 6:
            return p_tb_a

        key = (ga, gb, a_serves)
        if key in memo:
            return memo[key]

        if a_serves:
            # A's service game
            p_a_wins_game = p_hold_a
            result = (
                p_a_wins_game * _dp(ga + 1, gb, not a_serves)
                + (1.0 - p_a_wins_game) * _dp(ga, gb + 1, not a_serves)
            )
        else:
            # B's service game
            p_a_wins_game = p_break_a  # = 1 - p_hold_b
            result = (
                p_a_wins_game * _dp(ga + 1, gb, not a_serves)
                + (1.0 - p_a_wins_game) * _dp(ga, gb + 1, not a_serves)
            )

        memo[key] = result
        return result

    return _dp(0, 0, True)


# ── Set → Match (best of 3) ──────────────────────────────────────────────────

def match_win_prob_bo3(p_set: float) -> float:
    """P(win best-of-3 match) = p^2 * (3 - 2p)."""
    return p_set ** 2 * (3.0 - 2.0 * p_set)


def match_win_prob(p_set: float, best_of: int = 3) -> float:
    """P(win match) for best_of 3 or 5."""
    if best_of == 5:
        p = p_set
        return p ** 3 * (1 + 3 * (1 - p) + 6 * (1 - p) ** 2)
    return match_win_prob_bo3(p_set)


# ── Composite: stats → match probability ─────────────────────────────────────

@dataclass
class PlayerServeStats:
    """Rolling serve/return statistics for one player on a surface."""
    first_serve_in_pct: float      # fraction [0, 1]
    first_serve_won_pct: float
    second_serve_won_pct: float
    return_pts_won_pct: float
    bp_saved_pct: float
    bp_converted_pct: float
    ace_rate: float
    n_matches: int


def serve_point_win_prob(stats: PlayerServeStats) -> float:
    """Exact serve point win probability from first/second serve stats.

    P(win point on serve) = P(1st in) * P(win | 1st in) + P(1st out) * P(win | 2nd serve)
    """
    p = (
        stats.first_serve_in_pct * stats.first_serve_won_pct
        + (1.0 - stats.first_serve_in_pct) * stats.second_serve_won_pct
    )
    return max(0.45, min(0.80, p))


def serve_strength(stats: PlayerServeStats) -> float:
    """Composite serve strength rating (0-1 scale)."""
    return (
        0.4 * stats.first_serve_won_pct
        + 0.3 * stats.second_serve_won_pct
        + 0.2 * stats.first_serve_in_pct
        + 0.1 * stats.ace_rate
    )


def return_strength(stats: PlayerServeStats) -> float:
    """Composite return strength rating (0-1 scale)."""
    return (
        0.6 * stats.return_pts_won_pct
        + 0.4 * stats.bp_converted_pct
    )


def pressure_rating(stats: PlayerServeStats) -> float:
    """Composite pressure rating (0-1 scale)."""
    return (
        0.5 * stats.bp_saved_pct
        + 0.5 * min(1.0, stats.bp_converted_pct)
    )


def predict_match(
    stats_a: PlayerServeStats,
    stats_b: PlayerServeStats,
    best_of: int = 3,
    avg_return: float = 0.43,
    opp_adjustment: float = 0.50,
) -> Dict[str, float]:
    """Full analytical prediction: stats → point → game → set → match.

    Opponent-adjusted serve point probabilities:
      p_serve_a_adj = p_serve_a - opp_adjustment * (return_pts_won_b - avg_return)

    A strong returner (above avg) reduces the server's point win probability;
    a weak returner (below avg) increases it. The adjustment coefficient (0.50)
    is 2.5x the original, giving meaningful separation without amplifying noise.

    Returns dict with all intermediate probabilities.
    """
    # Base serve point win probabilities
    p_serve_a = serve_point_win_prob(stats_a)
    p_serve_b = serve_point_win_prob(stats_b)

    # Opponent-adjusted: stronger returner reduces server's point win prob
    p_serve_a_adj = p_serve_a - opp_adjustment * (stats_b.return_pts_won_pct - avg_return)
    p_serve_b_adj = p_serve_b - opp_adjustment * (stats_a.return_pts_won_pct - avg_return)
    p_serve_a_adj = max(0.40, min(0.82, p_serve_a_adj))
    p_serve_b_adj = max(0.40, min(0.82, p_serve_b_adj))

    # Game hold probabilities
    p_hold_a = game_win_prob(p_serve_a_adj)
    p_hold_b = game_win_prob(p_serve_b_adj)

    # Tiebreak probability
    p_tb_a = tiebreak_win_prob(p_serve_a_adj, p_serve_b_adj)

    # Set probability
    p_set_a = set_win_prob(p_hold_a, p_hold_b, p_tb_a)

    # Match probability
    p_match_a = match_win_prob(p_set_a, best_of)

    # Set 1 under 12.5 (= no tiebreak in set 1)
    p_s1_under = p_set_under_12_5(p_hold_a, p_hold_b)

    return {
        "p_serve_a": p_serve_a_adj,
        "p_serve_b": p_serve_b_adj,
        "p_hold_a": p_hold_a,
        "p_hold_b": p_hold_b,
        "p_break_a": 1.0 - p_hold_b,
        "p_break_b": 1.0 - p_hold_a,
        "p_tiebreak_a": p_tb_a,
        "p_set_a": p_set_a,
        "p_match_a": p_match_a,
        "p_set1_under_12_5": p_s1_under,
    }


# ── Monte Carlo simulation ──────────────────────────────────────────────────

def _sim_game(p_serve: float, rng: np.random.Generator) -> bool:
    """Simulate a single service game. Returns True if server holds."""
    pts_s, pts_r = 0, 0
    while True:
        if rng.random() < p_serve:
            pts_s += 1
        else:
            pts_r += 1
        if pts_s >= 4 and pts_s - pts_r >= 2:
            return True
        if pts_r >= 4 and pts_r - pts_s >= 2:
            return False


def _sim_tiebreak(p_a: float, p_b: float, rng: np.random.Generator) -> bool:
    """Simulate a tiebreak. Returns True if player A wins."""
    pts_a, pts_b = 0, 0
    point_num = 0
    while True:
        if point_num == 0:
            server_is_a = True
        else:
            server_is_a = ((point_num - 1) // 2) % 2 == 0

        p_win = p_a if server_is_a else (1.0 - p_b)
        if rng.random() < p_win:
            pts_a += 1
        else:
            pts_b += 1
        point_num += 1

        if pts_a >= 7 and pts_a - pts_b >= 2:
            return True
        if pts_b >= 7 and pts_b - pts_a >= 2:
            return False


def _sim_set(p_serve_a: float, p_serve_b: float, rng: np.random.Generator) -> tuple:
    """Simulate a set. Returns (a_wins: bool, games_a: int, games_b: int)."""
    ga, gb = 0, 0
    a_serves = True
    while True:
        if a_serves:
            if _sim_game(p_serve_a, rng):
                ga += 1
            else:
                gb += 1
        else:
            if _sim_game(p_serve_b, rng):
                gb += 1
            else:
                ga += 1
        a_serves = not a_serves

        # Check set completion
        if ga >= 6 and ga - gb >= 2:
            return True, ga, gb
        if gb >= 6 and gb - ga >= 2:
            return False, ga, gb
        if ga == 6 and gb == 6:
            tb = _sim_tiebreak(p_serve_a, p_serve_b, rng)
            if tb:
                return True, 7, 6
            else:
                return False, 6, 7


def simulate_match(
    p_serve_a: float,
    p_serve_b: float,
    n_simulations: int = 10000,
    best_of: int = 3,
    seed: int = 42,
) -> Dict[str, float]:
    """Monte Carlo match simulation.

    Returns dict with:
      - p_match_a: P(A wins match)
      - expected_total_games: mean total games
      - std_total_games: std of total games
      - p_straight_sets_a: P(A wins 2-0)
      - p_straight_sets_b: P(B wins 0-2)
      - p_three_sets: P(match goes to 3 sets)
      - games_distribution: dict of total_games -> frequency
    """
    rng = np.random.default_rng(seed)
    sets_needed = (best_of // 2) + 1

    wins_a = 0
    total_games_list = []
    set1_games_list = []
    straight_a = 0
    straight_b = 0
    three_sets = 0
    games_dist: Dict[int, int] = {}
    set1_games_dist: Dict[int, int] = {}

    for _ in range(n_simulations):
        sets_a, sets_b = 0, 0
        match_games = 0
        set_num = 0

        while sets_a < sets_needed and sets_b < sets_needed:
            a_wins_set, ga, gb = _sim_set(p_serve_a, p_serve_b, rng)
            set_games = ga + gb
            match_games += set_games
            if set_num == 0:
                set1_games_list.append(set_games)
                set1_games_dist[set_games] = set1_games_dist.get(set_games, 0) + 1
            set_num += 1
            if a_wins_set:
                sets_a += 1
            else:
                sets_b += 1

        total_games_list.append(match_games)
        games_dist[match_games] = games_dist.get(match_games, 0) + 1

        if sets_a == sets_needed:
            wins_a += 1
            if sets_b == 0:
                straight_a += 1
        else:
            if sets_a == 0:
                straight_b += 1

        if sets_a + sets_b == best_of:
            three_sets += 1

    n = n_simulations
    tg = np.array(total_games_list, dtype=float)
    s1g = np.array(set1_games_list, dtype=float)
    return {
        "p_match_a": wins_a / n,
        "expected_total_games": float(tg.mean()),
        "std_total_games": float(tg.std()),
        "p_straight_sets_a": straight_a / n,
        "p_straight_sets_b": straight_b / n,
        "p_three_sets": three_sets / n,
        "games_distribution": {k: v / n for k, v in sorted(games_dist.items())},
        "set1_games_distribution": {k: v / n for k, v in sorted(set1_games_dist.items())},
        "expected_set1_games": float(s1g.mean()),
    }


def p_set_tiebreak(p_hold_a: float, p_hold_b: float) -> float:
    """P(a set reaches tiebreak at 6-6) given hold probabilities.

    Uses DP over game states. A serves first.
    """
    memo: Dict[tuple, float] = {}

    def _dp(ga: int, gb: int, a_serves: bool) -> float:
        if ga >= 6 and ga - gb >= 2:
            return 0.0  # set ended without tiebreak
        if gb >= 6 and gb - ga >= 2:
            return 0.0  # set ended without tiebreak
        if ga == 6 and gb == 6:
            return 1.0  # tiebreak reached

        key = (ga, gb, a_serves)
        if key in memo:
            return memo[key]

        if a_serves:
            p_a_wins_game = p_hold_a
        else:
            p_a_wins_game = 1.0 - p_hold_b

        result = (
            p_a_wins_game * _dp(ga + 1, gb, not a_serves)
            + (1.0 - p_a_wins_game) * _dp(ga, gb + 1, not a_serves)
        )
        memo[key] = result
        return result

    return _dp(0, 0, True)


def p_set_under_12_5(p_hold_a: float, p_hold_b: float) -> float:
    """P(under 12.5 games in a set) = P(no tiebreak) = 1 - P(6-6)."""
    return 1.0 - p_set_tiebreak(p_hold_a, p_hold_b)


def prob_over_games(games_dist: Dict[int, float], line: float) -> float:
    """P(total games > line) from a games distribution."""
    return sum(freq for g, freq in games_dist.items() if g > line)


def prob_under_games(games_dist: Dict[int, float], line: float) -> float:
    """P(total games < line) from a games distribution."""
    return sum(freq for g, freq in games_dist.items() if g < line)
