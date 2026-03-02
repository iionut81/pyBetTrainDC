from __future__ import annotations

import math


def clamp_probability(p: float) -> float:
    return max(1e-9, min(1.0 - 1e-9, float(p)))


def k_from_fh_share(fh_share: float, default_k: float = 1.2) -> float:
    # Derived from cumulative hazard proportion by half time:
    # share ~= (45/90)^k = 0.5^k  =>  k = log(share)/log(0.5)
    if fh_share <= 0.0 or fh_share >= 1.0:
        return default_k
    k = math.log(fh_share) / math.log(0.5)
    return max(0.6, min(2.0, float(k)))


def p_goal_before_45(expected_total_goals: float, k: float) -> float:
    # Set integrated hazard at t=90 equal to expected goals from DC:
    # H(t) = xg_total * (t/90)^k
    # P(goal before 45) = 1 - exp(-H(45))
    h45 = max(0.0, expected_total_goals) * (0.5 ** max(0.1, k))
    return clamp_probability(1.0 - math.exp(-h45))

