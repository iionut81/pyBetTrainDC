from __future__ import annotations

from typing import Dict

import numpy as np


def run_monte_carlo(
    lambda_home: float,
    lambda_away: float,
    iterations: int = 50000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    hg = rng.poisson(lambda_home, size=iterations)
    ag = rng.poisson(lambda_away, size=iterations)

    home_win = hg > ag
    draw = hg == ag
    away_win = hg < ag

    p_1x = float(np.mean(home_win | draw))
    p_x2 = float(np.mean(away_win | draw))
    upset_1x = float(np.mean(away_win))
    upset_x2 = float(np.mean(home_win))

    # Binomial-like uncertainty proxy for each DC market.
    var_1x = p_1x * (1.0 - p_1x)
    var_x2 = p_x2 * (1.0 - p_x2)

    return {
        "mc_1X": p_1x,
        "mc_X2": p_x2,
        "upset_1X": upset_1x,
        "upset_X2": upset_x2,
        "variance_1X": float(var_1x),
        "variance_X2": float(var_x2),
    }

