from __future__ import annotations

"""
classification.py
Stages 4-5: turn a raw rank_value (p_cal_adj) into an approximate historical
percentile (for display) plus a descriptive label and a BET_ELIGIBLE flag,
using a market's historical_percentiles breakpoints — computed exclusively
from the historical DISTRIBUTION OF p_cal_adj itself (never hit rate, score,
or any derived value) for the market's POST_HARD_FILTER / PRE_VETO
population (see markets/tennis_set1_over_7_5.py for why: veto is a selection
filter, not something that should also define the statistical universe we
measure p_cal_adj against).

BET_ELIGIBLE requires historical_percentile >= BET_THRESHOLD_PERCENTILE
(default 80 — i.e. the historical top quintile). HIGH/MEDIUM/LOW/VERY_LOW are
informative labels, not "almost eligible" — being the best match available
today does not by itself make a match BET_ELIGIBLE.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

LABEL_TOP = "TOP_HISTORICAL_QUINTILE"
LABEL_HIGH = "HIGH"
LABEL_MEDIUM = "MEDIUM"
LABEL_LOW = "LOW"
LABEL_VERY_LOW = "VERY_LOW"

BET_THRESHOLD_PERCENTILE_DEFAULT = 80.0


def compute_percentiles(values: Iterable[float]) -> Dict[str, float]:
    """p0/p20/p40/p60/p80/p90/p95/p100 of a real historical rank_value sample.

    Recompute this (don't hand-edit the breakpoints) whenever the underlying
    historical dataset changes materially — e.g. after a retrain adds a
    meaningful amount of new backtest data.
    """
    series = np.asarray(list(values), dtype=float)
    quantiles = np.quantile(series, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    keys = ("p0", "p20", "p40", "p60", "p80", "p90", "p95", "p100")
    return dict(zip(keys, (float(q) for q in quantiles)))


def _historical_percentile(rank_value: float, percentiles: Dict[str, float]) -> float:
    """np.interp does not extrapolate: values below p0 clamp to 0, values
    above p100 clamp to 100 — never extrapolated past the observed range."""
    p0 = percentiles.get("p0", rank_value)
    p20 = percentiles.get("p20", p0)
    p40 = percentiles.get("p40", p20)
    p60 = percentiles.get("p60", p40)
    p80 = percentiles.get("p80", p60)
    p100 = percentiles.get("p100", p80)
    return float(np.interp(rank_value, [p0, p20, p40, p60, p80, p100], [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]))


def classify(
    rank_value: Optional[float],
    percentiles: Dict[str, float],
    bet_threshold_percentile: float = BET_THRESHOLD_PERCENTILE_DEFAULT,
) -> Tuple[Optional[float], str, bool]:
    """Returns (historical_percentile 0-100 or None, label, bet_eligible).

    Both the label and bet_eligible are derived from historical_percentile —
    never from hit rate, a 0-100 score, or any other derived value.
    """
    if rank_value is None or not percentiles:
        return None, "", False

    historical_percentile = _historical_percentile(rank_value, percentiles)
    bet_eligible = historical_percentile >= bet_threshold_percentile

    if historical_percentile >= 80.0:
        label = LABEL_TOP
    elif historical_percentile >= 60.0:
        label = LABEL_HIGH
    elif historical_percentile >= 40.0:
        label = LABEL_MEDIUM
    elif historical_percentile >= 20.0:
        label = LABEL_LOW
    else:
        label = LABEL_VERY_LOW

    return historical_percentile, label, bet_eligible
