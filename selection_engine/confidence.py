from __future__ import annotations

"""
confidence.py
Stage 6: confidence is NOT a win probability. It's a generic (market-agnostic)
read of how much to trust the score itself, based on data quality, agreement
between the 5 categories, stability, market compatibility, and whether a
contradiction was flagged.
"""

from typing import Dict

from selection_engine.types import CATEGORY_MAX, CategoryScore

HIGH_THRESHOLD = 75.0
MEDIUM_THRESHOLD = 50.0


def compute_confidence(
    category_scores: Dict[str, CategoryScore], data_quality: float, contradiction: bool
) -> str:
    values = [cs.value for cs in category_scores.values()]
    if not values:
        return "LOW"

    spread = max(values) - min(values)
    agreement = max(0.0, (CATEGORY_MAX - spread) / CATEGORY_MAX)
    stability = category_scores.get("stability")
    stability_ratio = (stability.value / CATEGORY_MAX) if stability else 0.5
    market_compat = category_scores.get("market_compatibility")
    market_compat_ratio = (market_compat.value / CATEGORY_MAX) if market_compat else 0.5

    conf_score = (
        data_quality * 25.0
        + stability_ratio * 25.0
        + agreement * 25.0
        + market_compat_ratio * 25.0
    )
    if contradiction:
        conf_score -= 15.0
    conf_score = max(0.0, min(100.0, conf_score))

    if conf_score >= HIGH_THRESHOLD:
        return "HIGH"
    if conf_score >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"
