from __future__ import annotations

"""
ranking.py
Orders qualified matches strictly by rank_value (p_cal_adj) descending.
Ties (equal rank_value — rare in practice) break by: data_quality, then the
diagnostic STABILITY category (never the primary signal, only a tie-breaker),
then match_id for a fully deterministic order.
"""

from typing import List

from selection_engine.types import MatchResult


def _sort_key(result: MatchResult):
    has_signal = result.rank_value is not None
    stability = result.category_scores.get("stability")
    stability_value = stability.value if stability else 0.0
    return (
        not has_signal,
        -(result.rank_value if has_signal else 0.0),
        -result.data_quality,
        -stability_value,
        result.match_id,
    )


def rank_matches(results: List[MatchResult]) -> List[MatchResult]:
    return sorted(results, key=_sort_key)
