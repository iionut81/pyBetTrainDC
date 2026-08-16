from __future__ import annotations

"""
ranking.py
Stage 7: order qualified matches by final_score, then stability, then
confidence, then contradiction count, then data quality (spec order).
"""

from typing import List

from selection_engine.types import MatchResult

_CONFIDENCE_RANK = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


def _sort_key(result: MatchResult):
    stability = result.category_scores.get("stability")
    stability_value = stability.value if stability else 0.0
    return (
        -result.final_score,
        -stability_value,
        -_CONFIDENCE_RANK.get(result.confidence, 0),
        int(result.contradiction),
        -result.data_quality,
    )


def rank_matches(results: List[MatchResult]) -> List[MatchResult]:
    return sorted(results, key=_sort_key)
