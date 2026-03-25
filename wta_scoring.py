from __future__ import annotations

"""
wta_scoring.py
Shared score parsing for WTA pipelines (train_wta, run_wta_daily).
"""


def parse_set1_games(score: str) -> int:
    """Total games in set 1 from a full match score string.

    Examples: ``6-2 6-4`` → 8, ``7-6(3) 6-3`` → 13.
    Returns -1 if the first set cannot be parsed.
    """
    try:
        set1 = str(score).split()[0]
        parts = set1.replace("(", "-").replace(")", "").split("-")
        return int(parts[0]) + int(parts[1])
    except (IndexError, ValueError, AttributeError):
        return -1
