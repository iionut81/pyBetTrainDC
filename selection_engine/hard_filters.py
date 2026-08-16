from __future__ import annotations

"""
hard_filters.py
Stage 2: market-configurable pass/fail checks (e.g. huge class gap, extreme
volatility, profile incompatible with the market). Runs after data
validation, before scoring.
"""

from typing import Optional

from selection_engine.types import MarketProfile, MatchInput


def apply_hard_filters(match: MatchInput, profile: MarketProfile) -> Optional[str]:
    """Return an elimination reason string, or None if the match survives."""
    for hard_filter in profile.hard_filters:
        reason = hard_filter(match)
        if reason is not None:
            return f"HARD_FILTER:{reason}"
    return None
