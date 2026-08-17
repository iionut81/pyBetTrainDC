from __future__ import annotations

"""
veto.py
Stage 2 (after hard filters): some situations must eliminate a match
regardless of its ranking signal (e.g. a critical hold mismatch that risks a
one-sided set). Vetoes are market-configurable, evaluated directly on raw
match data — before any diagnostic scoring or ranking happens.
"""

from typing import Optional

from selection_engine.types import MarketProfile, MatchInput


def apply_vetoes(match: MatchInput, profile: MarketProfile) -> Optional[str]:
    """Return an elimination reason string, or None if the match survives."""
    for veto in profile.vetoes:
        reason = veto(match)
        if reason is not None:
            return f"VETO:{reason}"
    return None
