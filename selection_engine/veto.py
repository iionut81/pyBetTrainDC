from __future__ import annotations

"""
veto.py
Stage 5: some situations must eliminate a match regardless of score (e.g. a
critical contradiction between raw stats and the matchup profile). Vetoes
are market-configurable, evaluated after scoring so they can see category
scores as well as raw match data.
"""

from typing import Dict, Optional

from selection_engine.types import CategoryScore, MarketProfile, MatchInput


def apply_vetoes(
    match: MatchInput, category_scores: Dict[str, CategoryScore], profile: MarketProfile
) -> Optional[str]:
    """Return an elimination reason string, or None if the match survives."""
    for veto in profile.vetoes:
        reason = veto(match, category_scores)
        if reason is not None:
            return f"VETO:{reason}"
    return None
