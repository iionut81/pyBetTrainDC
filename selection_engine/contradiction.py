from __future__ import annotations

"""
contradiction.py
Stage 4: catch cases where the raw sum looks good but the categories disagree
with each other (e.g. STATISTICS=19 while MATCHUP=8 and STABILITY=7). Flags
CONTRADICTION and applies a configurable penalty instead of trusting the sum.
"""

from typing import Dict, List, Tuple

from selection_engine.types import CategoryScore, MarketProfile


def detect_contradiction(
    category_scores: Dict[str, CategoryScore], profile: MarketProfile
) -> Tuple[bool, float, List[str]]:
    """Return (contradiction, penalty, notes)."""
    values = {name: cs.value for name, cs in category_scores.items()}
    if not values:
        return False, 0.0, []

    notes: List[str] = []
    contradiction = False

    highs = {n: v for n, v in values.items() if v >= profile.contradiction_high}
    lows = {n: v for n, v in values.items() if v <= profile.contradiction_low}
    if highs and lows:
        contradiction = True
        for hn, hv in highs.items():
            for ln, lv in lows.items():
                notes.append(f"- High {hn} ({hv:.0f}/20) but low {ln} ({lv:.0f}/20)")

    spread = max(values.values()) - min(values.values())
    if spread >= profile.contradiction_spread:
        contradiction = True
        notes.append(f"- Category scores disagree by {spread:.0f}/20 points")

    penalty = profile.contradiction_penalty if contradiction else 0.0
    return contradiction, penalty, notes
