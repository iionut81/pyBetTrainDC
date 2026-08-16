from __future__ import annotations

"""
data_validation.py
Stage 1 of the pipeline: check we have enough/fresh/consistent data before
any scoring happens. Never estimates or invents missing data — a gap here
is a straight elimination.
"""

from typing import Optional, Tuple

from selection_engine.types import MarketProfile, MatchInput


def validate(match: MatchInput, profile: MarketProfile) -> Tuple[Optional[str], float]:
    """Return (elimination_reason, data_quality).

    elimination_reason is None when the match has enough data to proceed.
    data_quality is 0-1, based on how many optional fields are present —
    it still applies even when the match qualifies, feeding into confidence.
    """
    for field in profile.required_fields:
        value = match.stats.get(field)
        if value is None:
            return "INSUFFICIENT_DATA", 0.0

    if profile.min_sample_size:
        for key in ("sample_size_a", "sample_size_b"):
            sample = match.meta.get(key)
            if sample is not None and sample < profile.min_sample_size:
                return "INSUFFICIENT_DATA", 0.0

    if profile.max_data_age_days is not None:
        age = match.meta.get("data_age_days")
        if age is not None and age > profile.max_data_age_days:
            return "INSUFFICIENT_DATA", 0.0

    if not profile.optional_fields:
        data_quality = 1.0
    else:
        present = sum(1 for f in profile.optional_fields if match.stats.get(f) is not None)
        data_quality = present / len(profile.optional_fields)

    return None, data_quality
