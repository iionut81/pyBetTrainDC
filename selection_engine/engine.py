from __future__ import annotations

"""
engine.py
Orchestrates the full pipeline for one market over N candidate matches:
DATA VALIDATION -> HARD FILTER/VETO -> rank_signal (e.g. p_cal_adj) ->
HISTORICAL PERCENTILE -> LABEL/BET_ELIGIBLE -> RANK -> TOP N / NO BET.

Selection, not prediction: the engine never invents a probability — it takes
whatever rank_signal_fn returns (typically a production model's own
calibrated probability for this exact market) and classifies/ranks matches
against real historical outcomes for that same signal.
"""

from typing import List

from selection_engine.classification import classify
from selection_engine.data_validation import validate
from selection_engine.hard_filters import apply_hard_filters
from selection_engine.ranking import rank_matches
from selection_engine.types import EngineResult, MarketProfile, MatchInput, MatchResult
from selection_engine.veto import apply_vetoes


def run_selection(matches: List[MatchInput], profile: MarketProfile) -> EngineResult:
    eliminated: List[MatchResult] = []
    qualified: List[MatchResult] = []

    for match in matches:
        reason, data_quality = validate(match, profile)
        if reason is not None:
            eliminated.append(
                MatchResult(
                    match_id=match.match_id,
                    competitors=match.competitors,
                    status="ELIMINATED",
                    elimination_reason=reason,
                    data_quality=data_quality,
                )
            )
            continue

        reason = apply_hard_filters(match, profile)
        if reason is not None:
            eliminated.append(
                MatchResult(
                    match_id=match.match_id,
                    competitors=match.competitors,
                    status="ELIMINATED",
                    elimination_reason=reason,
                    data_quality=data_quality,
                )
            )
            continue

        reason = apply_vetoes(match, profile)
        if reason is not None:
            eliminated.append(
                MatchResult(
                    match_id=match.match_id,
                    competitors=match.competitors,
                    status="ELIMINATED",
                    elimination_reason=reason,
                    data_quality=data_quality,
                )
            )
            continue

        # Diagnostics only from here on — computed for logging/analysis,
        # never used to decide ranking or BET eligibility.
        category_scores = {
            name: scorer(match) for name, scorer in profile.category_scorers.items()
        }

        rank_value = profile.rank_signal_fn(match) if profile.rank_signal_fn else None
        historical_percentile, label, bet_eligible = classify(
            rank_value, profile.historical_percentiles, profile.bet_threshold_percentile
        )

        qualified.append(
            MatchResult(
                match_id=match.match_id,
                competitors=match.competitors,
                status="QUALIFIED",
                category_scores=category_scores,
                rank_value=rank_value,
                historical_percentile=historical_percentile,
                label=label,
                bet_eligible=bet_eligible,
                data_quality=data_quality,
            )
        )

    ranked = rank_matches(qualified)
    eligible = [m for m in ranked if m.bet_eligible]
    top_picks = eligible[: profile.top_n]
    decision = "BET" if top_picks else "NO_BET"

    return EngineResult(
        market_id=profile.market_id,
        total_analyzed=len(matches),
        eliminated=eliminated,
        qualified=ranked,
        top_picks=top_picks,
        decision=decision,
        historical_p80=profile.historical_percentiles.get("p80"),
    )
