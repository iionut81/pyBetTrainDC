from __future__ import annotations

"""
engine.py
Orchestrates the full pipeline for one market over N candidate matches:
INPUT -> DATA VALIDATION -> HARD FILTERS -> SCORING -> CONTRADICTION -> VETO
-> CONFIDENCE -> RANKING -> TOP CANDIDATES -> FINAL VALIDATION -> BET/NO BET.
"""

from typing import List

from selection_engine.confidence import compute_confidence
from selection_engine.contradiction import detect_contradiction
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

        category_scores = {
            name: scorer(match) for name, scorer in profile.category_scorers.items()
        }
        if profile.score_fn is not None:
            # Ranking driven by the market's own chosen signal (e.g. a single
            # backtested-trustworthy category) — other categories are still
            # computed above and attached as diagnostics, but the composite
            # sum + contradiction check are bypassed entirely.
            raw_total = profile.score_fn(category_scores)
            contradiction, penalty, contradiction_notes = False, 0.0, []
        else:
            raw_total = sum(cs.value for cs in category_scores.values())
            contradiction, penalty, contradiction_notes = detect_contradiction(
                category_scores, profile
            )
        final_score = max(0.0, raw_total - penalty)

        reason = apply_vetoes(match, category_scores, profile)
        if reason is not None:
            eliminated.append(
                MatchResult(
                    match_id=match.match_id,
                    competitors=match.competitors,
                    status="ELIMINATED",
                    elimination_reason=reason,
                    category_scores=category_scores,
                    raw_total=raw_total,
                    contradiction=contradiction,
                    contradiction_penalty=penalty,
                    contradiction_notes=contradiction_notes,
                    final_score=final_score,
                    data_quality=data_quality,
                )
            )
            continue

        confidence = compute_confidence(category_scores, data_quality, contradiction)
        qualified.append(
            MatchResult(
                match_id=match.match_id,
                competitors=match.competitors,
                status="QUALIFIED",
                category_scores=category_scores,
                raw_total=raw_total,
                contradiction=contradiction,
                contradiction_penalty=penalty,
                contradiction_notes=contradiction_notes,
                final_score=final_score,
                data_quality=data_quality,
                confidence=confidence,
            )
        )

    ranked = rank_matches(qualified)
    top_picks = [m for m in ranked if m.final_score >= profile.minimum_score][: profile.top_n]
    if not top_picks and not profile.allow_no_bet and ranked:
        top_picks = ranked[: profile.top_n]
    decision = "BET" if top_picks else "NO_BET"

    return EngineResult(
        market_id=profile.market_id,
        total_analyzed=len(matches),
        eliminated=eliminated,
        qualified=ranked,
        top_picks=top_picks,
        decision=decision,
    )
