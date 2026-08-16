from __future__ import annotations

"""
output.py
Renders an EngineResult as the plain-text report format from the project spec.

"Rank Score" is deliberately not called "Score" or graded as "good/bad" in
isolation — for markets that set MarketProfile.score_fn (e.g.
TENNIS_SET1_OVER_7_5), it's a rescaled transform of one trusted signal
(currently p_cal_adj), not an independent composite quality judgment. It only
means something as a RANKING key ("this cleared more matches than that one"),
not as an absolute grade ("92/100 = great").
"""

from selection_engine.types import EngineResult, MatchResult

_RULE = "=" * 50
_DASH = "-" * 50


def _format_pick(rank: int, result: MatchResult) -> str:
    vs = " vs ".join(result.competitors)
    lines = [
        f"#{rank} {vs}",
        f"Rank Score: {result.final_score:.0f}/100 (ranking signal, not a quality grade)",
        f"Confidence: {result.confidence}",
        "",
    ]
    if result.strengths:
        lines.append("Strengths:")
        lines.extend(result.strengths)
        lines.append("")
    if result.risks:
        lines.append("Risks:")
        lines.extend(result.risks)
        lines.append("")
    return "\n".join(lines).rstrip()


def format_report(market_label: str, result: EngineResult) -> str:
    lines = [
        _RULE,
        f"MARKET: {market_label}",
        _RULE,
        "",
        f"MATCHES ANALYZED: {result.total_analyzed}",
        "",
        f"ELIMINATED: {len(result.eliminated)}",
        f"QUALIFIED: {len(result.qualified)}",
        "",
        _DASH,
        f"TOP {len(result.top_picks)}" if result.top_picks else "TOP CANDIDATES",
        _DASH,
        "",
    ]

    if result.top_picks:
        for i, pick in enumerate(result.top_picks, start=1):
            lines.append(_format_pick(i, pick))
            lines.append("")
            lines.append(_DASH)
            lines.append("")
    else:
        lines.append("(none clear the minimum score threshold)")
        lines.append("")

    if result.decision == "BET":
        lines.append("FINAL DECISION")
        lines.append("")
        lines.append(f"TOP PICK: {' vs '.join(result.top_picks[0].competitors)}")
        if len(result.top_picks) > 1:
            lines.append(f"SECOND PICK: {' vs '.join(result.top_picks[1].competitors)}")
    else:
        lines.append("FINAL DECISION: NO BET")

    lines.append(_RULE)
    return "\n".join(lines)
