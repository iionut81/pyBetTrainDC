from __future__ import annotations

"""
output.py
Renders an EngineResult as a plain-text report.

Always shows the top-ranked candidates (by rank_value) regardless of BET
eligibility — a match that's merely the best one available today is not the
same thing as a BET_ELIGIBLE match, and the report must never blur that
distinction (e.g. never say "Parry-Boisson is a bad match" when the honest
statement is "best available today, but below the historical top quintile").
Each displayed candidate carries its own label/bet_eligible so that's always
explicit, independent of whatever the final BET/NO BET decision turns out
to be.
"""

from selection_engine.types import EngineResult, MatchResult

_RULE = "=" * 50
_DASH = "-" * 50


def _format_pct(value) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def _format_candidate(rank: int, result: MatchResult) -> str:
    vs = " vs ".join(result.competitors)
    lines = [
        f"#{rank} {vs}",
        f"p_cal_adj: {_format_pct(result.rank_value)}",
        f"historical_percentile: {result.historical_percentile:.1f}"
        if result.historical_percentile is not None
        else "historical_percentile: n/a",
        f"label: {result.label or 'n/a'}",
        f"bet_eligible: {'YES' if result.bet_eligible else 'NO'}",
        "",
    ]
    if result.strengths:
        lines.append("Diagnostics (+):")
        lines.extend(result.strengths)
        lines.append("")
    if result.risks:
        lines.append("Diagnostics (-):")
        lines.extend(result.risks)
        lines.append("")
    return "\n".join(lines).rstrip()


def format_report(market_label: str, result: EngineResult, top_n: int = 2) -> str:
    n_veto = sum(1 for r in result.eliminated if (r.elimination_reason or "").startswith("VETO:"))
    n_eligible = sum(1 for r in result.qualified if r.bet_eligible)

    lines = [
        _RULE,
        f"MARKET: {market_label}",
        _RULE,
        "",
        f"MATCHES ANALYZED: {result.total_analyzed}",
        "",
        f"ELIMINATED: {len(result.eliminated)}",
        f"ELIMINATED BY VETO: {n_veto}",
        f"QUALIFIED: {len(result.qualified)}",
        f"BET ELIGIBLE: {n_eligible}",
        "",
        f"HISTORICAL P80: {_format_pct(result.historical_p80)}",
        "",
        _DASH,
        "",
    ]

    displayed = result.qualified[:top_n]
    if displayed:
        for i, candidate in enumerate(displayed, start=1):
            lines.append(_format_candidate(i, candidate))
            lines.append("")
            lines.append(_DASH)
            lines.append("")
    else:
        lines.append("(no qualified candidates)")
        lines.append("")

    lines.append(f"FINAL DECISION: {'BET' if result.decision == 'BET' else 'NO BET'}")
    lines.append("")
    if result.decision == "BET":
        lines.append(f"TOP PICK: {' vs '.join(result.top_picks[0].competitors)}")
        if len(result.top_picks) > 1:
            lines.append(f"SECOND PICK: {' vs '.join(result.top_picks[1].competitors)}")
    else:
        if displayed:
            best = displayed[0]
            best_vs = " vs ".join(best.competitors)
            lines.append(
                f"REASON: {best_vs} is the best available candidate, but it does not reach "
                "the historical top-quintile threshold required for BET eligibility."
            )
        else:
            lines.append("REASON: No qualified candidates were available to evaluate.")

    lines.append(_RULE)
    return "\n".join(lines)
