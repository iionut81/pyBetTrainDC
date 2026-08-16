from __future__ import annotations

"""
markets/tennis_set1_over_7_5.py
Market profile for TENNIS_SET1_OVER_7_5.

RANKING IS DRIVEN BY p_cal_adj ALONE (see score_from_p_cal_adj below), not by
a composite of all 5 categories. Backtested 2026-08-16 on 17,081 historical
WTA matches (backtest_selection_engine_wta.py):
  - p_cal_adj alone, bucketed by quantile, gave a clean monotonic hit-rate
    curve (80.7% -> 85.7% across 5 equal-size buckets).
  - The composite 5-category score's own bucket hit-rates were noisier
    (82.7 -> 84.6 -> 85.8 -> 86.4 -> 84.8 -> 85.7, non-monotonic) and did not
    beat p_cal_adj alone.
  - The surface bonus that used to sit in MARKET_COMPATIBILITY tested as
    literally no different with vs without it — removed for good.
  - VETO held up (79.0% hit rate when triggered vs 83.5% when not) and stays
    as an elimination gate.
FORM, MATCHUP, MARKET_COMPATIBILITY and STABILITY are still computed and
attached to every result as diagnostics (visible in the report's
strengths/risks) — they just no longer decide ranking or the minimum-score
threshold. Promote one back into score_fn only if a future backtest on more
data shows it adds real separation on its own; don't assume it does.

Thresholds here are illustrative defaults for exercising the engine end to
end — they are independent of, and simpler than, the tuned production logic
in wta_set1_filters.py / run_wta_daily.py. Do not treat these numbers as
validated betting thresholds.

Expected `MatchInput.stats` fields:
  p_hold_a, p_hold_b        (required) hold rate 0-1 for each player
  surface                   (required) "Hard" | "Clay" | "Grass"
  p_cal_adj                 (optional) 0-1, model's own calibrated P(set 1 over 7.5) —
                            preferred STATISTICS signal when available (e.g. from a
                            production pipeline's own Monte Carlo + calibration)
  expected_total_games      (optional) model-estimated total GAMES IN SET 1 (not the
                            whole match) — fallback STATISTICS signal for mock/demo data
                            that has no calibrated probability of its own
  recent_form_variance_a/b  (optional) 0-1, higher = more erratic recent serve form
  tiebreak_rate             (optional) 0-1, historical set-1 tie-break frequency
"""

from typing import Dict, Optional

from selection_engine.types import CategoryScore, MarketProfile, MatchInput

MARKET_ID = "TENNIS_SET1_OVER_7_5"

# Category scorers -----------------------------------------------------------


def score_form(match: MatchInput) -> CategoryScore:
    va = match.stats.get("recent_form_variance_a")
    vb = match.stats.get("recent_form_variance_b")
    variances = [v for v in (va, vb) if v is not None]
    if not variances:
        return CategoryScore(value=14.0, notes=["- No recent-form data available"])

    avg_var = sum(variances) / len(variances)
    value = max(0.0, min(20.0, 20.0 * (1.0 - avg_var)))
    notes = []
    if avg_var <= 0.15:
        notes.append(f"+ Consistent recent service form (variance {avg_var:.2f})")
    elif avg_var >= 0.40:
        notes.append(f"- Erratic recent form (variance {avg_var:.2f})")
    return CategoryScore(value=value, notes=notes)


def score_matchup(match: MatchInput) -> CategoryScore:
    hold_a = float(match.stats["p_hold_a"])
    hold_b = float(match.stats["p_hold_b"])
    min_hold = min(hold_a, hold_b)
    gap = abs(hold_a - hold_b)

    base = 12.0 if 0.62 <= min_hold <= 0.80 else max(0.0, 12.0 - abs(min_hold - 0.71) * 40)
    if gap <= 0.05:
        bonus = 8.0
    elif gap <= 0.10:
        bonus = 5.0
    elif gap <= 0.15:
        bonus = 2.0
    else:
        bonus = 0.0
    value = max(0.0, min(20.0, base + bonus))

    notes = []
    if gap <= 0.08:
        notes.append(f"+ Balanced hold levels (gap {gap:.2f})")
    elif gap > 0.15:
        notes.append(f"- Hold levels imbalanced (gap {gap:.2f})")
    if 0.62 <= min_hold <= 0.80:
        notes.append("+ Both players hold serve in the sweet-spot range")
    return CategoryScore(value=value, notes=notes)


#  p_cal_adj linear map bounds. First calibration (0.75/0.90) was an eyeball
#  guess and clipped ~50% of real matches to the max (0.90 sat at only the
#  53rd percentile of the real distribution) — collapsing exactly the
#  differentiation Top-N% selection needs among the best candidates.
#  Recalibrated 2026-08-16 on the 1st/99th percentile of p_cal_adj across
#  17,595 historical WTA matches (min 0.728, 1%=0.751, median 0.893,
#  99%=0.958, max 0.986).
P_CAL_ADJ_FLOOR = 0.75
P_CAL_ADJ_CEILING = 0.96


def score_statistics(match: MatchInput) -> CategoryScore:
    p_cal_adj = match.stats.get("p_cal_adj")
    if p_cal_adj is not None:
        p_cal_adj = float(p_cal_adj)
        span = P_CAL_ADJ_CEILING - P_CAL_ADJ_FLOOR
        value = max(0.0, min(20.0, 20.0 * (p_cal_adj - P_CAL_ADJ_FLOOR) / span))
        notes = []
        if value >= 15.0:
            notes.append(f"+ High calibrated Over 7.5 probability ({p_cal_adj:.0%})")
        elif value <= 8.0:
            notes.append(f"- Low calibrated Over 7.5 probability ({p_cal_adj:.0%})")
        return CategoryScore(value=value, notes=notes)

    expected_games = match.stats.get("expected_total_games")
    if expected_games is None:
        return CategoryScore(value=10.0, notes=["- No statistical signal available (p_cal_adj/expected_total_games)"])

    expected_games = float(expected_games)
    value = max(0.0, min(20.0, 20.0 * (expected_games - 9.0) / (13.0 - 9.0)))
    notes = []
    if value >= 15.0:
        notes.append(f"+ High expected total games ({expected_games:.1f})")
    elif value <= 8.0:
        notes.append(f"- Low expected total games ({expected_games:.1f})")
    return CategoryScore(value=value, notes=notes)


def score_market_compatibility(match: MatchInput) -> CategoryScore:
    # No surface bonus: backtested 2026-08-16, made no measurable difference
    # to hit rate (see module docstring) — removed rather than left in "just
    # in case".
    tb_rate = match.stats.get("tiebreak_rate")
    tb_component = (tb_rate * 10.0) if tb_rate is not None else 5.0
    value = max(0.0, min(20.0, 10.0 + tb_component))

    notes = []
    if tb_rate is not None:
        if tb_rate >= 0.25:
            notes.append(f"+ High tie-break tendency fits Over 7.5 ({tb_rate:.0%})")
        elif tb_rate < 0.10:
            notes.append(f"- Low tie-break tendency ({tb_rate:.0%})")
    return CategoryScore(value=value, notes=notes)


def score_stability(match: MatchInput) -> CategoryScore:
    hold_a = float(match.stats["p_hold_a"])
    hold_b = float(match.stats["p_hold_b"])
    min_hold = min(hold_a, hold_b)
    gap = abs(hold_a - hold_b)

    penalty = 0.0
    notes = []
    if min_hold < 0.55:
        penalty += 10.0
        notes.append(f"- Weak server drags down stability (min hold {min_hold:.2f})")
    if gap > 0.15:
        penalty += 8.0
        notes.append(f"- Large hold gap adds volatility (gap {gap:.2f})")
    elif gap > 0.10:
        penalty += 4.0

    value = max(0.0, min(20.0, 20.0 - penalty))
    if penalty == 0.0:
        notes.append("+ No internal volatility flags")
    return CategoryScore(value=value, notes=notes)


CATEGORY_SCORERS = {
    "form": score_form,
    "matchup": score_matchup,
    "statistics": score_statistics,
    "market_compatibility": score_market_compatibility,
    "stability": score_stability,
}


def score_from_p_cal_adj(category_scores: Dict[str, CategoryScore]) -> float:
    """Ranking score = STATISTICS alone, rescaled 0-20 -> 0-100.

    STATISTICS is p_cal_adj itself (or, for mock data with no p_cal_adj, the
    expected_total_games fallback — see score_statistics). FORM, MATCHUP,
    MARKET_COMPATIBILITY and STABILITY are computed and reported but do not
    feed this number. See module docstring for the backtest that justifies
    this over summing all 5 categories.
    """
    return category_scores["statistics"].value * 5.0

# Hard filters ----------------------------------------------------------------


def filter_low_expected_games(match: MatchInput) -> Optional[str]:
    expected_games = match.stats.get("expected_total_games")
    if expected_games is not None and float(expected_games) < 8.0:
        return "LOW_EXPECTED_GAMES"
    return None


def filter_extreme_hold_gap(match: MatchInput) -> Optional[str]:
    gap = abs(float(match.stats["p_hold_a"]) - float(match.stats["p_hold_b"]))
    if gap > 0.30:
        return "EXTREME_HOLD_GAP"
    return None


# Vetoes ------------------------------------------------------------------


def veto_blowout_risk(match: MatchInput, category_scores: Dict[str, CategoryScore]) -> Optional[str]:
    hold_a = float(match.stats["p_hold_a"])
    hold_b = float(match.stats["p_hold_b"])
    min_hold = min(hold_a, hold_b)
    gap = abs(hold_a - hold_b)
    if min_hold < 0.55 and gap > 0.15:
        return "CRITICAL_CONTRADICTION"
    return None


TENNIS_SET1_OVER_7_5_PROFILE = MarketProfile(
    market_id=MARKET_ID,
    sport="tennis",
    minimum_score=80.0,
    top_n=2,
    allow_no_bet=True,
    required_fields=["p_hold_a", "p_hold_b", "surface"],
    optional_fields=[
        "p_cal_adj",
        "expected_total_games",
        "recent_form_variance_a",
        "recent_form_variance_b",
        "tiebreak_rate",
    ],
    hard_filters=[filter_low_expected_games, filter_extreme_hold_gap],
    vetoes=[veto_blowout_risk],
    category_scorers=CATEGORY_SCORERS,
    score_fn=score_from_p_cal_adj,
)
