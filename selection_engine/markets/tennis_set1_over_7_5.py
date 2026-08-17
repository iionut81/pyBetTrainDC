from __future__ import annotations

"""
markets/tennis_set1_over_7_5.py
Market profile for TENNIS_SET1_OVER_7_5.

RANKING/ELIGIBILITY IS PERCENTILE-BASED ON p_cal_adj — the production
pipeline's own calibrated P(Set 1 Over 7.5) for this exact match — not a
0-100 composite/rescaled score. History (why, in order):

  2026-08-16: backtested a 5-category weighted composite score against
  p_cal_adj alone on 17,081 historical WTA matches. p_cal_adj alone gave a
  cleaner, monotonic hit-rate curve than the composite; the surface bonus in
  MARKET_COMPATIBILITY tested as literally no different with vs without.
  Simplified to score_fn = STATISTICS-only (still a 0-100 rescale of
  p_cal_adj) with a fixed minimum_score=80 cutoff.

  2026-08-17 (first pass): realized minimum_score=80 -> "p_cal_adj >= 91.8%"
  was an artifact of two unrelated design choices (an untuned placeholder
  cutoff of 80, rescaled through a percentile-normalized 0-100 map) — not a
  threshold ever shown to correlate with anything. Replaced with:
  rank_signal_fn returns raw p_cal_adj (no rescale at all), BET eligibility
  decided by comparing it against real historical percentile breakpoints.

  2026-08-17 (second pass): the first pass computed those breakpoints from
  the POST-VETO population (n=14,323) — wrong population. Veto is a
  selection filter (it decides which matches survive to be ranked); it must
  not also define the statistical universe rank_value is measured against,
  or the percentiles become entangled with an unrelated filtering decision.
  P_CAL_ADJ_HISTORICAL_PERCENTILES below is now computed from the
  POST_HARD_FILTER / PRE_VETO population (n=17,081) — data validation and
  hard filters applied, veto NOT applied. Only matches at or above the
  historical top quintile (p80 of THAT population) are BET_ELIGIBLE — being
  the best candidate available today is not the same thing; see
  classification.py.

FORM, MATCHUP, MARKET_COMPATIBILITY and STABILITY are still computed and
attached to every result as diagnostics (visible in the report) — they do
NOT feed rank_signal_fn, historical_percentiles, or BET eligibility. Promote
one back into the ranking signal only if a future backtest shows it adds
real separation on its own; don't assume it does.

Thresholds/hard-filters here are illustrative defaults for exercising the
engine end to end — independent of, and simpler than, the tuned production
logic in wta_set1_filters.py / run_wta_daily.py. Do not treat these numbers
as validated betting thresholds beyond what the backtest actually shows.

Expected `MatchInput.stats` fields:
  p_hold_a, p_hold_b        (required) hold rate 0-1 for each player
  surface                   (required) "Hard" | "Clay" | "Grass"
  p_cal_adj                 (optional) 0-1, model's own calibrated P(set 1 over 7.5) —
                            the ranking signal when available
  expected_total_games      (optional) model-estimated total GAMES IN SET 1 (not the
                            whole match) — fallback ranking signal for mock/demo data
                            with no calibrated probability of its own (rescaled to a
                            0-1 pseudo-probability; not a real probability)
  recent_form_variance_a/b  (optional) 0-1, higher = more erratic recent serve form
  tiebreak_rate             (optional) 0-1, historical set-1 tie-break frequency
"""

from typing import Dict, Iterable, Optional

from selection_engine.classification import compute_percentiles
from selection_engine.types import CategoryScore, MarketProfile, MatchInput

MARKET_ID = "TENNIS_SET1_OVER_7_5"

# Category scorers (diagnostics only — do not feed ranking) -------------------


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


def score_statistics(match: MatchInput) -> CategoryScore:
    """Diagnostic display of the same signal rank_signal_p_cal_adj uses for
    ranking — plain p_cal_adj * 20, no percentile rescale. Not used for
    ranking or eligibility (see module docstring)."""
    p_cal_adj = match.stats.get("p_cal_adj")
    if p_cal_adj is not None:
        p_cal_adj = float(p_cal_adj)
        value = max(0.0, min(20.0, p_cal_adj * 20.0))
        notes = []
        if p_cal_adj >= 0.90:
            notes.append(f"+ High calibrated Over 7.5 probability ({p_cal_adj:.0%})")
        elif p_cal_adj < 0.80:
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
    # to hit rate — removed rather than left in "just in case".
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

# Ranking signal --------------------------------------------------------------


def rank_signal_p_cal_adj(match: MatchInput) -> Optional[float]:
    """Raw ranking value — p_cal_adj itself, no rescale. Falls back to an
    expected_total_games-derived pseudo-probability only for mock/demo data
    that has no real calibrated probability; that fallback is NOT a real
    probability and exists purely so the demo can exercise the pipeline."""
    p_cal_adj = match.stats.get("p_cal_adj")
    if p_cal_adj is not None:
        return float(p_cal_adj)

    expected_games = match.stats.get("expected_total_games")
    if expected_games is not None:
        return max(0.0, min(1.0, (float(expected_games) - 9.0) / 4.0))

    return None


# Historical percentile breakpoints for rank_signal_p_cal_adj ----------------
# Computed 2026-08-17 from the POST_HARD_FILTER / PRE_VETO population — every
# historical WTA match in simulations/WTA/backtests/wta_predictions.csv that
# passed data_validation + hard_filters, BEFORE veto is applied (n=17,081).
# Veto is deliberately excluded from this population (see module docstring).
# Recompute with selection_engine.classification.compute_percentiles()
# whenever the dataset changes materially — do not hand-edit these numbers.
P_CAL_ADJ_HISTORICAL_PERCENTILES: Dict[str, float] = {
    "p0": 0.753523,
    "p20": 0.859453,
    "p40": 0.886749,
    "p60": 0.900665,
    "p80": 0.916333,
    "p90": 0.928661,
    "p95": 0.939216,
    "p100": 0.983489,
}

BET_THRESHOLD_PERCENTILE = 80.0


def recompute_historical_percentiles(p_cal_adj_values: Iterable[float]) -> Dict[str, float]:
    """Re-derive P_CAL_ADJ_HISTORICAL_PERCENTILES from a fresh sample (e.g.
    after a retrain adds meaningfully more backtest data). Does not mutate
    the module constant — copy the result in by hand once you've reviewed it,
    same as any other calibration change in this file."""
    return compute_percentiles(p_cal_adj_values)


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


def veto_blowout_risk(match: MatchInput) -> Optional[str]:
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
    top_n=2,
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
    rank_signal_fn=rank_signal_p_cal_adj,
    historical_percentiles=P_CAL_ADJ_HISTORICAL_PERCENTILES,
    bet_threshold_percentile=BET_THRESHOLD_PERCENTILE,
)
