from __future__ import annotations

"""
wta_set1_filters.py
Pure Set 1 Over 7.5 gate logic (blowout / competitive set). Shared by run_wta_daily
and analyze_wta_ablation.py so backtest ablation matches production.
"""

from typing import Any, Dict, Mapping


def merge_set1_o75_config(
    base: Mapping[str, Any],
    grass_overrides: Mapping[str, Any] | None,
    *,
    surface: str,
) -> Dict[str, Any]:
    """Start from base config; on Grass, shallow-merge grass_overrides (if any)."""
    out: Dict[str, Any] = dict(base)
    if surface.lower() == "grass" and grass_overrides:
        out.update(grass_overrides)
    return out


def eval_set1_o75_gates(
    p_hold_a: float,
    p_hold_b: float,
    expected_total_games: float,
    p_s1_7_cal: float,
    surface: str,
    tournament_level: str,
    round_id: int,
    o75_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate Set1 O7.5 filters. Returns flags and adjusted probability used for decisions."""
    o = o75_cfg
    elite_levels = tuple(o.get("elite_levels", ["WTA 1000", "Grand Slam", "WTA 500"]))
    is_clay = surface.lower() == "clay"
    is_lower_tier = tournament_level not in elite_levels

    hold_floor = float(o.get("hold_floor", 0.62))
    hold_strong_clay = float(o.get("hold_strong_clay", 0.66))
    hold_strong_default = float(o.get("hold_strong_default", 0.62))
    min_hold = hold_strong_clay if is_clay else hold_strong_default
    gap = abs(p_hold_a - p_hold_b)
    min_hold_val = min(p_hold_a, p_hold_b)
    holds_floor = min_hold_val >= hold_floor
    holds_strong = min_hold_val >= min_hold

    blowout_hold_weak = float(o.get("blowout_hold_weak", 0.62))
    blowout_hold_moderate = float(o.get("blowout_hold_moderate", 0.65))
    blowout_score = 0
    for hold in (p_hold_a, p_hold_b):
        if hold < blowout_hold_weak:
            blowout_score += 2
        elif hold < blowout_hold_moderate:
            blowout_score += 1
    if gap > float(o.get("gap_large", 0.08)):
        blowout_score += 2
    if max(p_hold_a, p_hold_b) > float(o.get("asym_server_high", 0.68)) and min_hold_val < float(
        o.get("asym_server_low", 0.60)
    ):
        blowout_score += 2
    clay_mb = float(o.get("clay_min_hold_blowout", 0.64))
    if is_clay:
        if min_hold_val < clay_mb:
            blowout_score += 2
        else:
            blowout_score += 1
    lower_mb = float(o.get("lower_tier_min_hold", 0.64))
    if is_lower_tier and min_hold_val < lower_mb:
        blowout_score += 1
    rnd_sf = int(o.get("round_semifinal", 4))
    rnd_f = int(o.get("round_final_plus", 5))
    if round_id == rnd_sf:
        blowout_score += 1
    elif round_id >= rnd_f:
        blowout_score += 2

    collapse_risk = min_hold_val < float(o.get("collapse_min_hold", 0.58))

    cg_tight = float(o.get("comp_gap_tight", 0.07))
    cg_loose = float(o.get("comp_gap_loose", 0.09))
    cg_mh = float(o.get("comp_min_hold_loose_gap", 0.64))
    competitive_set = holds_floor and (gap <= cg_tight or (gap <= cg_loose and min_hold_val >= cg_mh))

    p_s1_7_adj = float(p_s1_7_cal)
    cpl = float(o.get("clay_penalty_hold_lo", 0.64))
    cph = float(o.get("clay_penalty_hold_hi", 0.66))
    if is_clay:
        if min_hold_val < cpl:
            p_s1_7_adj -= float(o.get("clay_penalty_lo", 0.03))
        elif min_hold_val < cph:
            p_s1_7_adj -= float(o.get("clay_penalty_hi", 0.015))

    hc_eg = float(o.get("hc_exp_games", 25.0))
    hc_ps = float(o.get("hc_p_s1", 0.86))
    hc_mh = float(o.get("hc_min_hold", 0.65))
    hc_br = int(o.get("hc_blowout_rescue_at", 4))
    high_confidence = expected_total_games >= hc_eg and p_s1_7_adj >= hc_ps and min_hold_val >= hc_mh
    if high_confidence and blowout_score == hc_br:
        blowout_score -= 1

    rec_eg = float(o.get("rec_min_exp_games", 23.0))
    rec_ps = float(o.get("rec_min_p_s1", 0.81))
    rec_bm = int(o.get("rec_max_blowout", 3))
    rec_s1_7 = bool(
        expected_total_games >= rec_eg
        and p_s1_7_adj >= rec_ps
        and competitive_set
        and holds_floor
        and blowout_score <= rec_bm
        and not collapse_risk
    )

    el_eg = float(o.get("elite_exp_games", 24.5))
    el_ps = float(o.get("elite_p_s1", 0.84))
    el_bm = int(o.get("elite_max_blowout", 2))
    elite_pick = bool(
        expected_total_games >= el_eg
        and p_s1_7_adj >= el_ps
        and blowout_score <= el_bm
        and holds_strong
    )

    return {
        "p_s1_7_adj": p_s1_7_adj,
        "rec_s1_7": rec_s1_7,
        "elite_pick": elite_pick,
        "blowout_score": blowout_score,
        "competitive_set": competitive_set,
        "collapse_risk": collapse_risk,
        "holds_floor": holds_floor,
        "holds_strong": holds_strong,
        "min_hold_val": min_hold_val,
        "gap": gap,
    }
