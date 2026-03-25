"""Tests for wta_set1_filters (grass merge + stable defaults)."""

from wta_set1_filters import eval_set1_o75_gates, merge_set1_o75_config


def test_merge_grass_overrides():
    base = {"rec_max_blowout": 3, "hold_floor": 0.62}
    grass = {"rec_max_blowout": 2}
    m_hard = merge_set1_o75_config(base, grass, surface="Hard")
    m_grass = merge_set1_o75_config(base, grass, surface="Grass")
    assert m_hard["rec_max_blowout"] == 3
    assert m_grass["rec_max_blowout"] == 2
    assert m_grass["hold_floor"] == 0.62


def test_eval_gates_reasonable():
    o = {"elite_levels": ["WTA 1000", "Grand Slam", "WTA 500"]}
    g = eval_set1_o75_gates(
        p_hold_a=0.70,
        p_hold_b=0.69,
        expected_total_games=24.0,
        p_s1_7_cal=0.88,
        surface="Hard",
        tournament_level="WTA 1000",
        round_id=1,
        o75_cfg=o,
    )
    assert g["rec_s1_7"] in (True, False)
    assert g["blowout_score"] >= 0
