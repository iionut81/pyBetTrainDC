import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wta_set1_filters import eval_set1_o75_gates

PREDICTIONS = "simulations/WTA/backtests/wta_predictions.csv"

O75_CFG = {
    "elite_levels": ["WTA 1000", "Grand Slam", "WTA 500"],
    "hold_floor": 0.62,
    "hold_strong_clay": 0.66,
    "hold_strong_default": 0.62,
    "blowout_hold_weak": 0.62,
    "blowout_hold_moderate": 0.65,
    "gap_large": 0.08,
    "asym_server_high": 0.68,
    "asym_server_low": 0.60,
    "clay_min_hold_blowout": 0.64,
    "lower_tier_min_hold": 0.64,
    "collapse_min_hold": 0.58,
    "comp_gap_tight": 0.07,
    "comp_gap_loose": 0.09,
    "comp_min_hold_loose_gap": 0.64,
    "clay_penalty_hold_lo": 0.64,
    "clay_penalty_hold_hi": 0.66,
    "clay_penalty_lo": 0.03,
    "clay_penalty_hi": 0.015,
    "hc_exp_games": 25.0,
    "hc_p_s1": 0.86,
    "hc_min_hold": 0.65,
    "hc_blowout_rescue_at": 4,
    "rec_min_exp_games": 23.0,
    "rec_min_p_s1": 0.81,
    "rec_max_blowout": 3,
    "elite_exp_games": 24.5,
    "elite_p_s1": 0.84,
    "elite_max_blowout": 2,
    "round_semifinal": 4,
    "round_final_plus": 5,
}

GRAND_SLAMS = {"australian open", "roland garros", "wimbledon", "us open", "french open"}
WTA1000 = {
    "miami", "madrid", "rome", "cincinnati", "wuhan", "beijing", "indian wells",
    "toronto", "montreal", "doha", "dubai", "guangzhou", "tokyo", "san jose",
    "stuttgart", "eastbourne", "birmingham", "los angeles", "chicago", "tianjin",
    "zhuhai", "shenzhen", "new haven", "osaka"
}
WTA500 = {
    "berlin", "bad homburg", "strasbourg", "nottingham", "eastbourne",
    "birmingham", "washington", "san jose", "guangzhou", "seoul",
    "linz", "kremlin", "luxembourg", "coupe rogers"
}

def get_level(name):
    n = str(name).lower()
    for gs in GRAND_SLAMS:
        if gs in n:
            return "Grand Slam"
    for t in WTA1000:
        if t in n:
            return "WTA 1000"
    for t in WTA500:
        if t in n:
            return "WTA 500"
    return "WTA 250"

df = pd.read_csv(PREDICTIONS, low_memory=False)
df = df.dropna(subset=["p_hold_w", "p_hold_l", "p_set1_over_7_5", "y_set1_over_7_5", "round_id"])

# expected_total_games is missing from CSV (store_expected_games was off).
# Proxy derived from sample data: 14.8*(h_w+h_l)+1.81 ≈ model Markov value (±0.5 games)
# Validated: (0.79+0.81)*14.8+1.81=25.5 ≈ 25.49; (0.72+0.82)*14.8+1.81=24.6 ≈ 24.6
df["expected_total_games"] = 14.8 * (df["p_hold_w"] + df["p_hold_l"]) + 1.81

df["tournament_level"] = df["tourney_name"].apply(get_level)

rows = []
for _, r in df.iterrows():
    try:
        gates = eval_set1_o75_gates(
            p_hold_a=float(r["p_hold_w"]),
            p_hold_b=float(r["p_hold_l"]),
            expected_total_games=float(r["expected_total_games"]),
            p_s1_7_cal=float(r["p_set1_over_7_5"]),
            surface=str(r["surface"]),
            tournament_level=str(r["tournament_level"]),
            round_id=int(r["round_id"]),
            o75_cfg=O75_CFG,
        )
        rows.append({
            "surface": r["surface"],
            "elite_pick": gates["elite_pick"],
            "rec_s1_7": gates["rec_s1_7"],
            "blowout_score": gates["blowout_score"],
            "p_cal": r["p_set1_over_7_5"],
            "y": int(r["y_set1_over_7_5"]),
            "level": r["tournament_level"],
        })
    except Exception:
        continue

res = pd.DataFrame(rows)

print(f"\nTotal meciuri in backtest: {len(res)}")
print(f"Cu y_set1_over_7_5 = 1: {res['y'].sum()} ({res['y'].mean()*100:.1f}% baseline global)")

print("\n" + "="*65)
print("  BACKTEST O7.5 SET 1 — ALL vs ELITE_PICK vs REC")
print("="*65)

surfaces = ["Grass", "Clay", "Hard", "ALL"]
for surf in surfaces:
    if surf == "ALL":
        sub = res
    else:
        sub = res[res["surface"] == surf]
    if len(sub) == 0:
        continue

    all_n   = len(sub)
    all_hr  = sub["y"].mean() * 100
    ep = sub[sub["elite_pick"]]
    ep_n    = len(ep)
    ep_hr   = ep["y"].mean() * 100 if ep_n > 0 else float("nan")
    rec = sub[sub["rec_s1_7"]]
    rec_n   = len(rec)
    rec_hr  = rec["y"].mean() * 100 if rec_n > 0 else float("nan")

    print(f"\n  {surf}")
    print(f"    Toate meciurile:  HR {all_hr:.1f}%  (N={all_n:,})")
    print(f"    rec_s1_7=True:    HR {rec_hr:.1f}%  (N={rec_n:,})")
    print(f"    elite_pick=True:  HR {ep_hr:.1f}%  (N={ep_n:,})")

    if ep_n > 0:
        print(f"\n    Elite pick breakdown per p_cal bucket:")
        for lo, hi, label in [(0.84, 0.87, "0.84-0.87"),
                               (0.87, 0.90, "0.87-0.90"),
                               (0.90, 0.93, "0.90-0.93"),
                               (0.93, 1.01, "0.93+")]:
            bk = ep[(ep["p_cal"] >= lo) & (ep["p_cal"] < hi)]
            if len(bk) > 0:
                print(f"      p_cal {label}: HR {bk['y'].mean()*100:.1f}% (N={len(bk):,})")

print("\n" + "="*65)
print("  GRAND SLAM ONLY — elite_pick=True")
print("="*65)
for surf in ["Grass", "Clay", "Hard"]:
    sub = res[(res["surface"] == surf) & (res["level"] == "Grand Slam") & (res["elite_pick"])]
    if len(sub) > 0:
        print(f"  {surf} GS: HR {sub['y'].mean()*100:.1f}%  (N={len(sub):,})")
