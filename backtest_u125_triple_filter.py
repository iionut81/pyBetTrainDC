"""
Backtest: Triple Filter pentru U12.5 Set 2
==========================================
Aplica regulile noi si compara hit rate cu/fara filtre.

Nota: wta_predictions.csv contine date Set 1 (y_tiebreak = TB Set 1).
Folosim Set 1 ca proxy pentru U12.5 — corect deoarece:
  - 7-6 (TB) = 13 games = Over 12.5  (loss)
  - orice set fara TB = ≤12 games    = Under 12.5 (win)

Filtrul Elo/Markov gap si hold asymmetry pot fi testate direct.
Filtrul S1 TB→S2 pattern necesita date Set 2 (nu sunt in backtest CSV).
"""

import pandas as pd
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("simulations/WTA/backtests/wta_predictions.csv")

# ── Preprocess ─────────────────────────────────────────────────────────────
df["elo_markov_gap"] = (df["p_elo"] - df["p_markov"]).abs() * 100
df["hold_asym"]      = (df["p_hold_w"] - df["p_hold_l"]).abs()
df["has_elo"]        = df["p_elo"] > 0
df["u125_win"]       = (df["y_tiebreak"] == 0).astype(int)   # proxy U12.5

# ── Per-surface baseline ───────────────────────────────────────────────────
print("=" * 70)
print("BASELINE — tiebreak rate per suprafata (fara niciun filtru)")
print("=" * 70)
for surf, g in df.groupby("surface"):
    print(f"  {surf:<8} {len(g):>5} meciuri  |  TB rate: {g['y_tiebreak'].mean():.1%}  |  U12.5 rate: {g['u125_win'].mean():.1%}")

# ── Focus GRASS ────────────────────────────────────────────────────────────
grass = df[df["surface"] == "Grass"].copy()
print(f"\nTotal Grass in backtest: {len(grass)} meciuri")
print(f"Grass baseline U12.5 rate (fara niciun filtru): {grass['u125_win'].mean():.1%}")

# ── Helper ─────────────────────────────────────────────────────────────────
def report(label, subset, baseline_n, baseline_hr):
    n   = len(subset)
    hr  = subset["u125_win"].mean() if n > 0 else 0
    pct = n / baseline_n * 100
    print(f"  {label:<45} n={n:>4} ({pct:4.1f}%)  HR={hr:.1%}  delta={hr-baseline_hr:+.1%}")

# ── FILTRU 1: Prag p_tiebreak (tb_p_cal) ──────────────────────────────────
print("\n" + "=" * 70)
print("FILTRU 1 — Prag p_tiebreak (model U12.5 signal)")
print("=" * 70)
thresholds = [0.05, 0.0781, 0.10, 0.127, 0.15, 0.20]
base_n  = len(grass)
base_hr = grass["u125_win"].mean()

for t in thresholds:
    sub = grass[grass["p_tiebreak"] <= t]
    report(f"p_tiebreak ≤ {t:.3f}", sub, base_n, base_hr)

# ── FILTRU 2: Elo/Markov gap ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("FILTRU 2 — Elo/Markov gap (double guard)")
print("=" * 70)
# La pragul nostru operational: p_tiebreak ≤ 0.127 (tb_p_cal ≤ 12.73%)
operational = grass[grass["p_tiebreak"] <= 0.127].copy()
op_n  = len(operational)
op_hr = operational["u125_win"].mean()
print(f"\nBaza: p_tiebreak ≤ 0.127 → {op_n} picks, HR={op_hr:.1%}")

for gap_thresh in [20, 25, 30, 35, 40, 50]:
    sub = operational[
        (operational["elo_markov_gap"] <= gap_thresh) &
        (operational["has_elo"])
    ]
    report(f"  + Elo/Markov gap ≤ {gap_thresh}pp & elo>0", sub, op_n, op_hr)

# ── FILTRU 3: Hold asymmetry ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("FILTRU 3 — Hold asymmetry (structural dominance)")
print("=" * 70)
print(f"\nBaza: p_tiebreak ≤ 0.127 → {op_n} picks, HR={op_hr:.1%}")

for asym_thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    sub = operational[operational["hold_asym"] >= asym_thresh]
    report(f"  + hold_asym ≥ {asym_thresh:.2f}", sub, op_n, op_hr)

# ── FILTRU COMBINAT: toate regulile ───────────────────────────────────────
print("\n" + "=" * 70)
print("FILTRU COMBINAT — Pasul 1 complet (gap ≤ 35pp + elo>0 + asym ≥ 0.08)")
print("=" * 70)
combined = operational[
    (operational["elo_markov_gap"] <= 35) &
    (operational["has_elo"]) &
    (operational["hold_asym"] >= 0.08)
]
print(f"\nBaza ({op_n} picks, HR={op_hr:.1%}) → Combinat ({len(combined)} picks):")
report("Elo/Markov ≤35pp + elo>0 + asym≥0.08", combined, op_n, op_hr)

# ── BREAKDOWN pe praguri diferite cu filtru combinat ──────────────────────
print("\n" + "=" * 70)
print("BREAKDOWN — Prag + Filtru Combinat (iarbă)")
print("=" * 70)
print(f"{'Prag tb':<12} {'Fara filtru':>20} {'Cu filtru':>20} {'Delta HR':>10} {'Picks elim.':>12}")
print("-" * 78)

for t in [0.05, 0.0781, 0.10, 0.127]:
    sub_raw = grass[grass["p_tiebreak"] <= t]
    sub_flt = sub_raw[
        (sub_raw["elo_markov_gap"] <= 35) &
        (sub_raw["has_elo"]) &
        (sub_raw["hold_asym"] >= 0.08)
    ]
    n_raw = len(sub_raw);  hr_raw = sub_raw["u125_win"].mean() if n_raw else 0
    n_flt = len(sub_flt);  hr_flt = sub_flt["u125_win"].mean() if n_flt else 0
    elim  = n_raw - n_flt
    print(f"≤ {t:.4f}     {n_raw:>5} picks  {hr_raw:.1%}     {n_flt:>5} picks  {hr_flt:.1%}     {(hr_flt-hr_raw)*100:>+5.1f}pp     {elim:>5} ({elim/n_raw*100:.0f}%)")

# ── ANALIZA p_elo = 0 (no Elo data) ────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALIZA — Impactul p_elo = 0 (jucatoare fara date Elo)")
print("=" * 70)
op_elo0   = operational[~operational["has_elo"]]
op_elo_ok = operational[operational["has_elo"]]
print(f"  p_tiebreak ≤ 0.127, p_elo = 0  : n={len(op_elo0):>4}  HR={op_elo0['u125_win'].mean():.1%}  ← fara Elo")
print(f"  p_tiebreak ≤ 0.127, p_elo > 0  : n={len(op_elo_ok):>4}  HR={op_elo_ok['u125_win'].mean():.1%}  ← cu Elo")

# ── ANALIZA gap mare vs mic ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALIZA — HR cu gap mare vs gap mic")
print("=" * 70)
op_elo_ok2 = operational[operational["has_elo"]].copy()
low_gap  = op_elo_ok2[op_elo_ok2["elo_markov_gap"] <= 20]
mid_gap  = op_elo_ok2[(op_elo_ok2["elo_markov_gap"] > 20) & (op_elo_ok2["elo_markov_gap"] <= 35)]
high_gap = op_elo_ok2[op_elo_ok2["elo_markov_gap"] > 35]
print(f"  Gap ≤ 20pp  : n={len(low_gap):>4}  HR={low_gap['u125_win'].mean():.1%}")
print(f"  Gap 20-35pp : n={len(mid_gap):>4}  HR={mid_gap['u125_win'].mean():.1%}")
print(f"  Gap > 35pp  : n={len(high_gap):>4}  HR={high_gap['u125_win'].mean():.1%}  ← ar fi SKIP cu regula noua")

# ── NOTA despre S1 TB → S2 pattern ─────────────────────────────────────────
print("\n" + "=" * 70)
print("NOTA — S1 TB → S2 pattern")
print("=" * 70)
print("  Backtest-ul curent nu contine date Set 2 (doar Set 1 in wta_predictions.csv).")
print("  Pattern S1 TB → S2 a fost validat manual pe:")
print("    Navarro: 1/7 S2 TB dupa S1 TB pe iarba (14%) → confirma U12.5")
print("    Bondar:  0/3 S2 TB dupa S1 TB pe iarba (0%)  → confirma U12.5")
print("    Lys:     0/4 S2 TB dupa S1 TB pe iarba (0%)  → confirma U12.5")
print("    Kalinskaya: 2/3 = 67% → ar fi redus Kraus/Kal de la 8/10 la 6/10")
print("  Pentru backtest complet: necesita adaugare coloana y_tiebreak_s2 in pipeline.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
