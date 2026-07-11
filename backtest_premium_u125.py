"""
backtest_premium_u125.py
Compara definitia curenta vs definitii noi pentru premium_u125.

premium_CURRENT:  hold_asym > 0.15 AND min_hold < 0.50 AND p_tb_cal < 0.12
premium_BLOWOUT:  min_hold < 0.45 AND hold_asym > 0.20 AND p_tb_cal < 0.08
premium_STRUCT:   min_hold >= 0.55 AND hold_asym > 0.15 AND p_tb_cal < 0.10
premium_STRICT:   min_hold >= 0.55 AND hold_asym > 0.20 AND p_tb_cal < 0.08
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

# ── Load ───────────────────────────────────────────────────────────────────────

def parse_s2_tb(score):
    score = re.sub(r'\s*(RET|W/O|DEF|Def\.?|Abd\.?).*', '', str(score), flags=re.I).strip()
    sets = score.split()
    if len(sets) < 2:
        return None
    m = re.match(r'^(\d+)-(\d+)(\(\d+\))?$', sets[1])
    if not m:
        return None
    return float(m.group(3) is not None)

hist  = pd.read_csv(ROOT / "data/historical/wta_matches_combined.csv", low_memory=False)
preds = pd.read_csv(ROOT / "simulations/WTA/backtests/wta_predictions.csv", low_memory=False)

hist["match_date"]  = pd.to_datetime(hist["match_date"])
preds["match_date"] = pd.to_datetime(preds["match_date"])
hist["s2_tb"]       = hist["score"].apply(parse_s2_tb)

# Join
df = pd.merge(
    preds[["match_date","surface","tourney_name","round",
           "winner_name","loser_name","p_tiebreak","p_hold_w","p_hold_l","p_elo"]],
    hist[["match_date","surface","winner_name","loser_name","s2_tb"]],
    on=["match_date","surface","winner_name","loser_name"],
    how="inner",
).dropna(subset=["s2_tb"])

df["hold_asym"] = (df["p_hold_w"] - df["p_hold_l"]).abs()
df["min_hold"]  = df[["p_hold_w","p_hold_l"]].min(axis=1)
df["max_hold"]  = df[["p_hold_w","p_hold_l"]].max(axis=1)

# ── Defineste tipurile de premium ──────────────────────────────────────────────

df["prem_current"] = (
    (df["hold_asym"] > 0.15) &
    (df["min_hold"]  < 0.50) &
    (df["p_tiebreak"]< 0.12)
)

df["prem_blowout"] = (
    (df["min_hold"]  < 0.45) &
    (df["hold_asym"] > 0.20) &
    (df["p_tiebreak"]< 0.08)
)

df["prem_struct"] = (
    (df["min_hold"]  >= 0.55) &
    (df["hold_asym"] > 0.15) &
    (df["p_tiebreak"]< 0.10)
)

df["prem_strict"] = (
    (df["min_hold"]  >= 0.55) &
    (df["hold_asym"] > 0.20) &
    (df["p_tiebreak"]< 0.08)
)

# ── Functii raport ─────────────────────────────────────────────────────────────

def report(label, mask, df, min_n=30):
    sub = df[mask]
    if len(sub) < min_n:
        return f"  {label:<52} N={len(sub):4d}   n/a (insuficient)"
    hr  = 1 - sub["s2_tb"].mean()
    tb  = int(sub["s2_tb"].sum())
    return f"  {label:<52} N={len(sub):4d}   HR={hr*100:.1f}%   TB={tb}"

def report_surf(label, mask, df, min_n=30):
    lines = [report(f"{label} (ALL)", mask, df, min_n)]
    for surf in ["Hard","Clay","Grass"]:
        smask = mask & (df["surface"] == surf)
        lines.append(report(f"  {label} | {surf}", smask, df, min_n))
    return "\n".join(lines)


# ── Print ──────────────────────────────────────────────────────────────────────

print("=" * 75)
print("PREMIUM_U125 — COMPARATIE DEFINITII")
print("=" * 75)
print(f"  Total meciuri matched: {len(df)}")
print(f"  Baseline S2 TB rate:   {df['s2_tb'].mean()*100:.1f}%   HR baseline: {(1-df['s2_tb'].mean())*100:.1f}%")
print()

print("─" * 75)
print("REFERINTA — fara niciun filtru premium")
print("─" * 75)
print(report_surf("p_tb < 0.098 (operational)", df["p_tiebreak"] < 0.098, df))
print()

print("─" * 75)
print("CURRENT: hold_asym>0.15 AND min_hold<0.50 AND p_tb<0.12")
print("─" * 75)
print(report_surf("premium_CURRENT", df["prem_current"], df))
print()

# Subset: current premium AND within operational threshold
curr_op = df["prem_current"] & (df["p_tiebreak"] < 0.098)
print(report_surf("premium_CURRENT + p_tb<0.098", curr_op, df))
print()

print("─" * 75)
print("BLOWOUT: min_hold<0.45 AND hold_asym>0.20 AND p_tb<0.08")
print("─" * 75)
print(report_surf("premium_BLOWOUT", df["prem_blowout"], df))
print()

print("─" * 75)
print("STRUCTURAL: min_hold>=0.55 AND hold_asym>0.15 AND p_tb<0.10")
print("─" * 75)
print(report_surf("premium_STRUCT", df["prem_struct"], df))
print()

print("─" * 75)
print("STRICT: min_hold>=0.55 AND hold_asym>0.20 AND p_tb<0.08")
print("─" * 75)
print(report_surf("premium_STRICT", df["prem_strict"], df))
print()

print("=" * 75)
print("ANALIZA min_hold buckets (la p_tb<0.098)")
print("=" * 75)
sub_op = df[df["p_tiebreak"] < 0.098]
print(f"\n  {'min_hold bucket':<25} {'N':>5}  {'HR%':>7}  {'TB losses':>10}")
print("  " + "-" * 55)
buckets = [
    ("<0.40",   sub_op["min_hold"] < 0.40),
    ("0.40-0.45", (sub_op["min_hold"]>=0.40) & (sub_op["min_hold"]<0.45)),
    ("0.45-0.50", (sub_op["min_hold"]>=0.45) & (sub_op["min_hold"]<0.50)),
    ("0.50-0.55", (sub_op["min_hold"]>=0.50) & (sub_op["min_hold"]<0.55)),
    ("0.55-0.60", (sub_op["min_hold"]>=0.55) & (sub_op["min_hold"]<0.60)),
    ("0.60-0.65", (sub_op["min_hold"]>=0.60) & (sub_op["min_hold"]<0.65)),
    (">=0.65",   sub_op["min_hold"] >= 0.65),
]
for label, mask in buckets:
    s = sub_op[mask]
    if len(s) < 10:
        print(f"  {label:<25} {len(s):>5}  n/a")
        continue
    hr = 1 - s["s2_tb"].mean()
    print(f"  {label:<25} {len(s):>5}  {hr*100:>6.1f}%  {int(s['s2_tb'].sum()):>10}")

print()
print("=" * 75)
print("ANALIZA hold_asym buckets (la p_tb<0.098)")
print("=" * 75)
print(f"\n  {'hold_asym bucket':<25} {'N':>5}  {'HR%':>7}  {'TB losses':>10}")
print("  " + "-" * 55)
ab = [
    ("0.00-0.10", (sub_op["hold_asym"]>=0.00) & (sub_op["hold_asym"]<0.10)),
    ("0.10-0.15", (sub_op["hold_asym"]>=0.10) & (sub_op["hold_asym"]<0.15)),
    ("0.15-0.20", (sub_op["hold_asym"]>=0.15) & (sub_op["hold_asym"]<0.20)),
    ("0.20-0.25", (sub_op["hold_asym"]>=0.20) & (sub_op["hold_asym"]<0.25)),
    ("0.25-0.30", (sub_op["hold_asym"]>=0.25) & (sub_op["hold_asym"]<0.30)),
    (">=0.30",    sub_op["hold_asym"] >= 0.30),
]
for label, mask in ab:
    s = sub_op[mask]
    if len(s) < 10:
        print(f"  {label:<25} {len(s):>5}  n/a")
        continue
    hr = 1 - s["s2_tb"].mean()
    print(f"  {label:<25} {len(s):>5}  {hr*100:>6.1f}%  {int(s['s2_tb'].sum()):>10}")

print()
print("=" * 75)
print("COMBINATII: min_hold bucket X hold_asym la p_tb<0.098 (Clay)")
print("=" * 75)
sub_clay = df[(df["p_tiebreak"] < 0.098) & (df["surface"] == "Clay")]
print(f"\n  {'Combinatie':<40} {'N':>5}  {'HR%':>7}")
print("  " + "-" * 55)
combos = [
    ("min<0.50 + asym>0.15 (current)",
     (sub_clay["min_hold"]<0.50) & (sub_clay["hold_asym"]>0.15)),
    ("min<0.45 + asym>0.20 (blowout)",
     (sub_clay["min_hold"]<0.45) & (sub_clay["hold_asym"]>0.20)),
    ("min>=0.55 + asym>0.15 (struct)",
     (sub_clay["min_hold"]>=0.55) & (sub_clay["hold_asym"]>0.15)),
    ("min>=0.55 + asym>0.20 (strict)",
     (sub_clay["min_hold"]>=0.55) & (sub_clay["hold_asym"]>0.20)),
    ("min>=0.60 + asym>0.15",
     (sub_clay["min_hold"]>=0.60) & (sub_clay["hold_asym"]>0.15)),
    ("min 0.50-0.60 + asym>0.15",
     (sub_clay["min_hold"]>=0.50) & (sub_clay["min_hold"]<0.60) & (sub_clay["hold_asym"]>0.15)),
]
for label, mask in combos:
    s = sub_clay[mask]
    if len(s) < 10:
        print(f"  {label:<40} {len(s):>5}  n/a")
        continue
    hr = 1 - s["s2_tb"].mean()
    print(f"  {label:<40} {len(s):>5}  {hr*100:>6.1f}%  TB={int(s['s2_tb'].sum())}")

print()
print("Done.")
