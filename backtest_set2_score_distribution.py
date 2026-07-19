"""
Faza 1 din propunere: nu te uita doar la P(Under 12.5) agregat - descompune
in scorurile componente (6-0/6-1, 6-2/6-3, 6-4, 7-5, 7-6) si vezi daca
probabilitatea "sanatoasa" (dominata de scoruri comode) creste cu hold_asym,
sau daca vine mai ales din 7-5 (aproape-tiebreak).

Reuseste exact trailing hold rate din backtest_set1_vs_set2_u125.py.
"""
import re
import numpy as np
import pandas as pd

MIN_SVGMS = 40

df = pd.read_csv("data/historical/wta_matches_combined.csv", low_memory=False)
df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
df = df.dropna(subset=["match_date", "surface", "w_SvGms", "l_SvGms",
                        "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved", "score"])

df["breaks_conceded_winner"] = df["w_bpFaced"] - df["w_bpSaved"]
df["breaks_conceded_loser"] = df["l_bpFaced"] - df["l_bpSaved"]
df = df[(df["breaks_conceded_winner"] >= 0) & (df["breaks_conceded_loser"] >= 0)]

def surface_bucket(s):
    s = str(s)
    return s if s in ("Hard", "Clay", "Grass") else "Other"

df["surf"] = df["surface"].apply(surface_bucket)
df = df[df["surf"] != "Other"].reset_index(drop=True)
df["match_idx"] = df.index

SET_RE = re.compile(r"^(\d+)-(\d+)(\(\d+\))?$")

def get_set2_bucket(score):
    if not isinstance(score, str):
        return None
    sets = score.strip().split()
    if len(sets) < 2:
        return None
    m = SET_RE.match(sets[1].strip())
    if not m:
        return None
    g1, g2 = int(m.group(1)), int(m.group(2))
    total = g1 + g2
    hi, lo = max(g1, g2), min(g1, g2)
    if (hi, lo) in [(7, 6), (6, 7)]:
        return "7-6 (TB)"
    if total > 13 or total < 6:
        return None  # malformed
    if (hi, lo) == (7, 5):
        return "7-5"
    if (hi, lo) == (6, 4):
        return "6-4"
    if (hi, lo) in [(6, 2), (6, 3)]:
        return "6-2/6-3"
    if (hi, lo) in [(6, 0), (6, 1)]:
        return "6-0/6-1"
    return None

df["set2_bucket"] = df["score"].apply(get_set2_bucket)

# ---- trailing hold rate (identical methodology to prior scripts) ----
w_rows = df[["match_idx", "winner_name", "surf", "match_date", "w_SvGms", "breaks_conceded_winner"]].copy()
w_rows.columns = ["match_idx", "player", "surf", "date", "SvGms", "breaks_conceded"]
w_rows["role"] = "winner"
l_rows = df[["match_idx", "loser_name", "surf", "match_date", "l_SvGms", "breaks_conceded_loser"]].copy()
l_rows.columns = ["match_idx", "player", "surf", "date", "SvGms", "breaks_conceded"]
l_rows["role"] = "loser"
long_df = pd.concat([w_rows, l_rows], ignore_index=True)
long_df = long_df.sort_values(["player", "surf", "date"]).reset_index(drop=True)

g = long_df.groupby(["player", "surf"])
long_df["SvGms_shifted"] = g["SvGms"].shift(1).fillna(0)
long_df["breaks_shifted"] = g["breaks_conceded"].shift(1).fillna(0)
long_df["SvGms_cum_prior"] = long_df.groupby(["player", "surf"])["SvGms_shifted"].cumsum()
long_df["breaks_cum_prior"] = long_df.groupby(["player", "surf"])["breaks_shifted"].cumsum()
long_df["trailing_hold"] = 1.0 - (long_df["breaks_cum_prior"] / long_df["SvGms_cum_prior"].replace(0, np.nan))

pivot_hold = long_df.pivot(index="match_idx", columns="role", values="trailing_hold")
pivot_n = long_df.pivot(index="match_idx", columns="role", values="SvGms_cum_prior")

df = df.set_index("match_idx")
df["w_trail_hold"] = pivot_hold["winner"]
df["l_trail_hold"] = pivot_hold["loser"]
df["w_trail_n"] = pivot_n["winner"]
df["l_trail_n"] = pivot_n["loser"]
df = df.reset_index()

bt = df[(df["w_trail_n"] >= MIN_SVGMS) & (df["l_trail_n"] >= MIN_SVGMS)].copy()
bt["hold_asym"] = (bt["w_trail_hold"] - bt["l_trail_hold"]).abs()
bt["min_hold"] = bt[["w_trail_hold", "l_trail_hold"]].min(axis=1)
bt = bt.dropna(subset=["set2_bucket"])

ORDER = ["6-0/6-1", "6-2/6-3", "6-4", "7-5", "7-6 (TB)"]

print(f"Total matches with valid set2 bucket + trailing sample: {len(bt):,}\n")

bins = [0, 0.03, 0.06, 0.10, 0.15, 0.20, 1.0]
labels = ["0-3pp", "3-6pp", "6-10pp", "10-15pp", "15-20pp", "20pp+"]
bt["asym_bucket"] = pd.cut(bt["hold_asym"], bins=bins, labels=labels)

print("=== Distributia scorurilor Set 2, pe bucket de hold_asym (% din total) ===\n")
header = f"{'asym':10s}" + "".join(f"{o:>12s}" for o in ORDER) + f"{'n':>10s}" + f"{'U12.5':>10s}" + f"{'sanatos%':>10s}"
print(header)
for b in labels:
    gg = bt[bt["asym_bucket"] == b]
    if len(gg) == 0:
        continue
    dist = gg["set2_bucket"].value_counts(normalize=True) * 100
    row = f"{b:10s}"
    for o in ORDER:
        row += f"{dist.get(o, 0):11.1f}%"
    u125 = 100 - dist.get("7-6 (TB)", 0)
    healthy_share = (dist.get("6-0/6-1", 0) + dist.get("6-2/6-3", 0)) / u125 * 100 if u125 > 0 else 0
    row += f"{len(gg):10,}" + f"{u125:9.1f}%" + f"{healthy_share:9.1f}%"
    print(row)

print()
print("=== Aceeasi descompunere, doar pentru filtrul premium_u125-style (min_hold<0.50 & hold_asym>0.15) ===\n")
mask = (bt["min_hold"] < 0.50) & (bt["hold_asym"] > 0.15)
gg = bt[mask]
dist = gg["set2_bucket"].value_counts(normalize=True) * 100
for o in ORDER:
    print(f"  {o:10s}: {dist.get(o, 0):5.1f}%")
u125 = 100 - dist.get("7-6 (TB)", 0)
healthy_share = (dist.get("6-0/6-1", 0) + dist.get("6-2/6-3", 0)) / u125 * 100 if u125 > 0 else 0
borderline_share = dist.get("7-5", 0) / u125 * 100 if u125 > 0 else 0
print(f"\n  n={len(gg):,}  P(Under12.5)={u125:.1f}%")
print(f"  Din care 'sanatos' (6-0/6-1/6-2/6-3): {healthy_share:.1f}% din masa Under12.5")
print(f"  Din care 'la limita' (7-5, aproape TB): {borderline_share:.1f}% din masa Under12.5")
