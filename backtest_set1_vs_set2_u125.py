"""
Backtest: is "Under 12.5 games" (= no tiebreak) more stable/higher-HR in Set 1
than in Set 2, and how does it behave across hold_asym / min_hold buckets?

Uses the SAME trailing (no-lookahead) hold-rate methodology already validated
in backtest_breaks_market.py.
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

# ---- Parse score into set1/set2 tiebreak status ----
SET_RE = re.compile(r"^(\d+)-(\d+)(\(\d+\))?$")

def parse_sets(score):
    if not isinstance(score, str):
        return []
    sets = score.strip().split()
    return sets

def set_has_tb(set_str):
    m = SET_RE.match(set_str.strip())
    if not m:
        return None  # malformed / retirement marker etc.
    g1, g2, tb = int(m.group(1)), int(m.group(2)), m.group(3)
    if tb is not None:
        return True
    # no explicit tiebreak marker: 7-6/6-7 without parens (rare/old data) still counts as TB
    if (g1, g2) in [(7, 6), (6, 7)]:
        return True
    return False

def get_set_n_tb(score, n):
    sets = parse_sets(score)
    if len(sets) < n:
        return None
    return set_has_tb(sets[n - 1])

df["set1_tb"] = df["score"].apply(lambda s: get_set_n_tb(s, 1)).astype("boolean")
df["set2_tb"] = df["score"].apply(lambda s: get_set_n_tb(s, 2)).astype("boolean")

# ---- Trailing hold rate per player+surface (same method as backtest_breaks_market.py) ----
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

print(f"Matches with sufficient trailing sample: {len(bt):,}")
print()

# ---- Overall Set1 vs Set2 "no tiebreak" base rate ----
s1 = bt.dropna(subset=["set1_tb"])
s2 = bt.dropna(subset=["set2_tb"])
print(f"=== OVERALL 'no tiebreak' rate (= Under 12.5) ===")
print(f"Set 1: n={len(s1):,}  P(no TB) = {(~s1['set1_tb']).mean()*100:.2f}%")
print(f"Set 2: n={len(s2):,}  P(no TB) = {(~s2['set2_tb']).mean()*100:.2f}%")
print()

# ---- By hold_asym bucket ----
bins = [0, 0.03, 0.06, 0.10, 0.15, 0.20, 1.0]
labels = ["0-3pp", "3-6pp", "6-10pp", "10-15pp", "15-20pp", "20pp+"]

print("=== Set 1 — P(no TB) by hold_asym bucket ===")
s1 = s1.copy()
s1["bucket"] = pd.cut(s1["hold_asym"], bins=bins, labels=labels)
for b, gg in s1.groupby("bucket"):
    if len(gg) > 0:
        print(f"  {str(b):8s}: n={len(gg):6,}  P(no TB)={(~gg['set1_tb']).mean()*100:.2f}%")
print()

print("=== Set 2 — P(no TB) by hold_asym bucket ===")
s2 = s2.copy()
s2["bucket"] = pd.cut(s2["hold_asym"], bins=bins, labels=labels)
for b, gg in s2.groupby("bucket"):
    if len(gg) > 0:
        print(f"  {str(b):8s}: n={len(gg):6,}  P(no TB)={(~gg['set2_tb']).mean()*100:.2f}%")
print()

# ---- Premium-style filter: min_hold < 0.50 AND hold_asym > 0.15 (mirrors premium_u125 def) ----
premium_mask_s1 = (s1["min_hold"] < 0.50) & (s1["hold_asym"] > 0.15)
premium_mask_s2 = (s2["min_hold"] < 0.50) & (s2["hold_asym"] > 0.15)
print("=== 'premium_u125-style' filter (min_hold<0.50 & hold_asym>0.15) ===")
print(f"Set 1: n={premium_mask_s1.sum():,}  P(no TB) = {(~s1.loc[premium_mask_s1, 'set1_tb']).mean()*100:.2f}%")
print(f"Set 2: n={premium_mask_s2.sum():,}  P(no TB) = {(~s2.loc[premium_mask_s2, 'set2_tb']).mean()*100:.2f}%")
print()

# ---- By surface, premium filter ----
print("=== Premium filter, by surface ===")
for surf in ["Hard", "Clay", "Grass"]:
    m1 = premium_mask_s1 & (s1["surf"] == surf)
    m2 = premium_mask_s2 & (s2["surf"] == surf)
    print(f"  {surf:6s} Set1: n={m1.sum():5,} P(no TB)={(~s1.loc[m1,'set1_tb']).mean()*100:.2f}%   "
          f"Set2: n={m2.sum():5,} P(no TB)={(~s2.loc[m2,'set2_tb']).mean()*100:.2f}%")
