"""
Backtest: does the pre-match hold-rate asymmetry (same signal already used for
Winner / Under12.5 Set2) reliably predict WHO MAKES MORE BREAKS in a match?

Methodology:
- Build a long-format (match_idx, role, player, surface, date, SvGms, breaks_conceded)
  table from both winner and loser rows.
- For each player+surface, compute a TRAILING (shifted, no lookahead) cumulative
  hold rate = 1 - breaks_conceded_cum / SvGms_cum, using only matches strictly
  before the current one (vectorized groupby+shift+cumsum, no positional slicing).
- Require a minimum trailing sample (SvGms_cum >= MIN_SVGMS) for BOTH players.
- Predicted "more breaks maker" = player with the HIGHER trailing hold rate
  (her opponent, with the weaker serve, concedes more breaks -> she makes more).
- Actual "more breaks maker" = compare real breaks made by winner vs loser this match.
- Report hit rate overall and stratified by hold_asym bucket and by surface.
"""
import pandas as pd
import numpy as np

MIN_SVGMS = 40  # ~5-6 matches worth of service games, trailing sample floor

df = pd.read_csv("data/historical/wta_matches_combined.csv", low_memory=False)
df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
df = df.dropna(subset=["match_date", "surface", "w_SvGms", "l_SvGms",
                        "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"])

df["breaks_by_winner"] = df["l_bpFaced"] - df["l_bpSaved"]   # winner broke loser this many times
df["breaks_by_loser"] = df["w_bpFaced"] - df["w_bpSaved"]    # loser broke winner this many times
df["breaks_conceded_winner"] = df["breaks_by_loser"]         # winner's own serve broken this many times
df["breaks_conceded_loser"] = df["breaks_by_winner"]

df = df[(df["breaks_by_winner"] >= 0) & (df["breaks_by_loser"] >= 0)]

def surface_bucket(s):
    s = str(s)
    return s if s in ("Hard", "Clay", "Grass") else "Other"

df["surf"] = df["surface"].apply(surface_bucket)
df = df[df["surf"] != "Other"].reset_index(drop=True)
df["match_idx"] = df.index

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

# Merge back by explicit match_idx + role (no positional assumptions)
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

bt["pred_winner_makes_more"] = bt["w_trail_hold"] > bt["l_trail_hold"]
bt["actual_winner_makes_more"] = bt["breaks_by_winner"] > bt["breaks_by_loser"]
bt["actual_tie"] = bt["breaks_by_winner"] == bt["breaks_by_loser"]

bt_no_tie = bt[~bt["actual_tie"]].copy()
bt_no_tie["hit"] = bt_no_tie["pred_winner_makes_more"] == bt_no_tie["actual_winner_makes_more"]

print(f"Total historical matches (usable columns): {len(df):,}")
print(f"Matches with sufficient trailing sample (>= {MIN_SVGMS} SvGms both players): {len(bt):,}")
print(f"  of which non-tie (breaks_by_winner != breaks_by_loser): {len(bt_no_tie):,}")
print(f"  tie rate (equal breaks both sides): {bt['actual_tie'].mean()*100:.1f}%")
print()
print(f"OVERALL hit rate (predict who makes more breaks): {bt_no_tie['hit'].mean()*100:.2f}%")
print()

print("By surface:")
for surf, gg in bt_no_tie.groupby("surf"):
    print(f"  {surf:6s}: n={len(gg):6,}  HR={gg['hit'].mean()*100:.2f}%")
print()

print("By hold_asym bucket:")
bins = [0, 0.03, 0.06, 0.10, 0.15, 0.20, 1.0]
labels = ["0-3pp", "3-6pp", "6-10pp", "10-15pp", "15-20pp", "20pp+"]
bt_no_tie["asym_bucket"] = pd.cut(bt_no_tie["hold_asym"], bins=bins, labels=labels)
for bucket, gg in bt_no_tie.groupby("asym_bucket"):
    if len(gg) > 0:
        print(f"  {str(bucket):8s}: n={len(gg):6,}  HR={gg['hit'].mean()*100:.2f}%")
print()

bt_no_tie["pred_winner_wins_match"] = bt_no_tie["w_trail_hold"] > bt_no_tie["l_trail_hold"]
print(f"Sanity check - higher trailing hold rate == actual match winner: {bt_no_tie['pred_winner_wins_match'].mean()*100:.2f}%")
print(f"  (winner_name always sits in the 'winner' column by construction, so this checks")
print(f"   whether the player with the better trailing hold rate was the one who won)")
