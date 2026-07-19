"""
Backtest: does double-fault rate (and first-serve %) actually increase in
high-pressure rounds (Final, Semifinal) compared to earlier rounds -
WITHIN THE SAME PLAYER (paired comparison, so skill level cancels out)?
"""
import pandas as pd
import numpy as np

df = pd.read_csv("data/historical/wta_matches_combined.csv", low_memory=False)
df = df.dropna(subset=["round", "w_df", "l_df", "w_SvGms", "l_SvGms"])

EARLY = {"R128", "R64", "R32", "R16"}
LATE_QF = {"QF"}
LATE_SF = {"SF"}
LATE_F = {"F"}

def bucket(r):
    if r in EARLY:
        return "Early (R128-R16)"
    if r in LATE_QF:
        return "QF"
    if r in LATE_SF:
        return "SF"
    if r in LATE_F:
        return "F (Final)"
    return None

df["bucket"] = df["round"].apply(bucket)
df = df[df["bucket"].notna()]

w_rows = df[["winner_name", "bucket", "w_df", "w_SvGms", "w_1stServeIn_pct", "surface"]].copy()
w_rows.columns = ["player", "bucket", "df_count", "SvGms", "first_serve_in", "surface"]
l_rows = df[["loser_name", "bucket", "l_df", "l_SvGms", "l_1stServeIn_pct", "surface"]].copy()
l_rows.columns = ["player", "bucket", "df_count", "SvGms", "first_serve_in", "surface"]
long_df = pd.concat([w_rows, l_rows], ignore_index=True)
long_df = long_df.dropna(subset=["df_count", "SvGms"])

print("=== UNPAIRED (all players pooled) — DF per match by round ===")
agg = long_df.groupby("bucket")["df_count"].agg(["mean", "count"])
agg = agg.reindex(["Early (R128-R16)", "QF", "SF", "F (Final)"])
print(agg)
print()

print("=== UNPAIRED — DF per service game by round (normalizes for match length) ===")
long_df["df_per_svgm"] = long_df["df_count"] / long_df["SvGms"]
agg2 = long_df.groupby("bucket")["df_per_svgm"].agg(["mean", "count"])
agg2 = agg2.reindex(["Early (R128-R16)", "QF", "SF", "F (Final)"])
print(agg2)
print()

print("=== UNPAIRED — First Serve In % by round ===")
long_df_fs = long_df.dropna(subset=["first_serve_in"])
agg3 = long_df_fs.groupby("bucket")["first_serve_in"].agg(["mean", "count"])
agg3 = agg3.reindex(["Early (R128-R16)", "QF", "SF", "F (Final)"])
print(agg3)
print()

# ---- PAIRED WITHIN-PLAYER COMPARISON: Final vs Early rounds ----
print("=== PAIRED WITHIN-PLAYER: DF/match rate, Final vs Early rounds (same player, both buckets, min 3 matches each) ===")
MIN_MATCHES = 3

early = long_df[long_df["bucket"] == "Early (R128-R16)"].groupby("player").agg(
    early_df_mean=("df_count", "mean"), early_n=("df_count", "size"),
    early_svgm_mean=("df_per_svgm", "mean"))
finals = long_df[long_df["bucket"] == "F (Final)"].groupby("player").agg(
    final_df_mean=("df_count", "mean"), final_n=("df_count", "size"),
    final_svgm_mean=("df_per_svgm", "mean"))

paired = early.join(finals, how="inner")
paired = paired[(paired["early_n"] >= MIN_MATCHES) & (paired["final_n"] >= MIN_MATCHES)]
paired["diff_df_per_match"] = paired["final_df_mean"] - paired["early_df_mean"]
paired["diff_df_per_svgm"] = paired["final_svgm_mean"] - paired["early_svgm_mean"]

print(f"Players qualifying for paired comparison (>= {MIN_MATCHES} matches in both Early and Final rounds): {len(paired)}")
print(f"Mean DF/match  — Early rounds: {paired['early_df_mean'].mean():.3f}  |  Finals: {paired['final_df_mean'].mean():.3f}")
print(f"Mean within-player DIFFERENCE (Final - Early), DF/match: {paired['diff_df_per_match'].mean():+.3f}")
print(f"Mean within-player DIFFERENCE (Final - Early), DF/service-game: {paired['diff_df_per_svgm'].mean():+.4f}")
print(f"Players with HIGHER DF rate in finals than early rounds: {(paired['diff_df_per_match'] > 0).mean()*100:.1f}%")
print(f"Players with LOWER DF rate in finals than early rounds:  {(paired['diff_df_per_match'] < 0).mean()*100:.1f}%")

from scipy import stats
t_stat, p_val = stats.ttest_rel(paired["final_df_mean"], paired["early_df_mean"])
print(f"\nPaired t-test: t={t_stat:.3f}, p-value={p_val:.4f}")
