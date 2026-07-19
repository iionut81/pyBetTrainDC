"""
Backtest: does Glicko-2 (volatility-driven adaptive updates) predict
post-hot-streak matches better than our production SurfaceElo, which has
a fixed/decaying K-factor that's slow to react for experienced players
regardless of how "surprising" recent results are (the Badosa-type case)?

Walk-forward, no lookahead: for every match, predict with BOTH systems
using only pre-match state, then update both, then record whether either
player was "hot" (>=4 wins in her last 5 matches) going into this match.
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import glicko2

from wta_elo import SurfaceElo

HOT_STREAK_MIN_WINS = 4   # out of last 5
LAST_N = 5

df = pd.read_csv("data/historical/wta_matches_combined.csv", low_memory=False)
df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
df = df.dropna(subset=["match_date", "winner_id", "loser_id", "surface"]).copy()
df["winner_id"] = df["winner_id"].astype(int)
df["loser_id"] = df["loser_id"].astype(int)

def surface_bucket(s):
    s = str(s)
    return s if s in ("Hard", "Clay", "Grass") else "Other"

df["surf"] = df["surface"].apply(surface_bucket)
df = df[df["surf"] != "Other"].sort_values("match_date").reset_index(drop=True)
df["w_sets"] = pd.to_numeric(df.get("w_sets"), errors="coerce").fillna(0).astype(int)
df["l_sets"] = pd.to_numeric(df.get("l_sets"), errors="coerce").fillna(0).astype(int)
df["total_games"] = pd.to_numeric(df.get("total_games"), errors="coerce").fillna(0).astype(int)

elo = SurfaceElo()
glicko_players = {}  # (player_id, surf) -> glicko2.Player

def get_glicko(pid, surf):
    key = (pid, surf)
    if key not in glicko_players:
        glicko_players[key] = glicko2.Player()
    return glicko_players[key]

Q = math.log(10) / 400.0

def glicko_expect(p_a, p_b):
    """Standard Glicko expected-score formula, 1500-scale ratings/RD."""
    g_rd_b = 1.0 / math.sqrt(1.0 + 3.0 * Q**2 * p_b.rd**2 / math.pi**2)
    return 1.0 / (1.0 + 10.0 ** (-g_rd_b * (p_a.rating - p_b.rating) / 400.0))

recent_results = defaultdict(lambda: deque(maxlen=LAST_N))  # player_id -> deque of 1/0

records = []

for _, row in df.iterrows():
    w_id, l_id, surf = row["winner_id"], row["loser_id"], row["surf"]
    date = row["match_date"]

    # ── hot-streak flag from PRE-match history ──
    w_hist = recent_results[w_id]
    l_hist = recent_results[l_id]
    w_hot = len(w_hist) == LAST_N and sum(w_hist) >= HOT_STREAK_MIN_WINS
    l_hot = len(l_hist) == LAST_N and sum(l_hist) >= HOT_STREAK_MIN_WINS
    involves_hot = w_hot or l_hot

    # ── predictions BEFORE update (no leakage) ──
    p_elo = elo.predict(w_id, l_id, surf)

    gp_w = get_glicko(w_id, surf)
    gp_l = get_glicko(l_id, surf)
    p_gli = glicko_expect(gp_w, gp_l)

    if p_elo is not None:
        records.append({
            "date": date, "involves_hot": involves_hot,
            "p_elo": p_elo, "p_gli": p_gli, "outcome": 1,  # winner's perspective, outcome=1 always
        })

    # ── update Elo (production system) ──
    elo.update(w_id, l_id, surf, w_sets=row["w_sets"], l_sets=row["l_sets"],
               total_games=row["total_games"], match_date=date)

    # ── update Glicko-2 (one match = one rating period, simplification) ──
    gp_w.update_player([gp_l.rating], [gp_l.rd], [1])
    gp_l.update_player([gp_w.rating], [gp_w.rd], [0])

    # ── roll recent-results history AFTER this match ──
    recent_results[w_id].append(1)
    recent_results[l_id].append(0)

rec = pd.DataFrame(records)
rec = rec.dropna(subset=["p_elo", "p_gli"])

def brier(p, y):
    return np.mean((p - y) ** 2)

def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print(f"Total matches with predictions from both systems: {len(rec):,}")
print(f"  of which involve a 'hot' player (>=4/5 wins going in): {rec['involves_hot'].sum():,} "
      f"({rec['involves_hot'].mean()*100:.1f}%)")
print()

for label, sub in [("ALL matches", rec), ("Hot-streak matches", rec[rec["involves_hot"]]),
                   ("Non-hot matches", rec[~rec["involves_hot"]])]:
    if len(sub) == 0:
        continue
    print(f"=== {label} (n={len(sub):,}) ===")
    print(f"  Elo    : Brier={brier(sub['p_elo'], sub['outcome']):.4f}  LogLoss={logloss(sub['p_elo'], sub['outcome']):.4f}  mean_p={sub['p_elo'].mean():.3f}")
    print(f"  Glicko2: Brier={brier(sub['p_gli'], sub['outcome']):.4f}  LogLoss={logloss(sub['p_gli'], sub['outcome']):.4f}  mean_p={sub['p_gli'].mean():.3f}")
    print()
