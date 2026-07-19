"""
Fair rematch: Glicko-2 WITH the same 50% surface + 50% overall blend that our
production SurfaceElo already uses for predictions (SURFACE_BLEND=0.50).
Also extracts Badosa's Glicko-2 vs Sackmann-Elo trajectory to check whether
Glicko-2 closes the GAP Elo/Markov we've flagged repeatedly all week.
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import glicko2

from wta_elo import SurfaceElo

HOT_STREAK_MIN_WINS = 4
LAST_N = 5
SURFACE_BLEND = 0.50  # match production Elo's blend weight

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
gli_surf = {}   # (player_id, surf) -> glicko2.Player
gli_all = {}    # player_id -> glicko2.Player (cross-surface, like Elo's __ALL__)

def get_gli_surf(pid, surf):
    key = (pid, surf)
    if key not in gli_surf:
        gli_surf[key] = glicko2.Player()
    return gli_surf[key]

def get_gli_all(pid):
    if pid not in gli_all:
        gli_all[pid] = glicko2.Player()
    return gli_all[pid]

Q = math.log(10) / 400.0

def g_factor(rd):
    return 1.0 / math.sqrt(1.0 + 3.0 * Q**2 * rd**2 / math.pi**2)

def glicko_expect_blended(pid_a, pid_b, surf):
    sa, sb = get_gli_surf(pid_a, surf), get_gli_surf(pid_b, surf)
    aa, ab = get_gli_all(pid_a), get_gli_all(pid_b)
    r_a = SURFACE_BLEND * sa.rating + (1 - SURFACE_BLEND) * aa.rating
    r_b = SURFACE_BLEND * sb.rating + (1 - SURFACE_BLEND) * ab.rating
    rd_b = SURFACE_BLEND * sb.rd + (1 - SURFACE_BLEND) * ab.rd
    g = g_factor(rd_b)
    return 1.0 / (1.0 + 10.0 ** (-g * (r_a - r_b) / 400.0)), r_a, r_b

recent_results = defaultdict(lambda: deque(maxlen=LAST_N))
records = []
badosa_trace = []
BADOSA_ID = None
name_lookup = {}

for _, row in df.iterrows():
    w_id, l_id, surf = row["winner_id"], row["loser_id"], row["surf"]
    date = row["match_date"]
    name_lookup[w_id] = row["winner_name"]
    name_lookup[l_id] = row["loser_name"]
    if str(row["winner_name"]).strip().lower() == "paula badosa":
        BADOSA_ID = w_id
    if str(row["loser_name"]).strip().lower() == "paula badosa":
        BADOSA_ID = l_id

    w_hist, l_hist = recent_results[w_id], recent_results[l_id]
    w_hot = len(w_hist) == LAST_N and sum(w_hist) >= HOT_STREAK_MIN_WINS
    l_hot = len(l_hist) == LAST_N and sum(l_hist) >= HOT_STREAK_MIN_WINS
    involves_hot = w_hot or l_hot

    p_elo = elo.predict(w_id, l_id, surf)
    p_gli, r_a, r_b = glicko_expect_blended(w_id, l_id, surf)

    if p_elo is not None:
        records.append({"date": date, "involves_hot": involves_hot,
                         "p_elo": p_elo, "p_gli": p_gli, "outcome": 1})

    if w_id == BADOSA_ID or l_id == BADOSA_ID:
        is_w = (w_id == BADOSA_ID)
        badosa_trace.append({
            "date": date, "opponent": row["loser_name"] if is_w else row["winner_name"],
            "result": "W" if is_w else "L",
            "surface": surf,
            "badosa_glicko_blend": r_a if is_w else r_b,
        })

    elo.update(w_id, l_id, surf, w_sets=row["w_sets"], l_sets=row["l_sets"],
               total_games=row["total_games"], match_date=date)

    sw, sl = get_gli_surf(w_id, surf), get_gli_surf(l_id, surf)
    aw, al = get_gli_all(w_id), get_gli_all(l_id)
    sw.update_player([sl.rating], [sl.rd], [1]); sl.update_player([sw.rating], [sw.rd], [0])
    aw.update_player([al.rating], [al.rd], [1]); al.update_player([aw.rating], [aw.rd], [0])

    recent_results[w_id].append(1)
    recent_results[l_id].append(0)

rec = pd.DataFrame(records).dropna(subset=["p_elo", "p_gli"])

def brier(p, y): return np.mean((p - y) ** 2)
def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print(f"Total matches: {len(rec):,}  |  hot-streak involved: {rec['involves_hot'].sum():,} ({rec['involves_hot'].mean()*100:.1f}%)\n")
for label, sub in [("ALL", rec), ("Hot-streak", rec[rec["involves_hot"]]), ("Non-hot", rec[~rec["involves_hot"]])]:
    print(f"=== {label} (n={len(sub):,}) ===")
    print(f"  Elo (blended, production)  : Brier={brier(sub['p_elo'], sub['outcome']):.4f}  LogLoss={logloss(sub['p_elo'], sub['outcome']):.4f}")
    print(f"  Glicko2 (blended, fair)    : Brier={brier(sub['p_gli'], sub['outcome']):.4f}  LogLoss={logloss(sub['p_gli'], sub['outcome']):.4f}")
    print()

print("=" * 70)
print(f"BADOSA TRACE (player_id={BADOSA_ID}) — last 15 matches, blended Glicko rating")
print("=" * 70)
bt = pd.DataFrame(badosa_trace).tail(15)
print(bt.to_string(index=False))
