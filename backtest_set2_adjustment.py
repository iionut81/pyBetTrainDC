"""
Backtest: adjust_p_hold_set2 — impact pe U12.5 Set 2
======================================================
Verifică dacă ajustarea hold rates bazată pe scorul Set 1
îmbunătățește predicția TB în Set 2.

Sursa date: data/historical/wta_matches_combined.csv
    → coloana 'score' conține scorurile per set (ex: "7-6(3) 6-2")
    → coloana 'surface' pentru filtrare iarbă
    → coloane de serviciu pentru estimare hold rates

Logica:
1. Parsăm scorul Set 1 din 'score'
2. Determinăm dacă Set 2 a avut TB (actual outcome)
3. Calculăm hold rates din stats de serviciu din meci
4. Rulăm p_set_tiebreak cu hold-uri BASE vs AJUSTATE
5. Comparăm acuratețea predicției pentru Set 2
"""

import re
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from wta_markov import game_win_prob, p_set_tiebreak

# ── Config ─────────────────────────────────────────────────────────────────
CSV_PATH   = "data/historical/wta_matches_combined.csv"
SURFACE    = "Grass"          # None = toate suprafețele
TB_THRESH  = 0.10             # prag operational U12.5 (tb_p_cal ≤ 10%)
MIN_SERVE_GAMES = 4           # minim game-uri de serviciu per jucătoare

# ── Funcție ajustare (cu fix TB 7-6) ───────────────────────────────────────
def adjust_p_hold_set2(p_A, p_B, s1_score):
    """Ajustează hold rates pentru Set 2 pe baza scorului Set 1."""
    gA, gB = s1_score
    diff   = gA - gB
    aA = aB = 0.0

    if gA == 7 and gB == 6:          # A câștigă TB
        aA, aB = +0.04, -0.06
    elif gA == 6 and gB == 7:         # B câștigă TB
        aA, aB = -0.06, +0.04
    elif abs(diff) >= 2 and (gA == 7 or gB == 7):  # 7-5
        if diff > 0:
            aA, aB = +0.03, -0.05
        else:
            aA, aB = -0.05, +0.03
    elif abs(diff) >= 4:              # Blowout 6-0 / 6-1 / 6-2
        if diff > 0:
            aA, aB = -0.04, +0.02
        else:
            aA, aB = +0.02, -0.04

    return (
        max(0.1, min(0.9, p_A * (1 + aA))),
        max(0.1, min(0.9, p_B * (1 + aB))),
    )

# ── Parser scor ─────────────────────────────────────────────────────────────
def parse_sets(score_str):
    """
    Extrage lista de (winner_games, loser_games) per set.
    Exemple: "6-2 6-4" → [(6,2),(6,4)]
             "7-6(3) 6-1" → [(7,6),(6,1)]
             "6-3 3-6 6-4" → [(6,3),(3,6),(6,4)]
    Returnează [] dacă scorul e invalid / RET.
    """
    if not isinstance(score_str, str) or score_str.strip() == "":
        return []
    # Excludem meciuri abandonate
    if any(x in score_str.upper() for x in ("RET", "W/O", "DEF", "ABN", "UNF")):
        return []
    sets = []
    for tok in score_str.strip().split():
        m = re.match(r"^(\d+)-(\d+)(?:\(\d+\))?$", tok)
        if m:
            sets.append((int(m.group(1)), int(m.group(2))))
    return sets if len(sets) >= 2 else []

def set2_is_tb(s2_score):
    """True dacă Set 2 s-a terminat la tiebreak (7-6)."""
    gW, gL = s2_score
    return (gW == 7 and gL == 6) or (gW == 6 and gL == 7)

# ── Estimare hold rate din statistici serviciu ──────────────────────────────
def hold_from_stats(svpt, first_in, first_won, second_won, sv_gms):
    """
    Estimează game-level hold rate din statistici punct de serviciu.
    p_serve_point → game_win_prob (Markov exact).
    """
    if pd.isna(svpt) or svpt < 1 or pd.isna(sv_gms) or sv_gms < MIN_SERVE_GAMES:
        return None
    first_in  = float(first_in)  if not pd.isna(first_in)  else 0
    first_won = float(first_won) if not pd.isna(first_won) else 0
    second_won= float(second_won)if not pd.isna(second_won)else 0
    svpt      = float(svpt)

    p1_in   = first_in / svpt
    p1_won  = first_won / first_in if first_in > 0 else 0
    p2_won  = second_won / (svpt - first_in) if (svpt - first_in) > 0 else 0
    p_point = p1_in * p1_won + (1 - p1_in) * p2_won

    if not 0.3 < p_point < 0.85:
        return None
    return game_win_prob(p_point)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    if SURFACE:
        df = df[df["surface"] == SURFACE].copy()
        print(f"Surface filter '{SURFACE}': {len(df)} matches")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        sets = parse_sets(row.get("score", ""))
        if len(sets) < 2:
            skipped += 1
            continue

        s1, s2 = sets[0], sets[1]

        # Hold rates din statistici
        # WINNER → "A" (servo first in match = winner perspective in Sackmann)
        p_hold_w = hold_from_stats(
            row.get("w_svpt"), row.get("w_1stIn"), row.get("w_1stWon"),
            row.get("w_2ndWon"), row.get("w_SvGms")
        )
        p_hold_l = hold_from_stats(
            row.get("l_svpt"), row.get("l_1stIn"), row.get("l_1stWon"),
            row.get("l_2ndWon"), row.get("l_SvGms")
        )
        if p_hold_w is None or p_hold_l is None:
            skipped += 1
            continue

        # Predicție BASE
        tb_base = p_set_tiebreak(p_hold_w, p_hold_l)

        # Ajustare Set 1 → predicție AJUSTATĂ
        # NOTE: în Sackmann, winner a câștigat deci s1 = (w_games, l_games)
        # "A" în funcție = winner (jucătoare cu hold_w)
        p_adj_w, p_adj_l = adjust_p_hold_set2(p_hold_w, p_hold_l, s1)
        tb_adj  = p_set_tiebreak(p_adj_w, p_adj_l)

        # Outcome real Set 2
        actual_tb_s2 = 1 if set2_is_tb(s2) else 0

        # Tip Set 1 (pentru analiză)
        gW, gL = s1
        if (gW == 7 and gL == 6) or (gW == 6 and gL == 7):
            s1_type = "TB"
        elif abs(gW - gL) >= 4:
            s1_type = "Blowout"
        elif (gW == 7 or gL == 7):
            s1_type = "7-5"
        else:
            s1_type = "Normal"

        records.append({
            "surface":      row.get("surface"),
            "match_date":   row.get("match_date"),
            "winner":       row.get("winner_name"),
            "loser":        row.get("loser_name"),
            "score":        row.get("score"),
            "s1_score":     f"{s1[0]}-{s1[1]}",
            "s1_type":      s1_type,
            "s2_score":     f"{s2[0]}-{s2[1]}",
            "actual_tb_s2": actual_tb_s2,
            "p_hold_w":     round(p_hold_w, 4),
            "p_hold_l":     round(p_hold_l, 4),
            "tb_base":      round(tb_base, 4),
            "tb_adj":       round(tb_adj, 4),
            "delta_tb":     round(tb_adj - tb_base, 4),
            "p_adj_w":      round(p_adj_w, 4),
            "p_adj_l":      round(p_adj_l, 4),
        })

    print(f"Processed: {len(records)} matches | Skipped: {skipped}")
    res = pd.DataFrame(records)

    # ── ANALIZĂ GLOBALĂ ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ANALIZĂ GLOBALĂ — Set 2 TB rate per tip Set 1")
    print("="*70)
    grp = res.groupby("s1_type").agg(
        n          = ("actual_tb_s2", "count"),
        tb_rate    = ("actual_tb_s2", "mean"),
        tb_base    = ("tb_base", "mean"),
        tb_adj     = ("tb_adj", "mean"),
    ).round(4)
    print(grp.to_string())
    print(f"\nTotal Set 2 TB rate: {res['actual_tb_s2'].mean():.1%}")

    # ── ANALIZĂ U12.5 SIGNAL (tb_base ≤ threshold) ──────────────────────────
    print("\n" + "="*70)
    print(f"ANALIZĂ U12.5 — predicții cu tb_base ≤ {TB_THRESH}")
    print("="*70)
    sig = res[res["tb_base"] <= TB_THRESH].copy()
    sig_adj = sig[sig["tb_adj"] <= TB_THRESH].copy()

    if len(sig) == 0:
        print("Niciun pick la threshold.")
    else:
        hr_base = 1 - sig["actual_tb_s2"].mean()
        hr_adj  = 1 - sig_adj["actual_tb_s2"].mean() if len(sig_adj) > 0 else 0
        print(f"\n  Picks base (tb_base ≤ {TB_THRESH}):     n={len(sig):>5}  HR={hr_base:.1%}")
        print(f"  Picks adj  (tb_adj  ≤ {TB_THRESH}):     n={len(sig_adj):>5}  HR={hr_adj:.1%}")
        print(f"  Picks eliminate de adj:               n={len(sig)-len(sig_adj):>5}")
        print(f"  HR delta:                             +{(hr_adj-hr_base)*100:.2f}pp")

        # Per tip Set 1
        print(f"\n  Breakdown per tip Set 1:")
        for s1t in ["TB", "7-5", "Normal", "Blowout"]:
            sub = sig[sig["s1_type"] == s1t]
            if len(sub) == 0:
                continue
            sub_adj = sub[sub["tb_adj"] <= TB_THRESH]
            hr_b = 1 - sub["actual_tb_s2"].mean()
            hr_a = 1 - sub_adj["actual_tb_s2"].mean() if len(sub_adj) > 0 else float("nan")
            print(f"    {s1t:<10}: n={len(sub):>4}  HR_base={hr_b:.1%}  "
                  f"n_adj={len(sub_adj):>4}  HR_adj={hr_a:.1%}")

    # ── AJUSTARE — câte picks ELIMINATING/CONFIRMING ─────────────────────────
    print("\n" + "="*70)
    print("IMPACT AJUSTARE — direcție delta")
    print("="*70)
    adj_up   = res[res["delta_tb"] > 0.005]
    adj_down = res[res["delta_tb"] < -0.005]
    adj_none = res[res["delta_tb"].abs() <= 0.005]
    print(f"  Ajustare UP   (adj mai riscant):  n={len(adj_up):>5}  TB real={adj_up['actual_tb_s2'].mean():.1%}")
    print(f"  Ajustare DOWN (adj mai sigur):    n={len(adj_down):>5}  TB real={adj_down['actual_tb_s2'].mean():.1%}")
    print(f"  Fără ajustare:                    n={len(adj_none):>5}  TB real={adj_none['actual_tb_s2'].mean():.1%}")

    # ── SALVARE ───────────────────────────────────────────────────────────────
    out_path = "simulations/WTA/backtests/backtest_set2_adjustment.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSalvat: {out_path} ({len(res)} rânduri)")
    print("="*70)

if __name__ == "__main__":
    main()
