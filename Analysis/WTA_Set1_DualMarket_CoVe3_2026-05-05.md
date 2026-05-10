# CoVe Analysis — WTA Set 1 OVER 7.5
**Date:** 2026-05-05 | **Template:** 3.0.WTA_Set1_Over75.md v2.0
**Tournament:** Internazionali BNL d'Italia ROME, WTA 1000, Clay, Q2

---

## PRE-FILTER CHECKS

| Check | Status |
|-------|--------|
| Tournament NOT WTA 125/ITF | ✅ WTA 1000 Rome |
| Injury return >3 luni | ✅ Zheng revenit din feb 2026 (3+ luni activă) |
| p_cal_adj ≥ 80% | ✅ 84.53% |
| blowout_score < 8 | ✅ blowout = 4 |
| competitive_set = True OR expected_games > 23 | ✅ ambele |

---

## MODEL DATA

| Metric | Value |
|--------|-------|
| player_a | Anna Bondar |
| player_b | Qinwen Zheng |
| p_hold_a (Bondar) | 0.6465 |
| p_hold_b (Zheng) | 0.6685 |
| p_markov | 0.4562 (Bondar wins 45.6% points) |
| p_elo | 0.4017 |
| expected_games | **24.53** |
| p_cal_adj | **84.53%** |
| blowout_score | **4** |
| competitive_set | **True** |
| elite_pick | False |

---

## SCORING (/10)

**A. MODEL QUALITY (+2/3)**
- elite_pick = False → 0
- p_cal_adj = 84.53% ≥ 83% → +1
- competitive_set = True + expected_games = 24.53 > 24 → +1

**B. RANKING MATCH (+2/3)**
- Bondar: **#63** | Zheng: **#36**
- Ambele în #30-#80 → +2
- Gap = 27 → sub 50 → no penalty

**C. MATCHUP DYNAMICS (+1/2)**
- Bondar = clay grinder/specialist (career 57% WTA clay, beat Svitolina în Madrid 2026)
- Zheng = power baseliner pe clay
- Both clay specialists on clay → +1

**D. TOURNAMENT FORM (+2/2)**
- Bondar Q1 Rome: beat Pliskova **7-6(5)**, 6-2 → Set 1 = TB = 13 jocuri = Over 7.5 ✅
- Zheng Q1 Rome: beat Cornet **6-3**, 7-6(2) → Set 1 = 9 jocuri = Over 7.5 ✅
- Ambele produced Over 7.5 → +2

**E. INTUITION (+1/1)**
- H2H: Zheng 2-0, dar ultimul meci (Rome): 7-6, 6-4 → set 1 TB = 13 jocuri
- Bondar în formă maximă pe clay 2026 (7-2 clay record, beat Svitolina)
- Zheng revenită post-accidentare cot, încă adaptată → profil meci lung
- +1

**Penalizări:**
- elite_pick = False → -1

**TOTAL: 2 + 2 + 1 + 2 + 1 − 1 = 7/10**

---

## EV CHECK

Score = 7/10 → P_real = p_cal_adj − 3pp = **84.53% − 3% = 81.5%**

| Cotă | EV | Verdict |
|------|----|---------|
| 1.30 | (0.815 × 1.30) − 1 = **+5.9%** | ✅ STRONG BET |
| 1.25 | (0.815 × 1.25) − 1 = **+1.9%** | ❌ PASS (7/10 necesită ≥ 1.30) |
| 1.20 | (0.815 × 1.20) − 1 = **−2.2%** | ❌ PASS |

Per template: Score 7/10 → necesită cotă ≥ 1.30 + EV ≥ +3%.

---

## 🎾 PICK: Anna Bondar vs Qinwen Zheng — Set 1 O7.5

**CONDIȚIONAT de cotă ≥ 1.30**

```
Score: 7/10 | P_real: 81.5% | Fair odds: 1.227
Tournament: ROME WTA 1000, Q2 (Clay)
```

**Why bet:**
1. Model: p_cal_adj 84.5%, blowout=4, competitive_set=True, expected 24.53 jocuri
2. Ambele au produs Over 7.5 în Q1 azi: Bondar 7-6 vs Pliskova, Zheng 6-3 vs Cornet
3. H2H ultimul meci = 7-6 6-4 (set 1 TB = 13 jocuri)
4. Bondar clay specialist în formă maximă (#63), Zheng #36 revenită post-op elbow
5. EV +5.9% la cotă 1.30 ✅

**Why I lose:**
Zheng decide să termine rapid — serviciu puternic + forehand dominant, ia break dublu rapid, scor 6-1 sau 6-2. Probabilitate ~18% dat de forma Zheng post-accidentare impredictibilă.

---

## SUMMARY TABLE

| Market | Pick | Score | p_cal_adj | p_research | Cotă necesară | Action |
|--------|------|-------|-----------|------------|--------------|--------|
| Set 1 Over 7.5 | **Bondar vs Zheng** (ROME Q2) | **7/10** | 84.53% | **~81.5%** | **≥ 1.30** | ✅ BET dacă cotă ≥ 1.30 |
| Set 1 Under 12.5 | — | — | — | — | — | ❌ 0 picks |

---

*Analysis: 2026-05-05 | Template WTA CoVe v2.0 | Tournament: ROME WTA 1000 Qualifying*

Sources:
- [Tennis Tonic — Zheng vs Bondar Rome](https://tennistonic.com/tennis-news/992993/how-to-watch-zheng-vs-bondar-on-live-streaming-in-rome-on-tuesday/)
- [PB Tennis — Bondar clay form context](https://x.com/Probahis/status/2046392320271302800)
- [Action Network — Zheng vs Bondar odds](https://www.actionnetwork.com/tennis/wta-rome-odds-picks-expert-betting-predictions-for-kalinina-vs-kenin-and-zheng-vs-bondar-may-13)
