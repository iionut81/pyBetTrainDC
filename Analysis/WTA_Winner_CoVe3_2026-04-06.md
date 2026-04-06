# CoVe WTA Match Winner 3.2 — 2026-04-06
## Linz WTA 500 (Clay, R1) + Madrid 125 (Clay, R1)

---

## Match 1: Polina Kudermetova vs Mayar Sherif (Madrid 125, Clay)

### Model Block
- **Chosen side:** Mayar Sherif (predicted_winner = player_b)
- p_hold_a = 0.5398, p_hold_b = 0.6994
- chosen_hold = 0.6994, opponent_hold = 0.5398
- **hold_diff = 0.1596** ✅ (>0.15)
- p_markov_side = 1 - 0.1889 = **0.8111** ✅
- p_elo_side = 1 - 0.3383 = **0.6617** ✅
- p_cal_side = **0.7545** ✅ (>0.65)
- fair_odds = 1.3254
- market_odds: **unavailable**

**Hard filter:**
- p_cal_side ≥ 0.65? ✅ (0.7545)
- hold_diff ≥ 0.10? ✅ (0.1596)
- At least 2 of 3 agree? Markov 0.811 ✅, Elo 0.662 ✅, Cal 0.755 ✅ → **3/3** ✅
- Contradiction? Markov 0.811 vs Elo 0.662 → no contradiction (both above thresholds)
- **HARD FILTER: PASS** ✅

**Volatility flag:** opponent_hold = 0.5398 > 0.50 → no flag
**Hold quality:** chosen_hold 0.6994 (strong), hold_diff 0.16 (dominant profile)

**Base classification:**
- p_cal_side ≥ 0.70? ✅
- hold_diff ≥ 0.15? ✅
- p_markov_side ≥ 0.70? ✅
- → **BET** (pre-research)

### Research Block
- **Sherif 2026:** 7-6 overall, **6-3 on clay**. Two SF runs (Antalya, Dubrovnik). Clay is clearly her best surface. Ranked #102. — [PB Tennis](https://x.com/Probahis/status/2038938046112870511)
- **Kudermetova:** Lost to Fernandez 6-2, 6-1 at Charleston (got demolished). Required medical timeout. Poor clay form.
- **Surface:** Clay — Sherif's best surface. Career-high #31 as highest-ranked Egyptian ever.
- **Matchup:** Sherif's clay court grinding vs Kudermetova's inconsistency. Sherif has clay volume advantage.
- **Red flags:** None for Sherif. Kudermetova's health concern from Charleston is a red flag for her.
- **Data source:** tennisabstract/sackmann → slightly less reliable but model signals are strong (3/3 agree)

**Net research adjustment:** +0.03 (Sherif strong clay form, Kudermetova poor form)
**research_adjusted_probability = 0.7545 + 0.03 = 0.7845**

### Verdict Block
- **Pick:** Mayar Sherif
- **Model probability:** 75.4%
- **Research-adjusted probability:** **78.5%**
- **Fair odds:** 1.27
- **Price check:** unavailable
- **Action: VALUE ONLY** (strong structural play, but WTA 125 level + no price verification)
- **Reason:** 3/3 model signals agree. Sherif 6-3 on clay in 2026, dominant hold gap 0.16. Kudermetova demolished at Charleston.
- **How I lose:** Kudermetova raises her level on a fresh start at a new tournament, while Sherif's WTA 125 level opponents haven't tested her for bigger stages.
- **Sources:** [WTA Official](https://www.wtatennis.com/players/318711/mayar-sherif), [PB Tennis](https://x.com/Probahis/status/2038938046112870511)

---

## Match 2: Nuria Brancaccio vs Jessika Ponchet (Madrid 125, Clay)

### Model Block
- **Chosen side:** Jessika Ponchet (predicted_winner = player_b)
- p_hold_a = 0.5098, p_hold_b = 0.6369
- chosen_hold = 0.6369, opponent_hold = 0.5098
- **hold_diff = 0.1271** ✅ (>0.10)
- p_markov_side = 1 - 0.2456 = **0.7544** ✅
- p_elo_side = 1 - 0.5074 = **0.4926** ❌ (<0.55)
- p_cal_side = **0.6518** ✅ (barely above 0.65)
- fair_odds = 1.5342
- market_odds: **unavailable**

**Hard filter:**
- p_cal_side ≥ 0.65? ✅ (0.6518, barely)
- hold_diff ≥ 0.10? ✅ (0.1271)
- At least 2 of 3? Markov 0.754 ✅, Elo 0.493 ❌, Cal 0.652 ✅ → **2/3** ✅
- Contradiction? Markov 0.754 vs Elo 0.493 → gap 0.26. Markov > 0.70 and Elo < 0.45? No (Elo 0.493 > 0.45). No contradiction.
- **HARD FILTER: PASS** ✅ (but marginal)

**Volatility flag:** opponent_hold = 0.5098 > 0.50 → borderline, no formal flag
**Hold quality:** chosen_hold 0.637 (OK), hold_diff 0.127 (moderate, not dominant)

**Base classification:**
- p_cal_side ≥ 0.70? ❌ (0.652)
- → **VALUE ONLY** (pre-research)

### Research Block
- Both lower-ranked players at WTA 125 level. Limited research available.
- Elo disagrees with Markov → mixed signal.
- p_cal barely above threshold (0.652) → weak edge.

**Net research adjustment:** +0.00 (insufficient data)
**research_adjusted_probability = 0.6518**

### Verdict Block
- **Pick:** Jessika Ponchet
- **Model probability:** 65.2%
- **Research-adjusted probability:** 65.2%
- **Fair odds:** 1.53
- **Price check:** unavailable
- **Action: PASS**
- **Reason:** Marginal p_cal (0.652), Elo disagrees (0.493), WTA 125 level unreliable. Not enough edge.
- **How I lose:** N/A (PASS)

---

## Match 3: Katie Boulter vs Elena-Gabriela Ruse (Linz WTA 500, Clay)

### Model Block
- **Chosen side:** Elena-Gabriela Ruse (predicted_winner = player_b)
- p_hold_a = 0.5877, p_hold_b = 0.6430
- chosen_hold = 0.6430, opponent_hold = 0.5877
- **hold_diff = 0.0553** ❌ (<0.10)
- p_markov_side = 1 - 0.3834 = **0.6166** ✅
- p_elo_side = 1 - 0.4069 = **0.5931** ✅
- p_cal_side = **0.6088**
- fair_odds = 1.6426
- market_odds: **unavailable**

**Hard filter:**
- p_cal_side ≥ 0.65? ❌ (0.6088)
- **HARD FILTER: FAIL**

### Verdict Block
- **Action: PASS**
- **Reason:** p_cal_side 0.609 < 0.65 threshold. hold_diff 0.055 < 0.10. Too close to call. Boulter 11-5 in 2026 with title in Ostrava; Ruse 9-8. Both 0-0 on clay this year — first clay match of season for both. Neither has demonstrated clay form yet.
- **Sources:** [Tennis Tonic](https://tennistonic.com/tennis-news/981662/h2h-prediction-of-katie-boulter-vs-elena-gabriela-ruse-in-linz-with-odds-preview-pick-6th-april-2026/)

---

## Match 4: Francesca Jones vs Elizabeth Mandlik (Madrid 125, Clay)

### Model Block
- **Chosen side:** Francesca Jones (predicted_winner = player_a)
- p_hold_a = 0.6407, p_hold_b = 0.5741
- chosen_hold = 0.6407, opponent_hold = 0.5741
- **hold_diff = 0.0666** ❌ (<0.10)

**Hard filter:**
- hold_diff ≥ 0.10? ❌ (0.0666)
- **HARD FILTER: FAIL**

### Verdict Block
- **Action: PASS**
- **Reason:** hold_diff 0.067 < 0.10. Too balanced. p_cal 0.603 well below 0.65.

---

## Match 5: Sloane Stephens vs Tatjana Maria (Linz WTA 500, Clay)

### Model Block
- **Chosen side:** Sloane Stephens (predicted_winner = player_a)
- p_hold_a = 0.6259, p_hold_b = 0.5954
- chosen_hold = 0.6259, opponent_hold = 0.5954
- **hold_diff = 0.0305** ❌ (<0.10)

**Hard filter:**
- hold_diff ≥ 0.10? ❌ (0.0305)
- **HARD FILTER: FAIL**

### Verdict Block
- **Action: PASS**
- **Reason:** hold_diff 0.031 < 0.10. Essentially a coin flip. p_cal 0.548 well below threshold.

---

## Self-Verification

| Question | Answer |
|----------|--------|
| Did any match pass the hard filter? | Yes — Kudermetova/Sherif and Brancaccio/Ponchet |
| Is the chosen side stable? | Sherif yes (0.699 hold, 6-3 clay). Ponchet marginal. |
| Can the opponent create chaos? | Kudermetova can (but poor form). Brancaccio maybe. |
| Does matchup support model? | Sherif/Kudermetova yes (clay specialist vs struggling player). |
| Is there a real upset pattern? | Kudermetova had medical timeout at Charleston — not reliable. |
| Can chosen side lose 4 games in a row? | Sherif on clay — unlikely. Ponchet — possible. |

---

## SUMMARY

| Match | Side | p_cal | Hard Filter | Action |
|-------|------|-------|-------------|--------|
| Kudermetova vs **Sherif** | Sherif | 75.4% | ✅ PASS | **VALUE ONLY** |
| Brancaccio vs Ponchet | Ponchet | 65.2% | ✅ PASS (marginal) | **PASS** |
| Boulter vs Ruse | Ruse | 60.9% | ❌ FAIL | **PASS** |
| Jones vs Mandlik | Jones | 60.3% | ❌ FAIL | **PASS** |
| Stephens vs Maria | Stephens | 54.8% | ❌ FAIL | **PASS** |

### Today's pick: **Mayar Sherif to beat Polina Kudermetova — VALUE ONLY**

Strong structural play (3/3 model signals, dominant hold gap, clay specialist) but WTA 125 level limits confidence and no market odds available for price verification.

---

*CoVe WTA Winner 3.2 complete. 1 VALUE ONLY pick, 4 PASS. Hard filter correctly eliminated 3 matches with tiny hold gaps.*
