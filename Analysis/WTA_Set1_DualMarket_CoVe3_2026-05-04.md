# CoVe Analysis — WTA Set 1 Dual Market (Under 12.5 + Over 7.5)
**Date:** 2026-05-04 | **Template:** 1.0.3.WTA_over7.5_Under12.5.md v3.2
**Matches submitted:** 8 (toate ROME WTA 1000, Clay)
**Turneu:** Internazionali BNL d'Italia, Qualifying Round

---

## PRE-FILTER — TOȚI 8 CANDIDAȚI

### Market A: Under 12.5 Set 1

| Meci | hold_a | hold_b | Hold Gap | Gap≥0.12? | Pass? |
|------|--------|--------|----------|-----------|-------|
| Townsend vs Sramkova | 0.693 | 0.667 | 0.026 | ❌ | PASS — ambele >0.65 (Step A HARD PASS) |
| Potapova vs Begu | 0.604 | 0.652 | 0.048 | ❌ | PASS — gap <0.12 |
| Korneeva vs Seidel | 0.588 | 0.577 | 0.011 | ❌ | PASS — gap <0.12 |
| Blinkova vs Sherif | 0.591 | 0.592 | 0.001 | ❌ | PASS — UNSTABLE |
| Salkova vs Kraus | 0.475 | 0.496 | 0.021 | ❌ | PASS — gap <0.12 |
| Tomljanovic vs Lombardini | 0.570 | 0.553 | 0.017 | ❌ | PASS — gap <0.12 |
| Sasnovich vs Grabher | 0.673 | 0.599 | 0.074 | ❌ | PASS — gap <0.12 |
| Tagger vs Lamens | 0.541 | 0.544 | 0.003 | ❌ | PASS — UNSTABLE |

**Under 12.5: 0 picks.** Toate eșuează — fie ambele holds prea mari (Townsend-Sramkova), fie gap-ul hold prea mic pentru a genera breaks suficiente într-un singur sens.

---

### Market B: Over 7.5 Set 1 — Pre-filter

| Meci | hold_a | hold_b | blowout | p_cal_adj | Rec | Verdict |
|------|--------|--------|---------|-----------|-----|---------|
| **Townsend vs Sramkova** | 0.693 | 0.667 | **2** ✅ | **85.6%** ✅ | ✅ | 🔥 CANDIDAT |
| Potapova vs Begu | 0.604 | 0.652 | **5** ⚠️ | 82.0% ⚠️ | ❌ | ⚠️ VERIFICAT→PASS |
| Korneeva vs Seidel | 0.588 | 0.577 | **7** ❌ | 81.0% | ❌ | PASS (blowout≥7, p<82%) |
| Blinkova vs Sherif | 0.591 | 0.592 | **7** ❌ | 80.9% | ❌ | PASS (UNSTABLE + blowout) |
| Salkova vs Kraus | 0.475 | 0.496 | **7** ❌ | 80.6% | ❌ | PASS (blowout=7 + p<82%) |
| Tomljanovic vs Lombardini | 0.570 | 0.553 | **7** ❌ | 80.6% | ❌ | PASS (blowout=7 + p<82%) |
| Sasnovich vs Grabher | 0.673 | 0.599 | **5** ⚠️ | 80.4% | ❌ | PASS (p<82%) |
| Tagger vs Lamens | 0.541 | 0.544 | **7** ❌ | 80.4% | ❌ | PASS (UNSTABLE + blowout) |

---

## VERIFICARE POTAPOVA vs BEGU (eliminare confirmată)

**Context:** Potapova #38, Begu #184.
- Potapova: finală + semifinală în ultimele 2 turnee pe clay. A bătut Rybakina. În formă maximă.
- Begu: revenită după pauză voluntară de **1 an**. Madrid 2026: pierdut un set **0-6** față de Siegemund.

**Step A Over 7.5:** "Obvious mismatch → PASS"
- Ranking gap: #38 vs #184
- p_markov=0.3991 → Potapova câștigă 60% din puncte prin Markov
- Begu formă actuală (0-6 recent) <<< hold_b=0.6517 historical
- Gap efectiv >> 0.20 → **HARD PASS automat pe Over 7.5**

**Concluzie Potapova-Begu: PASS** — Potapova în formă + Begu dezintegrată = breadstick risc real.

---

## FULL CoVe — SINGURUL CANDIDAT VALID

### ✅ Taylor Townsend vs Rebecca Sramkova (ROME WTA 1000, Clay, Qualifying)

#### Step 1 — Model Data
| Metric | Value |
|--------|-------|
| p_hold_a (Townsend) | **0.6929** |
| p_hold_b (Sramkova) | **0.6672** |
| p_markov | 0.5670 (Townsend favorit) |
| p_elo | 0.5251 (Townsend ușor favorit) |
| expected_games | **24.48** |
| p_cal | 0.8564 |
| p_cal_adj | **85.6%** |
| blowout_score | **2** (minim risc) |
| competitive_set | **True** |
| elite_pick | False |
| recommended | ✅ True |

#### Market B — Over 7.5 Checklist

**STEP A — Anti-blowout:**
- Townsend hold=0.693 (> 0.45) ✅
- Sramkova hold=0.667 (> 0.45) ✅
- Gap = 0.026 (< 0.20) ✅ — nu e mismatch
- blowout_score=2 = model confirmă explicitcompetitive match
- **PASS ✅**

**STEP B — Hold Stability:**
- Ambele holds > 0.65 → 🔥 **PREMIUM** (template: "Both > 0.60 → Premium")
- Expected_games=24.48 → model confirmă many holds on both sides
- **PASS ✅**

**STEP C — Matchup Balance:**
- "Can both realistically reach 2-3 holds?" → BOTH hold >0.65 → DA, fiecare va ține 4-5 servicii
- Set likely: 6-3, 6-4, 7-5 sau 7-6 TB = minimum 9 games
- **VALID ✅**

**STEP D — Surface:**
- Clay → ⚠️ Downgrade per template. Apply -0.5 la scor.
- Dar: expected_games=24.48 pe clay compensează (nu teren rapid care taie punctele)
- Clay WTA 1000 qualifying = context valid per template (nu WTA 125/ITF)

**STEP E — Slow Starter Filter:**
- Townsend: ultimul meci Madrid: pierdut 4-6, 2-6 la Boulter (set 1 = 10 jocuri, pierdut)
- Dar: Pattern "pierdut Set 1 în 3 din 5 meciuri cu oponenți similari"? Insuficiente date → **+0pp** (ignorat per regulă)

**STEP F (echivalent Over) — Blowout Risk:**
- blowout_score=2 este cel mai mic din toate cele 8 meciuri
- competitive_set=True → model validat
- Nicio accidentare/oboseală identificată pentru niciunul

#### Step 2 — Research Context

**Taylor Townsend (#90 WTA):**
- 2026 overall: 9W-4L (69.2%)
- Serve: **5.15 ace/meci** — serviciu agresiv confirmat
- Hold=0.693 consistent cu 66.9% first serve wins
- Ultimul meci: pierdut lui Boulter 4-6, 2-6 → dar Boulter este mult mai sus în clasament
- Clay form: mediocru la Madrid (eliminat R1) → vine după înfrângere
- Qualifying Rome = motivată să avanseze

**Rebecca Sramkova (#122 WTA):**
- Recentă: a câștigat la Selekhmeteva pe clay (27 aprilie 2026)
- Miami qualifying: a câștigat 6-4, 6-2 vs Andreeva
- Activă pe clay în lunile recente (Challenger + qualifying)
- Hold=0.667 = consistent cu performanțele recente

**H2H:** Prima întâlnire directă între cele două. Nicio datorie H2H.

**Injury/Fatigue:** Nicio informație despre accidentare sau oboseală pentru niciunul.

**Scheduling:** Ora 18:00 ROME (seară) → condiții normale clay, nu căldură extremă.

#### Scoring (/10)

| Criteriu | Evaluare | Punctaj |
|----------|----------|---------|
| Hold Structure (ambele >0.65) | 🔥 PREMIUM | 3/3 |
| Matchup Fit (competitive, 24.48 expected) | ✅ Perfect balance | 2/2 |
| Gap Quality (gap mic = echilibrat) | ✅ Ideal Over 7.5 | 1.5/2 |
| Context (clay -0.5, qualifying -0.5) | ⚠️ Slight downgrade | 1/2 |
| Gut/Analyst | Model confirmat, blowout=2 | 0.5/1 |
| **TOTAL** | | **8/10** |

**Clay penalty aplicat → MODERATE (nu HIGH)**

#### Step 6 — Research Probability
| Factor | Constatare | pp |
|--------|------------|---|
| Surface clay | downgrade per template | −1 |
| Hold premium (ambele >0.65) | both can hold 4-5 times | +2 |
| Qualifying context | ușor mai volatil | −1 |
| blowout=2, competitive=True | model confirmat | +2 |
| No injury/fatigue | full fitness | +0 |
| **TOTAL** | | **+2pp** |

**p_research: 85.6% + 2pp = ~87%**

#### Final Verdict

**OVER 7.5 SET 1: MODERATE BET — 8/10**

**How I REALISTICALLY lose:**
Townsend servește prost în primele game-uri (5.15 ace/meci dar sub medie azi), Sramkova prinde ritm și face break dublu rapid → scor 6-2 sau 6-3 → sub 7.5 total. Pe clay, break-urile rapide sunt mai frecvente când una servește greoi. Probabilitate pierdere: ~13%.

---

## SUMMARY TABLE

| Market | Pick | Score | p_cal_adj | p_research | Confidence | Action |
|--------|------|-------|-----------|-----------|-----------|--------|
| Set 1 **Over 7.5** | **Townsend vs Sramkova** (ROME) | **8/10** | 85.6% | **~87%** | **MODERATE** | ✅ BET |
| Set 1 Under 12.5 | — | — | — | — | — | ❌ 0 picks |

---

## SELF-VERIFICATION

- [x] Analizat AMBELE piețe pentru toți 8 candidați ✅
- [x] Under 12.5: 0 picks — toate eșuează pe hold gap sau holds prea mari ✅
- [x] Over 7.5: Townsend-Sramkova singurul valid (blowout=2, competitive, p=85.6%) ✅
- [x] Potapova-Begu: PASS confirmat (Begu 0-6 recent, ranking gap masiv, blowout risc) ✅
- [x] Blowout_score≥7 + p_cal_adj<82% = PASS automat pentru 6 din 8 meciuri ✅
- [x] Clay surface penalty aplicat (−1pp) ✅
- [x] Qualifying context notat (−1pp) ✅
- [x] Cap +10pp respectat (aplicat +2pp) ✅
- [x] "How I lose" = scenariu real (nu eroare individuală) ✅
- [x] Surse inline citate ✅

---

*Analysis: 2026-05-04 | Template WTA CoVe v3.2 | Tournament: ROME WTA 1000 Qualifying*

Sources:
- [Townsend recent form — JustWomensSports](https://justwomenssports.com/reads/taylor-townsend-madrid-open-2026-first-round-loss-katie-boulter-results/)
- [Sramkova vs Selekhmeteva — Flashscore](https://www.flashscore.com/match/tennis/selekhmeteva-oksana-v7vDWRvO/sramkova-rebecca-WUQY7p9b/)
- [Potapova vs Begu context — XBets](https://xbets.ro/ponturi/tenis/anastasia-potapova-vs-irina-begu-wta-calificari-roma-4-05-2026/)
