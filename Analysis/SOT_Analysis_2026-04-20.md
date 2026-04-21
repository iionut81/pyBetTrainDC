# CoVe — Football Shots on Target (Per-Team Over/Under)
## Date: 2026-04-20 (Monday)
## Template v1.0 applied (prompt 1.0.6CoVe_SOT.md)
## Scaling factor: 1.90x | Lines: 2.5 / 3.5 / 4.5 / 5.5

---

## TODAY'S SLATE (3 matches, 24 evaluations)

| # | Match | Liga | Context | Key model output |
|---|---|---|---|---|
| 1 | Crystal Palace vs West Ham | E0 | Selhurst Park, 20:00 BST | CP λ_bk=4.72 / WH λ_bk=4.18 |
| 2 | Lecce vs Fiorentina | I1 | Via del Mare, 18:45 UTC | Lecce λ_bk=4.03 / Fio λ_bk=3.47 |
| 3 | Moreirense vs Estoril | P1 | Liga Portugal, 19:15 UTC | More λ_bk=3.75 / Est λ_bk=4.28 |

---

## STEP 0 — HARD PASS FILTERS

### A. Class Gap Check

| Match | Teams positioning | Verdict |
|---|---|---|
| Crystal Palace vs West Ham | Both mid-table (CP 10th area, WH 11-13), balanced | ✅ MODEL RELIABLE |
| Lecce vs Fiorentina | Lecce 18th (relegation) vs Fiorentina **15th** (struggling!) | ⚠️ Form gap, NOT class gap — but Fio 9 unbeaten, Lecce 4 straight losses |
| Moreirense vs Estoril | Both mid-table P1, balanced | ✅ MODEL RELIABLE |

**IMPORTANT REVISION:** Research shows Fiorentina is **15th in Serie A** (23% win rate), NOT top-6 as I initially assumed. Fio away form: 4W 5D 7L. The class-gap blind spot filter does NOT apply here — but **form momentum gap** does (Fio 9 unbeaten, Lecce 4 losses).

### B. Data Quality

All 6 teams have >= 15 matches in profile. ✅

### C. League Scaling Check

- E0 Crystal Palace-West Ham: scaling 1.90 well-calibrated → **TRUST**
- I1 Lecce-Fiorentina: scaling 1.90 OK for mid-table → **TRUST**
- P1 Moreirense-Estoril: scaling needs ~2.00 → **ADJUST +7pp on research side**

### D. Match State Risks

- **Crystal Palace:** Adam Wharton fitness scare. Jordan Ayew back. Eze form good.
- **West Ham:** 9/10 key players available, relegation-threatened so MUST attack away
- **Lecce:** Last match 1 SOT in 35% possession — AWFUL form
- **Fiorentina:** De Gea in goal, 4-3-3 attacking, 9 unbeaten
- **Moreirense:** **7 players out** (3 injured, 3 suspended, 1 more absent) — CHAOS
- **Estoril:** 2 absences only, relatively fit

---

## MATCH 1: Crystal Palace vs West Ham (E0)

### STEP 1 — HARD DATA

- **Crystal Palace 2025-26 avg SOT: 3.67 per match** ([FootyStats](https://footystats.org/clubs/crystal-palace-fc-143))
- **West Ham avg SOT: 3.7 per match** (last few matches)
- Crystal Palace home tempo: solid 3-4-3 attacking shape under Glasner
- West Ham 4-4-1-1 with Castellanos as striker + Summerville on wing

### STEP 2 — MODEL OUTPUT vs BOOKMAKER

| Team | Line | Model p_over | Book odds | Book implied | Edge |
|---|---|---|---|---|---|
| Crystal Palace | 2.5 | 78.9% | n/a* | — | — |
| Crystal Palace | **3.5** | **63.8%** | **1.55** | **64.5%** | **-0.7pp (FAIR)** |
| Crystal Palace | 4.5 | 48.1% | 2.15 | 46.5% | +1.6pp |
| Crystal Palace | 5.5 | 34.1% | 3.30 | 30.3% | +3.8pp |
| West Ham | 2.5 | 74.5% | n/a* | — | — |
| West Ham | 3.5 | 57.0% | 1.60 | 62.5% | -5.5pp |
| West Ham | 4.5 | 40.0% | 2.30 | 43.5% | -3.5pp |
| West Ham | 5.5 | 25.9% | 3.55 | 28.2% | -2.3pp |

\* Bookie shows lines starting at 3.5 for both teams

### STEP 3 — RESEARCH

- CP last match: **5 SOT vs Newcastle** (well above 3.5 line)
- CP home advantage + WH weakened defense (Diouf, Disasi still gelling)
- WH must attack to escape relegation → may push for SOT
- Palace has Yeremy Pino as creative threat behind Strand Larsen
- Both teams played weekend just prior → minor fatigue

### STEP 4 — SCORE

**Best pick: Crystal Palace O3.5 @ 1.55**

| Factor | Score |
|---|---|
| Lambda margin (4.72 - 3.5 = 1.22) | 2/3 |
| Attack/Defense (Eze+Mateta vs WH shaky D) | 2/2 |
| Home/Away context (home side O3.5) | 2/2 |
| Match state (both stable, CP fresh) | 1/2 |
| Intuition | 1/1 |
| **Total** | **8/10** HIGH |

**But edge = -0.7pp → FAIR pricing.** Score good but no value. **ODDS DEPENDENT**.

Sources: [Sports Mole](https://www.sportsmole.co.uk/football/crystal-palace/relegation-battle/preview/crystal-palace-vs-west-ham-prediction-team-news-lineups_595940.html), [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/crystal-palace-vs-west-ham-prediction-lineups-odds-2026-04-20/), [FootyStats](https://footystats.org/clubs/crystal-palace-fc-143)

---

## MATCH 2: Lecce vs Fiorentina (I1)

### STEP 1 — HARD DATA

- **Fiorentina 2025-26 avg SOT: 3.58 per match** ([FootyStats](https://footystats.org/clubs/acf-fiorentina-471))
- **Fiorentina away form: 4W-5D-7L** (16 matches, poor return)
- **Lecce last match: 1 SOT, 35% possession** — extremely poor away
- Lecce 18th, Fiorentina 15th — both mid-relegation zone
- Lecce 4 straight losses vs Fiorentina 9 unbeaten (5 Serie A)

### STEP 2 — MODEL OUTPUT vs BOOKMAKER

| Team | Line | Model p_over | Book odds | Book implied | Edge |
|---|---|---|---|---|---|
| Lecce | **2.5** | **72.0%** | **1.37** | **73.0%** | **-1.0pp (FAIR PERFECT)** |
| Lecce | 3.5 | 54.1% | 1.90 | 52.6% | +1.5pp |
| Lecce | 4.5 | 37.4% | 2.95 | 33.9% | +3.5pp |
| Fiorentina | 3.5 | 45.0% | 1.50 | 66.7% | -21.7pp ⚠️ |
| Fiorentina | 4.5 | 27.6% | 2.07 | 48.3% | -20.7pp ⚠️ |
| Fiorentina | 5.5 | 15.4% | 3.15 | 31.7% | -16.3pp ⚠️ |

### STEP 3 — RESEARCH

**Lecce situation:**
- Home advantage crucial — Via del Mare is tough atmosphere
- Must-win = will push for shots, may open up
- But Fio away form poor → could be low-energy match
- 4-2-3-1 formation = Banda + Pierotti wide + Stulic striker (attacking trio)

**Fiorentina paradox:**
- Bookmaker O3.5 @ 1.50 (66.7%) — **priced as if Fio dominates**
- Reality: 15th place, 23% win rate, 3.58 avg SOT (barely 3.5 line)
- Bookmaker likely pricing Fio's 9-unbeaten run + Gudmundsson/Piccoli
- **22pp gap to model is EXTREME** — suggests bookmaker reading momentum, not stats

**Why model disagrees with bookmaker on Fio:**
- Model sees Fio av SOT history (3.47 bk scale)
- Bookmaker sees last 5 matches form surge

### STEP 4 — SCORE

**Pick 1: Lecce O2.5 @ 1.37**

| Factor | Score |
|---|---|
| Lambda margin (4.03 - 2.5 = 1.53) | 3/3 |
| Attack/Defense (Lecce open attack vs Fio OK D) | 1/2 |
| Home/Away context (home O2.5) | 2/2 |
| Match state (must-win + home) | 2/2 |
| Intuition | 1/1 |
| **Total** | **9/10** HIGH |

Edge -1.0pp → **FAIR pricing.** Strong score but zero value.

**Pick 2: Fiorentina O3.5 @ 1.50 (INVERSE CHECK)**

Research cap: Fio away +10pp → model 45% + 10 = 55% max. Still below 66.7%. **Cannot bet Over at this price.**

Inverse: Fiorentina UNDER 3.5 @ 2.47 (implied 40.5%). Model says 55% UNDER probability. Edge +14.5pp?

**WAIT — but scaling 1.9x could be wrong for Fio's recent form.** If Fio is actually shooting 4.5/match in recent games, real lambda could be 4.5 not 3.47. That's +1 SOT adjustment.

**Verdict:** Fiorentina lines are the MOST VOLATILE pricing today. Too risky to bet either side. **SKIP.**

Sources: [Sports Mole](https://www.sportsmole.co.uk/football/lecce/preview/lecce-vs-fiorentina-prediction-team-news-lineups_595949.html), [Ratingbet](https://ratingbet.com/predictions/lecce-vs-fiorentina-prediction-expert-analysis-possible-lineups-april-20-2026/), [FootyStats Fiorentina](https://footystats.org/clubs/acf-fiorentina-471), [Freetips](https://www.freetips.com/football/lecce-vs-fiorentina-predictions-betting-tips-20260419-0012/)

---

## MATCH 3: Moreirense vs Estoril (P1)

### STEP 1 — HARD DATA

- Liga Portugal → model scaling **1.9 underfit** (should be ~2.0)
- **Moreirense: 7 players unavailable** (Alvaro Martinez, Vasco Sousa, Dinis Pinto, Maracas suspended, Kiko suspended, Michel injured, Nile John suspended)
- Estoril: 2 unavailable (Boma injured, Sanchez suspended)
- Both P1 mid-table, no class gap

### STEP 2 — MODEL OUTPUT vs BOOKMAKER

Applying **+7pp P1 scaling adjustment** (known scaling underfit):

| Team | Line | Raw Model | Adjusted (+7pp) | Book odds | Book implied | Adj Edge |
|---|---|---|---|---|---|---|
| Moreirense | **2.5** | 70.5% | **77.5%** | **1.29** | **77.5%** | **0.0pp (FAIR EXACT)** |
| Moreirense | 3.5 | 50.7% | 57.7% | 1.72 | 58.1% | -0.4pp |
| Moreirense | 4.5 | 32.5% | 39.5% | 2.55 | 39.2% | +0.3pp |
| Estoril | 3.5 | 57.9% | 64.9% | 1.50 | 66.7% | -1.8pp |
| Estoril | 4.5 | 41.5% | 48.5% | 2.05 | 48.8% | -0.3pp |
| Estoril | 5.5 | 27.7% | 34.7% | 3.10 | 32.3% | +2.4pp |

### STEP 3 — RESEARCH

- **Moreirense has 7 absentees** — HUGE negative factor on SOT
  - Kiko Bondoso (best creator) suspended
  - Nile John suspended
  - 3 squad players injured
  - = Reduce Moreirense SOT forecast by 0.5-1.0 (bk scale)
- Estoril at relatively full strength — stronger lineup
- H2H: 3-3 in last meeting (open play)
- Both teams lambda adjusted for P1 scaling

### STEP 4 — SCORE

**Pick: Moreirense O2.5 @ 1.29**

| Factor | Score |
|---|---|
| Lambda margin (3.75 - 2.5 = 1.25, after adj ~1.5) | 2/3 |
| Attack/Defense (Moreirense depleted!) | 0/2 |
| Home/Away context | 2/2 |
| Match state (More 7 absent = volatility) | 0/2 |
| Intuition (7 absences = risky) | 0/1 |
| **Total** | **4/10** — PASS |

**BIG RED FLAG:** Moreirense 7 absent players. Bookmaker 77.5% is aggressive given depleted squad. Model adjusted exactly matches (0pp edge) but the **risk is asymmetric** — Moreirense could shoot 1 SOT, not 3+, due to squad issues.

**Pick alternative: Estoril O3.5 @ 1.50**

Adjusted model 64.9%, book 66.7% → -1.8pp edge. Weaker edge, less appealing than Moreirense O2.5 on paper but Estoril has full squad.

| Factor | Score |
|---|---|
| Lambda margin (4.28 - 3.5 = 0.78) | 1/3 |
| Attack/Defense | 1/2 |
| Home/Away | 1/2 (away team) |
| Match state (full squad) | 2/2 |
| Intuition | 1/1 |
| **Total** | **6/10** — PASS (borderline) |

Sources: [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/moreirense-vs-estoril-prediction-lineups-odds-2026-04-20/), [Sports Mole](https://www.sportsmole.co.uk/football/moreirense/preview/moreirense-vs-estoril-prediction-team-news-lineups_595985.html), [Tips.GG](https://tips.gg/article/moreirense-vs-estoril-20-04-2026/)

---

## STEP 4 — CORRECTIONS TABLE

| Pick | Side | Line | Model | Research Adj | Edge | Score | Action |
|---|---|---|---|---|---|---|---|
| Lecce O2.5 | home | 2.5 | 72.0% | 0 | -1.0pp | 9/10 | **PASS (no edge)** |
| CP O3.5 | home | 3.5 | 63.8% | +2pp (home form) | +1.3pp | 8/10 | **PASS (marginal)** |
| More O2.5 | home | 2.5 | 70.5% | +7pp P1 - 5pp (7 absent) | -5pp | 4/10 | **PASS (chaos risk)** |
| CP O4.5 | home | 4.5 | 48.1% | 0 | +1.6pp | 6/10 | PASS (lambda margin too small) |
| CP O5.5 | home | 5.5 | 34.1% | 0 | +3.8pp | 3/10 | PASS (lambda < line) |
| Fio O3.5 | away | 3.5 | 45.0% | +10pp cap | 55% vs 67% = -12pp | — | **PASS (blind spot)** |
| Estoril O3.5 | away | 3.5 | 57.9% | +7pp P1 | -1.8pp | 6/10 | PASS (borderline) |

---

## STEP 5 — FINAL PICKS

### 🥇 TOP CANDIDATE (informational only — NO BET)

**Lecce Over 2.5 SOT @ 1.37**
- Score: 9/10 | Confidence: HIGH
- Model: 72.0% | Research: 72% | Fair odds: 1.39
- Edge: -1.0pp → **FAIR pricing, no value**
- Key stat: Lecce home 4-2-3-1 with Banda+Pierotti+Stulic trio, must-win relegation fight
- How it loses: Lecce parks bus expecting Fio onslaught, Piccoli/Gudmundsson dominate

### 🥈 Runner-up (informational only)

**Crystal Palace Over 3.5 SOT @ 1.55**
- Score: 8/10 | Confidence: MODERATE
- Model: 63.8% | Research: 66% | Fair odds: 1.52
- Edge: +1.3pp → **borderline value, too thin**
- Key stat: CP 3.67 avg SOT season + Eze/Mateta fit + home tempo
- How it loses: WH score early, CP chases with scattered shots, Wharton absent hurts creation

### SKIP LIST

- **Moreirense O2.5 @ 1.29** — 7 absentees = too risky despite 0pp edge
- **Fiorentina lines (all)** — 22pp gap vs bookmaker, too volatile
- **Estoril O3.5 @ 1.50** — borderline edge after adjustment, marginal value
- **All lines 4.5+** — lambda insufficient margin
- **All WH lines** — negative edge consistently

---

## FINAL VERDICT: **SKIP SOT AZI**

### Motive detaliate

**1. Zero picks cu edge >= +3pp + score >= 8/10**

Cele mai bune 2 picks (Lecce O2.5 9/10, CP O3.5 8/10) au edge < 1.5pp = **fair pricing**, zero value long-term.

**2. Market pare eficient azi**

Bookmakerul a pretuit corect aproape toate liniile echilibrate (CP O3.5 0.7pp off, Lecce O2.5 1pp off). Bookmakers have shot-on-target data on lock.

**3. Outliers suspicioase**

Fiorentina 22pp gap = blind spot (form momentum pricing vs historic lambda). Cannot safely bet either side.

**4. Squad disruption ignorata de model**

Moreirense 7 absentees — model doesn't capture this. Bookmaker might or might not have adjusted. Too much noise.

**5. Prima zi a modelului SOT**

Sunt in ziua 1 de model — trebuie validat mai intai pe meciuri monitorizate fara stake, inainte de a deploy real bankroll.

### RECOMANDARE: **0 RON pe SOT azi**

**Scop principal azi:** monitorizare + validation. Urmaresc rezultatele celor 3 meciuri si notez:
- Crystal Palace SOT actual vs predicted 4.72 (bk scale)
- Lecce SOT actual vs 4.03
- Moreirense SOT actual (cu 7 absenti) vs 3.75

### NEXT STEPS

**1. Tracking validation (automat via Flashscore re-import maine)**
- Compari fiecare SOT actual vs model prediction
- Calculezi eroare per team + per league
- Identificam patterns (P1 scaling confirmed? Fio form-pricing justified?)

**2. Urmatoarele zile**
- Meciuri cu **news breaking** (striker out last-minute) = posibil value
- Meciuri **mid-table echilibrate** fara class gap = model cel mai fiabil
- Linii **2.5 si 3.5** = safer zones decat 4.5+

**3. Model improvements (proiect future)**
- Integrare WElo pentru class adjustment
- Per-league scaling factor (nu global 1.9)
- Penalizare pentru squad depletion (N_absent * -0.3 SOT)

---

## APPENDIX: Per-match model outputs (complete)

Full 24-row evaluation in `simulations/SOT/evaluations/1.1_SOT_Evaluations.csv`.

Top picks by p_over (pre-edge filter):
1. CP O2.5: 78.9% (book n/a)
2. Estoril O2.5: 74.7% (book n/a)
3. WH O2.5: 74.5% (book n/a)
4. Lecce O2.5: 72.0% @ 1.37
5. Moreirense O2.5: 70.5% @ 1.29
6. Fiorentina O2.5: 65.3% (book n/a)
7. CP O3.5: 63.8% @ 1.55

---

## Sources

- [Crystal Palace FC Stats - FootyStats](https://footystats.org/clubs/crystal-palace-fc-143)
- [Crystal Palace vs West Ham - Sports Mole](https://www.sportsmole.co.uk/football/crystal-palace/relegation-battle/preview/crystal-palace-vs-west-ham-prediction-team-news-lineups_595940.html)
- [Crystal Palace vs West Ham - Sportsgambler](https://www.sportsgambler.com/betting-tips/football/crystal-palace-vs-west-ham-prediction-lineups-odds-2026-04-20/)
- [Lecce vs Fiorentina - Ratingbet](https://ratingbet.com/predictions/lecce-vs-fiorentina-prediction-expert-analysis-possible-lineups-april-20-2026/)
- [Lecce vs Fiorentina - Sports Mole](https://www.sportsmole.co.uk/football/lecce/preview/lecce-vs-fiorentina-prediction-team-news-lineups_595949.html)
- [Lecce vs Fiorentina - Freetips](https://www.freetips.com/football/lecce-vs-fiorentina-predictions-betting-tips-20260419-0012/)
- [Fiorentina Stats - FootyStats](https://footystats.org/clubs/acf-fiorentina-471)
- [Moreirense vs Estoril - Sportsgambler](https://www.sportsgambler.com/betting-tips/football/moreirense-vs-estoril-prediction-lineups-odds-2026-04-20/)
- [Moreirense vs Estoril - Sports Mole](https://www.sportsmole.co.uk/football/moreirense/preview/moreirense-vs-estoril-prediction-team-news-lineups_595985.html)
- [Moreirense vs Estoril - Tips.GG](https://tips.gg/article/moreirense-vs-estoril-20-04-2026/)