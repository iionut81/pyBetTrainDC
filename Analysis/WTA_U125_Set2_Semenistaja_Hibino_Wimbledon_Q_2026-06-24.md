# CoVe Analysis — U12.5 Set 2 | Wimbledon Qualifying 2026
## Darja Semenistaja vs Nao Hibino
**Data:** 2026-06-24 | **Ora:** 13:20 BST (14:20 CEST)
**Turneu:** Wimbledon Qualifying Round 2 — Grand Slam, Roehampton
**Suprafață:** Iarbă (outdoor, Roehampton Community Sports Centre)
**Analist:** AI Sports Analyst | **Surse:** TennisAbstract, TennisStats, Model (Markov+WElo)

---

## PASUL 1 — TRIPLE FILTER

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **0.0%** | ✅✅ MAX SIGNAL |
| p_markov | 0.6496 (64.96% Semenistaja) | — |
| p_elo | 0.5579 (55.79% Semenistaja) | — |
| Elo/Markov gap | **\|64.96 - 55.79\| = 9.17pp** | ✅ ≤ 35pp |
| p_elo = 0 | Nu | ✅ |
| UNSTABLE flag | Nu | ✅ |
| fatigue_flag_a | **TRUE** (Semenistaja) | ⚠️ context |
| hold_asym | 6.54pp | ✅ |
| blowout_score | 2/9 | Redus |
| data_source | tennisabstract/sackmann | — |

**PASUL 1: ✅ TRECUT**

---

## PASUL 2 — TENNISABSTRACT (iarbă) — 🔴 RED FLAG MAJOR

### Darja Semenistaja — Iarbă 2023-2026

**Sample: 1 MECI** 🔴 — ELIMINARE AUTOMATĂ (< 10 threshold)

| Meci | Turneu | Scor | S2 TB? |
|---|---|---|---|
| vs Sasnovich (L) | Wimbledon Q3 2025 | 7-5, 6-3 | ❌ NO |

**1 singur meci pe iarbă în toată cariera înregistrată (2023-2026).**
Hold rate 67.82% din model vine din **date clay/hard cu ajustare suprafață** — NU din grass-specific real.

---

### Nao Hibino — Iarbă 2023-2026

**Sample: 4 meciuri** 🔴 (sub ≥10 threshold)

| Meci | Turneu | Scor S1 | S1 TB? | Scor S2 | S2 TB? |
|---|---|---|---|---|---|
| vs Bolkvadze (W) | Birmingham Q1 2026 | **7-6(7)** | ✅ | 6-1 | ❌ NO |
| vs Valdmannova (W) | Birmingham Q2 2026 | 6-4 | ❌ | 6-2 | ❌ NO |
| vs Volynets (L) | s'Hertogenbosch Q1 2026 | 6-2 | ❌ | 6-2 | ❌ NO |
| vs Mboko (L) | Wimbledon Q1 2025 | 2-6 | ❌ | **7-6(7)** | ✅ YES |

**Hibino S2 TB pe iarbă: 1/4 = 25%** ⚠️ (>20% = risc moderat)
**S1 TB → S2: 0/1 = 0%** ✅ (când S1 TB, S2 a fost decisiv 6-1)

### Verdict Pasul 2

| | Semenistaja | Hibino |
|---|---|---|
| Sample iarbă | **1** 🔴 | **4** 🔴 |
| Threshold ≥10 | ❌ FAIL | ❌ FAIL |
| S2 TB iarbă | 0/1 = 0% (nesigur) | 1/4 = 25% ⚠️ |

**PASUL 2: ❌ PASS** — Semenistaja 1 meci iarbă = date insuficiente. Hold 67.82% calculat din clay/hard, nu grass.

---

## ⚠️ VERDICT: PASS

**Cu toate acestea, prezentăm analiza completă** — datele TennisStats sunt interesante și merită documentate pentru înțelegerea meciului.

---

## 1. MATCH CONTEXT

**Wimbledon Qualifying Round 2** — Roehampton, Londra.
Semenistaja seed **#7** în calificări (ranking 102-110). Este una din favoritele mari la calificare. Hibino (#247) = underdog clar.

**Condiții:** ~19-22°C, iarbă Roehampton — suprafață mai lentă decât Court-urile Wimbledon proper, ușor diferită față de clay dar rapidă față de hard.

---

## 2. PROFILURI JUCĂTOARE

### Darja Semenistaja (Letonia)
- **Rang:** #102 | **Vârstă:** 23 ani | **Elo:** 755
- **Stil:** Jucătoare agresivă de fundal, forehand puternic, carieră în ascensiune
- **Sezon 2026:** 50.0% win rate (17/34) — meciuri mixte
- **Last 12M:** 62.8% (49/78) — formă solidă pe parcurs
- **Form recent:** WWWWLWL — 4 victorii consecutive, 2 pierderi recente
- **DFs:** 1.15/meci ← EXCELENT, control solid al serviciului
- **Grass experience:** 1 singur meci (Wimbledon Q3 2025 L vs Sasnovich)
- **FATIGUE:** ⚠️ days_rest=1, last_3sets=TRUE, had_3sets_7d=TRUE — a jucat 3 seturi ieri!

### Nao Hibino (Japonia)
- **Rang:** #247 | **Vârstă:** 31 ani | **Înălțime:** 163cm, 58kg | **Elo:** 292
- **Stil:** Baseliner consistent, topspin, experiență vastă (veteran circuit)
- **Prize money:** $3.39M ← carieră lungă dar la nivel moderat
- **Sezon 2026:** 51.9% (14/27) — performanță decentă
- **DFs:** 2.75/meci (mai mult decât Semenistaja)
- **Grass experience:** 4 meciuri (Birmingham 125 + s'Hertogenbosch + Wimbledon Q)
- **Fatigued:** Nu — days_rest=1 dar ultimul meci în 2 seturi (fresh relativ)

---

## 3. DATE TENNISABSTRACT CONTEXT

### Semenistaja — Ultimele 5 meciuri (orice suprafață)
| Data | Turneu | Suprafață | Rezultat |
|---|---|---|---|
| 20260623 | Wimbledon Q | Iarbă | TBD (azi Q1) |
| 20260608 | Modena 125 | Clay | R16 L vs Samson (6-2, 7-5) |
| 20260608 | Modena 125 | Clay | R32 W vs Ribera (6-3, 2-6, 7-5) **← 3 seturi!** |
| 20260602 | Makarska 125 | Clay | SF W vs Avanesyan (6-1, 6-2) |
| 20260602 | Makarska 125 | Clay | R32 W vs Lukas (6-3, 5-7, 7-6) **← 3 seturi!** |

**OBSERVAȚIE CRITICĂ:** Semenistaja a jucat **2 meciuri în 3 seturi** în ultimele 2 săptămâni. Ieri a jucat Q1 Wimbledon (confirmat de model days_rest=1, last_3sets=TRUE). Ea vine cu oboseală reală.

### Hibino — Ultimele 5 meciuri pe iarbă/recent
A jucat la Birmingham 125 (Q1 W, Q2 W) și s'Hertogenbosch (Q1 L). Active pe iarbă în 2026 — mai obișnuită cu suprafața decât Semenistaja.

---

## 4. STATISTICI TENNISABSTRACT vs SACKMANN

**De ce hold 67.82% Semenistaja este suspect pe iarbă:**
- TennisAbstract: 1 meci iarbă (L vs Sasnovich 7-5, 6-3 → a pierdut S2!)
- Modelul calculează hold% cu decay 20 meciuri pe suprafață → cu 1 meci grass, fallback spre clay data
- Hold 67.82% = probably clay-heavy estimate, nu grass-specific

---

## 5. STATISTICI TENNISABSTRACT FINALE

### TennisStats (toate suprafețele, 2026)
| Statistică | Hibino | Semenistaja | Combinat |
|---|---|---|---|
| Aces/meci | 1.83 | 1.56 | 3.39 |
| DFs/meci | 2.75 | **1.15** ← excelent | 3.90 |
| **Over 12.5/set** | **15%** | **6%** ← minim | **11%** |
| TB/meci | 22% | 15% | 18% |
| Avg games/set | 9.11 | 9.09 | **9.10** |
| Set 2 Win Rate | 52% | 44% | — |
| Breaks/meci (BP won) | 2.5 | 3.1 | 5.6 |

**Semenistaja 6% Over 12.5/set** = una din cele mai mici rate din circuit. Pe TOATE suprafețele, ea aproape niciodată nu ajunge la TB. Semnal structural puternic — DAR de pe clay/hard.

**Combined: 11% Over 12.5/set → 89% U12.5** dacă extrapolăm la iarbă (nesigur).

---

## 6. CONDIȚIE FIZICĂ & OBOSEALĂ

### Semenistaja — ⚠️ FATIGUED REAL
- **days_rest = 1** → a jucat ieri (Q1 Wimbledon vs Siskova — victoria)
- **last_3sets = TRUE** → meciul de ieri a mers în 3 seturi!
- **had_3sets_7d = TRUE** → și înainte a mai avut 3 seturi (Modena)
- Tinereța (23 ani) compensează parțial
- **Impactul pe U12.5:** Semenistaja obosită → hold scade sub 67.82% → Hibino o rupe mai ușor → seturi decisive → AJUTĂ U12.5 paradoxal

### Hibino — ✅ Relatively Fresh
- days_rest = 1, dar ultimul meci în 2 seturi (nu fatigued)
- La 31 ani, experiență vastă în meciuri succesive

**Avantaj fizic: Hibino** — dar la calitate Semenistaja rămâne favorita clară.

---

## 7. CONTEXT MOTIVAȚIONAL & PSIHOLOGIC

### Semenistaja — ⬆️ MOTIVAȚIE MAXIMĂ
- Seed #7 în calificări = trebuie să ajungă în main draw → presiune de confirmare
- 23 ani, carieră în ascensiune, Wimbledon = Grand Slam de mare vizibilitate
- A câștigat Q1 (even if in 3 sets) → are momentul dar și oboseala

### Hibino — ↔️ LIBERĂ (nimic de pierdut)
- #247, veteran de 31 ani, Wimbledon qualifying = bonus
- A bătut adversare similare în calificări de turnee precedente
- H2H 0-2 vs Semenistaja = știe că e outsider

**H2H:** 0-2 pentru Hibino, ambele pe clay/hard (W100 Wiesbaden 2025, W25 Cancun 2022). Semenistaja dominantă în ambele.

---

## 8. STIL DE JOC & TACTICI

**Semenistaja pe iarbă:** Jucătoare de fundal agresivă — forehand puternic, dar cu 1 singur meci pe iarbă în CV, comportamentul real este necunoscut. Teoretic: 1.56 aces/meci + 1.15 DFs/meci = servici controlat.

**Hibino pe iarbă:** 163cm, 4 meciuri pe iarbă. Topspin clasic, rezistentă, returnuri bune. Nu este o jucătoare naturală de iarbă dar s-a descurcat în calificări la Birmingham. Hibino câștigă prin consistență și longevitate schimburi.

**Factorul CRUCIAL:** Semenistaja este structural superioară (#102 vs #247) dar FĂRĂ experiență reală pe iarbă. Hibino cu 4 meciuri pe iarbă are mai multă acomodare cu suprafața.

---

## 9. CoVe SCORING — U12.5 SET 2

### Factori confirmare ✅
| Factor | Valoare | Semnal |
|---|---|---|
| Model tb_p_cal | **0.0%** | ✅✅✅ |
| TennisStats Over 12.5 combinat | **11%** | ✅✅ |
| Semenistaja 6% Over 12.5 | Record circuit | ✅✅ |
| Elo/Markov gap | 9.17pp | ✅✅ |
| Hibino S1 TB → S2 | 0/1 = 0% | ✅ |
| Semenistaja DFs | 1.15/meci | ✅ |

### Factori risc / PASS reasons 🔴
| Factor | Valoare | Semnal |
|---|---|---|
| **Semenistaja grass sample** | **1 MECI** | 🔴🔴 ELIMINARE |
| **Hibino grass sample** | **4 meciuri** | 🔴 SUB THRESHOLD |
| Hold Semenistaja | 67.82% = clay/hard estimate | 🔴 nevalidat pe iarbă |
| Hibino S2 TB iarbă | 1/4 = 25% | ⚠️ risc moderat |
| Semenistaja fatigue | 3 seturi ieri | ⚠️ |

### SCOR FINAL

**PASS** — sample insuficient pe iarbă pentru ambele jucătoare.

Dacă ignori regula de sample și te bazezi EXCLUSIV pe TennisStats + model: maxim **5/10** (risc prea mare fără date grass-specifice, Semenistaja necunoscută pe iarbă).

---

## 10. PREDICȚIE CÂȘTIGĂTOARE

**Semenistaja câștigă: ~68-72%**
- Ranking masiv (#102 vs #247)
- Elo 755 vs 292 = diferență enormă de calitate
- H2H 2-0 Semenistaja
- Fatigue este singurul wildcard care o poate surprinde pe Hibino

---

## 11. VERDICT FINAL

| Market | Status | Scor | Decizie |
|---|---|---|---|
| **U12.5 Set 2** | **PASS** | **N/A** | **❌ Nu recomandăm** |

**Motivul PASS:** Semenistaja are 1 singur meci pe iarbă în TennisAbstract. Hold rate 67.82% vine din clay/hard data. Comportamentul ei real pe iarbă este necunoscut. Regula triple filter (sample ≥10) protejează de exact acest scenariu.

**Ce ar fi necesar pentru un pick valid:** minim 8-10 meciuri pe iarbă în TennisAbstract pentru Semenistaja → în prezent nu există.

---

## SURSE

- [TennisAbstract JS — Darja Semenistaja](https://www.tennisabstract.com/jsmatches/DarjaSemenistaja.js)
- [TennisAbstract JS — Nao Hibino](https://www.tennisabstract.com/jsmatches/NaoHibino.js)
- [TennisStats H2H — Semenistaja vs Hibino](https://www.tennisstats.com)
- [Tennis.com — Semenistaja vs Siskova Q1 Wimbledon 2026](https://www.tennis.com/tournaments/wimbledon/matches/d-semenistaja-vs-a-siskova-2026-06-23)
- [Olympics.com — Wimbledon 2026 Qualifying Draw](https://www.olympics.com/en/news/wimbledon-2026-qualifying-draw-order-of-play-schedule-results)
- Model Markov+WElo: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-06-24)
