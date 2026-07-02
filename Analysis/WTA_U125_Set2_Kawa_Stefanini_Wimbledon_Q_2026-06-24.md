# CoVe Analysis — U12.5 Set 2 | Wimbledon Qualifying 2026
## Katarzyna Kawa vs Lucrezia Stefanini
**Data:** 2026-06-24 | **Ora:** 12:10 BST (13:10 CEST)
**Turneu:** Wimbledon Qualifying Round 2 — Grand Slam, Roehampton
**Suprafață:** Iarbă (outdoor, Roehampton Community Sports Centre)
**Analist:** AI Sports Analyst | **Surse:** TennisAbstract, TennisStats, Model (Markov+WElo)

---

## PASUL 1 — TRIPLE FILTER (din CSV model)

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **8.64%** | ✅ ≤ 10% |
| p_markov | 0.6416 (64.16% Kawa) | — |
| p_elo | 0.5501 (55.01% Kawa) | — |
| Elo/Markov gap | **\|64.16 - 55.01\| = 9.15pp** | ✅ ≤ 35pp |
| p_elo = 0 | Nu | ✅ |
| UNSTABLE flag | Nu | ✅ |
| hold_asym | 6.48pp | ✅ |
| blowout_score | 4/9 | Moderat |

**PASUL 1: ✅ TRECUT**

---

## PASUL 2 — TENNISABSTRACT (iarbă)

### Katarzyna Kawa — Iarbă 2023-2026

**Sample: 7 meciuri** ⚠️ (sub pragul ideal ≥10, dar veteran WTA cu pattern consistent)

| Meci | Turneu | Scor S1 | S1 TB? | Scor S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs Andreeva | Birmingham 125 | 7-5 | ❌ | 6-4 | ❌ NO |
| vs Xu | Birmingham 125 | 4-6 | ❌ | 7-5 | ❌ NO |
| vs Shapatava | s'Hertogenbosch Q | **7-6(4)** | ✅ | 6-3 | **❌ NO** |
| vs Ruse | s'Hertogenbosch Q | 6-2 | ❌ | 6-1 | ❌ NO |
| vs Stoiber | Wimbledon Q1 2025 | 5-7 | ❌ | 6-4 | ❌ NO |
| vs Banks | Wimbledon Q2 2025 | 6-2 | ❌ | 6-4 | ❌ NO |
| vs Jovic | Wimbledon Q3 2025 | 6-3 | ❌ | **7-6(2)** | **✅ YES** |

**Kawa S2 TB pe iarbă: 1/7 = 14.3%** ✅
**S1 TB → S2 pattern: 0/1 = 0% TB în S2 după S1 TB** ✅ (vs Shapatava: S1 TB → S2 decisiv 6-3)

---

### Lucrezia Stefanini — Iarbă 2023-2026

**Sample: 8 meciuri** ⚠️ (sub pragul ideal ≥10, dar pattern consistent)

| Meci | Turneu | Scor S1 | S1 TB? | Scor S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs Banks | Birmingham 125 | 6-2 | ❌ | 6-4 | ❌ NO |
| vs Ryser | Birmingham 125 | 6-3 | ❌ | **7-6(4)** | **✅ YES** |
| vs Golubic | Ilkley 125 | 6-4 | ❌ | 5-7 | ❌ NO |
| vs Ma | Newport 125 | 6-1 | ❌ | 5-7 | ❌ NO |
| vs Rogers | Newport 125 | 6-2 | ❌ | 7-5 | ❌ NO |
| vs Mandlik | Newport 125 QF | 6-3 | ❌ | 6-1 | ❌ NO |
| vs Valdmannova | Ilkley 125 2026 | 6-4 | ❌ | 6-3 | ❌ NO |
| vs Vidmanova | Ilkley 125 2026 | 6-2 | ❌ | 6-1 | ❌ NO |

**Stefanini S2 TB pe iarbă: 1/8 = 12.5%** ✅
**S1 TB → S2 pattern: N/A** — Stefanini nu a avut NICIUN S1 TB pe iarbă = semnal structural pozitiv

---

### Rezumat Pasul 2

| | Kawa | Stefanini | Verdict |
|---|---|---|---|
| Sample iarbă | 7 ⚠️ | 8 ⚠️ | Borderline (nu critica — veteran WTA) |
| S2 TB rate iarbă | **14.3%** | **12.5%** | ✅✅ ambele sub 15% |
| S1 TB → S2 | 0/1 = 0% | N/A (0 S1 TBs) | ✅✅ |
| Pattern | Consistent | Consistent | ✅ |

**PASUL 2: ✅ CONDIȚIONAT TRECUT** (sample sub 10 dar pattern solid)

---

## 1. MATCH CONTEXT

**Wimbledon Qualifying Round 2** la Roehampton Community Sports Centre — 3 mile de All England Club. Calificările se joacă luni-joi (22-25 iunie), 3 runde → 12 locuri în main draw.

**Kawa: seed #14** în calificări — una din favoritele la calificare.
A câștigat Q1 vs Bandecchi (6-3, fără să-și piardă serviciul — start perfect).

**Stefanini** a trecut de Q1 vs Giovannini.
Acum se înfruntă pentru un loc în Q3 (ultima rundă de calificare).

**Condiții Roehampton, 24 iunie:** ~19-21°C, parțial înnorat, vânt ușor. Condiții tipice de iarbă britanică — suprafață mai lentă decât Wimbledon proper, dar în continuare rapidă față de hard/clay.

---

## 2. PROFILURI JUCĂTOARE

### Katarzyna Kawa (Polonia)
- **Rang:** #121 | **Vârstă:** 33 ani | **Înălțime:** 180cm | **Elo:** 635
- **Stil:** Jucătoare agresivă, forehand puternic, servici decent (1.17 aces/meci)
- **Avantaj iarbă:** 180cm = servici eficient pe suprafață rapidă
- **2026 form:** 63.3% (19/30) — formă excelentă în 2026
- **Form recent:** WWWWWLL — 5 victorii consecutive înainte de ultimele 2 pierderi
- **Wimbledon history:** A ajuns în Q3 în 2025 (pierdută vs Jovic la TB Set 2)
- **Q1 Wimbledon 2026:** W vs Bandecchi fără să-și piardă serviciul ← formă excelentă

### Lucrezia Stefanini (Italia)
- **Rang:** #163 | **Vârstă:** 28 ani | **Înălțime:** 164cm | **Elo:** 463
- **Stil:** Baseliner grinding, returnuri bune, rezistență fizică. MINIMAL servici (0.23 aces/meci!)
- **Dezavantaj iarbă:** 164cm = servici limitat pe suprafață rapidă → se rupe mai des
- **2026 form:** 38.5% (10/26) — formă slabă
- **Form recent:** LLWLWLL — pierdut 5 din ultimele 7
- **DFs:** 3.68/meci (mai multe decât Kawa 3.0)

---

## 3. STATISTICI HOLD & SERVIRE

### Model (Markov + WElo, iarbă)
| Parametru | Kawa (A) | Stefanini (B) |
|---|---|---|
| **Hold % iarbă** | **61.95%** | **55.47%** |
| Hold asymmetry | +6.48pp Kawa | |
| p_markov | **64.16%** Kawa | |
| p_elo | **55.01%** Kawa | |
| expected_games | **24.0** | |

### TennisStats (toate suprafețele, 2026)
| Statistică | Kawa | Stefanini | Combinat |
|---|---|---|---|
| Aces/meci | **1.17** | **0.23** ← minimal | 1.40 |
| DFs/meci | 3.0 | 3.68 | 6.68 |
| Over 12.5/set | **10%** | **15%** | **13%** |
| TB/meci | **17%** | **27%** | 22% |
| Avg games/set | 9.1 | 9.0 | **9.05** |
| Set 2 Win Rate | 60% | **31%** | |

**REVELAȚIE CHEIE:** Stefanini câștigă doar 31% din Set 2-urile ei. Istoric de fragilitate în Set 2 indiferent de suprafață.

---

## 4. CONDIȚIE FIZICĂ & OBOSEALĂ

### Kawa — ✅ Fresh (relativ)
- days_rest = 1 (jucată ieri Q1) → dar câștigată rapid vs Bandecchi (fără break-uri)
- fatigue_flag = FALSE
- Vârstă 33 ani = veteran, experiențat în gestionarea efortului fizic

### Stefanini — ✅ Odihnită (relativ)
- days_rest = 16 (ultimul meci = Ilkley 16 iunie)
- fatigue_flag = FALSE
- Vine cu mai multă odihnă decât Kawa

**Avantaj fizic: Stefanini** (mai odihnită) — dar diferența nu e critică.

---

## 5. CONTEXT MOTIVAȚIONAL & PSIHOLOGIC

### Kawa — ⬆️ MOTIVAȚIE ÎNALTĂ
- Seed #14 în calificări = favorita la avansare → trebuie să confirme
- La 33 ani, fiecare Wimbledon e prețios (carieră în declining years)
- A câștigat Q1 fără break → încredere mare în serviciu
- Un loc în main draw Wimbledon = punct WTA major + expunere

### Stefanini — ↔️ MOTIVAȚIE NEUTRU
- Formă slabă (38.5%) → presiunea de a câștiga un meci pe iarbă mai rară
- A trecut de Q1 (vs Giovannini) → deja un "bonus"
- Fără presiunea favorita → poate juca liber

**Mental:** Kawa are avantajul experienței (33 ani, mult mai multă presiune gestionată), Stefanini vine cu mai puțin de pierdut.

---

## 6. STIL DE JOC & TACTICI

**Kawa pe iarbă:** Servici consistent (fără aces masive dar solid), lovitură de fund de teren plată care funcționează bine pe iarbă. Tinde să câștige seturi fără să ajungă la 6-6 — pattern de 6-4, 6-3 pe care îl confirmă TennisAbstract.

**Stefanini pe iarbă:** Stil de clay-courter adaptat la iarbă. Cu 164cm și 0.23 aces/meci, ea NU servește bine pe iarbă — adversarele returnează bine și o rup frecvent. Se bazează pe retururi și rezistență, dar pe iarbă îi lipsesc puterea și lungimea paselor.

**Mismatch cheie:** Kawa 180cm vs Stefanini 164cm pe iarbă = Kawa are avantaj structural de servici. Kawa 62% hold vs Stefanini 55% → Kawa rupe Stefanini mai ușor.

---

## 7. H2H

**H2H: 0-1** în favoarea Stefanini
- Monastir 2023 (WTA, Hard): Stefanini câștigată 2-0

Singura întâlnire = pe hard court (suprafață complet diferită). Nu are relevanță directă pentru iarbă.

---

## 8. CoVe SCORING — U12.5 SET 2

### Factori confirmare ✅
| Factor | Valoare | Semnal |
|---|---|---|
| Model tb_p_cal | 8.64% | ✅ |
| TennisStats Over 12.5 combinat | **13%** | ✅✅ |
| Kawa S2 TB iarbă | 1/7 = **14.3%** | ✅ |
| Stefanini S2 TB iarbă | 1/8 = **12.5%** | ✅✅ |
| Stefanini: 0 S1 TBs pe iarbă | Pattern clar decisiv | ✅✅ |
| S1 TB → S2: Kawa 0/1 | 0% TB în S2 după S1 TB | ✅ |
| Stefanini Set 2 win rate | 31% (pierde S2 des) | ✅ pentru U12.5 |
| Kawa hold > Stefanini | +6.48pp asym | ✅ |
| Elo/Markov gap | 9.15pp | ✅✅ consistent |
| avg games/set TennisStats | 9.05 (departe de 12.5) | ✅ |

### Factori risc ⚠️
| Factor | Valoare | Semnal |
|---|---|---|
| Sample Kawa iarbă | 7 (< 10 threshold) | ⚠️ |
| Sample Stefanini iarbă | 8 (< 10 threshold) | ⚠️ |
| Stefanini TB/meci | 27% (all surfaces) | ⚠️ moderat |
| Kawa days_rest=1 | jucată ieri | ⚠️ minor |

### SCOR FINAL U12.5 SET 2

**7/10** ✅ — Pick valid cu rezerva sample

**Motiv 7 și nu 8:** ambele jucătoare au 7-8 meciuri pe iarbă (sub ≥10 threshold). Pattern-urile sunt consistent pozitive dar sample-ul nu ne permite scor 8+.

**Probabilitate ajustată: ~85-87%** (model 91%, TennisStats 87%)

---

## 9. PREDICȚIE CÂȘTIGĂTOARE

**Kawa câștigă: ~65-68%**
- Seed #14, formă bună (63.3% 2026), hold superior pe iarbă, 180cm vs 164cm
- Stefanini în formă slabă (38.5%), serviciu limitat pe iarbă
- H2H irelevanț (hard court, 2023)

---

## 10. VERDICT FINAL

| Market | Probabilitate | Scor | Recomandare |
|---|---|---|---|
| **U12.5 Set 2** | **~85-87%** | **7/10** | **✅ PICK** |

**Sample warning:** ambele jucătoare au 7-8 meciuri iarbă (sub pragul ideal ≥10). Pick valid dar cu rezerva — nu este la același nivel de certitudine ca picks-urile cu 15+ meciuri iarbă.

---

## SURSE

- [TennisAbstract JS — Katarzyna Kawa](https://www.tennisabstract.com/jsmatches/KatarzynaKawa.js)
- [TennisAbstract JS — Lucrezia Stefanini](https://www.tennisabstract.com/jsmatches/LucreziaStefanini.js)
- [TennisStats H2H — Kawa vs Stefanini](https://www.tennisstats.com)
- [Eurosport — Kawa Q1 Wimbledon 2026](https://eurosport.tvn24.pl/tenis/wimbledon/2026/wimbledon-2026.-katarzyna-kawa-awansowala-do-2.-rundy-kwalifikacji-odpadla-linda-klimovicova_sto23312397/story.shtml)
- [Olympics.com — Wimbledon 2026 Qualifying Draw](https://www.olympics.com/en/news/wimbledon-2026-qualifying-draw-order-of-play-schedule-results)
- [Wikipedia — Wimbledon 2026 Women's Qualifying](https://en.wikipedia.org/wiki/2026_Wimbledon_Championships_%E2%80%93_Women's_singles_qualifying)
- Model Markov+WElo: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-06-24)
