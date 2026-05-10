# CoVe Analysis — Corners UNDER 12.5
**Date:** 2026-05-04 | **Template:** 1.0.3CoVe_Corners.md v1.5
**Matches submitted:** 6 | **Data source:** soccerstats.com (I1, SP2, SP1) + RO1 = 404

---

## STEP 0 — DATA FETCH

### I1 — Serie A
| Team | FOR/g | AGS/g |
|------|-------|-------|
| Cremonese | 3.41 | **6.29** |
| Lazio | 3.71 | 4.29 |
| AS Roma | 5.06 | 3.38 |
| Fiorentina | 4.56 | 4.65 |

### SP2 — LaLiga 2
| Team | FOR/g | AGS/g |
|------|-------|-------|
| Almeria | 4.84 | 4.35 |
| Mirandes | 4.32 | 5.00 |

### SP1 — LaLiga
| Team | FOR/g | AGS/g |
|------|-------|-------|
| Sevilla FC | 4.97 | 4.55 |
| Real Sociedad | 5.85 | 4.67 |

### RO1 — Liga 1
**soccerstats.com/table.asp?league=romania&tid=cr → 404. Date indisponibile.**
Alternativă (windrawwin) → 403.
Matches Rapid-CFR + Slobozia-Hermannstadt = **date statistice neverificabile → SKIP.**

---

## STEP 0B — MISMATCH CALCULATIONS

| Meci | Liga | exp_H | exp_A | Total | Mismatch | Verdict |
|------|------|-------|-------|-------|----------|---------|
| Cremonese vs Lazio | I1 | (3.41+4.29)/2=**3.85** | (3.71+6.29)/2=**5.00** | 8.85 | **1.15** | ❌ HARD PASS |
| Roma vs Fiorentina | I1 | (5.06+4.65)/2=**4.86** | (4.56+3.38)/2=**3.97** | 8.83 | **0.89** | ❌ HARD PASS |
| Almeria vs Mirandes | SP2 | (4.84+5.00)/2=**4.92** | (4.32+4.35)/2=**4.34** | 9.26 | **0.585** | ⚠️ la limită |
| Sevilla vs Real Sociedad | SP1 | (4.97+4.67)/2=**4.82** | (5.85+4.55)/2=**5.20** | 10.02 | **0.38** | ✅ trece |
| Rapid vs CFR Cluj | RO1 | — | — | — | — | ⛔ DATE LIPSĂ |
| Slobozia vs Hermannstadt | RO1 | — | — | — | — | ⛔ DATE LIPSĂ |

---

## PRE-FILTER RESULTS

### ❌ ELIMINATED — MISMATCH > 0.6

**Cremonese vs Lazio (I1) — mismatch 1.15**
- Lazio expected 5.00 cornere, Cremonese 3.85
- Cremonese concede 6.29 AGS/meci — adversarii atacă constant în box
- Lazio va domina complet faza ofensivă pe flanc
- Model arată 88.9% BUT mismatch override automat
- **HARD PASS**

**Roma vs Fiorentina (I1) — mismatch 0.89**
- Roma 5.06 FOR + expected 4.86 cornere
- Roma atacă din central dar cu presență mare pe flanc (Saelemaekers, El Shaarawy)
- Fiorentina expected doar 3.97 → Roma dominant unilateral
- Model arată 88.4% BUT mismatch override
- **HARD PASS**

---

## FULL CoVe — SUPRAVIEȚUITORI MISMATCH

### MATCH 1: Almeria vs Mirandes (SP2) ⚠️

**Step 1 — Model data**
| Metric | Value |
|--------|-------|
| λ | 9.38 |
| Almeria FOR/g | 4.84 |
| Mirandes FOR/g | 4.32 |
| exp_home | 4.92 |
| exp_away | 4.34 |
| Total | 9.26 |
| Mismatch | 0.585 (sub 0.6, dar la limită) |
| p_cal | 84.2% |

**Step A:** Almeria 4.84, Mirandes 4.32 — ambele în zona 4-5 → ✅ GOOD (nu GOLD, ambele >4)
**Step B:** Total 9.26 → ✅ GOOD (9-10)
**Step E:** SP2 = 85.0% hit rate → ✅ bună ligă

**Step C2 — Tactical style:**
- Almeria: LaLiga2, stil direct, nimic specific inverted. **MIXED → +0pp**
- Mirandes: echipă compactă, defensivă. **MIXED → +0pp**

**Step C3 — Weather:**
- Almeria, mai 2026, seară → probabil cald și uscat → **+1pp (favorabil)**

**Step C4-B — Psychology (CRITICAL):**
- **Almeria (3rd, 67pts) — PROMOTION CHASE**: Must win pentru a rămâne în cursa pentru promovare automată (locul 2). Vor ataca → **−2pp**
- **Mirandes (20th, 3pts de safety)** — LUPTĂ PENTRU SALVARE: Trebuie să puncteze. Vor lupta. Dacă pierd, practic retrogradați. → **−2pp**
- **Total C4-B: −4pp**

**Step D — Game state:**
- Ambele echipe cu mize uriașe → joc intens, presiune ridicată
- Almeria favorită dar Mirandes disperată → game state impredictibil
- Posibil scenariu: Almeria 1-0 early → Mirandes atacă → corners spike

**Tabel ajustări:**
| Factor | Constatare | pp |
|--------|------------|----|
| C2 style | mixed ambele | +0 |
| C3 weather | Almeria, dry, warm | +1 |
| C4-B Almeria | promotion must-win | −2 |
| C4-B Mirandes | relegation survival | −2 |
| **TOTAL** | | **−3pp** |

**p_research: 84.2% − 3pp = ~81.2%** → Sub pragul de 82%.

**Quick Score:**
- A (baseline both 4-5): +2
- B (total 9.26 ✅): +2
- C2 (mixed): +0
- C3 (weather ok): +1
- D+E (SP2 good, stakes high): +1 (ligă bună, dar contextul presiunii reduce)
- C4 (context contra Under): **−1**
- **Total: 5/10** → **PASS**

**Verdict: PASS** — mismatch la limită + ambele echipe cu presiuni maxime → risc prea mare.

---

### MATCH 2: Sevilla vs Real Sociedad (SP1)

**Step 1 — Model data**
| Metric | Value |
|--------|-------|
| λ | 9.44 |
| Sevilla FOR/g | 4.97 |
| Real Sociedad FOR/g | 5.85 |
| exp_home | 4.82 |
| exp_away | 5.20 |
| Total | 10.02 |
| Mismatch | 0.38 |
| p_cal | 81.4% |

**Step A:** Sevilla 4.97 (sub 5), Real Sociedad 5.85 — Real Sociedad aproape de 6 → **+2** (decent dar nu GOLD)
**Step B:** Total 10.02 → ⚠️ BORDERLINE (10-11.5 range)
**Step E:** SP1 = 83.3% hit rate → ✅ bună ligă

**Step C2 — Tactical style:**
- **Sevilla** (18th, disperați): stil fluctuant, 5 înfrângeri din 6. Vor ataca haotic. Cross-heavy probable în disperare. → **−3pp** (traditional wing attacks + haos)
- **Real Sociedad** (8th, 43pts): stil posesie combinat cu atacuri pe flanc (Oyarzabal, Kubo). 5.85 FOR = creează cornere constant. Scored in 23/24 matches. → **−2pp** (attacking, consistent)

**Step C3 — Weather:**
- Sevilla, mai 2026, 19:00 UTC → probabil cald, uscat → **+0 (neutral)**

**Step C4-A — Injuries Sevilla:**
- Azpilicueta (muscle), Marcão (wrist) out — ambii defensivi/fundași → **−1pp** (defensivă mai fragilă = adversarul atacă mai ușor)

**Step C4-B — Psychology:**
- **Sevilla (18th, 34pts = RELEGATION)**: Pierd 5 din 6. Must-win sau retrogradare iminentă. DISPERARE = atacuri haotice pe flanc, risc mare de open play cu cornere. → **−3pp**
- **Real Sociedad**: Confortabili la 8th, nicio presiune. Joc liber, posibil relaxați defensive. → **+0pp**
- **Total C4-B: −3pp**

**Tabel ajustări:**
| Factor | Constatare | pp |
|--------|------------|----|
| C2 Sevilla | haos + cross-heavy disperare | −3 |
| C3 weather | Sevilla, dry | +0 |
| C4-A injuries | Azpilicueta + Marcão out | −1 |
| C4-B Sevilla | RELEGATION survival, disperare | −3 |
| **TOTAL** | | **−7pp** (cap la −10pp) |

**p_research: 81.4% − 7pp = ~74.4%** → Cu mult sub 82%.

**Quick Score:**
- A (Sevilla 4.97, Real Sociedad 5.85): +2
- B (total 10.02 borderline): +1
- C2 (Sevilla haos + Sociedad atacant): **−1** (traditional wings ambele)
- C3 (neutral): +0
- D+E (SP1 bună, dar Sevilla relegate = presiune = chaos): +1
- C4 (−7pp context, major negativ): **−1**
- **Total: 2/10** → **HARD PASS**

**Verdict: HARD PASS** — Sevilla în relegate cu stil haotic + Real Sociedad 5.85 FOR cu 10.02 expected total. Nu există marjă de siguranță.

---

### MATCH 3 & 4: Rapid vs CFR Cluj / Slobozia vs Hermannstadt (RO1)

**soccerstats.com 404 pentru RO1 — date FOR/AGAINST neverificabile.**

Per regula obligatorie din template: "NICIODATA nu accepta o cifra unica din web search snippet fara confirmare."

**Rapid vs CFR Cluj context:** Rapid 2nd (~56pts), CFR 4th (~53pts) → ambele în cursa pentru Europa/titlu. Derby românesc major = **−2pp C4-B** + risc mismatch necalculat.

**Concluzie: SKIP — date insuficiente.** Imposibil de calculat mismatch fără FOR/AGS verificate.

---

## SELF-VERIFICATION

- [x] Fetched soccerstats.com I1, SP2, SP1 → OK
- [x] RO1 → 404, windrawwin → 403 → corect skipuit (2 meciuri)
- [x] Mismatch calculat cu formula exactă pentru 4 meciuri disponibile
- [x] Cremonese-Lazio (1.15) și Roma-Fiorentina (0.89) eliminate înainte de model
- [x] Almeria-Mirandes: mismatch < 0.6 dar context C4-B anunat (−4pp) → sub 82%
- [x] Sevilla-Real Sociedad: mismatch OK dar Sevilla relegare + haos → p_research 74% → HARD PASS
- [x] Citat surse inline pentru context
- [x] Nicio narrative bias — concluzia onestă chiar dacă 0 picks

---

## FINAL PICKS — CORNERS U12.5

### **0 picks recomandate pentru 04.05.2026**

| Meci | Motiv eliminare |
|------|----------------|
| Cremonese vs Lazio | Mismatch 1.15 → HARD PASS automat |
| Roma vs Fiorentina | Mismatch 0.89 → HARD PASS automat |
| Almeria vs Mirandes | Mismatch borderline + ambele echipe presiune maximă → p_research 81% |
| Sevilla vs Real Sociedad | Sevilla relegare = haos + Real Sociedad 5.85 FOR → p_research 74% |
| Rapid vs CFR Cluj | Date RO1 indisponibile (soccerstats 404) |
| Slobozia vs Hermannstadt | Date RO1 indisponibile (soccerstats 404) |

**Zi slabă pentru Corners U12.5.** Top 2 picks după probabilitate (Cremonese-Lazio 88.9%, Roma-Fiorentina 88.4%) eliminate ambele pe mismatch — demonstrând că probabilitatea din model nu înlocuiește verificarea Step 0.

**Recomandare:** Nu paria Corners U12.5 astăzi. Verifică Goals sau WTA dacă există candidați.

---

*Analysis: 2026-05-04 | Template CoVe v1.5 | Data: soccerstats.com I1/SP2/SP1*

Sources:
- [Soccerstats Italia](https://www.soccerstats.com/table.asp?league=italy&tid=cr)
- [Soccerstats SP2](https://www.soccerstats.com/table.asp?league=spain2&tid=cr)
- [Soccerstats SP1](https://www.soccerstats.com/table.asp?league=spain&tid=cr)
- [Almeria vs Mirandes preview — BetMines](https://betmines.com/match-preview/almeria-vs-mirandes-prediction-match-preview-and-analysis-la-liga-2-04-05-2026)
- [Sevilla vs Real Sociedad preview — BetMines](https://betmines.com/match-preview/sevilla-vs-real-sociedad-prediction-match-preview-and-analysis-laliga-04-05-2026)
- [LaLiga2 standings — Wikipedia](https://en.wikipedia.org/wiki/2025%E2%80%9326_Segunda_Divisi%C3%B3n)
- [LaLiga standings — Wikipedia](https://en.wikipedia.org/wiki/2025%E2%80%9326_La_Liga)
