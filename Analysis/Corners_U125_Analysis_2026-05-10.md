# CoVe — Football Corners UNDER 12.5
## Template: v1.6 | Date: 2026-05-10 (Duminică)
## Model: 61 meciuri → 38 recomandate

---

## STEP 0 — DECISION TREE λ (v1.6)

| Meci | λ | Decizie |
|------|---|---------|
| Hellas Verona vs Como | 7.97 | ✅ λ < 9 → TRUST MODEL |
| Cremonese vs Pisa | 8.38 | ✅ λ < 9 → TRUST MODEL |
| AC Milan vs Atalanta | 8.56 | ✅ λ < 9 → (dead rubber check: ambele cu miză → PROCEED) |
| Parma vs Roma | 8.03 | ✅ λ < 9 → (Roma FOR=5.03 < 6, dar mismatch 1.19 HOME/AWAY → borderline) |
| Barcelona vs Real Madrid | 10.79 | ❌ λ > 10.5 → HARD PASS |
| Cordoba vs Granada | 9.29 | ❌ FOR > 6 (Cordoba 6.08) → HARD PASS |
| SP1 cluster (Mallorca, Athletic, Oviedo) | 9.4–9.9 | ⚠️ Zona 9-10.5 → mismatch > 0.6 → HARD PASS |

**Lecție aplicată azi:** Verona-Como și Cremonese-Pisa ar fi fost HARD PASS cu v1.5 (mismatch OVERALL 0.97 și 0.93). Sub v1.6 cu λ < 9 → trec direct la CoVe.

---

## MECI 1 — HELLAS VERONA vs COMO
### I1 (Serie A) | Matchday ~37 | 15:30 CET | Stadio Bentegodi, Verona

---

### Date model
| Parametru | Valoare |
|-----------|---------|
| λ | **7.97** → < 9 → TRUST MODEL |
| p_cal | **88.7%** |
| p_cal_adj | **88.7%** |
| Fair odds | **1.127** |

### Step 0 — Mismatch HOME/AWAY (footystats.org)

| Stat | Verona (HOME) | Como (AWAY) |
|------|---------------|-------------|
| FOR/meci | 4.65 | 3.76 |
| AGAINST/meci | 3.71 | 4.35 |

- exp_home = (4.65 + 4.35) / 2 = **4.50**
- exp_away = (3.76 + 3.71) / 2 = **3.74**
- **mismatch = 0.77** → relevant DAR λ=7.97 < 9 → skip mismatch per v1.6

### Date empirice istorice
| Stat | Verona HOME | Como (overall) |
|------|-------------|----------------|
| Match corners AVG | **8.36** | **7.28** |
| Over 12.5 | **0% (0/17!)** | **3% (1/35)** |
| Under 12.5 | **100% acasă** | **97%** |

### Context
- **Verona**: 19th, 20 pts, must-win → atacă agresiv
- **Como**: 6th, 61% posesie, Fàbregas style INVERTED → puțin cornere despite dominanță
- Como away FOR: **3.76** (< 4) → sub GOLD threshold
- C2 Style: Como INVERTED → **+5pp boost**
- C4-B Verona must-win: **−2pp** (dart pe flanc în min 70+ dacă pierd)

### Quick Score
| Criteriu | Puncte |
|---------|--------|
| A. Baseline (Como away 3.76 < 4) | +2 |
| B. λ=7.97 EXCELLENT | +2 |
| C2. Como inverted 61% posesie | +1 |
| C3. Vreme Verona mai, uscat | +1 |
| D+E. I1 83.1%, game state moderat | +1 |
| C4. Forma: 0/17 Verona home + Como style | +1 |
| **TOTAL** | **8/10** |

### Research probability
- p_cal: 88.7%
- C2 +5pp, C4-B −2pp, C4-C +3pp = +6pp net
- p_research: **~93%** (cap la 95%)
- Fair odds: **1.127** | BET la cotă **≥ 1.07**

### Verdict
**✅ INFORMED BET** — excepție justificată față de mismatch-ul OVERALL. Como INVERTED style + 0/17 Verona home = semnal empiric excepțional.

**How I lose:** Verona pierde 0-1 în min 60 → se aruncă disperat → 5-6 cornere în ultimele 20 minute → total 13+. Prob: ~7%.

Surse:
- [footystats.org — Hellas Verona](https://footystats.org/italy/hellas-verona-fc)
- [footystats.org — Como](https://footystats.org/italy/como-1907)
- [soccerstats.com — I1 corners](https://www.soccerstats.com/table.asp?league=italy&tid=cr)

---

## MECI 2 — CREMONESE vs PISA
### I1 (Serie A) | Matchday ~37 | azi CET | Stadio Zini, Cremona

---

### Date model
| Parametru | Valoare |
|-----------|---------|
| λ | **8.38** → < 9 → TRUST MODEL |
| p_cal | **87.3%** |
| Fair odds | **1.146** |

### Step 0 — Mismatch HOME/AWAY (footystats.org)

| Stat | Cremonese (HOME) | Pisa (AWAY) |
|------|-----------------|------------|
| FOR/meci | **3.47** 🔥 < 4 GOLD | **3.59** 🔥 < 4 GOLD |
| AGAINST/meci | 8.88 − 3.47 = **5.41** | 8.83 − 3.59 = **5.24** |

- exp_home = (3.47 + 5.24) / 2 = **4.355**
- exp_away = (3.59 + 5.41) / 2 = **4.50**
- **mismatch = 0.145** → EXCELLENT (dar λ < 9 → skip per v1.6 oricum)

**Nota:** mismatch OVERALL soccerstats = 0.93 (date inflate de away-form slabă a lui Cremonese). HOME/AWAY split = 0.145. Lecția v1.6 aplicată corect.

### Date empirice istorice
| Stat | Cremonese HOME | Pisa AWAY |
|------|----------------|-----------|
| Match corners AVG | **8.88** | **8.83** |
| Over 12.5 | **12% (2/17)** | **12% (2/17)** |
| **Under 12.5** | **88%** | **88%** |

Ambele 88% — aliniat perfect cu p_cal = 87.3%.

### Context clasament
- **Cremonese**: 18th, 28 pts → must-win acasă
- **Pisa**: 20th (LAST), 18 pts → 10 puncte în spate → practic retrogradată → joacă compact, fără urgență
- Dacă Cremonese marchează devreme → meci controlat → UNDER favorabil
- Dacă Cremonese nu marchează → frustration spike posibil la finalul meciului

### Quick Score
| Criteriu | Puncte |
|---------|--------|
| A. GOLD — ambele < 4 FOR | +3 |
| B. λ=8.38 EXCELLENT | +2 |
| C2. Neutru (ambele medii) | +0 |
| C3. Vreme Cremona mai, uscat | +1 |
| D+E. I1 83.1%, risc moderat | +1 |
| C4. −2pp must-win, +2pp forma | +0 |
| **TOTAL** | **7/10** |

### Research probability
- p_cal: 87.3%
- C4 net: +1pp (forma) − 2pp (must-win) + 2pp (forma confirm) = +1pp
- p_research: **88.3%**
- Fair odds: **1.132** | BET la cotă **≥ 1.10**

### Verdict
**✅ BET MODERATE (7/10)** — profil GOLD (ambele < 4 FOR), mismatch real 0.145, 88% empiric Under.

**How I lose:** Cremonese 0-1 în min 65 → atacă disperat → 4-5 cornere consecutive → total 13+. Prob: ~12%.

Surse:
- [footystats.org — Cremonese stats](https://footystats.org/italy/us-cremonese)
- [footystats.org — Pisa stats](https://footystats.org/italy/ac-pisa-1909)
- [soccerstats.com — I1 corners](https://www.soccerstats.com/table.asp?league=italy&tid=cr)

---

## TABEL FINAL

| Meci | Liga | λ | p_cal | p_research | Score | Cotă min | Verdict |
|------|------|---|-------|------------|-------|---------|---------|
| **Verona vs Como** | I1 | 7.97 | 88.7% | ~93% | **8/10** | ≥ 1.07 | ✅ BET |
| **Cremonese vs Pisa** | I1 | 8.38 | 87.3% | 88.3% | **7/10** | ≥ 1.10 | ✅ BET |
| Barcelona vs Real Madrid | SP1 | 10.79 | — | — | — | — | ❌ λ > 10.5 |
| Cordoba vs Granada | SP2 | 9.29 | — | — | — | — | ❌ FOR > 6 |
| Toate SP1 cluster | SP1 | 9.4–9.9 | — | — | — | — | ❌ mismatch > 1.0 |

---

## NOTA METODOLOGICĂ — LECTIA ZILEI (v1.6)

Ambele meciuri ar fi fost **HARD PASS** cu CoVe v1.5 (mismatch OVERALL > 0.6).
Sub CoVe v1.6 cu decision tree λ-based:
- λ < 9 → TRUST MODEL → skip mismatch → direct CoVe tactic/contextual
- Mismatch HOME/AWAY corect: Verona 0.77, Cremonese 0.145 (vs OVERALL 0.97 și 0.93)
- Corecția a eliminat 2 FALSE POSITIVE HARD PASS și a găsit 2 picks valide.

**Eroarea din v1.5:** Mismatch OVERALL inflata de away-form slabă a echipelor de jos din clasament (Cremonese are AWAY AGAINST = 6.94 dar HOME AGAINST = doar 5.41).

---

## SURSE PRINCIPALE

- [footystats.org — Serie A corner stats](https://footystats.org/italy)
- [soccerstats.com — I1 corners table](https://www.soccerstats.com/table.asp?league=italy&tid=cr)
- [CoVe template v1.6](../Prompts/1.0.1CoVe_Corners.md)
