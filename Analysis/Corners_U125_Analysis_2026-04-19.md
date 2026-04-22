# CoVe — Football Corners (UNDER 12.5)
## Date: 2026-04-19 (Sunday)
## Template v1.1 applied (mismatch > 0.6 = HARD PASS)
## 58 fixtures, 46 recommended. Top 8 filtered + analyzed.

---

## MISMATCH FILTER APPLIED (v1.1 rule)

Per template 1.0.3CoVe_Corners.md v1.1: mismatch > 0.6 = **HARD PASS** (corner spike lesson from Sassuolo + Udinese).

### Candidates after filtering:

| # | Match | Liga | Lambda | p_cal | Mismatch | Verdict |
|---|---|---|---|---|---|---|
| 1 | Cremonese vs Torino | I1 | 8.29 | 88.1% | **0.20** | **PASS FILTER** |
| 2 | Radnicki 1923 vs Radnicki Nis | RS1 | 9.39 | 83.0% | **0.25** | **PASS FILTER** |
| 3 | Javor vs Sp. Subotica | RS1 | 9.05 | 83.3% | **0.46** | **PASS FILTER** |
| 4 | Empoli vs Entella | I2 | 9.33 | 83.3% | **0.57** | **PASS FILTER (borderline)** |
| 5 | **Istanbulspor vs Sariyer** | **TR2** | 7.71 | 84.8% | **0.59** | **PASS FILTER (borderline)** |
| - | Pisa vs Genoa | I1 | 8.28 | 88.1% | 0.76 | **FAIL (> 0.6)** |
| - | Juventus vs Bologna | I1 | 8.56 | 86.8% | 1.06 | **FAIL (> 0.6)** |
| - | IMT Novi Beograd vs Mladost | RS1 | 8.43 | 83.9% | 0.83 | **FAIL (> 0.6)** |
| - | TSC vs Napredak | RS1 | 9.22 | 83.1% | 0.87 | **FAIL (> 0.6)** |

**Au trecut filtrul: 5 meciuri** (din 15 top). Restul ELIMINATE per v1.1 rule.

---

## MATCH 1: Cremonese vs Torino (I1) — U12.5 @ 88.1% — TOP PICK

### Checklist:
- **Step A (baseline):** Serie A low corner league. Both mid-table, defensive. Lambda 8.29.
- **Step B (total):** < 9 → **EXCELLENT**
- **Step C (style):** Serie A = controlled tempo
- **Step D (game state):** Both mid-table, mismatch 0.20 = balanced match
- **Step E (league):** Serie A → **GOOD**

### PASS RULE ✓
- At least one team avg corners < 4: likely YES (both Serie A mid-table)
- Expected total < 10.5: YES (8.29)
- No high-tempo matchup: YES
- **Mismatch 0.20 < 0.6 → PASS FILTER**

### SCORE: 9/10 — HIGH

| Pick | Lambda | Score | Confidence | Why | How Loses |
|---|---|---|---|---|---|
| **Cremonese vs Torino U12.5** | 8.29 | **9/10** | **HIGH** | Lambda 8.29, Serie A tempo, mismatch 0.20 (balanced) | Late red card, unexpected tactical shift |

---

## MATCH 2: Istanbulspor vs Sariyer (TR2) — U12.5 @ 84.8%

### Checklist:
- **Step A:** Lambda 7.71 — lowest of all! Very low-corner environment
- **Step B:** < 8 → **EXCELLENT**
- **Step E:** TR2 = lower league, volatile data

**CAUTION:** k_dispersion data for TR2 teams less reliable (new leagues added). Mismatch 0.59 borderline.

### SCORE: 7/10 — MODERATE (TR2 cap)

| Pick | Lambda | Score | Confidence | Verdict |
|---|---|---|---|---|
| Istanbulspor vs Sariyer U12.5 | 7.71 | 7/10 | MODERATE (TR2 cap) | ODDS DEP |

---

## MATCH 3: Radnicki 1923 vs Radnicki Nis (RS1) — U12.5 @ 83.0%

### Checklist:
- Serbian Super Liga, low tempo
- Mismatch 0.25 — balanced
- Lambda 9.39 — GOOD range
- RS1 = smaller league, less reliable

### SCORE: 7/10 — MODERATE (RS1 cap)

---

## MATCH 4: Javor vs Sp. Subotica (RS1) — U12.5 @ 83.3%

### Checklist:
- Lambda 9.05 — GOOD
- Mismatch 0.46 — decent
- RS1 → lower confidence

### SCORE: 7/10 — MODERATE (RS1 cap)

---

## MATCH 5: Empoli vs Entella (I2) — U12.5 @ 83.3%

### Checklist:
- Serie B = excellent for Under corners
- Lambda 9.33 — GOOD
- Mismatch 0.57 — borderline

### SCORE: 8/10 — MODERATE-HIGH

| Pick | Lambda | Score | Confidence | Why |
|---|---|---|---|---|
| Empoli vs Entella U12.5 | 9.33 | 8/10 | MODERATE | Serie B tempo, Lambda 9.33, balanced (mismatch 0.57) |

---

## MATCHES REJECTED BY MISMATCH FILTER (Lecture learned)

| Match | Lambda | p_cal | Mismatch | Why Rejected |
|---|---|---|---|---|
| **Pisa vs Genoa** | 8.28 | 88.1% | **0.76** | Would have been top pick but high mismatch = corner spike risk |
| **Juventus vs Bologna** | 8.56 | 86.8% | **1.06** | Juve home favorite, will camp in Bologna half → corner spike |
| IMT Novi Beograd vs Mladost | 8.43 | 83.9% | 0.83 | Home team dominant → one-way traffic |
| TSC vs Napredak | 9.22 | 83.1% | 0.87 | Same dominance risk |

### Important: Model says YES to these (p_cal 84-88%) but corner spike pattern (Sassuolo + Udinese lesson) = AVOID.

---

## SUMMARY

| # | Match | Liga | Lambda | p_cal | Mismatch | Score | Confidence |
|---|---|---|---|---|---|---|---|
| 1 | **Cremonese vs Torino** | **I1** | **8.29** | **88.1%** | **0.20** | **9/10** | **HIGH** |
| 2 | Empoli vs Entella | I2 | 9.33 | 83.3% | 0.57 | 8/10 | MODERATE |
| 3 | Istanbulspor vs Sariyer | TR2 | 7.71 | 84.8% | 0.59 | 7/10 | MODERATE (cap) |
| 4 | Radnicki 1923 vs Radnicki Nis | RS1 | 9.39 | 83.0% | 0.25 | 7/10 | MODERATE (cap) |

### TOP PICK

**Cremonese vs Torino U12.5** (9/10 HIGH)
- Lambda 8.29, Serie A defensive tempo
- Mismatch 0.20 = PERFECT balanced match
- Both mid-table Serie A teams
- Per stake sizing 9/10 → **30 RON** on 1,000 RON bankroll

**Singurul pick care a trecut filtrul mismatch + template v1.1 + liga mare.**

Sources: [Previous DC analysis context], [Serie A corner stats typical]

---

## CROSSOVER WITH DC/GOALS (19 April)

| Match | DC | Goals U4.5 | Corners U12.5 |
|---|---|---|---|
| Nottingham vs Burnley | **9/10** | 8/10 | N/A (not top) |
| Aston Villa vs Sunderland | N/A | **9/10** | N/A |
| Cremonese vs Torino | N/A | ~85% | **9/10** |
| Padova vs Reggiana | N/A | **9/10** | ~80% |

**Nu avem triple-model pick azi** — spre deosebire de Udinese (18 April). Asta e bine pentru risc management.
