# CoVe — Royal Union Saint-Gilloise vs YR KV Mechelen
## Belgium Pro League | Championship Playoff GW7 | 2026-05-10, 19:30 CET
## Piețe analizate: Total Shots U26.5 | Union SOT U7.5 | Mechelen SOT U3.5 + Goals context

---

## DATE MODEL SOT (run_sot_daily.py)

| Side | Linie | p_over (model) | p_under (model) | Recommended |
|------|-------|----------------|-----------------|-------------|
| Union HOME | O2.5 SOT | **79.99%** | 20.01% | ✅ True |
| Union HOME | O3.5 SOT | **58.42%** | 41.58% | False |
| Union HOME | O4.5 SOT | **57.40%** | 42.60% | False |
| Union HOME | O5.5 SOT | **35.06%** | 64.94% | False |
| **Union HOME** | **O7.5 SOT (extrapolat)** | **~8-12%** | **~88-92%** | — |
| Mechelen AWAY | O2.5 SOT | 48.87% | **51.13%** | False |
| **Mechelen AWAY** | **O3.5 SOT** | **26.51%** | **73.49%** | False |
| Mechelen AWAY | O4.5 SOT | 26.91% | 73.09% | False |

**Nota extrapolation Union U7.5:**
Model arată: 5.5+ = 35.06%. Gap-ul 4.5→5.5 = 57→35 = −22pp. Extrapolând:
6.5+ ≈ 13%, 7.5+ ≈ 6%. Deci **P(Union SOT < 7.5) ≈ 88-92%**.

---

## DATE FOOTYSTATS

### Standings context (CRITIC)
| Echipă | Poz | Pts | MP | Miză |
|--------|-----|-----|-----|------|
| Club Brugge | 1 | 81 | 37 | Titlu |
| **Union SG** | **2** | **79** | **36** | **MUST WIN — titlu la 2 pts** |
| KV Mechelen | 8 | 49 | 36 | Europa Playoff |

**Union este în cursa titlului** — 2 puncte în spatele lui Brugge. Acesta este cel mai important meci al sezonului. Vor ataca masiv.

### Form recentă
| Echipă | Ultimele 5 | PPG |
|--------|------------|-----|
| Union SG | W W D W **L** | **2.19** |
| KV Mechelen | **L L L L W** | **1.36** |

Union HOME: W W W W D (neînvinși acasă în 18 meciuri!). PPG home = **2.78**.
Mechelen AWAY: W L L D L — formă slabă.

### Shots & SOT (FootyStats)
| Stat | Union SG (overall) | Mechelen (overall) |
|------|--------------------|--------------------|
| Shots/match | **14.75** | **9.92** |
| SOT/match | **5.28** | **3.64** |
| SOT Over 3.5 | 75% | **42%** |
| SOT Over 4.5 | 61% | 39% |
| SOT Over 5.5 | 47% | 28% |
| SOT Over 6.5 | **28%** | 8% |

---

## PIAȚA 1 — UNION SAINT-GILLOISE UNDER 7.5 SOT

### Analiza
| Sursă | Estimare p(Under 7.5) |
|-------|----------------------|
| Model SOT (extrapolat) | **88-92%** |
| FootyStats overall (over 6.5 = 28%) | ~72-80% |
| Context: HOME match, title race | ↓ (mai multe shots la home) |
| Context: Zorgane INJURED (scorer cheie) | ↑ (mai puțin SOT fără creator de play) |
| Context: Mechelen defensive away | ↑ (mai puțin SOT Union dacă Mechelen parchează autobuzul) |

**Estimare finală: P(Union < 7.5 SOT) ≈ 78-85%**

**Risc principal:** Union MUST WIN pentru titlu → vor trage mult mai mult decât în meciurile normale. Dacă rămân la 1-0 sau 0-0 la pauză → vor escalada în a 2-a repriză → spike SOT.

**Injury flag:** Zorgane (CM creator, 4 goluri sezon) ❌ INDISPONIBIL + Burgess (CB) ❌ INDISPONIBIL

**Verdict: MODERATE BET la cotă ≥ 1.20.** Sub 1.15 → no value.
Fair odds ≈ 1/(0.81) = **1.235**

---

## PIAȚA 2 — KV MECHELEN UNDER 3.5 SOT

### Analiza
| Sursă | Estimare p(Under 3.5) |
|-------|----------------------|
| Model SOT (direct) | **73.5%** |
| FootyStats overall (over 3.5 = 42%) | **58%** |
| Context: AWAY match vs top-2 team | ↑ (mai puține shots away) |
| Context: Mechelen L L L L W recent | ↑ (echipă în colaps de formă) |
| Context: Union neînvinși acasă 18 match | ↑ (Union controlează) |
| Context: Mechelen avg shots away < 9.92 | ↑ |

**Reconciliere:** FootyStats 58% e OVERALL (include și HOME matches unde Mechelen trage mai mult). AWAY context → model's 73.5% e mai precis.

**Estimare finală: P(Mechelen < 3.5 SOT) ≈ 70-75%**

**Mechelen recent form:** 6-1 vs Club Brugge (masacru), 1-4 vs Sint-Truiden, 1-2 vs Anderlecht. Formă colapsată.

**Verdict: BET la cotă ≥ 1.33.** Fair odds ≈ 1/(0.72) = **1.389**

---

## PIAȚA 3 — TOTAL SHOTS UNDER 26.5

### Calcul
| Echipă | Estimare shots (context specific) |
|--------|----------------------------------|
| Union HOME | ~16-18 (peste media 14.75 — titlu chase) |
| Mechelen AWAY | ~7-9 (sub media 9.92 — formă slabă, vs top team) |
| **Combined estimate** | **~23-27** |

**Distribution analysis (FootyStats):**
- Union over 14.5 shots: 61% → Union ≥ 15 shots în 61% meciuri
- Union over 15.5 shots: 53% → Union ≥ 16 în 53%
- Mechelen over 10.5 shots: 42% → Mechelen ≥ 11 în 42%

Scenariul OVER 26.5 shots:
- Union 16+ (53% prob) + Mechelen 11+ (42% prob) = ~22% combined → posibil dar nu dominant
- Union 15 + Mechelen 12+ = 27+ → posibil

**Estimare P(Total Shots < 26.5):**
- Dacă Union 14-15 shots + Mechelen 9-10 shots = 23-25 → UNDER ✅
- Dacă Union 17+ shots (title pressure) + Mechelen 10+ = 27+ → OVER ❌

**Context title race:** Union MUST WIN = atacă agresiv = MAI MULTE shots → risc OVER crescut față de normal.

**Estimare finală: P(Total Shots < 26.5) ≈ 55-65%**

**VERDICT: PASS sau BET MIC** — valoare marginală. Dacă cotă ≥ 1.55, pot lua, dar riscul titlu-chase e real.
Fair odds ≈ 1/(0.60) = **1.67**

---

## GOALS CONTEXT CoVe

### Date cheie
| Stat | Union (home) | Mechelen (away) |
|------|-------------|-----------------|
| Scored/match | 1.61 overall (1.94 home) | 1.22 overall (1.29 away) |
| Conceded/match | 0.58 overall (0.33 home) | 1.42 overall (1.65 away) |
| xG | 1.69 (1.85 home) | 1.21 (1.19 away) |
| xGA | 1.01 (0.82 home) | 1.71 (1.73 away) |
| Clean sheets | 47% (67% home!) | 25% |
| Over 2.5 | 36% Union, 42% Mechelen | avg 39% H2H |

### Expected goals această partidă
- Union attack (home xG) = **1.85**
- Mechelen conceded away (xGA) = **1.73**
- Expected Union goals = (1.85 + 1.73) / 2 = **1.79**

- Mechelen attack (away xG) = **1.19**
- Union conceded at home (xGA) = **0.82**
- Expected Mechelen goals = (1.19 + 0.82) / 2 = **1.005**

- **Expected total goals = 1.79 + 1.005 = ~2.80**

### Odds market vs model
| Piață | Cotă | Implied prob | Model prob | Edge |
|-------|------|-------------|------------|------|
| Over 2.5 | 1.60 | 62.5% | ~55-60% | ⚠️ piața supraestimează |
| Under 2.5 | 2.30 | 43.5% | ~40-45% | ≈ fair |

H2H record: Over 2.5 = 56% din 16 meciuri → **56% istoric**
Dar context playoff TITLE CHASE: Union va ataca → mai probabil Over 2.5

**Concluzie Goals:** xG = 2.80 → borderline 2.5. Piața la 1.60 pentru Over 2.5 pare corect prețuită. **Nu recomandăm goal markets** — valoare limitată.

---

## H2H CORNERS CONTEXT
- Match corners avg H2H: **10.47** (sus față de liga avg 10.17)
- Union FOR: 5.78, Mechelen AGAINST: 6.17 → Union va domina în cornere
- **Corners Over 12.5**: 24% H2H (3/16 → nu recomandăm Under 12.5 la aceste echipe)
- Mechelen AGAINST corners: 6.17/meci → echipă care lasă mulți cornere → risc pentru Union Under corners

---

## INJURII (CONFIRMATE DIN FOOTYSTATS LINEUP)
| Jucător | Echipă | Rol | Impact SOT |
|---------|--------|-----|-----------|
| **Adem Zorgane** | Union | CM creator (4 goluri) | ❌ INDISPONIBIL → −1 SOT estimat |
| **Christian Burgess** | Union | CB | ❌ INDISPONIBIL → defense weaker |

Zorgane absent = mai puțin creativitate centrală. Dar Rodriguez (9 goluri) și Florucz (7 goluri) disponibili → impactul limitat.

---

## TABEL FINAL — VERDICT PER PIAȚĂ

| Piață | P(Win) model | P(Win) context | Fair odds | Cotă min BET | Verdict |
|-------|-------------|----------------|-----------|-------------|---------|
| **Union SG U7.5 SOT** | 88-92% | **~80%** | **1.25** | ≥ 1.20 | ✅ BET |
| **Mechelen U3.5 SOT** | **73.5%** | **~72%** | **1.39** | ≥ 1.33 | ✅ BET |
| Total Shots U26.5 | ~60% | ~58% | **1.67** | ≥ 1.55 | ⚠️ MARGINAL |
| Over 2.5 Goals | ~55% | ~60% | fair | piața 1.60 | ⚠️ SKIP |

### PICKS RECOMANDATE

**Pick 1: KV Mechelen Under 3.5 SOT** ← strongest per model (73.5%)
- Mechelen L L L L W, colaps de formă, away vs top-2 team, avg SOT 3.64 overall → mult mai mic away
- Fair odds 1.39 → dacă piața oferă 1.40+ → VALUE

**Pick 2: Union SG Under 7.5 SOT** ← secondary (80-85%)
- Union va trage mult (title chase) dar 7.5 e un prag înalt
- Zorgane absent ajută ușor
- Fair odds 1.25 → dacă piața oferă 1.22+ → ușor value

**How I lose pick 1 (Mechelen U3.5):** Mechelen marchează devreme (1-0) → Union atacă disperat → Mechelen pe counter → 5-6 shots → 4+ SOT. Prob: ~27%.

**How I lose pick 2 (Union U7.5):** Union 0-0 la pauză → escaladare masivă → 8-9 SOT în 60-90min. Prob: ~15-20%.

---

## SURSE

- [FootyStats — Union SG vs KV Mechelen](https://footystats.org/belgium/royal-union-saint-gilloise-vs-yr-kv-mechelen)
- [SOT Model output](../simulations/SOT/evaluations/1.1_SOT_Evaluations.csv)
- [Belgium Pro League standings](https://footystats.org/belgium/pro-league)
