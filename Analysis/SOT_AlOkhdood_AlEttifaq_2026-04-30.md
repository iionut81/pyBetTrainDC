# Al Okhdood vs Al Ettifaq — SOT Markets Analysis

**Data:** 30 aprilie 2026, ora 21:30 IST
**Stadion:** Prince Hathloul Bin Abdulaziz Sport City Stadium, Najran (Al Okhdood home)
**Competiție:** Saudi Pro League — Round 30

---

## 1. Match Context

| Factor | Al Okhdood | Al Ettifaq |
|---|---|---|
| Poziție | **#17** (zonă retrogradare) | **#7** |
| Record sezon | 4W-?-? (29 meciuri) | 12W-6D-11L |
| Goluri marcate | **23** (0.79/match) | 41 (1.41/match) |
| Goluri primite | 61 (2.10/match) | 50 (1.72/match) |
| Forma ultimul 5 | L-L-W-L-L | D-L-W-L-L |
| **Goluri ult. 5** | **DOAR 1!** ⚠️ | 6 |
| Home/Away record | 2W-2D-6L acasă (last 10) | 0W-0D-5L away (last 5) |

**KEY INSIGHT:** Al Okhdood **NU A MARCAT în 4 din ultimele 5 meciuri** (1 gol total). Atac extrem de slab.
**Al Ettifaq:** **0 victorii în ultimele 5 deplasări**. Wijnaldum (14 goluri) singurul threat.

**Suspendări/Absențe:**
- Al Ettifaq: Jack Hendry suspendat (apărător central)

---

## 2. Pickurile tale & Analiză EV

### 🎯 Pick #1: Al Okhdood OVER/UNDER 2.5 SOT @ 1.30

**Model SOT v2.1:**
- λ_bk Al Okhdood: 4.048 (model)
- p_over = 64.1% / p_under = 35.9%
- Fair odds OVER = 1.56, UNDER = 2.79

**Adjusted (recent form):**
- Doar 1 gol în 5 meciuri = ~0.2 goluri/meci recent
- Estimare λ recent ≈ 2.0-2.5 SOT (echipă în criză ofensivă)
- **P(over 2.5) realist:** 40-50%
- **P(under 2.5) realist:** 50-60%

**EV @ cota 1.30:**

| Side | Implied | Real est | EV | Verdict |
|---|---|---|---|---|
| OVER 2.5 @ 1.30 | 76.9% | 40-50% | **−40%** | ❌ NO BET |
| UNDER 2.5 @ 1.30 | 76.9% | 50-60% | **−28%** | ❌ NO BET |

**🚨 VERDICT: PASS** — cota 1.30 e prea mică pentru oricare side. Nu există value chiar dacă realitatea e mai aproape de UNDER. Piața preprețuie UNDER 2.5 (datorită formei slabe), dar nu suficient pentru cota 1.30.

---

### 🎯 Pick #2: Al Ettifaq UNDER 6.5 SOT @ 1.36

**Model SOT v2.1:**
- λ_bk Al Ettifaq: 5.472
- k_dispersion: 4.92 (high dispersion = wider distribution)

**Calcul matematic:**

Poisson(λ=5.472):
- P(X≤6) = ~69%

NB(λ=5.472, k=4.92) — high dispersion:
- Variance = 5.472 + 5.472²/4.92 = 5.47 + 6.09 = 11.55 (std=3.40)
- Mai mult mass în coadă → P(X≤6) ușor mai mare = **~72-75%**

**Adjusted recent form:**
- 6 goluri / 5 meciuri = 1.2 g/match (decent)
- 0 win in last 5 away (subperformanță)
- Wijnaldum dependent — single point of failure
- λ_real estimat: 4.5-5.5 (sub model)
- **P(UNDER 6.5) realist: 75-80%**

**EV @ cota 1.36:**
- Implied: 73.5%
- Real est: 75-80%
- **EV = 0.78 × 1.36 - 1 = +6%** (POZITIV ușor)

**✅ VERDICT: VALUE BET** — small positive EV, dar margine decentă. Recommend dacă bankroll permite.

---

### 🎯 Pick #3: Total UNDER 10.5 SOT @ 1.42

**Model SOT v2.1:**
- λ_total = λ_Okhdood + λ_Ettifaq = 4.048 + 5.472 = **9.52**
- (sub linia 10.5 = mean below)

**Calcul:**

Poisson(λ=9.52):
- P(X≤10) ≈ 64% (Normal approx: Z = (10.5-9.52)/3.09 = 0.317)

NB combined (high variance):
- Variance ~17-19 (std~4.2)
- Mai mult mass tail → P(X≤10) ≈ **65-70%**

**Adjusted recent form:**
- λ_real total = 2.5 (Okhdood real) + 5.0 (Ettifaq real) = ~7.5 SOT
- P(UNDER 10.5 | λ=7.5) Poisson = ~85%
- Conservative: **75-80%**

**EV @ cota 1.42:**
- Implied: 70.4%
- Real est: 75-80%
- **EV = 0.78 × 1.42 - 1 = +11%** (POZITIV moderat)

**✅ VERDICT: VALUE BET** — EV mai mare decât #2, datorită combinării celor două forme slabe.

---

## 3. Comparație finală

| Pick | Cota | Real prob | Fair odds | EV | Verdict |
|---|---|---|---|---|---|
| ❌ Al Okhdood 2.5 (any side) | 1.30 | 50-60% | 1.67-2.00 | **−28% to −40%** | **PASS** |
| ✅ Al Ettifaq UNDER 6.5 | 1.36 | 75-80% | 1.25-1.33 | **+6%** | BUY (mic) |
| ✅ Total UNDER 10.5 | 1.42 | 75-80% | 1.25-1.33 | **+11%** | **BUY (best)** |

---

## 4. Recomandare finală

### 🥇 **Best Single: Total SOT UNDER 10.5 @ 1.42** (8/10)

**Motive:**
1. **Două echipe în formă slabă:** Al Okhdood goalless 4/5, Al Ettifaq 0 wins 5 away
2. **Match defensiv așteptat** — Al Okhdood bunker (worst attack), Al Ettifaq vulnerable defensiv (lipsă Hendry)
3. **Predicție expert:** 1-3 score → 4 goluri = match cu shots dar nu shower of SOT
4. **EV +11%** = clear value

### 🥈 **Alternativă: Al Ettifaq UNDER 6.5 @ 1.36** (7/10)

**Motive:**
- EV mai mic (+6%) dar mai sigur (single team, less variance)
- Singularly Wijnaldum threat — restul echipei underperform

### ⚠️ **NU paria: Al Okhdood 2.5 @ 1.30**

Cota 1.30 nu oferă value indiferent de side. Piața deja prețuiește situația.

### ❌ **AVOID: Combo Total + Al Ettifaq UNDER**

Highly correlated — dacă Ettifaq sub 6.5, automat tend Total sub 10.5. Cota 1.36 × 1.42 = 1.93 NU compensează corelația. Single bet > combo.

---

## 5. Stake recommendation

- **Total UNDER 10.5 @ 1.42** — 2-3% bankroll (confidence 8/10, EV +11%)
- **Al Ettifaq UNDER 6.5 @ 1.36** — 1-2% bankroll dacă vrei pariere dublă (NU combo)
- Total combined exposure: max 4-5% bankroll

---

## Surse internet

- [Khelnow - Al Okhdood vs Al Ettifaq preview](https://khelnow.com/football/world-football-al-okhdood-vs-al-ettifaq-preview-202604)
- [Scoreaxis - Match stats](https://www.scoreaxis.com/match/al-ettifaq-vs-al-okhdood/)
- [Sofascore - Al-Okhdood team](https://www.sofascore.com/football/team/al-okhdood/336456)
- [SPL Official - 2025/26 season](https://www.spl.com.sa/en/teams/3083/al-okhdood/overview)
- [FBref - Al-Ettifaq stats](https://fbref.com/en/squads/11be4c0a/Al-Ettifaq-Stats)
- [Wikipedia - 2025-26 Saudi Pro League](https://en.wikipedia.org/wiki/2025%E2%80%9326_Saudi_Pro_League)

---

**Generat:** 30 aprilie 2026
**Autor:** Claude Code SOT Analysis