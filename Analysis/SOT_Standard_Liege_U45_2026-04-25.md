# CoVe — Standard Liège Sub 4.5 SOT (away vs Genk)
## Date: 2026-04-25 (Saturday) | Kickoff: 16:15 UTC / 19:15 RO
## Match: KRC Genk vs Standard Liège
## Tournament: Belgian Pro League — **Title Playoffs Round 5** (NOT regular season!)
## Venue: Cegeka Arena, Genk
## Template: Prompts/1.0.6CoVe_SOT.md v1.0

---

## STEP 0 — HARD PASS Filters

| Filter | Status | Note |
|---|---|---|
| Class gap | ✅ Mid vs mid (Genk top playoff, Standard 3rd) | Model OK |
| Strong-team-UNDER trap | ✅ Standard not strong = nu se aplică | |
| Data quality | ✅ B1 full data | |
| Referee/match state | ⚠️ **TITLE PLAYOFFS** = boost +20% per "derby" rule | RED FLAG |
| Weather | OK | |

---

## STEP 1 — Model Data

| Metric | Standard Liège (away) |
|---|---|
| λ_our (Flashscore) | **1.735** |
| **λ_bk (bookmaker scale)** | **2.95** |
| Scaling factor (B1) | 1.7 |
| Elo multiplier | 1.0 (no Elo boost) |
| k_dispersion | 7.65 (moderate variance) |
| Model P(O4.5) | 28.2% |
| **Model P(U4.5)** | **71.8%** |

**Lambda margin check (CoVe Step A pentru UNDER):**
- Rule: "trust UNDER dacă λ_bk < line - 1.0"
- 4.5 - 1.0 = 3.5
- λ_bk **2.95 < 3.5** ✅ pass
- Margin: **1.55 SOT below line** = SAFE

---

## STEP 2 — RESEARCH

### Standard Liège recent away SOT (last 3 verified)

| Adversar | Loc | Rezultat | SOT raw |
|---|---|---|---|
| Antwerp | away | L 1-2 | **3** ✅ U4.5 hit |
| OH Leuven | away | W 3-1 | **7** ❌ U4.5 MISS |
| Charleroi | away | W 2-1 | **2** ✅ U4.5 hit |

**Hit rate U4.5 last 3 away: 2/3 = 66.7%** (vs market implied 81%)

⚠️ Sample mic dar arată VOLATILITATE (variance 2-7, std deviation mare).

[Source: [Soccerstats](https://www.soccerstats.com/teamstats.asp?league=belgium&stats=u7301-standard-liege)]

### Genk defensive profile (home)

- xGA 1.44/match (decent)
- 5.69 SOT atinși/meci (overall medie atac, nu apărare)
- Home form: 6W 4D 5L (mediu)
- **NU este elite defensive home** = Standard va genera SOT

### Match Context — CRITICAL ⚠️

**B1 Title Playoffs Round 5:**
- Genk: 29 pts, **TOP** (vs Westerlo)
- Standard: 26 pts, **3rd** — **3 pts în spate**
- Standard: ultim meci PIERDUT 1-2 cu Antwerp
- Standard a alternat W/L în playoff = inconsistent

**🚨 KEY: Standard MUST WIN to stay in title race**
- Per CoVe template: "Title race urgency → −2"
- Per CoVe template: "Must-win game → −2"
- Per "Derby/rival match" rule: SOT boost +20%
- TITLE PLAYOFFS intensitate = mai aproape de derby decât regular season

[Sources: [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/genk-vs-standard-liege-prediction-lineups-odds-2026-04-25/), [BetMines](https://betmines.com/match-preview/genk-vs-standard-liege-prediction-match-preview-and-analysis-pro-league-25-04-2026)]

### Lineups & Tactics

**Standard Liège (3-4-2-1):**
- GK: Epolo
- Defense: Dierckx, Hautekiet, Homawoo
- Wing-backs: Lawrence, Karamoko
- Midfield: Nielsen, Mortensen
- Attacking mids: Abid, Nguene
- Striker: **Timothé Nkada**

**3-4-2-1 = formație ofensivă** cu 3 atacanți (1 striker + 2 trequartisti) → genera mai multe shots decât 5-3-2 defensiv.

**Genk:** No major injuries except Lawal (out)
**Standard:** No injuries reported ✅

### H2H Recent
- Standard a câștigat ULTIMELE 2 confruntări vs Genk
- Standard psihologic încrezătoare în acest matchup

---

## STEP 3 — Self-Verification

| Check | Answer |
|---|---|
| Class-gap filter aplicat? | ✅ Mid-vs-mid OK |
| Folosit λ_bk (nu λ_our)? | ✅ 2.95 |
| Verificat data quality? | ✅ B1 OK |
| Recent 5 SOT actual? | ⚠️ Doar 3 verified, 2/3 hit |
| Considerat early red card? | Nu specific, dar B1 mediu |
| **Match state risk** | ❌ **TITLE PLAYOFFS = MUST WIN** |
| Striker availability | ✅ Nkada disponibil |

---

## STEP 4 — Final Question

### "Poate Standard să NU ajungă la 4.5 SOT?"

**Pro U4.5 (Standard NU atinge 4.5):**
1. ✅ λ_bk 2.95 (mult sub 4.5)
2. ✅ Standard recent vs Antwerp doar 3 SOT, vs Charleroi doar 2 SOT
3. ✅ Genk top defensive context, nu lasă spațiu mare
4. ✅ B1 medie SOT/team away ~2.5
5. ✅ Standard NU e elite atacant (Nkada bun, dar nu Lukaku)

**Contra U4.5 (Standard ATINGE 4.5):**
1. ❌ **TITLE PLAYOFFS** — Standard MUST WIN
2. ❌ Recent vs OH Leuven: **7 SOT** — proof Standard poate exploda
3. ❌ Standard pierde teren în titlu, push agresiv
4. ❌ 3-4-2-1 formație ofensivă, nu defensivă
5. ❌ Genk apărare DOAR mediu (5W home doar)
6. ❌ H2H: Standard câștigat ultimele 2 vs Genk = nu se intimidează
7. ❌ Standard pe 0-2 sau pierde în repriza 2 → 5-6 SOT chase

---

## STEP 5 — SCORE & VERDICT

```
Lambda margin : 3/3 (1.55 SOT below line, SAFE static)
Attack/defense: 1/2 (-1: Genk apărare doar mediu, Standard atac decent)
Home/away     : 1/2 (Standard away typical lower, dar must-win)
Match state   : 0/2 (-2: TITLE PLAYOFFS + must-win + chasing 3pts)
Intuition     : 0/1 (recent OH Leuven 7 SOT spike example)
TOTAL         : 5/10 — DOWNGRADE
```

### 🎯 Verdict: **MARGINAL PASS / RISKY**

**Real probability estimate: 70-78%** (vs market implied 81%)

**Factori care fac U4.5 RISKY:**
1. **Title playoffs intensity** — Standard pressing pentru titlu
2. **Recent 7 SOT spike vs OH Leuven** = pattern offensive când deschis
3. **Must-win = Standard va atacca** — dacă Genk conduce, Standard îl chase
4. **3-4-2-1 formație** = 3 atacanți potențial = volum mai mare

**Factori protectivi:**
1. λ_bk 2.95 << 4.5 (margin static SAFE)
2. Recent 2-3 SOT vs Antwerp/Charleroi (poate fi norma)
3. Genk OK home defensive (apărare decentă)

---

## 🎯 RECOMANDARE

### ⚠️ **U4.5 SOT Standard @ 1.23 = MARGINAL PASS**

**Motive:**
- Cota 1.23 nu reflectă risc TITLE PLAYOFFS
- Sample recent 2/3 hit U4.5 (66.7%) sub 80% market
- Variance demonstrată (2 vs 7 SOT extremes)

### Alternative pe acest meci:

**Mai SAFE: U3.5 Standard @ 1.55**
- λ_bk 2.95 vs line 3.5 = margin 0.55 (mai mic safety)
- BUT cota mai mare reflectă risc
- Hit rate Poisson 65.9% vs implied 64.5%
- Fair price, slight value

**MAI BUN — total combined:**
**Total Suturi U9.5 @ 1.52** (din ecran)
- Combined λ_bk = 5.45 + 2.95 = 8.40
- Poisson(8.40) P(U9.5) = ~67%
- Market implied 65.8%
- Edge **+1pp** = fair-mic value
- Distrabuit pe combined = **mai stabil decât per-team**

### 🧠 Alternative MAI BUNE de azi (dacă vrei value real):

| Pick | Cotă | EV | Confidence |
|---|---|---|---|
| **Benfica O3.5 SOT** | 1.13+ | +5% | HIGH 9/10 |
| **Toulouse-Monaco U12.5 Corners** | 1.20+ | +5-7% | HIGH 9/10 |
| **Entella-Padova U4.5 Goals** | 1.10+ | +5% | PREMIUM 10/10 |

### Lecție din sesiune

**Cotele mici (1.20-1.30) NU înseamnă safe.** Dacă context dictează (title playoffs, must-win), cota poate ascunde risc real.

Per CoVe v1.0: pentru UNDER pe SOT, **rule de bază**:
> "Only trust if lambda_bk < line - 1.0 AND opponent defense elite AND team in clear decline"

Pe Standard:
- ✅ λ_bk 2.95 < 3.5 (line - 1.0)
- ❌ Genk apărare NU elite home (only 6W home)
- ❌ Standard NU e in decline (3rd in playoff, just lost 1-2 to Antwerp)

→ Doar 1 din 3 condiții îndeplinite = **NU bet U4.5**

---

## 📚 Sources

- [Genk vs Standard Liège preview (Sportsgambler)](https://www.sportsgambler.com/betting-tips/football/genk-vs-standard-liege-prediction-lineups-odds-2026-04-25/)
- [Match preview (BetMines)](https://betmines.com/match-preview/genk-vs-standard-liege-prediction-match-preview-and-analysis-pro-league-25-04-2026)
- [Standard Liège stats (Soccerstats)](https://www.soccerstats.com/teamstats.asp?league=belgium&stats=u7301-standard-liege)
- [Standard Liège FBref](https://fbref.com/en/squads/33c6b26e/Standard-Liege-Stats)
- [KRC Genk Footystats](https://footystats.org/clubs/krc-genk-533)
