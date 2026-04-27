# CoVe — SOT Per-Team: Sønderjyske vs FC Midtjylland (DK1)
## Date: 2026-04-23 (Joi)
## Template v1.0 (1.0.6CoVe_SOT.md) | Model v2.1 calibrated
## Stadium: Sydbank Park | 14:00 UTC | Danish Superliga

---

## 🎯 MATCH CONTEXT

**Sønderjyske (struggling)** vs **FC Midtjylland (top-2 contender)** — Danish Superliga clash.

**Key storyline:**
- **Sønderjyske CRISIS:** just lost 6-0 vs Brondby (!!) — 2 straight defeats
- **Midtjylland IN FORM:** 2 straight wins, latest 2-1 vs AGF
- **Clear class gap:** Midtjylland elite vs Sønderjyske floundering
- BTTS 71% expected, Over 2.5 goals 62% expected

---

## 🎯 USER REQUEST: Midtjylland Sub 6.5 SOT @ 1.40

Răspuns scurt: **HARD PASS — class-gap blind spot** (same logic ca Anderlecht Under).

Detaliu mai jos.

---

## STEP 0 — HARD PASS FILTERS

### A. Class Gap Check 🔴 CRITICAL

**Midtjylland top-2 AWAY vs Sønderjyske lower mid HOME** = **EXTREME CLASS GAP AWAY SCENARIO**.

Per template:
> "Top-6 team AWAY vs bottom-10 team → model UNDERESTIMATES top team SOT → **HARD PASS any UNDER bet on favorite**"

**EXACT scenariul blind spot.**

Context reinforcer:
- Sønderjyske 6-0 LOSS vs Brondby last (defensive disaster)
- Midtjylland 2 wins in a row (momentum)
- Midtjylland wins probability 62% per analytics
- Sonderjyske conceded 19 in last 10 matches
- **Midtjylland will ATTACK HEAVILY** vs Sønderjyske's broken defense

### B. Match State
- ⚠️ Sonderjyske mental crisis (6-0 loss + 2 defeats)
- ✅ Midtjylland confident, attacking
- No derby, no weather issues

### C. Data Quality
- Both teams 30+ matches DK1 ✅
- DK1 scaling 1.80 (config validated)

### D. League Scaling
- DK1 = 1.80 per config
- Model uses this ✅

---

## STEP 1 — MODEL OUTPUT

| Side | Team | λ_our | λ_bk | k | Elo |
|---|---|---|---|---|---|
| home | Sønderjyske | 2.69 | 4.85 | 9.01 | 0.99 |
| away | Midtjylland | 2.60 | **4.67** | 13.64 | **1.08** |
| **Total** | — | — | **9.52** | — | — |

---

## STEP 2 — BOOKMAKER vs MODEL (critical gap)

### Sønderjyske (home) bookmaker lines: 2.5/3.5/4.5

| Line | Over | Under | No-vig Over | Model Over | Edge vs NV |
|---|---|---|---|---|---|
| 2.5 | 1.35 | 3.05 | 69.3% | 80.0% | **+10.7pp** ⚠️ SUSPICIOUS |
| 3.5 | 1.85 | 1.88 | 50.4% | 65.4% | **+14.9pp** ⚠️ FALSE |
| 4.5 | 2.82 | 1.40 | 33.2% | 50.0% | **+16.8pp** ⚠️ FALSE |

⚠️ **Bookmaker prices Sønderjyske ~3.5 SOT expected** (lines 2.5/3.5/4.5 centered on 3.5).
**Model says 4.85** — **model OVER-predicts Sønderjyske by ~1.3 SOT!**

Why? Model doesn't capture "weak home team PRESSED into defensive shell" scenario.

### Midtjylland (away) bookmaker lines: 4.5/5.5/6.5

| Line | Over | Under | No-vig Over | Model Over | Edge vs NV |
|---|---|---|---|---|---|
| 4.5 | 1.50 | 2.50 | 62.5% | 48.2% | **-14.3pp** ⚠️ FALSE |
| 5.5 | 1.98 | 1.78 | 47.3% | 33.3% | **-14.0pp** ⚠️ FALSE |
| **6.5** | **2.82** | **1.40** | **33.2%** | **21.6%** | **-11.6pp** ⚠️ FALSE |

⚠️ **Bookmaker prices Midtjylland ~5.5 SOT expected** (away lines 4.5/5.5/6.5).
**Model says 4.67** — **model UNDER-predicts Midtjylland by ~0.8 SOT!**

**Classic class-gap blind spot confirmed.**

### Total match

| Line | Over | Under | No-vig Over | Model Over | Edge |
|---|---|---|---|---|---|
| 8.5 | 1.65 | 2.15 | 56.6% | 61.1% | +4.5pp |
| 9.5 | 2.10 | 1.67 | 44.3% | 48.1% | +3.8pp |
| 10.5 | 2.80 | 1.40 | 33.3% | 35.7% | +2.4pp |

**Total model ~9.5 vs bookmaker ~9** — aliniat. Errors per-team cancel in total.

---

## 🎯 FOCUS: Midtjylland Sub 6.5 SOT @ 1.40

### Edge analysis

**Model raw P(Midtjylland < 6.5) = 78.4%**

| Metric | Valoare |
|---|---|
| Model raw | 78.4% |
| Model adjusted (class gap -10pp Under) | **~68%** |
| Bookmaker implied | 71.4% |
| No-vig fair | 66.8% |
| **Edge vs implied** | **-3.4pp** ❌ |
| **Edge vs no-vig** | **+1.2pp** (near fair) |

### Interpretare

**Pure model:** Ar sugera Under 6.5 = value (+7pp vs implied).

**Adjusted class gap:**
- Bookmaker implies Midtjylland lambda ~5.5 (not 4.67)
- La λ=5.5, P(Under 6.5) = 67.5%
- La λ=6.0 (Midtjylland hot + sonderjyske defensive crisis), P(Under 6.5) = 60%
- **Real P(Under 6.5) probabil 60-68%**

At 1.40 = need >= 71.4% to be positive EV.
**Real probability 60-68% = NEGATIVE EV.**

### VERDICT: Midtjylland Sub 6.5 @ 1.40 = **HARD PASS**

Motive:
1. **Class gap blind spot** (Midtjylland top AWAY, model underpredicts)
2. **Sonderjyske defensive crisis** = more quality chances pt Midtjylland
3. **Midtjylland hot form** (2 wins) + attacking mindset
4. **Bookmaker fair-priced** at 1.40 = 71.4% implied, real prob ~65% = NEGATIVE
5. Edge după ajustare = **-3 până la -7pp**

---

## STEP 3 — EXTERNAL RESEARCH

### Sønderjyske (home, 11-12th DK1)

**Source:** [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/sonderjyske-vs-fc-midtjylland-prediction-lineups-odds-2026-04-23/), [Soccervital](https://www.soccervital.com/sonderjyske-vs-midtjylland-soccer-prediction-jgg28846g.html)

**Form crisis:**
- **Lost 6-0 to Brondby last match** (defensive collapse)
- 2 consecutive defeats
- Last 10: 2W-4D-4L = 20% win rate
- Scored 10, conceded 19 in last 10 (1.00 avg scored, 1.9 conceded)

**Prediction context:**
- 62% Midtjylland wins
- Analytics favor Midtjylland heavily

### FC Midtjylland (away, top-2)

**Form:**
- **2 wins straight** (latest 2-1 home vs AGF)
- Title contender / CL push
- Last 10: 4W-4D-2L = 40% win rate
- Scored 14, conceded 9 in last 10 (1.4/0.9)
- **Net positive goal difference** in recent form

**Style:**
- 4-3-3 attacking, European experience
- Historically DK1 top finishers
- Hot away form

### Implications for SOT

**Midtjylland expected to generate MANY shots:**
- Sonderjyske defensive broken (6-0 loss)
- Midtjylland in form + attacking system
- Class gap away scenario
- **Real Midtjylland SOT: 5-7 range** (not 4.67 per model)

**Sonderjyske expected to generate FEW shots:**
- Pressed into defensive mode
- Morale low after 6-0
- Counter-attack only realistic
- **Real Sonderjyske SOT: 3-4 range** (not 4.85 per model)

---

## ⚡ SCORING

### Midtjylland Sub 6.5 @ 1.40

| Factor | Score |
|---|---|
| Lambda margin (4.67 vs 6.5 = -1.83) — PASSES strict rule (< line-1) | 3/3 |
| Class-gap blind spot (Midtjylland top away) | 0/2 ❌ BLOCKED |
| Match state (Midtjylland momentum) | 0/2 (contra) |
| Home/Away context (top away favors Over) | 0/2 |
| Gut feel | 0/1 |
| **TOTAL** | **3/10** — **HARD PASS** |

Lambda margin passes strict rule BUT class gap + match state override model signal.

---

## 🎯 Alternative picks pe acest meci

### ✅ OPTION A: **Sonderjyske Sub 3.5 SOT @ 1.88** (class gap reverse — weak home pressed)

- Model raw P(U3.5) = 34.6%
- Class gap adjustment: Sonderjyske pressed → lambda drops to ~3.0
- Adjusted P(U3.5) = **55-60%**
- Implied @ 1.88 = 53.2%
- **Edge: +2-7pp POSITIVE**
- ⚠️ But "weak home pressed" not formally in template — moderate confidence

### ✅ OPTION B: **Sonderjyske Sub 4.5 SOT @ 1.40**

- Model raw P(U4.5) = 50%
- Class gap adjustment: Sonderjyske real ~3.5 → P(U4.5) = **72-78%**
- Implied @ 1.40 = 71.4%
- **Edge: +1-7pp** (marginal but positive)
- Safer line (4.5 easier to hit Under)

### ✅ OPTION C: **Midtjylland Peste 4.5 @ 1.50**

- Model raw 48.2%
- Class gap adjustment: +10pp → **58%**
- Implied 66.7%
- **Edge: -8pp NEGATIV** (bookmaker prices this correctly)

### ❌ OPTION D: Total Over 8.5 @ 1.65

- Model 61.1%, implied 60.6% → +0.5pp near fair
- No value

---

## 🎯 FINAL VERDICT

### ❌ **Midtjylland Sub 6.5 @ 1.40 — HARD PASS**

Motiv: **Class-gap blind spot** (identical Anderlecht pattern). Modelul subestimează Midtjylland SOT. Bookmakerul prețuiește corect.

### ✅ Alternative viabile:

**🏆 BEST: Sonderjyske Sub 4.5 SOT @ 1.40**
- Class-gap reverse: weak home pressed → fewer SOT
- Model adjusted ~75% Under
- Fair ~1.33, offered 1.40 = +3-5pp edge
- Stake: **10 RON (1%)**

**🥈 MODERATE: Sonderjyske Sub 3.5 @ 1.88** (higher variance)
- Model adjusted ~57% Under
- Fair ~1.75, offered 1.88 = +3-5pp edge
- Stake: 5 RON (0.5%)

### ❌ SKIP:

- ❌ Midtjylland Sub 6.5/5.5/4.5 — class gap false signals
- ❌ Sonderjyske Over 2.5/3.5/4.5 — model OVER-predicts (will be pressed)
- ❌ Midtjylland Over 4.5/5.5/6.5 — bookmaker already prices correctly, no value
- ⚠️ Total O/U — near fair, no edge

---

## 🧠 LECȚIE MAJORĂ

**Același pattern ca Anderlecht (ieri) sau Fiorentina-Lecce (20 aprilie):**

Când echipa TOP joacă AWAY vs echipa SLABĂ HOME:
- Model → subestimează top team SOT (Under value fake)
- Model → supraevalueaza weak home SOT (Over value fake)
- **Realitatea:** weak team pressed, puține SOT; top team dominant, multe SOT

**Regulă practică confirmată:**
- ✅ BET: Weak home team UNDER (reverse class gap, model over-estimates weak team)
- ❌ SKIP: Top away team UNDER (model under-estimates top team)

---

## 📋 PORTFOLIO azi (revizuit cu acest match)

| Pick | Stake | Bookmaker | Edge |
|---|---|---|---|
| **PSV Under 9.5 SOT** | 20 RON (2%) | @ 1.45 | +9-23pp |
| **STV Over 3.5 SOT** | 15 RON (1.5%) | @ 1.38 | +4.6pp |
| **Total STV-And O8.5** | 15 RON (1.5%) | @ 1.70 | +4.6pp |
| **Sonderjyske Sub 4.5 SOT** | 10 RON (1%) | @ 1.40 | +3-5pp |
| **TOTAL** | **60 RON (6%)** | — | — |

Sub limita 8% daily ✅. **20 RON disponibil pentru WTA/Corners.**

---

## Sources

**Preview & lineups:**
- [Sportsgambler Sonderjyske vs Midtjylland](https://www.sportsgambler.com/betting-tips/football/sonderjyske-vs-fc-midtjylland-prediction-lineups-odds-2026-04-23/)
- [Soccervital prediction](https://www.soccervital.com/sonderjyske-vs-midtjylland-soccer-prediction-jgg28846g.html)
- [MyBets analysis](https://www.mybets.today/soccer-predictions/match-prediction-analysis-sonderjyske-vs-midtjylland-betting-tip-dii28846i/)
- [Sofascore FC Midtjylland](https://www.sofascore.com/football/team/fc-midtjylland/1289)
- [Fox Sports match](https://www.foxsports.com/soccer/danish-superliga-sonderjyske-vs-midtjylland-apr-23-2026-game-boxscore-729921)

**League context:**
- [Wiki 2025-26 Danish Superliga](https://en.wikipedia.org/wiki/2025%E2%80%9326_Danish_Superliga)
- [OddsPortal Superliga](https://www.oddsportal.com/football/denmark/superliga/)
- [Flashscore DK1](https://www.flashscore.com/football/denmark/superliga/)
