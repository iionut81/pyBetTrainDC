# CoVe — SOT Per-Team: Oxford United vs Wrexham (E1)
## Date: 2026-04-21 (Marți)
## Template v1.0 (1.0.6CoVe_SOT.md) | Model v2.1 calibrated
## Stadium: Kassam Stadium, Oxford | 14:45 BST | EFL Championship

---

## CONTEXT

**Oxford United (22nd, relegation fight)** vs **Wrexham (7th, playoff push)** — Championship matchday cu miză masivă pentru ambele echipe.

**PICK analizat:** **Wrexham Sub 4.5 SOT @ 1.47**

Model (v2.1 calibrated):
- lambda_our 1.689 → lambda_bk **3.55** (scaling 2.1 E1)
- Elo_multiplier 1.049 (Wrexham uşor favorit)
- **P(Wrexham > 4.5 SOT) = 27.82%**
- **P(Wrexham <= 4 SOT, adică Under 4.5) = 72.18%**
- Fair odds Under 4.5: **1.39**

---

## STEP 0 — HARD PASS FILTERS

### A. Class Gap Check ⚠️ CRITICAL

| Team | Poziție | Status |
|---|---|---|
| Wrexham | **7th** Championship (40% win rate) | Playoff push, 4 pts din zona 6 |
| Oxford Utd | **22nd** (3rd-from-bottom) | Relegation zone |

Per prompt v1.0:
> "Top-6 team AWAY vs bottom-10 team → **model UNDERESTIMATES top team SOT** → HARD PASS any UNDER bet on favorite, only OVER is usable"

🔴 **ATENTIE:** Wrexham (effectively top-6) AWAY vs bottom-3 Oxford = **BLIND SPOT**. Prompt spune **HARD PASS pentru UNDER pe favorit**.

Contra-argument: Calibration v2.1 + Elo 1.049x au atenuat partial underestimation-ul, dar NU complet (elasticity 0.15 × (Wrexham attack - Oxford defence) e limitata).

### B. Data Quality
- Wrexham 42 meciuri E1 sezonul 2025-26 ✅
- Oxford 42 meciuri ✅
- Model E1 cu 15,520 training predictions (cea mai mare lige)

### C. League Scaling
E1 = 2.10 (calibrat din Coventry, confirmat)

### D. Match State
- Oxford: **disperat să scape de retrogradare** → va presa acasă
- Wrexham: **playoff push** → NU va parca autobuzul
- **Ambele echipe motivate să atace** → poate scădea probabilitatea Under

---

## STEP 1 — HARD DATA (research)

### Wrexham 2025-26 Championship

([FootyStats](https://footystats.org/clubs/wrexham-fc-1837), [FBref](https://fbref.com/en/squads/dad7970b/Wrexham-Stats)):

- **17W-13D-12L** (42 matches)
- **7th in table** (40% win rate, 64 points)
- **Away form: 8W-7D-6L** (GOOD — 42 wins puncte pe teren)
- **Avg SOT per match: 3.80** (ALL competitions)
- Total 464 shots in 42 games = 11.32 shots/match
- Phil Parkinson (manager stabil, fără schimbări)

### Comparație cu model

| Metric | Model | Research | Diff |
|---|---|---|---|
| Avg SOT (season) | 3.55 (bk scale) | **3.80** | model **-0.25 SOT** sub avg |
| Expected > 4.5 | 27.82% | ~35-40% (estimat) | **possible under-prediction** |

**Key insight:** Model sub-prezice Wrexham's SOT cu ~0.3 SOT. Per regula clasă-gap, trebuie **research upgrade +5pp** pe OVER → **p_under scade la ~67%**.

### Wrexham squad (Oxford match)

([Sports Mole](https://www.sportsmole.co.uk/football/wrexham/championship-promotion-race/preview/oxford-utd-vs-wrexham-prediction-team-news-lineups_595968.html), [FootballPredictions](https://footballpredictions.net/predicted-lineups-oxford-united-v-wrexham-21-04-2026)):

**Formație: 3-4-2-1**
- GK: Danny Ward
- DEF: Dominic Hyam, Dan Scarr, Callum Doyle
- MID: Issa Kabore, George Dobson, Matty James, George Thomason
- ATT: Oliver Rathbone, Josh Windass, **Sam Smith** (striker)

**Absențe:**
- Jay Rodríguez (ankle injury) — BUT he's backup attacker, minim impact pe SOT
- Restul: squad complet sănătos ✅

**Interpretare formație 3-4-2-1:**
- 3 fundași centrali + wing-backs Kabore (width)
- 2 creative attacking mids (Rathbone + Windass) behind Smith
- Moderate attacking (NU ultra-defensiv 5-4-1 precum Leuven)
- **Structura permite volum moderat de suturi** — nu park-bus style

### Oxford United (home, 22nd)

Coach: Matt Bloomfield
Formation: 4-2-3-1
- Absent: Tyler Goodrham, Brian De Keersmaecker
- Oxford va presa acasă pentru puncte vitale

---

## STEP 2 — EXTERNAL RESEARCH

### Phil Parkinson context ([ESPN](https://www.espn.com/soccer/story/_/id/48387379/wrexham-phil-parkinson-referees-west-brom-complain), [Goal.com](https://www.goal.com/en-us/lists/phil-parkinson-issues-wrexham-promotion-warning-as-playoffs-near/bltb933839cc575d848))

- **Manager stabil** (fără schimbări) — NU volatility risk
- Recent complaint despre referees = sub tensiune but not chaotic
- **Promovare push to the wire** = determinare masivă
- Wrexham a învins **Stoke 1-0** recent (win vital)

### Psihologie meci

**Pro UNDER 4.5:**
- Parkinson pragmatic coach — not overly attacking
- 3-4-2-1 compact, nu agresiv
- Wrexham în Championship de 2 sezoane — respectă tactici sigure
- Presiune playoff = **NU expune defensiv** (nu risca înfrângere cu 1-2 goluri)

**Contra UNDER 4.5:**
- **Oxford 22nd = defensiv fragil** (cel mai mic rang)
- Wrexham avg 3.80 SOT >> linia 4.5 / 2 = peste medie single match
- Playoff push = motivat să atace OFFENSIV
- 3 sezoane in rise (Netflix franchise) = ambiție

### Tactical fit

**Wrexham va genera SOT dacă:**
- Oxford cedează early goal → deschide pentru counter
- Windass + Rathbone rezistă presiune și livrează pentru Smith
- Wing-backs Kabore/Spencer avansează pe flancuri

**Wrexham va avea SOT scăzut dacă:**
- Oxford presează early și domină posesia
- Smith e izolat (striker singur)
- Wrexham joacă defensiv (park & counter) la 0-0

---

## STEP 3 — MARKET ANALYSIS

### Cotele

**Wrexham Under 4.5 SOT @ 1.47** (din întrebarea ta)

Calcul implied & no-vig (asumând Peste 4.5 la ~2.80):
- Implied with vig: 1/1.47 = **68.0%**
- Assuming Over 4.5 @ 2.80 → 35.7% implied
- Sum: 68.0% + 35.7% = 103.7% → margin 3.7%
- **No-vig Under 4.5: 65.6%**

### Edge calculation

| Source | P(Under 4.5) | Edge vs implied 68% |
|---|---|---|
| Model raw (v2.1) | 76.4% | +8.4pp |
| Model calibrated | **72.2%** | **+4.2pp** |
| Research-adj (class-gap penalty -5pp) | **67.2%** | **-0.8pp** |

**After class-gap adjustment: edge ≈ 0pp (FAIR).**

---

## STEP 4 — SCORING

### Factor breakdown

| Factor | Score |
|---|---|
| Lambda margin (3.55 vs 4.5 = -0.95, passes strict rule) | 2/3 |
| Attack/Defense (Wrexham solid attack vs Oxford cedat) | 1/2 |
| Away context (λ 3.55 ~ away avg 3.80, close to line) | 1/2 |
| Match state (both need points = open game) | 1/2 |
| Intuition (Parkinson pragmatic = controlled tempo) | 1/1 |
| **TOTAL** | **6/10 — ODDS DEP** |

**Score borderline** — la 6/10 pariezi DOAR dacă edge > 3pp AFTER research.

### Post-research edge: ~0pp → PASS per scoring rule

---

## STEP 5 — FINAL VERDICT

### 🎯 Wrexham Sub 4.5 SOT @ 1.47

**Score:** 6/10 MODERATE
**Model calibrated P(Under):** 72.2%
**Research-adjusted P(Under):** **67.2%** (-5pp class-gap penalty)
**Implied probability:** 68.0%
**No-vig fair:** 65.6%

**Edge (adjusted vs implied):** -0.8pp  
**Edge (adjusted vs no-vig):** +1.6pp

**Verdict: ⚠️ FAIR PRICING — NO STRONG VALUE**

### Motive

1. **Clasă-gap blind spot activ** — Wrexham favorit away vs Oxford last-place. Model sub-prezice SOT favorit.
2. **Avg Wrexham 3.80 SOT** — linia 4.5 e doar 0.7 peste medie (aproape de line = variance)
3. **Ambele echipe atacă** — Oxford desperat de puncte, Wrexham playoff
4. **Formatie 3-4-2-1** — nu ultra-defensivă (ca Leuven 5-4-1)
5. **Nicio schimbare semnificativă** (no manager change, clean squad)

### PASS Reasons

- Edge post-adjustment ~0pp — no value
- 1.47 = cota mică pentru risc clasă-gap
- Wrexham away form GOOD (nu echipă în criză)
- Score borderline 6/10

### Daca TREBUIE sa pariezi

- **Stake minim:** 5-10 RON (0.5-1%) doar DACĂ bookie >= 1.50
- La **1.47**, aproape break-even → **PASS**

---

## ALTERNATIV: Check Over 4.5 Wrexham

Cota Peste 4.5: probabil 2.50-2.80
- Model P(Over 4.5) = 27.8%
- Research-adj: 32.8% (class-gap boost +5pp)
- Implied @ 2.80 = 35.7%
- Edge: **-3pp** (still negative)

**Over 4.5 NOT value either.**

---

## 🎯 RECOMANDARE FINALĂ

**PASS Oxford vs Wrexham SOT markets.**

Model aliniat cu bookmaker după class-gap adjustment. **Niciun side nu oferă value >3pp.**

### Dacă vrei expunere pe meciul ăsta, alternativ:

- **Wrexham Winner @ odds pe meci** (separat, non-SOT)
- **Total match goals/corners** (check alte CoVe modele)
- **Wrexham win + BTTS** (ambele echipe motivate = likely gol)

---

## 🧠 LECȚII LEARNED

1. **Clasă-gap rule applies on 7th vs 22nd too** (not just top-3 vs bottom-3)
2. **Championship cu ~3.80 SOT avg** → linia 4.5 e aproape de medie, high variance
3. **Elo multiplier 1.05 small** nu elimină complet class-gap blind spot
4. **Fair pricing sub 1.50** = skip majoritatea cazurilor (low profit vs risk)

---

## Sources

**Preview & lineups:**
- [Sports Mole — Oxford vs Wrexham](https://www.sportsmole.co.uk/football/wrexham/championship-promotion-race/preview/oxford-utd-vs-wrexham-prediction-team-news-lineups_595968.html)
- [Sportsgambler — Oxford vs Wrexham](https://www.sportsgambler.com/betting-tips/football/oxford-vs-wrexham-prediction-lineups-odds-2026-04-21/)
- [FootballPredictions lineups](https://footballpredictions.net/predicted-lineups-oxford-united-v-wrexham-21-04-2026)
- [Last Word On Football — team news](https://lastwordonsports.com/football/2026/04/20/team-news-form-predicted-line-up-oxford-united-welcome-wrexham/)

**Wrexham stats:**
- [FootyStats Wrexham](https://footystats.org/clubs/wrexham-fc-1837)
- [FBref Wrexham](https://fbref.com/en/squads/dad7970b/Wrexham-Stats)
- [Wrexham Wiki 2025-26](https://en.wikipedia.org/wiki/2025%E2%80%9326_Wrexham_A.F.C._season)

**Manager context:**
- [ESPN — Parkinson referee complaint](https://www.espn.com/soccer/story/_/id/48387379/wrexham-phil-parkinson-referees-west-brom-complain)
- [Goal.com — Wrexham playoff push](https://www.goal.com/en-us/lists/phil-parkinson-issues-wrexham-promotion-warning-as-playoffs-near/bltb933839cc575d848)

**Injuries:**
- [Transfermarkt Wrexham](https://www.transfermarkt.co.uk/wrexham-afc/sperrenundverletzungen/verein/1112)
- [BeSoccer Wrexham](https://www.besoccer.com/team/injuries-suspensions/wrexham)
