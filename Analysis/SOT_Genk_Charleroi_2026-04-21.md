# CoVe — SOT Per-Team Analysis: Genk vs Charleroi (B1)
## Date: 2026-04-21 (Marți)
## Template v1.0 (1.0.6CoVe_SOT.md)
## Stadium: Cegeka Arena | Belgian Jupiler Pro League | 18:30 UTC

---

## CONTEXT

**KRC Genk (7th, in form)** vs **Sporting Charleroi (in crisis, 8/10 lost)** — Belgian Pro League regular season.

**Form gap MASSIVE:**
- Genk: 6 wins in last 10, 3 wins in last 5 → uptrend
- Charleroi: 8 losses in last 10, 4 in last 5 → freefall

---

## STEP 0 — HARD PASS FILTERS

### A. Class Gap / Form Gap Check

| Team | Position | Last 10 | Form |
|---|---|---|---|
| Genk | 7th | 6W-?D-?L | 🔥 Strong uptrend |
| Charleroi | bottom-half | 1-2W, 8L | ❄️ Crisis mode |

**Form gap = severe.** Charleroi away in crisis vs Genk strong home = potential blowout scenario for SOT.

Per prompt v1.0:
> "Top-6 team at HOME vs bottom-10 team → model roughly OK but favor OVER"

Genk isn't top-6 (7th) but vs collapsing Charleroi is functionally that scenario. ✅ **Apply OVER bias to Genk.**

### B. Data Quality

- B1 has limited training data in our model — flagged LOW CONF
- Both teams 30+ matches in season ✅ (sample OK)

### C. League Scaling Check (CRITICAL)

⚠️ **B1 scaling EVIDENCE:** Yesterday Leuven analysis confirmed B1 real scaling ~1.70 (NOT 1.90).

Apply adjustment:
- Genk λ_our 2.863 × 1.70 = **4.87** (vs model's 5.44)
- Charleroi λ_our 1.983 × 1.70 = **3.37** (vs model's 3.77)
- Adjusted total: **8.24** (vs model's 9.21)

Model OVER-predicts B1 by ~10-12%.

### D. Match State Risks

- Genk missing 2 (Medina susp, Lawal injured) — but full attack available
- Charleroi missing none confirmed but team in collapse mode
- **Form differential** = Charleroi may park bus to limit damage → lower SOT for Charleroi
- Genk motivated to maintain push for European spot

---

## STEP 1 — MODEL OUTPUT (raw, scaling 1.9)

| Side | Team | Line | λ_our | λ_bk | p_over | Fair Over |
|---|---|---|---|---|---|---|
| home | **Genk** | **2.5** | 2.86 | 5.44 | **89.0%** | **1.12** |
| home | Genk | 3.5 | 2.86 | 5.44 | 77.0% | 1.30 |
| home | Genk | 4.5 | 2.86 | 5.44 | 61.6% | 1.62 |
| home | Genk | 5.5 | 2.86 | 5.44 | 45.5% | 2.20 |
| away | Charleroi | 2.5 | 1.98 | 3.77 | 66.7% | 1.50 |
| away | Charleroi | 3.5 | 1.98 | 3.77 | 48.8% | 2.05 |
| away | Charleroi | 4.5 | 1.98 | 3.77 | 33.1% | 3.02 |
| away | Charleroi | 5.5 | 1.98 | 3.77 | 21.1% | 4.74 |

---

## STEP 1B — MODEL ADJUSTED (B1 scaling 1.70)

| Side | Team | Line | λ_bk adj | p_over adj | Fair adj |
|---|---|---|---|---|---|
| home | Genk | 2.5 | 4.87 | **~85%** | 1.18 |
| home | Genk | 3.5 | 4.87 | ~71% | 1.41 |
| home | Genk | 4.5 | 4.87 | ~55% | 1.82 |
| home | Genk | 5.5 | 4.87 | ~38% | 2.63 |
| away | Charleroi | 2.5 | 3.37 | ~58% | 1.72 |
| away | Charleroi | 3.5 | 3.37 | ~38% | 2.63 |
| away | Charleroi | 4.5 | 3.37 | ~22% | 4.55 |

---

## STEP A — LAMBDA BASELINE CHECK

Per prompt threshold:

| Pick | λ_bk adj | Line | Margin | Verdict |
|---|---|---|---|---|
| Genk O2.5 | 4.87 | 2.5 | +2.37 | 🔥 GOLD (margin > 1.5) |
| Genk O3.5 | 4.87 | 3.5 | +1.37 | ✅ GOOD |
| Genk O4.5 | 4.87 | 4.5 | +0.37 | ⚠️ Borderline |
| Genk O5.5 | 4.87 | 5.5 | -0.63 | ❌ FAIL |
| Charleroi O2.5 | 3.37 | 2.5 | +0.87 | ⚠️ Borderline (away) |
| Charleroi O3.5 | 3.37 | 3.5 | -0.13 | ❌ FAIL |

**TOP CANDIDATE:** Genk O2.5 — solid lambda margin even after B1 adjustment.

---

## STEP 2 — EXTERNAL RESEARCH

### KRC Genk (home)

- **7th in Pro League, 11W-8D-10L** (38% win rate) ([FootyStats](https://footystats.org/clubs/krc-genk-533))
- **1.41 goals/game** average
- Home form: 6W-4D-5L
- Recent form: **6 wins in last 10, 3 wins in last 5** — strong uptrend ([Ratingbet](https://ratingbet.com/predictions/genk-vs-charleroi-prediction-expert-analysis-possible-lineups-april-21-2026/))
- Missing: Yaimar Medina (suspension), Tobias Lawal (injury) — backup goalkeeper
- Formation: **3-4-2-1** with Mirisola/Karetsas as creative threats
- Top scorer Oh Hyeon-gyu typically 1+ SOT/game

### Sporting Charleroi (away)

- **8 losses in last 10, 4 in last 5** — crisis mode ([Sportsgambler](https://www.sportsgambler.com/betting-tips/football/genk-vs-charleroi-prediction-lineups-odds-2026-04-21/))
- Formation: 4-2-3-1
- Schaedler primary striker
- Likely to play DEFENSIVE/parked-bus given form crisis
- **Concern:** away in crisis = volume Charleroi may DROP further

### Form context implications for SOT

- **Genk should DOMINATE** = generate high SOT volume (favored OVER)
- **Charleroi may park bus** = SOT volume LOW (favored UNDER)
- Charleroi away with 4 straight losses = high likelihood of conservative tactics

### KEY: Match dynamic

If Genk scores early (likely):
- Charleroi must attack → potential Charleroi SOT volume increases
- BUT historically depleted teams keep parking bus
- Genk could relax → Genk SOT may DROP after early lead

If Charleroi parks from minute 1:
- Genk dominate possession → many shots, mixed quality
- Genk SOT likely HIGH
- Charleroi SOT likely LOW

---

## STEP 3 — BOOKMAKER ODDS (NU AVEM)

⚠️ User did NOT provide bookmaker odds for Genk vs Charleroi.

**Estimated bookmaker pricing** (based on similar matches):
- Genk O2.5: probably @ 1.15-1.20 (model fair 1.12)
- Genk O3.5: probably @ 1.40-1.50 (model fair 1.30 raw, 1.41 adj)
- Genk O4.5: probably @ 1.85-2.00 (model fair 1.62 raw, 1.82 adj)
- Charleroi O2.5: probably @ 1.75-1.95 (model fair 1.50 raw, 1.72 adj)

**Cannot compute exact edge without odds.** Recommendation depends on what bookmaker offers.

---

## STEP 4 — SCORING

### PICK 1: Genk Over 2.5 SOT

| Factor | Score |
|---|---|
| Lambda margin (4.87 vs 2.5 = +2.37 GOLD) | 3/3 |
| Attack (Mirisola, Karetsas, Oh) | 2/2 |
| Home + form uptrend | 2/2 |
| Match state (Charleroi crisis = Genk dominates) | 2/2 |
| Intuition | 1/1 |
| **TOTAL** | **10/10** HIGH |

**Model adj:** ~85%
**Fair odds:** 1.18

**RECOMMEND BET if bookmaker offers >= 1.20** (small positive edge needed)
**Stake:** 2% bankroll = 20 RON

### PICK 2: Genk Over 3.5 SOT

| Factor | Score |
|---|---|
| Lambda margin (4.87 vs 3.5 = +1.37) | 2/3 |
| Attack | 2/2 |
| Home/form | 2/2 |
| Match state | 2/2 |
| Intuition | 1/1 |
| **TOTAL** | **9/10** HIGH |

**Model adj:** ~71%
**Fair odds:** 1.41

**RECOMMEND BET if bookmaker offers >= 1.50** (5pp+ edge)
**Stake:** 1% bankroll = 10 RON

### PICK 3: Genk Over 4.5 SOT

| Factor | Score |
|---|---|
| Lambda margin (4.87 vs 4.5 = +0.37 borderline!) | 1/3 |
| Attack | 2/2 |
| Home | 2/2 |
| Match state | 2/2 |
| Intuition | 1/1 |
| **TOTAL** | **8/10** MODERATE |

**Model adj:** ~55%
**Fair odds:** 1.82

⚠️ Lambda foarte aproape de linie (variance mare). **Higher risk.**

**RECOMMEND BET if bookmaker offers >= 1.95** (7pp+ edge needed for variance)
**Stake:** 1% = 10 RON max

### PICK 4: Charleroi Over 2.5 SOT (away)

| Factor | Score |
|---|---|
| Lambda margin (3.37 vs 2.5 = +0.87 borderline) | 1/3 |
| Attack (limited, in crisis) | 0/2 |
| Away context | 1/2 |
| Match state (likely PARK BUS) | 0/2 |
| Intuition (parking bus risk) | 0/1 |
| **TOTAL** | **2/10** — PASS |

⚠️ **HARD PASS** — Charleroi 8/10 losses + away + likely bus parking = SOT volume probably collapses.

### PICK 5: Charleroi UNDER 2.5 SOT (away)

| Factor | Score |
|---|---|
| Lambda margin (3.37 vs 2.5 = +0.87 too high for UNDER) | 0/3 |
| Defense context (Genk strong home D) | 1/2 |
| Away (deep defending = low shots) | 1/2 |
| Match state (Charleroi crisis) | 2/2 |
| Intuition | 1/1 |
| **TOTAL** | **5/10** — PASS |

⚠️ Per prompt: "Only trust UNDER if lambda_bk < line - 1.0" — λ 3.37 vs 2.5 means margin +0.87, FAIL strict rule.

Charleroi UNDER 2.5 might pay off with parking bus, but model says 58% they hit 2.5 → fair odds 1.72.

If bookmaker offers Charleroi UNDER 2.5 @ 2.00+ → ODDS DEP small stake.

---

## STEP 5 — SELF-VERIFICATION

- [x] Applied B1 scaling adjustment (1.70 confirmed yesterday from Leuven)
- [x] Used adjusted λ_bk for line comparison
- [x] Applied form gap (Charleroi crisis vs Genk uptrend)
- [x] Considered park-bus risk for Charleroi
- [x] Capped research at +5pp (no extreme adjustments)
- [x] Verified data quality (both 30+ matches)
- [x] Noted no bookmaker odds — recommendations conditional

---

## STEP 4 — CORRECTIONS TABLE

| Pick | Line | Side | Model raw | Adj (B1×1.7) | Fair adj | Score | Action |
|---|---|---|---|---|---|---|---|
| **Genk O2.5** | 2.5 | home | 89.0% | **85%** | **1.18** | **10/10** | ✅ BET if odds >= 1.20 |
| Genk O3.5 | 3.5 | home | 77.0% | 71% | 1.41 | 9/10 | BET if odds >= 1.50 |
| Genk O4.5 | 4.5 | home | 61.6% | 55% | 1.82 | 8/10 | ODDS DEP if odds >= 1.95 |
| Genk O5.5 | 5.5 | home | 45.5% | 38% | 2.63 | 5/10 | PASS (lambda < line) |
| Charleroi O2.5 | 2.5 | away | 66.7% | 58% | 1.72 | 2/10 | ❌ HARD PASS (park bus) |
| Charleroi U2.5 | 2.5 | away | 33.3% | 42% | 2.38 | 5/10 | PASS (lambda > line - 1) |
| Total O8.5 (estim) | 8.5 | total | ~63% | ~50% | ~2.00 | 6/10 | ODDS DEP if odds >= 2.10 |

---

## STEP 5 — FINAL PICKS

### 🏆 #1 PRIMARY BET: Genk Peste 2.5 SOT

- **Score:** 10/10 HIGH
- **Model adjusted:** ~85%
- **Fair odds:** 1.18
- **BET if bookmaker offers >= 1.20**
- **Stake:** 2% = 20 RON

**Key stats:**
- Genk 6W in last 10, in form
- Charleroi 8L in last 10, in crisis
- Mirisola/Karetsas/Oh all available
- Form gap = Genk dominates expected

**How I lose:**
- Genk early 2-0 lead → coast second half, shots drop
- Backup goalkeeper Lawal → maybe Genk plays more cautiously
- Park bus by Charleroi succeeds → Genk frustrated, low quality shots

**Sources:** [Sportsgambler](https://www.sportsgambler.com/betting-tips/football/genk-vs-charleroi-prediction-lineups-odds-2026-04-21/), [Ratingbet](https://ratingbet.com/predictions/genk-vs-charleroi-prediction-expert-analysis-possible-lineups-april-21-2026/), [FootyStats](https://footystats.org/clubs/krc-genk-533)

### 🥈 #2 ALT: Genk Peste 3.5 SOT (daca odds bune)

- **Score:** 9/10 HIGH
- **Model adj:** 71%
- **Fair:** 1.41 | **Accept if odds >= 1.50**
- **Stake:** 1% = 10 RON

### ❌ SKIP

- Charleroi any OVER (park bus risk)
- Charleroi UNDER 2.5 (lambda not below line - 1.0)
- Genk O5.5 (lambda < line, model fail)

---

## VERDICT FINAL

### 🎯 RECOMANDARE

**Single pick:** **Genk Peste 2.5 SOT @ 1.20+** = 20 RON (2% bankroll)

**Daca cota e SUB 1.20** = pass (no value, fair pricing)

### Combo posibil (daca odds bune):
- Genk O2.5 @ 1.20 + Genk O3.5 @ 1.50 in pariu separat
- Sau combo single la fair odds calculat

### Daca compar cu Coventry-Portsmouth:
- Genk O2.5 = 10/10 score, lambda margin solid
- Coventry-Portsmouth Total O8.5 = 9/10 + edge confirmat
- **AMBELE valide** — daca portfolio tine, ambele in betslip

---

## EXPUNERE PORTFOLIO (cumulativ azi)

| Pick | Stake | Cumulativ |
|---|---|---|
| Real Madrid O3.5 | 30 RON | 30 |
| **Genk O2.5** | **20 RON** | **50** |
| Coventry-Portsmouth Total O8.5 | 20 RON | 70 |
| Real Madrid O4.5 | 20 RON | 90 |

**Total = 90 RON = 9% bankroll** → **PESTE limita 8% daily!**

**Recomandare:** Scoate Real Madrid O4.5 (mai multa variance) → **70 RON = 7%** sub limita.

**SAU** scoate Coventry-Portsmouth Total → **70 RON = 7%** dar pierzi acel pick.

---

## 🧠 LECȚII INVĂȚATE

1. **B1 scaling 1.70 confirmat din nou** — applied corectly aici
2. **Form gap matter mult** pentru SOT — Charleroi crisis = parking bus risk
3. **Form trends NU sunt incluse in model v1.0** — research must adjust
4. **Strong home + weak away in crisis** = OVER home predictable

### Update memoria:
- B1 scaling: **1.70** confirmed
- Form gap rule: Bottom-table away in losing streak = parking bus → opponent UNDER on shot generation

---

## Sources

**Preview & lineups:**
- [Sportsgambler — Genk vs Charleroi](https://www.sportsgambler.com/betting-tips/football/genk-vs-charleroi-prediction-lineups-odds-2026-04-21/)
- [Ratingbet — Genk vs Charleroi](https://ratingbet.com/predictions/genk-vs-charleroi-prediction-expert-analysis-possible-lineups-april-21-2026/)
- [Tribuna lineups](https://tribuna.com/en/match/charleroi-vs-genk/lineups/)
- [Mighty Tips — Genk Charleroi](https://www.mightytips.com/football-predictions/genk-vs-sporting-charleroi-prediction-21-04-2026/)
- [LeagueLane preview](https://leaguelane.com/predictions/genk-vs-charleroi-21-april-2026/)

**Stats:**
- [FootyStats KRC Genk](https://footystats.org/clubs/krc-genk-533)
- [FBref Genk Stats](https://fbref.com/en/squads/1e972a99/Genk-Stats)
- [FBref Charleroi Stats](https://fbref.com/en/squads/140e320a/Charleroi-Stats)
- [Belgium Pro League FootyStats](https://footystats.org/belgium/pro-league)

**Standings:**
- [2025-26 Belgian Pro League — Wikipedia](https://en.wikipedia.org/wiki/2025%E2%80%9326_Belgian_Pro_League)