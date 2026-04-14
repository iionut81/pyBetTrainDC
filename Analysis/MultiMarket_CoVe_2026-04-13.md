# CoVe Multi-Market v1.0 — 2026-04-13

**Data sources:**
- DC Model: Dixon-Coles 20 leagues (retrained 13 April on Flashscore, 22,730 matches)
- Goals Model: Dixon-Coles per league (retrained 13 April)
- Corners Model: Negative Binomial (retrained 13 April, 18,915 corners history)
- Tennis Model: Markov + Surface Elo
- Research: Fresh internet per match from individual CoVe analyses today

---

## STEP 1: RUN ALL CHECKLISTS

### ⚽ Tondela vs Gil Vicente — DC X2 @ 1.25

*(Full analysis in `DC_Analysis_Tondela_GilVicente_2026-04-13.md`)*

| Step | Check | Result |
|------|-------|--------|
| A — Form | Gil Vicente 4W-2D-4L last 10. 4L all vs top teams (Porto, Benfica, Estoril, Santa Clara). | ⚠️ 4L = template PASS threshold, but context: 0L vs weaker teams |
| B — Away split | Away: 4W-6D-4L = 71% X2 coverage. | ✅ |
| C — Opponent | Tondela **17th**, **1 win in 14 at home** (7.1%!), 0.64 goals/game home. New coach (5-0 loss first match). President-coach scandal. | 🔥🔥 GOLD |
| D — Odds | 1.25 → ✅ Good range | ✅ |
| E — Context | Not derby. Gil Vicente fights for Europe. Tondela in total chaos. | ✅ |

**Score:** Form 1 + Away 2 + Opponent 2 + Context 2 + Gut 1 = **8/10**

**Final Q:** "Can Tondela WIN outright?" → 1 home win in 14 (7.1%). 0.64 goals/game. New coach with 5-0 loss. **~12% chance.** → ✅ DC VALID

**Sources:** [Sports Mole](https://www.sportsmole.co.uk/football/tondela/preview/tondela-vs-gil-vicente-prediction-team-news-lineups_595558.html), [FootyStats Gil Vicente](https://footystats.org/clubs/gil-vicente-fc-183), [Record — Feio](https://www.record.pt/futebol/futebol-nacional/liga-betclic/tondela/detalhe/oficial-goncalo-feio-e-o-novo-treinador-do-tondela)

| Metric | Value |
|--------|-------|
| Model | 84.3% |
| Research | +3pp = **87.3%** |
| Implied (1.25) | 80.0% |
| **Edge** | **+7.3%** |

---

### ⚽ Manchester United vs Leeds — DC 1X @ 1.16

*(Full analysis in `DC_Analysis_2026-04-13.md`)*

| Step | Check | Result |
|------|-------|--------|
| A — Form | Man Utd 1L in 10 under Carrick. 3rd place, CL race. | ✅ AUTO QUALIFY |
| B — Home | Unbeaten vs Leeds at home since 1981. | 🔥 |
| C — Opponent | Leeds 15th-16th, 0 goals in 4 PL, 1 away win in 15. | 🔥 |
| D — Odds | **1.16** → ⚠️ Thin (1.08-1.15 = thin for single bet) | ⚠️ -1 penalty |
| E — Context | **Derby** (Roses) → ⚠️ -1. **4-5 defenders out** (Mazraoui, De Ligt, Maguire susp, Dorgu, Dalot doubtful) → ❌ template: "2+ defenders missing = AUTO PASS" | ❌ |

**Score:** Form 3 + Home 2 + Opponent 2 + Context 0 + Gut 0 = **7/10** → **Borderline**
After penalties: Derby -1 + Odds <1.16 thin = **effectively 6/10 → PASS** for single bet.

**Key issue:** 4-5 defenders out + derby + odds 1.16 = too much risk for too little profit (+16% of stake).

---

### ⚽ Metaloglobus vs Csikszereda — U4.5 @ 1.10

| Step | Check | Result |
|------|-------|--------|
| A — Goal baseline | Lambda 2.78. Both RO1 lower-table. | ✅ Good |
| B — Matchup | Both mediocre quality. | ✅ Balanced |
| C — xG | Lambda 2.78 < 3.0. | ✅ |
| D — Game state | RO1 play-out. | ✅ Low urgency |
| E — League | RO1 → small league, moderate reliability. | ⚠️ |

**Score:** Baseline 2 + xG 2 + Matchup 2 + State 2 + Gut 0 = **8/10**

**BUT:** Odds **1.10** → ⚠️ Daily filter says "Odds < 1.10 = PASS." At exactly 1.10, it barely qualifies. Profit = +10% of stake = very thin.

| Metric | Value |
|--------|-------|
| Model p_cal | 88.3% |
| Implied (1.10) | 90.9% |
| **Edge** | **-2.6%** (no edge!) |

Market prices this correctly. No value despite high score.

---

### ⚽ Fiorentina vs Lazio — U12.5 Corners @ 1.12

*(Full analysis in `Corners_U115_Analysis_2026-04-13.md`)*

| Step | Check | Result |
|------|-------|--------|
| A — Baseline | Lazio 3.6 corners FOR (16th Serie A). Combined ~8.3. | 🔥 |
| B — Total | Lambda 7.81. Est ~8.3. | 🔥 EXCELLENT |
| C — Style | Sarriball = central. BOTH missing wide players (Zaccagni, Maldini, Marusic for Lazio; Lamptey, Dodo, Solomon for Fiorentina). | 🔥🔥 |
| D — State | Fiorentina relegation = some pressure. | ⚠️ |
| E — League | I1 Serie A → ✅ | ✅ |

**Score:** Baseline 3 + Total 2 + Style 2 + State 1 + Gut 1 = **9/10** 🔥

| Metric | Value |
|--------|-------|
| Model p_cal | 90.3% |
| Implied (1.12) | 89.3% |
| **Edge** | **+1.0%** (thin) |

---

### ⚽ Metaloglobus vs Csikszereda — U12.5 Corners @ 1.12

*(Full analysis in `Corners_U115_CoVe_2026-04-13.md`)*

| Step | Check | Result |
|------|-------|--------|
| A — Baseline | Csikszereda away **2.1 corners FOR** (🔥 GOLD) | 🔥 |
| B — Total | Lambda 8.90. Est ~8.5. | 🔥 |
| C — Style | RO1 play-out = ultra-cautious. Both low-quality. | ✅ |
| D — State | Low urgency, play-out. | ✅ |
| E — League | RO1 → smaller league. | ⚠️ |

**Score:** Baseline 3 + Total 2 + Style 2 + State 2 + Gut 0 = **9/10** 🔥

| Metric | Value |
|--------|-------|
| Model p_cal | 82.5% |
| Implied (1.12) | 89.3% |
| **Edge** | **-6.8%** (negative! market overprices Under) |

---

### 🎾 Rakhimova vs Kalieva — U12.5 Set 1 @ 1.12

*(Full analysis in `WTA_Set1_DualMarket_CoVe3_2026-04-13.md`)*

| Step | Check | Result |
|------|-------|--------|
| A — Hold | Rakhimova **0.409** = ELITE (<0.42 🔥🔥) | 🔥 |
| B — Surface | Rouen **WTA 250 Clay** → stricter thresholds | ⚠️ |
| C — Gap | 0.234 ≥ 0.18 strict threshold ✅ | ✅ |
| D — Context | Not a final. Kalieva 16-6 in form. | ✅ |

**Score:** 9/10 raw → **capped MODERATE** (WTA 250 Clay rule)

| Metric | Value |
|--------|-------|
| Est. prob | ~93% |
| Implied (1.12) | 89.3% |
| **Edge** | **+3.7%** |

---

## STEP 1 SUMMARY

| # | Match | Market | Score | Prob | Odds | Edge | Pass? |
|---|-------|--------|-------|------|------|------|-------|
| 1 | **Gil Vicente X2** | DC | **8** | 87.3% | **1.25** | **+7.3%** | ✅ |
| 2 | **Fiorentina/Lazio** | U12.5 Corners | **9** | 90.3% | 1.12 | +1.0% | ✅ (thin odds) |
| 3 | **Metaloglobus/Csikszereda** | U12.5 Corners | **9** | 82.5% | 1.12 | -6.8% | ⚠️ No edge |
| 4 | **Rakhimova/Kalieva** | U12.5 Tennis | **9** | ~93% | 1.12 | +3.7% | ✅ (WTA 250 cap) |
| 5 | Metaloglobus/Csikszereda | U4.5 Goals | 8 | 88.3% | 1.10 | -2.6% | ⚠️ No edge |
| 6 | Man Utd 1X | DC | 6 | 84.9% | 1.16 | — | ❌ PASS (4 def out + derby) |

---

## STEP 3: SELF-VERIFICATION

| Check | Answer |
|-------|--------|
| Market-specific numbers? | ✅ Lambda, hold %, corners FOR, DC coverage |
| Internet crosscheck? | ✅ All picks researched with URLs in individual CoVe files |
| Penalties applied? | ✅ Man Utd: derby -1, defenders -1. WTA 250 cap. |
| Research cap? | ✅ Max +3pp (Gil Vicente) |
| Contradictions? | ⚠️ Metaloglobus corners: model 82.5% but odds imply 89.3% = market thinks Under is even more likely than our model. |

### FINAL QUESTIONS:

| Pick | Question | Answer |
|------|---------|--------|
| Gil Vicente X2 | Can Tondela WIN? | 1W/14 home (7.1%), new coach 5-0 loss, 0.64 goals/game. **No.** |
| Fiorentina corners | 13+ corners? | Lazio 6.58 away total, both missing wide players, Sarriball central. **~8%.** |
| Metaloglobus corners | 13+ corners? | Csikszereda 2.1 corners away, play-out. **~10%.** |
| Rakhimova U12.5 | Both hold 5-6? | Rakhimova 0.409: P(5+) = 4%. **No.** |
| Man Utd 1X | Can Leeds WIN? | 0 goals in 4, 1 away win/15. **Low but 4 defenders out = vulnerability.** |

---

## STEP 5: STRATEGY FILTER

| Pick | Prob | Odds | Primary (≥85% + ≥1.25)? | Daily (≥82% + ≥1.10)? | Edge? |
|------|------|------|------------------------|-----------------------|-------|
| **Gil Vicente X2** | 87.3% | **1.25** | ✅ **(87.3% ≥ 85%, 1.25 ≥ 1.25)** | ✅ | **+7.3%** |
| Fiorentina corners | 90.3% | 1.12 | ❌ (odds < 1.25) | ✅ | +1.0% |
| Rakhimova U12.5 | ~93% | 1.12 | ❌ (odds < 1.25) | ✅ | +3.7% |
| Metaloglobus corners | 82.5% | 1.12 | ❌ | ✅ | -6.8% ❌ |
| Metaloglobus goals | 88.3% | 1.10 | ❌ | ⚠️ Barely | -2.6% ❌ |
| Man Utd 1X | 84.9% | 1.16 | ❌ | PASS | — |

---

## STEP 6: TOP 2 PICKS

### 🥇 PICK 1: Tondela vs Gil Vicente — DC X2 @ 1.25

| Metric | Value |
|--------|-------|
| **Score** | **8/10** |
| Confidence | **STRONG** |
| Model | 84.3% |
| Research | **87.3%** |
| Fair odds | 1.15 |
| Market odds | **1.25** |
| **Edge** | **+7.3%** |

**Key stat:** Tondela = **cel mai slab gazda din Liga Portugal** — 1 victorie din 14 acasa (7.1%), 0.64 goluri/meci, antrenor nou cu 5-0 in primul meci, scandal presedinte-antrenor. Gil Vicente concede 1.00/meci (elite) si lupta pentru Europa.

**Risk:** Gil Vicente au 4L/10 (dar toate vs top teams). Away form inconsistenta (1W in 8 recent). Daca Tondela au o zi buna cu disperarea relegarii... dar au 0.64 goluri/game home.

**🎓 Why this is PICK 1:** Singurul pick care califica pentru **strategia principala** (prob ≥85% + odds ≥1.25). Edge +7.3% = real value. Odds 1.25 = profit decent (+25% of stake).

---

### 🥈 PICK 2: Fiorentina vs Lazio — U12.5 Corners @ 1.12

| Metric | Value |
|--------|-------|
| **Score** | **9/10** 🔥 |
| Confidence | **HIGH** |
| Model | 90.3% |
| Research | **90.3%** |
| Fair odds | 1.11 |
| Market odds | **1.12** |
| **Edge** | **+1.0%** |

**Key stat:** Lazio away 6.58 total corners. **BOTH teams missing 3+ wide players** (Zaccagni, Maldini, Marusic for Lazio; Lamptey, Dodo, Solomon for Fiorentina). Sarriball = central play. Lambda 7.81 = margin 4.7 to line.

**Risk:** Fiorentina in relegation = could push late from wings (Gosens available at LB). Edge thin (+1%) — piata pricing corect.

**🎓 Why this is PICK 2:** Highest CoVe score (9/10) and highest model probability (90.3%). Structural certainty — both teams physically cannot generate corners without wingers. But odds 1.12 = thin profit.

---

## ACCUMULATOR (optional)

| Leg | Match | Market | Odds | Score |
|-----|-------|--------|------|-------|
| 1 | **Gil Vicente X2** | DC | 1.25 | 8 |
| 2 | **Fiorentina/Lazio** | U12.5 Corners | 1.12 | 9 |
| 3 | **Rakhimova/Kalieva** | U12.5 Tennis | 1.12 | 9 |
| **Combined** | | | **1.57** | |

Combined prob: 87.3% × 90.3% × 93% = **73.3%**
Combined fair odds: 1 / 0.733 = **1.364**
Combined market: **1.57**
**Combined edge: +15.1%**

⚠️ Accumulator NOT for main strategy (single bet 25%). Side bet only (2-5% stake).

---

## INVERSE MARKET CHECK

| Pair | Conflict? | Resolution |
|------|----------|-----------|
| Metaloglobus: Goals U4.5 + Corners U12.5 | ✅ Compatible | Low goals = low corners (same direction) |
| Man Utd 1X + defensive crisis | ⚠️ Conflict | DC needs defense, but 4 defenders out = contradiction → PASS correct |

---

## 🎓 KEY LESSONS

### 1. Odds matter more than score
Fiorentina corners (score 9, odds 1.12) vs Gil Vicente DC (score 8, odds 1.25):
- Fiorentina: +12% profit, +1% edge
- Gil Vicente: **+25% profit, +7.3% edge**
Score 9 > score 8, but **Gil Vicente is the better BET** because odds give more value.

### 2. Negative edge = no bet even with high score
Metaloglobus corners (score 9, p_cal 82.5%) at odds 1.12 (implied 89.3%) = **-6.8% edge.** The bookmaker already prices Under MORE aggressively than our model. No value.

### 3. AUTO PASS rules save you
Man Utd 1X looked good (84.9%, unbeaten since 1981 vs Leeds). But template AUTO PASS (4 defenders out) + derby -1 + thin odds = correctly eliminated. **Discipline > narrative.**

---

## Sources

All sources cited in individual CoVe analysis files:
- `DC_Analysis_Tondela_GilVicente_2026-04-13.md`
- `DC_Analysis_2026-04-13.md`
- `Corners_U115_Analysis_2026-04-13.md`
- `Corners_U115_CoVe_2026-04-13.md`
- `WTA_Set1_DualMarket_CoVe3_2026-04-13.md`

---

*CoVe Multi-Market v1.0 complete. 6 picks, 4 markets. TOP 2: Gil Vicente X2 @ 1.25 (PICK 1, edge +7.3%, qualifies PRIMARY strategy) + Fiorentina corners U12.5 @ 1.12 (PICK 2, score 9, thin edge). Man Utd PASS (4 defenders out). Sources in individual files.*