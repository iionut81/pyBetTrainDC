# CoVe Corners v1.0 — Under 11.5 / Under 12.5 — 2026-04-12

**Data sources:**
- Model: Negative Binomial, walk-forward on 12,079 matches (16 leagues)
- Fixtures: 47 (Flashscore + API-Football)
- Evaluations: 47 matches, 33 recommendations
- Calibration: `corners_calibration.csv`
- Backtest audit 2026-04-06: SP2 90.6%, I1 83.6%, F1 83.7%

---

## STEP 1 — CORNER BASELINE

### Top 15 (sorted by p_cal)

| # | Match | League | Lambda | k_disp | p_cal | Fair Odds | U11.5 Zone |
|---|-------|--------|--------|--------|-------|-----------|-----------|
| 1 | **Granada vs Cultural Leonesa** | SP2 | **7.52** | 25.5 | **96.2%** | 1.04 | 🔥 Safe |
| 2 | **Braga vs Arouca** | P1 | **6.60** | 30.1 | **94.2%** | 1.06 | 🔥 Safe |
| 3 | **Mirandes vs Castellon** | SP2 | **8.25** | 25.5 | **94.0%** | 1.06 | 🔥 Safe |
| 4 | **Huesca vs Dep. La Coruna** | SP2 | **8.34** | 25.5 | **93.7%** | 1.07 | 🔥 Safe |
| 5 | **Cadiz vs Andorra** | SP2 | **8.53** | 25.5 | **93.0%** | 1.07 | 🔥 Safe |
| 6 | **Estoril vs Porto** | P1 | **7.02** | 30.1 | **92.6%** | 1.08 | 🔥 Safe |
| 7 | **Racing Santander vs Almeria** | SP2 | **8.90** | 25.5 | **91.6%** | 1.09 | 🔥 Safe |
| 8 | **Como vs Inter** | I1 | **7.47** | 30.9 | **90.7%** | 1.10 | ⚠️ Flagged |
| 9 | **Parma vs Napoli** | I1 | **7.66** | 30.9 | **90.0%** | 1.11 | 🔥 Safe |
| 10 | **Genoa vs Sassuolo** | I1 | **7.91** | 30.9 | **88.9%** | 1.12 | ✅ Good |
| 11 | Alverca vs Casa Pia | P1 | 8.24 | 30.1 | 87.3% | 1.15 | ✅ Good |
| 12 | Spezia vs Mantova | I2 | 8.35 | 24.5 | 86.5% | 1.16 | ✅ Good |
| 13 | Mallorca vs Rayo | SP1 | 8.48 | 26.5 | 84.9% | 1.18 | ❌ **FLAGGED** |

---

## STEP 2 — RESEARCH

### ⚠️ CRITICAL FLAG: Mallorca vs Rayo — RESEARCH CONTRADICTS MODEL

**Model says:** lambda 8.48, p_cal 84.9% Under 11.5

**Research says:**
- Rayo Vallecano avg **10.83 total corners/game** — **2nd highest in La Liga!** — [APWin](https://www.apwin.com/team/rayo-vallecano/corners/)
- Rayo last 5 games averaged **12.20 corners.** — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/mallorca-vs-rayo-vallecano-prediction-lineups-odds-2026-04-12/)
- Mallorca last 7 home games: all over 9.5 total corners.

**Verdict:** Model UNDERESTIMATES Rayo's corner profile. **12+ corners is MORE likely than not.** → ❌ **PASS on corners** (goals Under still valid — different market!)

---

### 🥇 Parma vs Napoli — U12.5 (lambda 7.66, p_cal 90.0%) — **TRIPLE SIGNAL**

**Corner stats:**
- Napoli away: **4.78 corners/game** (drops from 5.90 home). Only 3.60 corners won in last 5 away. — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/parma-vs-napoli-prediction-lineups-odds-2026-04-12/)
- Parma: conceded **under 5.5 corners in last 6 home games.** — [DailySports](https://dailysports.net/predictions/parma-vs-napoli-prediction-h2h-and-probable-lineups-12042026/)
- Combined estimate: **~8.9 total corners.**

**Style:** Conte pragmatic = controls possession centrally, few wing crosses. Parma without Pellegrino = no attacking pressure = no corners forced.

**Checklist:**
- Baseline: 3 (lambda 7.66 🔥)
- Expected total: 2 (~8.9 = excellent)
- Style: 2 (Conte central control, Parma toothless)
- Game state: 1 (title race = Napoli controls but pushes late)
- League: 1 (I1 83.6%)
- **Score: 9/10** 🔥

**🔥 TRIPLE SIGNAL:** DC X2 score **10** + Goals U4.5 score **9** + Corners U12.5 score **9**. Three independent models confirm: Napoli controls, low goals, low corners.

---

### 🥇 Granada vs Cultural Leonesa — U11.5 (lambda 7.52, p_cal 96.2%)

**Corner stats:**
- Granada: avg 4.2 corners/game. Under 5.5 corners in 4 straight games. — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/granada-vs-cultural-leonesa-prediction-lineups-odds-2026-04-12/)
- Cultural Leonesa: 2.6 corners FOR, 4.1 against. Under 5.5 conceded in last 7. — [Betaminic](https://www.betaminic.com/statistics/corner-stats-la-liga/)
- Combined: **~7.5 total** (margin 4.0 to 11.5!)

**SP2 = best league for corners Under** (90.6% backtest, k_disp=25.5 = reliable).

**Checklist:**
- Baseline: 3 (lambda 7.52 🔥)
- Expected total: 2 (<9 = excellent)
- Style: 2 (SP2 low tempo, Cultural barely attacks)
- Game state: 2 (Granada at home, comfortable)
- League: 1 (SP2 90.6% 🔥)
- **Score: 10/10** 🔥

---

### 🥇 Genoa vs Sassuolo — U12.5 (lambda 7.91, p_cal 88.9%)

**Corner stats:**
- Genoa: 3.4 won, 5.1 conceded = 8.5 total per game. — [Betaminic](https://www.betaminic.com/statistics/corner-stats-serie-a/)
- Sassuolo away: only **2.50 corners/game** (very low). Total match avg: 8.27. — [Corner-Stats](https://corner-stats.com/sassuolo/italy/team/1524)
- Combined at Genoa: **~8.5 total.**

**Style:** Both low-corner teams. Sassuolo promoted back to Serie A, grinding. Neither generates wide pressure.

**Checklist:**
- Baseline: 3 (lambda 7.91 🔥)
- Expected total: 2 (<9)
- Style: 2 (both low-corner, Sassuolo especially poor away)
- Game state: 1 (Genoa 14th vs Sassuolo 10th, moderate stakes)
- League: 1 (I1 83.6%)
- **Score: 9/10**

---

### ⚠️ Como vs Inter — U12.5 (lambda 7.47, p_cal 90.7%) — FLAGGED

**Corner stats:**
- Inter: avg **7.2 corners/game** (1st-2nd in Serie A!). **Wing-backs Dimarco + Dumfries** deliver crosses constantly. — [StatsChecker](https://www.statschecker.com/stats/corners-per-game/serie-a-corner-stats)
- Como: 4.7 corners at home, 3.9 away. — [BetOnCorners](https://www.betoncorners.com/teams/como-calcio-corners/)
- Combined estimate: **~9.9 total** — higher than lambda suggests.

**⚠️ FLAG:** Inter are a **crossing-heavy team** (3-5-2 with overlapping wing-backs). Lambda 7.47 may UNDERESTIMATE Inter's corner generation. Inter avg 7.2 corners/game suggests combined total closer to 10-11, not 7.5.

**Checklist:**
- Baseline: 2 (lambda 7.47 good but Inter avg 7.2 = risk)
- Expected total: 1 (~9.9 = borderline)
- Style: 0 (Inter = crossing machine ❌)
- Game state: 2 (Como mid-table, moderate)
- League: 1 (I1)
- **Score: 6/10** → **PASS** (research contradicts model)

---

## STEP 3 — FINAL QUESTION: "Can this reach 12+ corners?"

| Match | 12+ Risk | Verdict |
|-------|---------|---------|
| **Granada/Cultural** | ~4%. SP2, lambda 7.52, both low attacking | ✅ **Under VALID** |
| **Parma/Napoli** | ~8%. Conte controls, Parma toothless | ✅ **Under VALID** |
| **Genoa/Sassuolo** | ~10%. Both low-corner, Sassuolo 2.5 away | ✅ **Under VALID** |
| **Como/Inter** | ~25%. Inter 7.2 avg corners = crossing dominant | ❌ **PASS** |
| **Mallorca/Rayo** | ~40%. Rayo 10.83 avg, last 5 = 12.20! | ❌ **PASS** |

---

## FINAL SCORECARD

| Pick | Baseline (/3) | Total (/2) | Style (/2) | Context (/2) | League (/1) | **TOTAL** |
|------|--------------|-----------|-----------|-------------|------------|-----------|
| **Granada/Cultural** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Mirandes/Castellon** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Huesca/Dep. La Coruna** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Braga/Arouca** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Parma/Napoli** | 3 | 2 | 2 | 1 | 1 | **9/10** 🔥 |
| **Cadiz/Andorra** | 3 | 2 | 2 | 1 | 1 | **9/10** |
| **Genoa/Sassuolo** | 3 | 2 | 2 | 1 | 1 | **9/10** |
| Estoril/Porto | 3 | 2 | 2 | 1 | 1 | **9/10** |
| Racing/Almeria | 3 | 2 | 1 | 1 | 1 | **8/10** |
| Como/Inter | 2 | 1 | 0 | 2 | 1 | **6/10** ❌ PASS |
| Mallorca/Rayo | 2 | 0 | 0 | 2 | 1 | **5/10** ❌ PASS |

---

## TRIPLE SIGNAL: PARMA vs NAPOLI

| Model | Pick | Score | p_cal |
|-------|------|-------|-------|
| **DC** | X2 @ ~1.35 | **10** | 83.1% |
| **Goals U4.5** | U4.5 | **9** | 90.1% |
| **Corners U12.5** | lambda=7.66 | **9** | 90.0% |

**Parma vs Napoli = cel mai consistent pick al zilei pe 3 modele independente.** Conte controleaza, Parma toothless, low goals, low corners.

---

## ⚠️ RESEARCH CONTRADICTIONS (IMPORTANT)

| Match | Model | Research | Action |
|-------|-------|---------|--------|
| **Mallorca/Rayo** | U11.5 p_cal 84.9% | Rayo avg 10.83 corners, last 5 = 12.20 | ❌ **PASS corners** (goals Under still valid) |
| **Como/Inter** | U12.5 p_cal 90.7% | Inter avg 7.2 corners, crossing-heavy 3-5-2 | ❌ **PASS corners** |

**Lesson:** Goals Under and Corners Under are different markets. A match can be low-scoring (few goals) but high-corner (lots of wing play). Mallorca/Rayo = perfect example: avg ~2 goals but avg ~11 corners.

---

## STRATEGY FILTER

| Pick | p_cal | Typical Odds | Sweet Spot 1.25+? |
|------|-------|-------------|-------------------|
| SP2 matches (5x) | 91-96% | 1.04-1.09 | ❌ Odds prea mic |
| I1 matches (Parma, Genoa) | 89-90% | 1.11-1.12 | ❌ Sub 1.25 |
| P1 matches (Braga, Estoril) | 92-94% | 1.06-1.08 | ❌ Odds prea mic |

**Corners Under nu califica pentru single bet strategy** — odds tipic 1.04-1.12. Excelente pentru **accumulator**.

---

## Sources

- [APWin — Rayo Vallecano corners](https://www.apwin.com/team/rayo-vallecano/corners/)
- [SportsGambler — Mallorca vs Rayo](https://www.sportsgambler.com/betting-tips/football/mallorca-vs-rayo-vallecano-prediction-lineups-odds-2026-04-12/)
- [SportsGambler — Parma vs Napoli](https://www.sportsgambler.com/betting-tips/football/parma-vs-napoli-prediction-lineups-odds-2026-04-12/)
- [DailySports — Parma vs Napoli](https://dailysports.net/predictions/parma-vs-napoli-prediction-h2h-and-probable-lineups-12042026/)
- [SportsGambler — Granada vs Cultural](https://www.sportsgambler.com/betting-tips/football/granada-vs-cultural-leonesa-prediction-lineups-odds-2026-04-12/)
- [StatsChecker — Serie A corners](https://www.statschecker.com/stats/corners-per-game/serie-a-corner-stats)
- [BetOnCorners — Inter](https://www.betoncorners.com/teams/inter-milan-corners/)
- [BetOnCorners — Como](https://www.betoncorners.com/teams/como-calcio-corners/)
- [Betaminic — Serie A corner stats](https://www.betaminic.com/statistics/corner-stats-serie-a/)
- [Corner-Stats — Sassuolo](https://corner-stats.com/sassuolo/italy/team/1524)
- [Betaminic — La Liga 2 corners](https://www.betaminic.com/statistics/corner-stats-la-liga/)
- Corners backtest audit 2026-04-06: SP2 90.6%, I1 83.6%

---

*CoVe Corners v1.0 complete. 47 fixtures, 33 recommendations. SP2 domina top (5 picks, all 91%+). TRIPLE SIGNAL on Parma/Napoli (DC+Goals+Corners). Two model contradictions flagged and eliminated (Mallorca/Rayo, Como/Inter). Sources cited.*
