# CoVe Corners v1.0 — Under 11.5 / Under 12.5 — 2026-04-11

**Data sources:**
- Model: Negative Binomial, walk-forward on 12,079 matches (16 leagues)
- Fixtures: 88 (Flashscore + API-Football)
- Evaluations: 88 matches, 68 recommendations (47 unique)
- Calibration: `corners_calibration.csv` — perfectly calibrated (gap max 0.5pp)
- Backtest audit 2026-04-06: 79.5% hit rate global, SP2 90.6%, I1 83.6%, F1 83.7%

---

## STEP 1 — CORNER BASELINE

### Top 20 (deduplicated, sorted by p_cal)

| # | Match | League | Lambda | k_disp | p_cal | Fair Odds | U11.5 Zone |
|---|-------|--------|--------|--------|-------|-----------|-----------|
| 1 | **Burgos vs Gijon** | SP2 | **7.85** | 25.5 | **95.3%** | 1.05 | 🔥 Safe |
| 2 | **Malaga vs Las Palmas** | SP2 | **7.98** | 25.5 | **94.9%** | 1.05 | 🔥 Safe |
| 3 | **Leganes vs Albacete** | SP2 | **8.71** | 25.5 | **92.4%** | 1.08 | 🔥 Safe |
| 4 | **Milan vs Udinese** | I1 | **7.42** | 30.9 | **90.9%** | 1.10 | 🔥 Safe |
| 5 | **Cordoba vs Zaragoza** | SP2 | **9.08** | 25.5 | 90.8% | 1.10 | ✅ Good |
| 6 | AVS vs Guimaraes | P1 | 7.99 | 30.1 | 88.4% | 1.13 | 🔥 Safe |
| 7 | **Torino vs Verona** | I1 | **8.11** | 30.9 | **88.0%** | 1.14 | 🔥 Safe |
| 8 | **Coventry vs Sheff Wed** | E1 | 8.08 | 77.3 | 88.0% | 1.14 | 🔥 Safe |
| 9 | Pescara vs Sampdoria | I2 | 8.57 | 24.5 | 85.6% | 1.17 | 🔥 Safe |
| 10 | Avellino vs Catanzaro | I2 | 8.83 | 24.5 | 84.5% | 1.18 | 🔥 Safe |
| 11 | Elche vs Valencia | SP1 | 8.58 | 26.5 | 84.5% | 1.18 | 🔥 Safe |
| 12 | **Cagliari vs Cremonese** | I1 | **8.93** | 30.9 | 84.3% | 1.19 | ✅ Good |
| 13 | Estrela vs Sporting | P1 | 8.91 | 30.1 | 83.8% | 1.19 | ✅ Good |
| 14 | Santa Clara vs Rio Ave | P1 | 8.95 | 30.1 | 83.6% | 1.20 | ✅ Good |
| 15 | Sudtirol vs Modena | I2 | 9.07 | 24.5 | 83.5% | 1.20 | ✅ Good |

---

## STEP 2 — RESEARCH

### 🥇 Milan vs Udinese — U12.5 (lambda=7.42, p_cal=90.9%)

**Corner stats:**
- Milan: avg **3.7 corners FOR**, 4.2 conceded. **Under 6.5 corners in 4 straight home matches.** — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/ac-milan-vs-udinese-prediction-lineups-odds-2026-04-11/)
- Udinese: avg 4.70 corners against in last 10 away. — [SempreMilan](https://sempremilan.com/ac-milan-udinese-preview-2026-team-news-prediction)
- Combined estimate: 3.7 + 4.7 = **~8.4 total** (well under 12.5)
- **Lambda = 7.42** — lowest in Serie A today!

**⚠️ FLAG:** 5 of last 6 H2H had **over 10.5 corners**. — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/ac-milan-vs-udinese-prediction-lineups-odds-2026-04-11/)

**Style:** Milan possession-based (53.3%, 508 passes/game). Udinese defensive. Not a crossing-heavy matchup.

**Context:** Milan 3rd, pushing for title. Udinese 11th. Milan full squad available for first time. — [SempreMilan](https://sempremilan.com/ac-milan-udinese-preview-2026-team-news-prediction)

**Checklist:**
- Baseline: 3 (lambda 7.42 🔥)
- Expected total: 2 (<9 excellent)
- Style: 1 (⚠️ H2H 5/6 over 10.5 = corner-heavy pattern)
- Game state: 1 (Milan push for title = pressure)
- League: 1 (I1 = 83.6% backtest)
- **Score: 8/10** (downgraded from 10 for H2H corner pattern)

---

### 🥇 Torino vs Verona — U12.5 (lambda=8.11, p_cal=88.0%)

**Corner stats:**
- Torino: avg **3.9 corners FOR**, 4.5 against. Total ~8.4. — [StatsChecker](https://www.statschecker.com/stats/corners-per-game/serie-a-corner-stats)
- Verona: avg **3.6 corners FOR**, 4.7 against. Total ~8.3. — [StatsChecker](https://www.statschecker.com/stats/corners-per-game/serie-a-corner-stats)
- Combined: ~8.4 total. Lambda = 8.11. Well under 12.5.
- Torino awarded avg **3.30 corners in last 10 games** (very low). — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/torino-vs-hellas-verona-prediction-lineups-odds-2026-04-11/)
- Corners against Verona **<5.5 in last 5 matches**. — [SportsGambler](https://www.sportsgambler.com/betting-tips/football/torino-vs-hellas-verona-prediction-lineups-odds-2026-04-11/)

**Style:** Torino under new coach D'Aversa — pragmatic, 1-0 wins. Verona 19th, defensive, low possession (43%). Neither generates wide pressure.

**Context:** Torino 12th improving. Verona 19th, 9 pts from safety = will sit deep = fewer corners for both.

**Checklist:**
- Baseline: 3 (lambda 8.11 🔥)
- Expected total: 2 (<9 excellent)
- Style: 2 (both low-tempo, Verona deep block)
- Game state: 2 (Verona defensive, low urgency for corners)
- League: 1 (I1 83.6%)
- **Score: 10/10** 🔥

---

### 🥇 Burgos vs Gijon — U12.5 (lambda=7.85, p_cal=95.3%)

**Corner stats:**
- SP2 = best league for corners Under (90.6% backtest, k_disp=25.5 = reliable)
- Lambda = 7.85 (Premium)
- Burgos defensive team. Gijon 0.79 goals/game = minimal attacking pressure = minimal corners.

**Checklist:**
- Baseline: 3 (lambda 7.85 🔥)
- Expected total: 2 (<9)
- Style: 2 (SP2 low tempo)
- Game state: 2 (comfortable)
- League: 1 (SP2 90.6% 🔥)
- **Score: 10/10** 🔥

---

### Coventry vs Sheffield Wed — U12.5 (lambda=8.08, p_cal=88.0%)

- E1 Championship. Lambda = 8.08 (safe for U12.5).
- BUT: E1 is crossing-heavy league → higher corner variance.
- k_dispersion = 77.3 (tight) → reliable prediction.
- **Score: 7/10** (E1 league profile ⚠️)

---

## STEP 3 — FINAL QUESTION: "Can this reach 12+ corners?"

| Match | 12+ Risk | Verdict |
|-------|---------|---------|
| **Burgos/Gijon** | ~5%. SP2, lambda 7.85, both low attacking | ✅ **Under VALID** |
| **Torino/Verona** | ~8%. Both avg ~3.5-4.0 corners, Verona deep block | ✅ **Under VALID** |
| **Milan/Udinese** | ~15%. H2H 5/6 over 10.5 corners flagged! | ⚠️ **Under VALID but flagged** |
| **Malaga/Las Palmas** | ~5%. SP2, lambda 7.98 | ✅ **Under VALID** |
| **Leganes/Albacete** | ~7%. SP2, lambda 8.71 | ✅ **Under VALID** |

---

## FINAL SCORECARD

| Pick | Baseline (/3) | Total (/2) | Style (/2) | Context (/2) | League (/1) | **TOTAL** |
|------|--------------|-----------|-----------|-------------|------------|-----------|
| **Torino/Verona** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Burgos/Gijon** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Malaga/Las Palmas** | 3 | 2 | 2 | 2 | 1 | **10/10** 🔥 |
| **Leganes/Albacete** | 3 | 2 | 2 | 1 | 1 | **9/10** |
| **Milan/Udinese** | 3 | 2 | 1 | 1 | 1 | **8/10** (H2H flag) |
| Coventry/Sheff Wed | 3 | 2 | 1 | 1 | 0 | **7/10** (E1 profile) |
| Cagliari/Cremonese | 2 | 2 | 1 | 1 | 1 | **7/10** |

---

## STRATEGY FILTER

| Pick | p_cal | Typical Odds | Sweet Spot 1.25+? |
|------|-------|-------------|-------------------|
| SP2 matches (Burgos, Malaga, Leganes) | 92-95% | 1.05-1.08 | ❌ Odds prea mic |
| I1 matches (Milan, Torino, Cagliari) | 84-91% | 1.10-1.19 | ❌ Sub 1.25 |
| E1 matches (Coventry) | 88% | 1.14 | ❌ Sub 1.25 |

**Corners Under nu califica pentru single bet strategy** — odds-urile sunt tipic 1.05-1.19. Dar **excelente structural** pentru accumulator sau side bets.

---

## TRIPLE SIGNAL: TORINO vs VERONA

| Model | Pick | Score | p_cal |
|-------|------|-------|-------|
| **DC** | 1X @ 1.22 | **10** | 89.9% |
| **Goals U4.5** | U4.5 | **7** | 84.6% |
| **Corners U12.5** | lambda=8.11 | **10** | 88.0% |

**Torino vs Verona are TRIPLE SIGNAL pe 3 modele independente** — cel mai consistent pick al zilei. Torino 1X @ 1.22 ramane candidat pentru TODAY'S PICK.

---

## Sources

- [SportsGambler — Milan vs Udinese](https://www.sportsgambler.com/betting-tips/football/ac-milan-vs-udinese-prediction-lineups-odds-2026-04-11/)
- [SempreMilan — Milan vs Udinese](https://sempremilan.com/ac-milan-udinese-preview-2026-team-news-prediction)
- [BettingAcademy — Milan vs Udinese](https://www.bettingacademy.co.uk/stats/match/italy/serie-a/milan/udinese/W65QKx7DlYqAw/preview)
- [FootballWhispers — Milan vs Udinese](https://footballwhispers.com/blog/ac-milan-vs-udinese-prediction-11-04-2026/)
- [SportsGambler — Torino vs Verona](https://www.sportsgambler.com/betting-tips/football/torino-vs-hellas-verona-prediction-lineups-odds-2026-04-11/)
- [StatsChecker — Serie A corners](https://www.statschecker.com/stats/corners-per-game/serie-a-corner-stats)
- [SoccerStats — Serie A corners](https://www.soccerstats.com/table.asp?league=italy&tid=cr)
- [WinDrawWin — Serie A corners](https://www.windrawwin.com/us/soccer-stats/corners/italy-serie-a/)
- Corners backtest audit 2026-04-06: SP2 90.6%, I1 83.6%, calibration gap max 0.5pp

---

*CoVe Corners v1.0 complete. 88 fixtures, 47 unique recommendations. SP2 dominate top. Torino/Verona triple signal (DC + Goals + Corners). Sources cited inline and at end.*
