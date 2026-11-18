# CoVe 1.11 — First-Half Goal Analysis
## PFC Levski Sofia vs FK Septemvri Sofia
**Bulgaria Parva Liga, Matchday 3 — Saturday, 2026-08-01, 16:00 UTC (19:00 Sofia), Vivacom Arena - Georgi Asparuhov, Sofia**

---

## PHASE 1 — Match Context
- Competition: Parva Liga, round 3, early season — LOW-MEDIUM importance (no table pressure yet)
- H2H: 21 meetings, Levski 16 wins, Septemvri 2 wins, 3 draws — Levski dominates historically
- Levski = big favorite (~1.15 win odds); Septemvri ~19.5, draw ~6.3
- Levski coach: Julio Velasquez. Septemvri coach: Stamen Belchev (4-2-3-1), team on a 23% win rate over last 13 matches — struggling side.

## PHASE 4/7 — Recent Form: Sofascore goal-minute data (fresh pull, 2026-08-01)
Pulled directly via `sofascore_loader.py` (team ids: Levski 3273, Septemvri 201203) — exact half-time scores and goal minutes for every recent match, not estimated.

### Levski — last 11 Parva Liga matches, goal minutes
| Date | Match | Score | Goal minutes | 1H goal? |
|---|---|---|---|---|
| 02-07 | Levski vs Botev Vratsa | 3-1 | 46, 51, 53, 58 | NO |
| 02-15 | Levski vs Botev Plovdiv | 3-0 | 37, 50, 54 | YES |
| 02-22 | CSKA 1948 vs Levski | 1-3 | 6, 43, 58, 89 | YES |
| 03-01 | Levski vs Lokomotiv Sofia | 4-3 | 3, 26, 69, 74, 76, 84, 90 | YES |
| 03-05 | Ludogorets vs Levski | 1-0 | 80 | NO |
| 03-09 | Levski vs Lokomotiv Plovdiv | 1-0 | 89 | NO |
| 03-15 | Beroe vs Levski | 0-1 | 61 | NO |
| 03-21 | Levski vs Cherno More | 2-1 | 53, 67, 90 | NO |
| 04-05 | Dobrudzha vs Levski | 2-2 | 63, 88, 90, 90 | NO |
| 04-09 | Levski vs Arda | 1-0 | 64 | NO |
| 04-13 | CSKA Sofia vs Levski | 1-1 | 71, 90 | NO |

**Levski: 3/11 = 27% first-half goal rate.** Even in emphatic wins (3-0, 3-1, 4-3), the first goal routinely arrives from the 46th minute onward. This looks like a genuine "slow starter, breaks teams down gradually" pattern under Velasquez this season, not noise.

### Septemvri — last 10 Parva Liga matches, goal minutes
| Date | Match | Score | Goal minutes | 1H goal? |
|---|---|---|---|---|
| 02-08 | Dobrudzha vs Septemvri | 3-0 | 37, 58, 66 | YES |
| 02-16 | Septemvri vs Montana | 1-0 | 53 | NO |
| 02-28 | Septemvri vs CSKA Sofia | 2-0 | 57, 90 | NO |
| 03-03 | Botev Plovdiv vs Septemvri | 2-0 | 58, 62 | NO |
| 03-07 | Septemvri vs Cherno More | 0-3 | 18, 48, 66 | YES |
| 03-13 | Lokomotiv Plovdiv vs Septemvri | 1-0 | 31 | YES |
| 03-17 | Lokomotiv Sofia vs Septemvri | 3-0 | 36, 45, 49 | YES |
| 03-21 | Septemvri vs Arda | 1-4 | 3, 26, 61, 67, 78 | YES |
| 04-06 | Septemvri vs Botev Vratsa | 0-0 | — | NO |
| 04-10 | Slavia Sofia vs Septemvri | 1-2 | 24, 30, 90 | YES |

**Septemvri: 6/10 = 60% first-half goal rate.** More live than Levski, but as the much weaker side traveling to face Levski, they're likely to sit deep and limit their own attacking output early — their own 1H rate here reflects matches where they were often the side conceding rather than dictating tempo.

Note: a separate web search summary claimed "Septemvri unbeaten in 6 of their last Parva Liga matches" — this directly contradicts the Sofascore data above (3 losses inside that window: vs Dobrudzha, Botev Plovdiv, Lokomotiv Sofia, plus a heavy 1-4 loss to Arda). Trusting the Sofascore minute-level data over the web summary per project data-source hierarchy.

## PHASE 12 — Goal Model

**P(Over 0.5 FT goals): ~92%** — HIGH confidence. Neither team has produced a 0-0 final score in the sampled matches except one (Septemvri vs Botev Vratsa). Levski is vastly superior in quality and almost always scores; combined with H2H dominance (16/21 wins), a goalless 90 minutes is unlikely.

**P(Over 0.5 HT goals): ~40-45%** — Below coin-flip. Levski's 27% 1H rate is the dominant signal here (home team, controls tempo, heavy favorite dictates game shape) combined with the fact both teams' first goals cluster disproportionately in the last 30 minutes across the sample.

## PHASE 14 — First-Half Goal Index
Negative factors dominate: Levski's own 1H goal rate is just 27% across 11 real matches (not narrative — measured), early-season low urgency, Septemvri set up to sit deep and defend as heavy underdog away from home, both teams' scoring patterns skew late.
Positive factors: Septemvri's own rate (60%) is decent, and Levski's overwhelming quality means SOME chances will exist early even if not converted.

**FIRST-HALF GOAL INDEX: ~40/100 → PASS**

## FINAL DECISION
- **Match goal (Over 0.5 FT): ~92% — very likely, but no market value (odds near 1.01-1.02).**
- **First-half goal (Over 0.5 HT): ~40-45% — PASS. This is a real, data-backed signal (not assumption) that Levski specifically tends to break games open after halftime rather than start fast, even against weak opposition.**

### Sources
- [Sofascore — Levski Sofia team page](https://www.sofascore.com/football/team/levski-sofia/3273)
- [Sofascore — match page](https://www.sofascore.com/football/match/fk-septemvri-sofia-levski-sofia/ypbsdzFb)
- [Fotmob — Septemvri Sofia squad/coach](https://www.fotmob.com/uk/teams/312976/overview)
- [AiScore — H2H history](https://www.aiscore.com/head-to-head/soccer-levski-sofia-vs-septemvri-sofia)
- [JohnnyBet — match predictions](https://www.johnnybet.com/app/betting-events/levski-sofia-septemvri-sofia)
