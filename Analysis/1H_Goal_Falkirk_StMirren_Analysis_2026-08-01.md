# CoVe 1.11 — First-Half Goal Analysis
## Falkirk FC vs St Mirren FC
**Scottish Premiership, Matchday 1 (2026/27 season opener) — Saturday, 2026-08-01, 15:00, Falkirk Stadium**

---

## PHASE 1 — Match Context
- Season opener, Scottish Premiership 2026/27, broadcast live on Sky Sports
- Falkirk finished 6th in their first top-flight campaign since 2009/10 last season; St Mirren finished 11th, survived via play-off (beat Partick Thistle 2-1 on aggregate)
- Odds tight: Falkirk ~1.96, Draw ~3.7, St Mirren ~3.65 — genuinely competitive match, no big favorite
- Falkirk's top scorer Barney Stewart left for West Brom (attacking loss)
- St Mirren signed Samuel Ramos (attacking midfielder) and Eseosa Sule (striker) specifically to fix a scoring problem — they scored 10 fewer goals than any other top-flight team last season

## PHASE 4/7 — Key storyline: goals are Falkirk's whole identity
Falkirk's 38 league games last season produced **112 goals** (≈2.9 goals/game combined) — the most entertaining, highest-scoring team in the division. St Mirren were the opposite (worst attack in the league), hence the two attacking signings this summer specifically to address it.

**Preseason form:**
- Falkirk: League Cup wins 5-0 vs Edinburgh City, 3-0 vs Stranraer, 1-0 friendly vs KRC Genk; losses to Ayr United and on penalties to Alloa Athletic
- St Mirren: 4-goal wins vs Dumbarton, Cove Rangers, East Kilbride, Arbroath (lower-tier opposition); losses 1-0 to Dunfermline, 3-1 to Hearts

## PHASE 8 — Squad Availability
- **Falkirk:** "Clean bill of health" entering the league season. Aidan Nesbitt, Scott Bain, Keelan Adams out but Jack McMillan has recovered from a pre-season injury. No major attacking absence besides the permanent loss of Stewart (transferred).
- **St Mirren:** Jonah Alunga (forward) out — ACL injury from April, re-signed on a new deal but not yet back.

## Sofascore goal-minute data (fresh pull, 2026-08-01, team ids: Falkirk 2363, St Mirren 2359)

### Falkirk — last 12 Premiership matches, goal minutes
| Date | Match | Score | Goal minutes | 1H goal? |
|---|---|---|---|---|
| 12-27 | Dundee vs Falkirk | 1-0 | 45 | YES |
| 01-03 | Falkirk vs Aberdeen | 1-0 | 58 | NO |
| 01-10 | St Mirren vs Falkirk | 0-2 | 55, 61 | NO |
| 01-14 | Falkirk vs Celtic | 0-1 | 43 | YES |
| 01-24 | Falkirk vs Hibernian | 4-1 | 17, 43, 59, 83, 89 | YES |
| 02-01 | Celtic vs Falkirk | 2-0 | 39, 62 | YES |
| 02-04 | Livingston vs Falkirk | 1-2 | 6, 25, 47 | YES |
| 02-11 | Falkirk vs Dundee | 1-0 | 55 | NO |
| 02-14 | Falkirk vs Dundee Utd | 2-3 | 17, 24, 53, 55, 83 | YES |
| 02-21 | Hearts vs Falkirk | 1-0 | 45 | YES |
| 02-28 | Falkirk vs Kilmarnock | 5-1 | 1, 28, 36, 40, 50, 56 | YES |
| 03-14 | Aberdeen vs Falkirk | 1-1 | 73, 89 | NO |

**Falkirk: 8/12 = 67% first-half goal rate.**

### St Mirren — last 11 Premiership matches, goal minutes
| Date | Match | Score | Goal minutes | 1H goal? |
|---|---|---|---|---|
| 01-14 | Hearts vs St Mirren | 2-0 | 60, 80 | NO |
| 01-20 | Livingston vs St Mirren | 1-1 | 56, 89 | NO |
| 01-31 | St Mirren vs Dundee | 0-0 | — | NO |
| 02-03 | St Mirren vs Hearts | 1-0 | 88 | NO |
| 02-11 | Kilmarnock vs St Mirren | 4-3 | 8, 43, 45, 48, 76, 82, 90 | YES |
| 02-14 | Hibernian vs St Mirren | 2-0 | 42, 66 | YES |
| 02-21 | St Mirren vs Motherwell | 0-5 | 14, 30, 49, 58, 66 | YES |
| 02-28 | Livingston vs St Mirren | 1-1 | 44, 76 | YES |
| 03-03 | Dundee Utd vs St Mirren | 2-1 | 8, 47, 83 | YES |
| 03-15 | St Mirren vs Rangers | 0-1 | 31 | YES |
| 03-21 | Falkirk vs St Mirren | 1-2 | 15, 18, 63 | YES |

**St Mirren: 7/11 = 64% first-half goal rate.**

**Actual H2H this season** (2 meetings): Falkirk 1-2 St Mirren (goals at 15', 18', 63' — fast start) and St Mirren 0-2 Falkirk (goals at 55', 61' — no 1H goal that time). 1/2 recent H2H meetings had a first-half goal, but the one that did had it very early (inside 20 minutes).

## PHASE 12 — Goal Model
Only one true 0-0 final score across 23 combined sampled matches (St Mirren vs Dundee, 31 Jan). Both teams score heavily and concede heavily — this is a genuinely open, high-event league fixture, reinforced by Falkirk's 112-goals-in-38-games identity and St Mirren's summer attacking reinforcements.

**P(Over 0.5 FT goals): ~93-95%** — HIGH confidence.
**P(Over 0.5 HT goals): ~68-70%** — combining the measured rates (67%/64%, avg ≈65%) with the qualitative context (Falkirk's high-tempo identity, St Mirren's new attacking pieces, Falkirk full-strength attack) nudging slightly above the raw average.

## PHASE 14 — First-Half Goal Index
Positive: both teams show real, measured first-half scoring rates (~65% combined), Falkirk's whole season identity is goals, St Mirren invested specifically in attacking upgrades, Falkirk has no fresh injury concerns, one recent H2H meeting had 2 goals inside 20 minutes.
Negative: St Mirren's own defensive record is genuinely poor away from home which could mean Falkirk dominates territory but needs time to break them down in some games (variance exists — not every Falkirk match is fast, e.g. 3 of their last 12 had no goal before 45').

**FIRST-HALF GOAL INDEX: ~68/100 → INTERESTING** (below the 80% CoVe BET threshold, but a genuinely live, data-backed positive signal — the opposite conclusion from the Levski/Septemvri case).

## FINAL DECISION
- **Match goal (Over 0.5 FT): ~93-95% — very likely, essentially no market value at typical odds.**
- **First-half goal (Over 0.5 HT): ~68-70% — INTERESTING, not quite a confident BET (below 80%), but a real positive lean backed by measured data + context, not a PASS.**

### Sources
- [St Mirren FC — ticket info, Falkirk away day](https://www.stmirren.com/news/ticket-info-falkirk-st-mirren-1-aug-2026)
- [Sky Sports — Falkirk 2026/27 Premiership fixtures](https://www.skysports.com/football/news/11793/13551062/falkirk-scottish-premiership-2026-27-fixtures-and-schedule)
- [OneFootball — opening-day form meets cup goal spree](https://onefootball.com/en/news/falkirk-v-st-mirren-opening-day-form-meets-cup-goal-spree-43211027)
- [Sofascore — Falkirk FC team page](https://www.sofascore.com/football/team/falkirk-fc/2363)
