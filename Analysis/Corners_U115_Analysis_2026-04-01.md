# Corners Under 11.5 — LaLiga2 (SP2) — 2026-04-01
## CoVe Verified | Sources Cited

---

## STEP 1: Data Analysis

**Market:** Under 11.5 total corners (max 11 corners in the match).
**Model:** Negative Binomial (k=200 ≈ Poisson) trained on Flashscore/football-data.co.uk history.
**Note:** Model is calibrated for U12.5. U11.5 probabilities are computed from raw NB CDF — no Platt calibration available for this line. Probabilities may be slightly overconfident.

### Model Output — U11.5 Derived from U12.5 Pipeline

| # | Match | Lambda | P(U12.5) raw | P(U11.5) raw | P(X=12) | Buffer | Fair Odds U11.5 |
|---|-------|--------|-------------|-------------|---------|--------|-----------------|
| 1 | Burgos CF vs Ceuta | 8.37 | 91.3% | **85.6%** | 5.7% | 3.13 | 1.169 |
| 2 | Andorra vs Malaga | 8.64 | 89.6% | **83.2%** | 6.4% | 2.86 | 1.201 |
| 3 | Huesca vs Cultural Leonesa | 9.33 | 84.5% | **76.6%** | 8.0% | 2.17 | 1.306 |
| 4 | Racing Santander vs Gijon | 9.68 | 81.6% | **72.9%** | 8.7% | 1.82 | 1.372 |

**SP2 League Parameters:** μ = 9.33 corners/match, k = 200, tempo = 0.967.

### Team Corner Profiles (from model training data)

| Team | Home For | Home Against | Away For | Away Against | N_home | N_away |
|------|---------|-------------|---------|-------------|--------|--------|
| Burgos CF | 3.70 | 3.90 | 2.56 | 5.67 | 10 | 9 |
| Ceuta | 5.00 | 5.00 | 3.60 | 5.60 | 5 | 5 |
| Andorra | 6.00 | 2.00 | 4.80 | 6.80 | **5** | **5** |
| Malaga | 5.00 | 5.00 | 3.50 | 6.00 | 11 | 8 |
| Huesca | 5.00 | 5.50 | 4.00 | 3.89 | 10 | 9 |
| Cultural Leonesa | 4.80 | 4.60 | 2.60 | 6.20 | **5** | **5** |
| Racing Santander | **6.50** | **4.70** | 5.27 | 4.36 | 10 | 11 |
| Gijon | 4.67 | 4.56 | 2.89 | 6.11 | 9 | 9 |

### External Corner Averages (APWin, full season)

| Team | Avg Corners/Game | League Rank (of 22) |
|------|-----------------|---------------------|
| Racing Santander | **10.79 (11.93 HOME)** | **1st** |
| Sporting Gijon | 9.76 (10.2 home) | 6th |
| Huesca | 9.62 (9.36 home) | 8th |
| Andorra | 9.40–9.59 | 9th |
| Ceuta | 9.28 (lower away) | 10th |
| Cultural Leonesa | 9.03 | 14th |
| Malaga | **8.56** | **20th** |
| Burgos CF | **8.21** | **22nd (last)** |

---

## STEP 2: External Research + Match Context

### Match 1: Burgos CF vs Ceuta — Lambda 8.37 | Buffer 3.13

**Burgos CF (5th, 53 pts):**
- **4 wins in last 5 matches.** Not conceded in last 5 official matches.
- Unbeaten in 7 consecutive home league games.
- Corner profile: LAST in the league (8.21 avg). At home: 3.7 for + 3.9 against = **7.6 total corners per home game** from model training data.
- Style: Defensive, compact, low-block. Teams that sit deep don't generate crosses or force opponents wide → fewer corners for both sides.

**Ceuta (8th–9th, 47 pts):**
- Lost 5 of last 6 away matches. Conceding 2+ goals per away match.
- Away corner profile: 3.6 for + 5.6 against = 9.2 total in away games.
- Key: Ceuta's away weakness means they don't control games → fewer attacking corners for them; opponents (Burgos) don't need to attack aggressively → fewer corners overall.

**Matchup dynamics:** Defensive home favorite dominating a weak away side = controlled, low-tempo match. Burgos won't overcommit forward; Ceuta can't sustain pressure away. Expected scoreline: 1-0, 2-0.

**How I lose:** Ceuta somehow push forward desperately (they need points for playoff chase) and both teams trade corners. Burgos, despite being defensive, have occasional high-corner home games when facing pressing opponents.

*Sources: [DailySports - Burgos vs Ceuta](https://dailysports.net/predictions/burgos-vs-ceuta-prediction-head-to-head-and-probable-lineups-01-april-2026/), [ESPN](https://www.espn.com/soccer/matchstats/_/gameId/748884), [APWin - SP2 Corners](https://www.apwin.com/league/spain/segunda-division/standings/corners/)*

---

### Match 2: Andorra vs Malaga — Lambda 8.64 | Buffer 2.86

**Andorra (13th, 42 pts):**
- Mixed form: 4W-4L-2D in last 10. Recent 4-0 vs Cultural Leonesa (outlier).
- Home corner profile from model: 6.0 for + 2.0 against = **8.0 total** — but **only 5 home games in sample.** The 2.0 corners conceded at home is suspiciously low and likely not sustainable.
- APWin overall average: 9.40–9.59 corners/game.

**Malaga (4th, 55 pts):**
- Strong form: 6W in last 10. Last result: 0-0 vs Leganés.
- **20th in league for corners (8.56 avg).** Malaga is a LOW-CORNER team — they don't generate many corners and don't concede many either.
- Away corner profile: 3.5 for + 6.0 against = 9.5 total in away games.
- Top scorer: Chupe (16 goals) — threat through central play, not wing-based crossing.

**Matchup dynamics:** Malaga controls possession and plays through the center (Chupe, Larrubia). They don't rely on wing play → fewer crosses → fewer corners. Andorra mid-table, no urgent motivation. 0-0 or 1-0 type game.

**⚠️ FLAG:** Andorra's home sample is only 5 games. The 2.0 corners conceded at home is almost certainly a small-sample artifact. With more data, this would regress toward 4-5.

**How I lose:** Andorra attack wide at home, Malaga absorb pressure and counter. Total corners creep above 11. Also, if Malaga's 0-0 draw pattern breaks and they push forward, corner count rises.

*Sources: [SportsGambler - Andorra vs Malaga](https://www.sportsgambler.com/betting-tips/football/fc-andorra-vs-malaga-prediction-lineups-odds-2026-04-01/), [ESPN](https://www.espn.com/soccer/match/_/gameId/748877/malaga-fc-andorra), [APWin](https://www.apwin.com/league/spain/segunda-division/standings/corners/)*

---

### Match 3: Huesca vs Cultural Leonesa — Lambda 9.33 | Buffer 2.17

**Huesca (20th, 31 pts):**
- Relegation zone. Recent losses: Granada 4-2, Almería 1-3, Malaga 5-3. New coach Oltra can't stop the slide.
- Missing: Toni Abad (RB), Jesús Álvarez (MF), Diego Aznar, Joaquín Fernández — **4 injuries.**
- Home corner profile: 5.0 for + 5.5 against = **10.5 total at home.** High concession rate because they lose shape.
- APWin: 9.62 avg, 9.36 at home.

**Cultural Leonesa (22nd, 28 pts):**
- Dead last (effectively). Winless under coach Rubén de la Barrera (1 draw in 5).
- Recent 0-4 loss to Andorra. Missing: Matia Barzic (CB). Captain Rodri Suárez back from suspension.
- Away corner profile: 2.6 for + 6.2 against = **8.8 total in away games.**
- APWin: 9.03 avg (14th in league).

**Matchup dynamics:** Relegation six-pointer = high emotional stakes but low quality. Both teams are bad. Huesca will push at home but are disorganized (4 injuries, 3 consecutive losses with 8 goals conceded). Cultural Leonesa sit deep away (only 2.6 corners for on the road). Expected: scrappy, disjointed 1-1 or 2-1 with moderate corners.

**⚠️ FLAG:** Huesca concede 5.5 corners per home game — opponents attack them because they're vulnerable. Buffer of only 2.17 is moderate. Lambda 9.33 = league average. This is on the edge.

**How I lose:** Relegation desperation → Huesca press constantly, get 6-7 corners themselves, Cultural Leonesa get 5 on counters = 11-12 total. Huesca's high-scoring (and high-conceding) recent trend (4-2, 1-3, 5-3) suggests open, chaotic games that generate corners.

*Sources: [Sport.es - Huesca vs Cultural Leonesa preview](https://www.sport.es/es/noticias/huesca/previa-sd-huesca-cultural-leonesa-128641965), [APWin - predictions](https://www.apwin.com/predictions/huesca-vs-cultural-leonesa-prediction-segunda-division-01-04-2026/), [StatsHub](https://statshub.sportradar.com/sportradar/en/match/61623738)*

---

### Match 4: Racing Santander vs Gijon — Lambda 9.68 | Buffer 1.82

**Racing Santander (1st, 59 pts):**
- League leaders but **2 consecutive losses** (4-0 to Albacete, 2-0 to Zaragoza). Lead cut to 1 point.
- **#1 in the league for corners: 10.79 avg, 11.93 AT HOME.**
- Home corner profile: 6.5 for + 4.7 against = **11.2 total corners at home from model data.**
- Style: Aggressive, attacking at El Sardinero. High pressing → high corner count.

**Sporting Gijon (10th, 46 pts):**
- Mixed form, winless in consecutive outings. 1 clean sheet in last 9 games.
- 6th in league for corners (9.76 avg, 10.2 at home).
- Away corner profile: 2.89 for + 6.11 against = 9.0 total in away games.

**🚨 CRITICAL FLAG:** Racing Santander's home corner average is **11.93** — this is **ABOVE** the Under 11.5 line. The model lambda (9.68) significantly underestimates the empirical home average. Why?
- Model blends 80% empirical + 20% league mean (9.33). This pulls lambda down.
- Gijon's low away corner production (2.89 for) also pulls total down.
- But Racing home is a corner machine: 6.5 corners for themselves + opponents get 4.7 = 11.2 even in model data.
- With 2 consecutive losses, Racing will be desperate to attack → more corners.

**How I lose (heavily):** Racing push hard at home after 2 losses, generate 7-8 corners. Gijon concede 6.11 corners per away game (opponents get 6.11 against them). Total: 12+ easily.

*Sources: [SportsKeeda - Racing vs Gijon](https://www.sportskeeda.com/football/racing-de-santander-vs-sporting-gijon-prediction-betting-tips-april-1st-2026), [SportsGambler](https://www.sportsgambler.com/betting-tips/football/racing-club-santander-vs-sporting-gijon-prediction-lineups-odds-2026-04-01/), [APWin - Racing Santander Corners](https://www.apwin.com/team/real-racing-club-de-santander/standings/corners/)*

---

## STEP 3: Self-Verification (CoVe Checklist)

### Q1: "Did I analyze objectively the specific numbers, dates, or timelines?"

**YES — with caveats.**
- Lambda, P(U11.5), buffer — all computed from exact NB(lambda, k=200) CDF. Cross-verified against raw P(U12.5) from CSV.
- P(U11.5) = P(U12.5) − P(X=12). Values confirmed via scipy.
- **Caveat:** U11.5 probabilities are RAW (no Platt calibration). The U12.5 calibration inflates raw → cal (e.g., 91.3% → 93.0% for Burgos). If similar calibration applied to U11.5, the probabilities could be 2-3pp higher. But I cannot assume this — calibration at different probability levels works differently.
- Andorra and Cultural Leonesa have **only 5 home/away games** in the model training sample → corner profiles unreliable.

### Q2: "Did I crosscheck with other info from internet? Are sources reliable?"

**YES.**
- APWin corner stats: industry-standard, based on full season (29-32 matches per team). Confirmed with corner-stats.com and FootyStats.
- Match previews from ESPN, DailySports, SportsKeeda, Sport.es — reliable sports media.
- **Key crosscheck:** Racing Santander's home corner average (11.93 from APWin) vs model lambda (9.68) = **2.25 corner gap.** This is the largest model-vs-reality discrepancy in the dataset. **FLAG CONFIRMED.**
- Burgos CF 22nd (last) in league for corners — aligns with model lambda 8.37.
- Malaga 20th in league for corners — aligns with model lambda 8.64 for their away match.

### Q3: "Did I make assumptions or just analyze?"

**Assumptions made:**
1. **Assumed** U11.5 raw probability is a reasonable proxy without calibration. Could be 2-3pp off in either direction.
2. **Assumed** Andorra's 2.0 home corners against (5-game sample) is unsustainable. This is an educated assumption — league average is ~4.5 corners against, and no team sustains 2.0 over 20+ games.
3. **Assumed** Racing's 2 consecutive losses will increase their attacking intensity → more corners. This is behavioral, not statistical. Could be wrong if they play cautiously to protect their lead.
4. **Did NOT assume** specific injury impacts on corner counts — this is too speculative (injuries affect goals more directly than corners).

### Q4: "What details did I include in my recommendations for picks?"

Each pick includes:
- Model probability (raw NB CDF) and fair odds
- Buffer (how many corners below the line)
- Team corner profiles (model training data + external APWin data)
- Match context (form, league position, motivation)
- Style analysis (defensive vs attacking, width vs central play)
- "How I lose" scenario
- Specific flags where model contradicts empirical data

---

## STEP 4: Corrections — What Changed and Why

| Match | Draft Assessment | Correction | Reason |
|-------|-----------------|------------|--------|
| Burgos vs Ceuta | Strong pick (85.6%) | **CONFIRMED #1** | No contradictions. Model (lambda 8.37) aligns with APWin (Burgos last in league for corners). Defensive home team, weak away opponent. |
| Andorra vs Malaga | Solid pick (83.2%) | **CONFIRMED #2 with warning** | Malaga is genuinely low-corner (20th). But Andorra home sample = 5 games. h_against = 2.0 is likely noise. Real probability probably 80-83%. |
| Huesca vs Cultural | Moderate pick (76.6%) | **CONFIRMED #3 but borderline** | Lambda 9.33 = league average. Buffer only 2.17. Relegation stakes add chaos. Huesca recent games are HIGH-SCORING (4-2, 1-3, 5-3) which correlates with higher corners. Probability realistically 73-77%. |
| Racing vs Gijon | Model says 72.9% | **🚨 REMOVED** | Racing HOME avg is 11.93 corners — ABOVE the 11.5 line. Model lambda 9.68 underestimates by 2.25 corners. After 2 consecutive losses, Racing will attack aggressively at El Sardinero. This is the clearest model-vs-reality contradiction in the dataset. |

**What changed:**
- Racing vs Gijon: **REMOVED.** Model data (lambda 9.68, P=72.9%) was already the weakest pick, but external data reveals a fundamental contradiction: Racing's home corner average exceeds the Under 11.5 line. The model's lambda is artificially depressed by the 20% league-mean blend and Gijon's low away corner production. In practice, El Sardinero is a corner-generating venue.
- Andorra vs Malaga: **Added sample-size warning.** Probability still solid but less certain than Burgos-Ceuta.
- Huesca vs Cultural: **Added relegation-chaos flag.** Huesca's recent open, high-scoring games suggest elevated corner risk.

---

## STEP 5: Final Top 3 Picks — Under 11.5 Corners

### #1 — Burgos CF vs Ceuta — CORNERS UNDER 11.5 ✅

| Metric | Value |
|--------|-------|
| Model P(U11.5) | **85.6%** |
| Fair Odds | 1.169 |
| Lambda | 8.37 |
| Buffer | 3.13 corners |
| Confidence | **HIGH** |

**Why #1:** This is the cleanest pick in the dataset. Burgos CF is **last in LaLiga2 for corners per game (8.21 avg)**. At home, their games average only 7.6 total corners from model training data. They've kept 5 clean sheets in a row — a sign of controlled, low-tempo football that suppresses corner counts for both sides. Ceuta have lost 5 of 6 away games and won't generate significant attacking pressure.

**Model-reality alignment:** Lambda 8.37 ↔ APWin 8.21 avg ↔ Model profile 7.6 home total. All three data points agree: this is a low-corner match.

**How I lose:** Ceuta, fighting for playoff positioning (47 pts, 8th place), push forward unexpectedly. Burgos counter aggressively rather than sitting deep. Total reaches 12+. Probability: ~14%.

*Sources: [APWin SP2 Corners](https://www.apwin.com/league/spain/segunda-division/standings/corners/), [DailySports](https://dailysports.net/predictions/burgos-vs-ceuta-prediction-head-to-head-and-probable-lineups-01-april-2026/)*

---

### #2 — Andorra vs Malaga — CORNERS UNDER 11.5 ✅

| Metric | Value |
|--------|-------|
| Model P(U11.5) | **83.2%** |
| Fair Odds | 1.201 |
| Lambda | 8.64 |
| Buffer | 2.86 corners |
| Confidence | **MEDIUM-HIGH** |

**Why #2:** Malaga is **20th in LaLiga2 for corners (8.56 avg)** — a fundamentally low-corner team. Their attack runs through central players (Chupe, Larrubia), not wing play. Less wing play = fewer crosses = fewer corners. Their last game was a 0-0 draw vs Leganés — the definition of a low-event match.

**Caution:** Andorra's home sample in model training is only 5 games. The 2.0 corners conceded at home is almost certainly noise — expect regression toward 4-5 corners against. This would push the "real" lambda toward 9.0-9.5, reducing the effective buffer to ~2.0-2.5. Still under 11.5 but tighter than the model suggests.

**How I lose:** Andorra attack wide at home (they scored 4 goals vs Cultural Leonesa recently — an outlier but shows offensive potential). If Malaga is forced to defend deep and Andorra get 7+ corners, total could hit 12. Probability: ~17%.

*Sources: [APWin SP2 Corners](https://www.apwin.com/league/spain/segunda-division/standings/corners/), [ESPN](https://www.espn.com/soccer/match/_/gameId/748877/malaga-fc-andorra)*

---

### #3 — Huesca vs Cultural Leonesa — CORNERS UNDER 11.5 ⚠️

| Metric | Value |
|--------|-------|
| Model P(U11.5) | **76.6%** |
| Fair Odds | 1.306 |
| Lambda | 9.33 |
| Buffer | 2.17 corners |
| Confidence | **MEDIUM** |

**Why #3:** Cultural Leonesa produce only **2.6 corners per away game** (the lowest away production in this dataset). They are last in the table, winless under their coach, and won't generate sustained attacking pressure. This should keep the total manageable despite Huesca's higher home environment (9.36 avg at home per APWin).

**Caution — Relegation chaos:** Huesca's last 3 games: 4-2, 1-3, 5-3. These are open, chaotic, high-event matches. High-scoring games correlate with higher corner counts because both teams push forward. Lambda 9.33 equals the league average — no edge from the matchup itself. Buffer is only 2.17.

**How I lose:** Huesca, desperate at home (20th, 31 pts), press relentlessly. Cultural Leonesa defend deep but concede 6.2 corners per away game. Huesca get 6-7 corners + Cultural get 5 on transitions = 12 total. Alternatively, the game opens up like Huesca's recent fixtures and corners pile up in the chaos. Probability: ~23%.

*Sources: [Sport.es](https://www.sport.es/es/noticias/huesca/previa-sd-huesca-cultural-leonesa-128641965), [APWin](https://www.apwin.com/predictions/huesca-vs-cultural-leonesa-prediction-segunda-division-01-04-2026/)*

---

### NOT Recommended:

| Match | P(U11.5) | Why Removed |
|-------|---------|-------------|
| **Racing Santander vs Gijon** | 72.9% | Racing HOME corner avg = **11.93** (APWin) — ABOVE the 11.5 line. Model lambda 9.68 underestimates by ~2.25 corners. League leaders after 2 losses will attack aggressively at El Sardinero. Model-reality contradiction too large. |

---

## Summary Table

| Rank | Match | P(U11.5) | Buffer | Confidence | Key Factor |
|------|-------|---------|--------|------------|------------|
| **1** | Burgos CF vs Ceuta | 85.6% | 3.13 | HIGH | Burgos last in SP2 for corners (8.21 avg), 5 clean sheets |
| **2** | Andorra vs Malaga | 83.2% | 2.86 | MED-HIGH | Malaga 20th for corners (8.56 avg), central play style |
| **3** | Huesca vs Cultural Leonesa | 76.6% | 2.17 | MEDIUM | Cultural away: only 2.6 corners/game; relegation risk |
| ~~4~~ | ~~Racing vs Gijon~~ | ~~72.9%~~ | ~~1.82~~ | ~~REMOVED~~ | ~~Racing home avg 11.93 > line 11.5~~ |

---

*Analysis generated: 2026-04-01*
*Model: Negative Binomial Corners (SP2, k=200 ≈ Poisson, trained on Flashscore + football-data.co.uk)*
*Line: Under 11.5 (derived from U12.5 model — no separate calibration)*
*CoVe: 1 removed (Racing Santander home corners > line), 3 confirmed*
*Sources cited inline per match*
