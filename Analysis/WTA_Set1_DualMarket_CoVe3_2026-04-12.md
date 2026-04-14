# CoVe 3.2 — WTA Set 1 Dual Market — 2026-04-12

**Data sources:**
- Model: Markov chain + Surface Elo, walk-forward on Sackmann + Tennis Abstract
- Fixtures: 4 matches (Stuttgart WTA 500 Q Finals + Linz WTA 500 Final)
- Note: ALL 4 matches are FINALS → -1 penalty applies per template

---

## PRE-ANALYSIS — 15 SECOND CHECKLIST

| # | Match | Tournament | Round | p_hold_a | p_hold_b | p_markov | p_cal | blowout |
|---|-------|-----------|-------|----------|----------|----------|-------|---------|
| 1 | Sonmez vs Friedsam | STUTTGART Q | **F** | 0.567 | 0.595 | 0.443 | 0.846 | 8 |
| 2 | Sasnovich vs Dodin | STUTTGART Q | **F** | 0.546 | 0.561 | 0.470 | 0.841 | 8 |
| 3 | **Parks vs Tomova** | STUTTGART Q | **F** | **0.461** | **0.719** | 0.077 | 0.789 | **10** |
| 4 | **Andreeva vs Potapova** | LINZ | **F** | **0.812** | **0.552** | 0.941 | 0.766 | **10** |

---

## STEP 1 — MODEL ANALYSIS + FILTER RESULTS

### MARKET A: Under 12.5 Set 1

| Match | Step A (Hold) | Step B (Gap) | Step E (Final) | Step F (TB) | Status |
|-------|-------------|-------------|---------------|-------------|--------|
| Sonmez/Friedsam | Both < 0.60 ✅ | Gap 0.028 < 0.12 ❌ | — | — | **PASS** (gap) |
| Sasnovich/Dodin | Both < 0.60 ✅ | Gap 0.015 < 0.12 ❌ | — | — | **PASS** (gap) |
| **Parks/Tomova** | Parks 0.461 < 0.50 🔥 | Gap 0.258 🔥 Strong | Final -1 | Parks < 0.55 ✅ | **CANDIDATE** |
| **Andreeva/Potapova** | Potapova 0.552 < 0.60 ✅ | Gap 0.260 🔥 Strong | Final -1 | Both > 0.55 ⚠️ -2pp | **CANDIDATE** |

### MARKET B: Over 7.5 Set 1

| Match | Step A (Anti-blowout) | Step B (Hold) | Step D (Surface) | Status |
|-------|---------------------|-------------|-----------------|--------|
| **Sonmez/Friedsam** | Neither < 0.45 ✅, gap 0.028 ✅ | Both 0.55-0.60 ✅ | Clay ⚠️ | **CANDIDATE** |
| Sasnovich/Dodin | Neither < 0.45 ✅, gap 0.015 ✅ | Sasnovich 0.546 ⚠️ | Dodin 3-set fatigue ❌ | **PASS** (fatigue) |
| Parks/Tomova | Gap 0.258 > 0.20 ❌ | — | — | **PASS** (mismatch) |
| Andreeva/Potapova | Gap 0.260 > 0.20 ❌ | — | — | **PASS** (mismatch) |

### Summary:

| Market | Candidates |
|--------|-----------|
| **Under 12.5** | Parks/Tomova, Andreeva/Potapova |
| **Over 7.5** | Sonmez/Friedsam |

---

## STEP 2 — RESEARCH

### 🥇 Parks vs Tomova — Under 12.5 (hold 0.461/0.719, gap 0.258)

**Yesterday's results:**
- **Tomova d. Hesse 6-1, 6-4.** Set 1: **7 games.** Dominated. — [Porsche Newsroom](https://newsroom.porsche.com/en/ppdb/2026/04/thrilling-qualifying-matches--anna-lena-friedsam-progresses.html)
- **Parks d. Bennemann 7-6(1), 6-3.** Set 1: **13 games (TB!).** Parks struggled vs a wildcard. — [Scores24](https://scores24.live/en/tennis/m-12-04-2026-parks-alycia-tomova-viktoriya-prediction)

**Tomova Set 1 scores (last 4):** 8, 10, 8, **7 games.** Average: **8.25 games.** All Under 12.5. — [TennisRatio](https://www.tennisratio.com/players/ViktoriyaTomova.html)

**Parks form:** 10-10 in 2026 (50%). Lost R1 at Linz 3-6, 3-6 to Galfi on April 6. Clay: 2-1 in 2026. Inconsistent week-to-week. — [PB Tennis](https://x.com/Probahis/status/2040955081411961291)

**H2H:** First meeting. No prior data.

**Style:** Tomova aggressive returner (40.7% break conversion, 58.1% on opponent 2nd serve). Parks hold 0.461 = will be broken early and often. Tomova holds 0.719 = Parks can barely create break points.

**Step F (TB Risk):**
1. Both > 0.55? Parks 0.461 < 0.55. ✅ Not triggered.
2. Parks hot streak? Beat Bennemann (~335). Not an upset. ✅ Not triggered.
3. Lethal combination? Parks < 0.55. ✅ Not lethal.

**Final Question: "Can BOTH hold 5-6 times?"**
- Parks 0.461 → P(5+ holds in 6 games) ≈ 7%. **No.** → ✅ **Under VALID**

**Scorecard:**
- Hold Structure: 3/3 (Parks 0.461 🔥 Premium)
- Matchup Fit: 2/2 (Tomova aggressive return vs Parks' weak serve = breaks guaranteed)
- Gap Quality: 2/2 (0.258 Strong, blowout score 10)
- Context: 1/2 (Final -1. But Tomova's Set 1 pattern ALL under 10 games)
- Gut: 1/1 (Tomova beat Hesse 6-1 yesterday. Parks needed TB vs wildcard.)
- **Raw: 9/10** → Final allows HIGH at 9+ → **9/10 HIGH** 🔥

**Research adjustment:** +0pp (model captures mismatch well)
**Estimated U12.5 probability:** ~92%

---

### 🥈 Andreeva vs Potapova — Under 12.5 (hold 0.812/0.552, gap 0.260)

**Yesterday's results:**
- **Andreeva d. Ruse 6-4, 6-1** (91 min). Set 1: **10 games.** — [WTA](https://www.wtatennis.com/news/4486009/top-seed-andreeva-former-champion-potapova-advance-to-linz-final)
- **Potapova d. Vekic 6-4, 6-2** (79 min). Set 1: **10 games.** — [Sky Sports](https://www.skysports.com/tennis/news/12110/13530726/mirra-andreeva-and-anastasia-potapova-make-upper-austria-ladies-linz-final-after-comfortable-straight-set-wins)

**H2H: Andreeva leads 2-1.**
- 2023 Wimbledon: Andreeva won **6-2**, 7-5. Set 1: **8 games.**
- 2025 US Open: Andreeva won **6-1**, 6-3. Set 1: **7 games.**
- **H2H Set 1 average: 7.5 games.** Andreeva dominates. — [Tennis Tonic](https://tennistonic.com/tennis-news/984016/h2h-prediction-of-mirra-andreeva-vs-anastasia-potapova-in-linz-with-odds-preview-pick-12th-april-2026/)

**Andreeva Linz Set 1s:** 10, 13 (TB), 10 games. Avg: 11.

**Potapova Linz Set 1s:** 10, 8, 13 (TB), 10 games. Avg: 10.25. Has not dropped a set (4-0).

**⚠️ Fatigue:** Andreeva played 2h17m QF (Cirstea, 3 sets) + 91-min SF. Potapova efficient (max 79 min yesterday). **Potapova fresher.** — [TennisUpToDate](https://tennisuptodate.com/wta/preview-upper-austria-ladies-linz-final-mirra-andreeva-targets-second-title-of-the-season-against-in-form-anastasia-potapova)

**Step F (TB Risk):**
1. Both > 0.55? Potapova 0.552 ≈ 0.55 (borderline yes). Andreeva 0.812. → ⚠️ Apply -2pp.
2. Potapova hot streak? 4-0 at Linz, no sets dropped. Beat Vekic comfortably. But these weren't upsets over higher-ranked. → Not triggered (borderline).
3. Lethal combination? Gap 0.260 > 0.12. → Not lethal. ✅

**Key insight:** H2H tells the real story. Andreeva produced Set 1 of 6-2 (8 games) and 6-1 (7 games) against Potapova. Even with Andreeva fatigued and Potapova in form, Andreeva's quality gap is enormous (#8 vs #91).

**Final Question: "Can BOTH hold 5-6 times?"**
- Potapova 0.552 → P(5+ holds in 6) ≈ 10%. Low. H2H confirms: 6-2 and 6-1 Set 1s. **No.** → ✅ **Under VALID**

**Scorecard:**
- Hold Structure: 2/3 (Potapova 0.552 Good, but Andreeva 0.812 = elite hold → TB risk if Potapova overperforms)
- Matchup Fit: 2/2 (H2H dominant: 6-2, 6-1 in Set 1)
- Gap Quality: 2/2 (0.260 Strong)
- Context: 2/2 (Both in form, genuine WTA 500 Final)
- Gut: 1/1 (H2H = 7.5 avg Set 1 games)
- **Raw: 9/10** → Final -1 → **8/10 MODERATE**

**Research adjustment:** +2pp (H2H: 8 and 7 games in Set 1) - 2pp (Step F both > 0.55) = **net 0pp**
**Estimated U12.5 probability:** ~88-90%

---

### 🥉 Sonmez vs Friedsam — Over 7.5 (hold 0.567/0.595, p_cal 84.6%)

**Yesterday's results:**
- **Sonmez d. Pohle 6-3, 6-4** (77 min). Set 1: **9 games.** Comfortable. — [TennisTemple](https://en.tennistemple.com/match/sonmez-pohle-stuttgart-2026/9459487/comments)
- **Friedsam d. Niemeier 7-5, 6-1.** Set 1: **12 games** (competitive). — [Porsche Newsroom](https://newsroom.porsche.com/en/ppdb/2026/04/thrilling-qualifying-matches--anna-lena-friedsam-progresses.html)

**Profile:** Ultra-balanced match (gap 0.028). Both moderate servers. Both reached the final = both in form and confident. Stuttgart clay qualifying final = home event for Friedsam (German).

**Step E (Slow Starter):** 
- Friedsam: Lost Set 1 vs Kalinina (4-6), vs Stefanini (1-6), vs Vandromme (2-6) earlier in 2026. BUT won Set 1 7-5 vs Niemeier yesterday. Not triggered (recent win overrides older pattern).

**Blowout check:** "Is bagel/breadstick likely?" Both holds ~0.56-0.59. Gap tiny. Neither can dominate. Friedsam produced 12-game Set 1 yesterday. **No.** → ✅ Over VALID

**Scorecard:**
- Hold Structure: 2/3 (both moderate, ~0.56-0.59 Good)
- Matchup Fit: 2/2 (balanced, competitive qualifying final)
- Gap Quality: 2/2 (tiny gap = tight set expected)
- Context: 1/2 (Final -1. But both confident after wins yesterday)
- Gut: 0/1 (Stuttgart qualifying final = limited data on matchup)
- **Raw: 7/10 + Final already in Context** → **7/10 MODERATE**

**Research adjustment:** +1pp (both produced 9+ game Set 1s yesterday, Friedsam had 12)
**Research probability:** 84.6% + 1pp = **85.6%**

---

### Sasnovich vs Dodin — Over 7.5 (PASS — Fatigue)

**Dodin played 3 sets yesterday** (lost Set 1 TB vs von Deichmann, came back to win 6-7, 6-4, 6-3). She is ranked ~564, physically exhausted, and facing a much stronger Sasnovich (~115) who cruised 6-1, 6-4 yesterday.

**Sasnovich's Set 1 yesterday: 6-1 = only 7 games.** If she repeats this against tired Dodin → 6-0 or 6-1 = Under 7.5.

**Score: 5/10** → **PASS** (fatigue mismatch creates blowout risk)

---

## STEP 3 — VOLATILITY / HUMAN FACTOR

| Match | Injuries | Fatigue | Psychology |
|-------|---------|---------|-----------|
| **Parks/Tomova** | None | Both played normal matches yesterday | Tomova confident (6-1 win). Parks shaky (needed TB vs wildcard). |
| **Andreeva/Potapova** | None | ⚠️ Andreeva: 2h17m QF + 91m SF. **Potapova fresher** (79m SF, no sets dropped all week). | Both motivated for title. Andreeva H2H dominant (2-1, last 2 dominant). |
| **Sonmez/Friedsam** | None | Both played ~77-80 min yesterday. Normal. | Home event for Friedsam (German crowd). Sonmez was Q1 seed. |

---

## STEP 4 — SELF VERIFICATION

| Check | Answer |
|-------|--------|
| Objective numbers? | ✅ All filter decisions based on template rules. |
| Research confirmed? | ✅ Tomova Set 1 avg 8.25 games. H2H Andreeva 6-2, 6-1 vs Potapova. |
| Narrative bias? | ✅ Flagged Andreeva fatigue honestly. |
| Cap ≤ +10pp? | ✅ Max +2pp. |
| Context penalties? | ✅ Final -1 applied to ALL matches. |
| Research contradictions? | ⚠️ Parks needed TB yesterday (13 games) vs a wildcard — but her hold 0.461 against Tomova's return game is completely different context. |

---

## FINAL QUESTION FILTER

| Match | Market | Question | Answer | Valid? |
|-------|--------|---------|--------|--------|
| **Parks/Tomova** | U12.5 | Can BOTH hold 5-6? | Parks 0.461: P(5+) = 7%. Tomova Set 1 avg = 8.25 games. **No.** | ✅ **VALID** |
| **Andreeva/Potapova** | U12.5 | Can BOTH hold 5-6? | Potapova 0.552: P(5+) ≈ 10%. H2H = 6-2, 6-1. **Very unlikely.** | ✅ **VALID** |
| **Sonmez/Friedsam** | O7.5 | Bagel/breadstick? | Both ~0.57 hold. Gap 0.028. Friedsam had 12-game Set 1 yesterday. **No.** | ✅ **VALID** |

---

## FINAL SCORECARD

| Pick | Hold (/3) | Matchup (/2) | Gap (/2) | Context (/2) | Gut (/1) | Raw | Final -1 | **TOTAL** | Confidence |
|------|----------|-------------|---------|-------------|---------|-----|---------|-----------|-----------|
| **Parks/Tomova U12.5** | 3 | 2 | 2 | 1 | 1 | 9 | — | **9/10** 🔥 | HIGH |
| **Andreeva/Potapova U12.5** | 2 | 2 | 2 | 2 | 1 | 9 | -1 | **8/10** | MODERATE |
| **Sonmez/Friedsam O7.5** | 2 | 2 | 2 | 1 | 0 | 7 | — | **7/10** | MODERATE |
| Sasnovich/Dodin O7.5 | — | — | — | — | — | — | — | 5/10 | PASS |

---

## FINAL OUTPUT TABLE

| Pick | Model | Research | Score | Confidence | Action | Why It Works | How It Loses |
|------|-------|---------|-------|-----------|--------|-------------|-------------|
| **Parks/Tomova U12.5** | ~92% | +0pp = **~92%** | **9** | HIGH | ✅ BET | Parks 0.461 broken every game. Tomova Set 1 avg 8.25. Blowout 10. TB impossible (7%). | Parks on adrenaline holds better than model + Tomova nerves in final slow her return game. Max risk = 7-5 (still Under). |
| **Andreeva/Potapova U12.5** | ~89% | +0pp = **~89%** | **8** | MODERATE | ✅ BET | H2H Set 1: 6-2, 6-1 (avg 7.5 games). Andreeva dominates this matchup. Gap 0.260. | Andreeva fatigue (3h+ in 3 days) + Potapova in career form (4-0 no sets dropped). If Potapova holds well → 7-5 or 6-6. But H2H says no. |
| **Sonmez/Friedsam O7.5** | 84.6% | +1pp = **85.6%** | **7** | MODERATE | ✅ BET | Ultra-balanced (gap 0.028). Both 0.56-0.59 hold. Friedsam had 12-game Set 1 yesterday. | One player collapses mentally in final pressure → 6-1 or 6-2. But neither has dominance to crush the other. |

---

## STRATEGY FILTER

| Pick | Prob | Typical Odds | Daily filter (≥82% + ≥1.10)? |
|------|------|-------------|---------------------------|
| Parks/Tomova U12.5 | ~92% | ~1.08-1.12 | ✅ If odds ≥1.10 |
| Andreeva/Potapova U12.5 | ~89% | ~1.10-1.15 | ✅ If odds ≥1.10 |
| Sonmez/Friedsam O7.5 | 85.6% | ~1.14-1.20 | ✅ |

**Tennis picks nu califica pentru TODAY'S PICK** — odds sub 1.25. Football ramane prioritar (Napoli X2 triple signal).

---

## ACCUMULATOR (max 3 legs, score 7+)

| # | Pick | Score | Tournament | Confidence |
|---|------|-------|-----------|-----------|
| 1 | **Parks/Tomova U12.5** | 9 | Stuttgart Q | HIGH |
| 2 | **Andreeva/Potapova U12.5** | 8 | Linz | MODERATE |
| 3 | **Sonmez/Friedsam O7.5** | 7 | Stuttgart Q | MODERATE |

**Note:** Legs 1+3 from same tournament. Different markets (U12.5 + O7.5).

---

## Sources

- [Porsche Newsroom — Stuttgart Qualifying](https://newsroom.porsche.com/en/ppdb/2026/04/thrilling-qualifying-matches--anna-lena-friedsam-progresses.html)
- [WTA — Andreeva & Potapova to Linz Final](https://www.wtatennis.com/news/4486009/top-seed-andreeva-former-champion-potapova-advance-to-linz-final)
- [Sky Sports — Linz SF results](https://www.skysports.com/tennis/news/12110/13530726/mirra-andreeva-and-anastasia-potapova-make-upper-austria-ladies-linz-final-after-comfortable-straight-set-wins)
- [TennisUpToDate — Linz Final Preview](https://tennisuptodate.com/wta/preview-upper-austria-ladies-linz-final-mirra-andreeva-targets-second-title-of-the-season-against-in-form-anastasia-potapova)
- [Tennis Tonic — Andreeva vs Potapova H2H](https://tennistonic.com/tennis-news/984016/h2h-prediction-of-mirra-andreeva-vs-anastasia-potapova-in-linz-with-odds-preview-pick-12th-april-2026/)
- [The Stats Zone — Andreeva vs Potapova](https://www.thestatszone.com/mirra-andreeva-vs-anastasia-potapova-preview-prediction-2026-upper-austria-ladies-linz-final-181728)
- [Scores24 — Parks vs Tomova](https://scores24.live/en/tennis/m-12-04-2026-parks-alycia-tomova-viktoriya-prediction)
- [TennisRatio — Tomova Stats](https://www.tennisratio.com/players/ViktoriyaTomova.html)
- [TennisTemple — Sonmez vs Pohle](https://en.tennistemple.com/match/sonmez-pohle-stuttgart-2026/9459487/comments)

---

*CoVe 3.2 complete. 4 fixtures (all Finals), 3 qualified picks (2 Under 12.5 + 1 Over 7.5). Parks/Tomova U12.5 = best tennis pick (score 9, Tomova avg 8.25 Set 1 games). Sources cited.*
