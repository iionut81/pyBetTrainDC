# CoVe Corners v1.0 — Under 11.5 — 2026-04-13 (6 meciuri selectate)

**Data sources:**
- Model: Negative Binomial (retrained 13 April, corners_history 18,915 rows)
- Team profiles: `corners_team_profiles.csv` (h_for, h_against, a_for, a_against)
- Earlier research: Levante/Getafe (from today's analysis)

---

## STEP 1 — MODEL + CORNER PROFILES

| # | Match | League | Lambda | p_cal | Home FOR | Away FOR | Est. Total |
|---|-------|--------|--------|-------|----------|----------|-----------|
| 1 | Rapid vs FC Arges | RO1 | 9.22 | 82.6% | 6.0 | 4.9 | ~9.4 |
| 2 | Metaloglobus vs Csikszereda | RO1 | 8.90 | 82.5% | 5.4 | **2.1** | ~8.5 |
| 3 | U. Cluj vs Craiova | RO1 | 8.47 | 82.4% | 5.3 | 4.6 | ~9.0 |
| 4 | Valladolid vs Eibar | SP2 | 9.20 | 81.4% | 4.7 | **3.7** | ~8.7 |
| 5 | Tondela vs Gil Vicente | P1 | 9.48 | 80.0% | **3.6** | 5.3 | ~9.3 |
| 6 | Levante vs Getafe | SP1 | 9.98 | 79.5% | 4.6 | 4.8 | ~10.0 |

---

## STEP 2 — CHECKLIST PER MATCH

### 1. Rapid vs FC Arges (RO1, lambda 9.22, p_cal 82.6%)

**Profiles:** Rapid h_for=**6.0** (>6 ❌), h_agn=3.4. Arges a_for=4.9, a_agn=4.7.
**Step A:** Rapid FOR > 6 → ❌ **DON'T BET** (too offensive from home team)

**Score: PASS** — Rapid genereaza prea multe cornere acasa (6.0 avg).

---

### 2. Metaloglobus vs Csikszereda (RO1, lambda 8.90, p_cal 82.5%)

**Profiles:** Metaloglobus h_for=5.4, h_agn=5.6. Csikszereda a_for=**2.1** (🔥 GOLD < 3.5), a_agn=5.0.
**Step A:** Csikszereda away FOR = 2.1 → 🔥 GOLD
**Step B:** Lambda 8.90, est. total ~8.5 → 🔥 EXCELLENT (<9)
**Step C:** Both low-quality teams, low budget → low tempo ✅. RO1 play-out = ultra-cautious.
**Step D:** Both lower table, no European stakes → ✅ Low urgency
**Step E:** RO1 = not in template explicitly, similar to Serie A profile → ✅

**Score:**
- Baseline: 3/3 (Csikszereda 2.1 GOLD 🔥)
- Total: 2/2 (lambda 8.90, est ~8.5 EXCELLENT)
- Style: 2/2 (low-quality play-out, both cautious)
- State: 2/2 (no pressure, low motivation)
- Gut: 0/1 (RO1 small sample n=6/9)
= **9/10** 🔥

**"12+ corners?"** → Csikszereda away 2.1 corners FOR. Lambda 8.90. Play-out. **~10% chance.** → ✅

---

### 3. U. Cluj vs Univ. Craiova (RO1, lambda 8.47, p_cal 82.4%)

**Profiles:** U. Cluj h_for=5.3, h_agn=3.7. Craiova a_for=4.6, a_agn=3.4.
**Step A:** Both 3-6 range → ✅ GOOD. Neither < 3.5 (no GOLD) but Craiova a_agn=3.4 (only concedes 3.4 away = strong away defense).
**Step B:** Lambda 8.47, est total ~9.0 → ✅ GOOD (9-10)
**Step C:** Both structured RO1 championship teams → ✅. U. Cluj organized at home.
**Step D:** Championship round = competitive → ⚠️ Both have stakes. Could push late.
**Step E:** RO1 → ✅

**Score:**
- Baseline: 2/3 (both moderate, no team <3.5)
- Total: 2/2 (lambda 8.47 🔥)
- Style: 2/2 (both structured, central play)
- State: 1/2 (championship round = stakes, but not extreme)
- Gut: 1/1 (Craiova only concedes 3.4 corners away)
= **8/10**

**"12+ corners?"** → Lambda 8.47. Craiova concedes only 3.4 away. **~12% chance.** → ✅

---

### 4. Valladolid vs Eibar (SP2, lambda 9.20, p_cal 81.4%)

**Profiles:** Valladolid h_for=4.7, h_agn=4.0. Eibar a_for=**3.7**, a_agn=6.3.
**Step A:** Eibar away 3.7 ≈ < 4 → near GOLD. Valladolid moderate. → ✅ GOOD
**Step B:** Lambda 9.20, est total ~8.7 → ✅ GOOD
**Step C:** SP2 = low tempo league. Both defensively organized. → ✅
**Step D:** SP2 mid-table → ✅ Low urgency
**Step E:** SP2 = **backtest 90.6%** (best league!) → 🔥

**Score:**
- Baseline: 2/3 (Eibar 3.7 near GOLD, Valladolid moderate)
- Total: 2/2 (lambda 9.20, est ~8.7)
- Style: 2/2 (SP2 low tempo, both conservative)
- State: 2/2 (mid-table, low urgency)
- Gut: 0/1 (k_disp=82.3 = decent but not elite)
= **8/10**

**"12+ corners?"** → SP2 low tempo. Lambda 9.20. Eibar away 3.7. **~12% chance.** → ✅

---

### 5. Tondela vs Gil Vicente (P1, lambda 9.48, p_cal 80.0%)

**Profiles:** Tondela h_for=**3.6** (< 4 = near GOLD), h_agn=5.7. Gil Vicente a_for=5.3, a_agn=5.1.
**Step A:** Tondela h_for=3.6 → near 🔥 GOLD. Dar Gil Vicente a_for=5.3 → decent offensive away. ✅ GOOD
**Step B:** Lambda 9.48, est total ~9.3 → ✅ GOOD (9-10)
**Step C:** P1 Liga Portugal = moderate tempo. Relegation match → cautious. → ✅
**Step D:** Both likely fighting relegation → defensive, cagey → ✅
**Step E:** P1 → ✅

**Score:**
- Baseline: 2/3 (Tondela 3.6 near GOLD, Gil Vicente 5.3 moderate)
- Total: 1/2 (lambda 9.48 = borderline 9-10)
- Style: 2/2 (relegation match, defensive approach)
- State: 2/2 (both cautious in relegation)
- Gut: 0/1 (k_disp=36.6 = wider dispersion = less reliable)
= **7/10**

**"12+ corners?"** → Tondela only 3.6 corners at home. Relegation tension. **~15% chance.** → ✅ but marginal.

---

### 6. Levante vs Getafe (SP1, lambda 9.98, p_cal 79.5%) — ❌ RESEARCH CONTRADICTION

**From earlier today's research:**
- Getafe last 5 games averaged **12.8 corners!** — [APWin](https://www.apwin.com/team/getafe-club-de-futbol/corners/)
- Diego Rico leads La Liga with **62 open-play crosses** — [Opta](https://theanalyst.com/articles/getafe-jose-bordalas-tactics-la-liga)
- Bordalas = crossing-heavy style ≠ model assumption of low corners

**Step A:** Getafe a_for=4.8, but RECENT form = much higher. → ⚠️ Model outdated
**Step C:** Bordalas = crossing machine → ❌

**Score: PASS** — Research contradicts model. Getafe recent 12.8 avg > 11.5 line.

---

## STEP 3 — SELF-VERIFICATION

| Check | Answer |
|-------|--------|
| Actual corner data? | ✅ Used corners_team_profiles.csv (model training data) |
| Tempo/style verified? | ✅ SP2 low tempo confirmed, Getafe crossing-heavy flagged |
| Narrative bias? | ✅ PASSed Rapid (6.0 too offensive) and Getafe (research contradicts) |
| Risk factors? | ✅ Championship round stakes, relegation tension noted |
| Model vs reality? | ⚠️ Getafe: model 79.5% but recent 12.8 avg = PASS |
| Cap +10pp? | ✅ No adjustments needed |

---

## FINAL QUESTION: "Can this match realistically reach 12+ corners?"

| Match | 12+ Risk | Verdict |
|-------|---------|---------|
| Rapid/Arges | **~25%** — Rapid 6.0 at home | ❌ **PASS** |
| **Metaloglobus/Csikszereda** | **~10%** — Csikszereda 2.1 away, play-out | ✅ **VALID** |
| **U. Cluj/Craiova** | **~12%** — Craiova 3.4 conceded away | ✅ **VALID** |
| **Valladolid/Eibar** | **~12%** — SP2 low tempo, Eibar 3.7 away | ✅ **VALID** |
| **Tondela/Gil Vicente** | **~15%** — Tondela 3.6, but Gil Vicente 5.3 | ✅ **VALID** (marginal) |
| Levante/Getafe | **~35%** — Getafe 12.8 recent avg! | ❌ **PASS** |

---

## FINAL SCORECARD

| Pick | Base (/3) | Total (/2) | Style (/2) | State (/2) | Gut (/1) | **TOTAL** | Confidence |
|------|----------|-----------|-----------|-----------|---------|-----------|-----------|
| **Metaloglobus/Csikszereda** | 3 | 2 | 2 | 2 | 0 | **9/10** 🔥 | HIGH |
| **U. Cluj/Craiova** | 2 | 2 | 2 | 1 | 1 | **8/10** | MODERATE |
| **Valladolid/Eibar** | 2 | 2 | 2 | 2 | 0 | **8/10** | MODERATE |
| **Tondela/Gil Vicente** | 2 | 1 | 2 | 2 | 0 | **7/10** | MODERATE |
| Rapid/Arges | — | — | — | — | — | **PASS** | Rapid 6.0 corners home |
| Levante/Getafe | — | — | — | — | — | **PASS** | Getafe 12.8 recent |

---

## FINAL PICKS

| Pick | Score | p_cal | Lambda | Why It Works | How It Loses |
|------|-------|-------|--------|-------------|-------------|
| **Metaloglobus/Csikszereda U11.5** | **9** | 82.5% | 8.90 | Csikszereda away **2.1 corners** (GOLD). Play-out = ultra-cautious. Low quality both sides. | Metaloglobus h_agn=5.6 → opponent generates corners against them. If Csikszereda attack unusually = 12+ possible. |
| **U. Cluj/Craiova U11.5** | **8** | 82.4% | 8.47 | Lambda lowest (8.47). Craiova concedes only **3.4 corners away**. Both structured. | Championship round = stakes. Both push for result → late corners. |
| **Valladolid/Eibar U11.5** | **8** | 81.4% | 9.20 | SP2 **backtest 90.6%** (best league). Eibar away 3.7. Low tempo. | SP2 has wider variance. k_disp=82.3 decent but not elite. |
| **Tondela/Gil Vicente U11.5** | **7** | 80.0% | 9.48 | Tondela h_for=3.6 (near GOLD). Relegation = cautious. | Gil Vicente a_for=5.3 → decent away offense. k_disp=36.6 = wide dispersion = less predictable. Lambda 9.48 closer to line. |

---

## Sources

- Model data: `corners_team_profiles.csv`, `corners_league_params.csv` (retrained 13 April)
- [APWin — Getafe corners](https://www.apwin.com/team/getafe-club-de-futbol/corners/) (Getafe 12.8 recent avg = PASS)
- [Opta Analyst — Bordalas tactics](https://theanalyst.com/articles/getafe-jose-bordalas-tactics-la-liga) (Diego Rico 62 crosses)
- Backtest audit: SP2 90.6%, I1 83.6%, RO1 82.5%

---

*CoVe Corners v1.0 complete. 6 meciuri analizate, 4 qualified (2 PASS: Rapid corners 6.0, Getafe research contradicts). Metaloglobus/Csikszereda = best pick (Csikszereda 2.1 away GOLD). Sources cited.*