# CoVe 3.2 — WTA Set 1 Dual Market — 2026-04-13

**Data sources:**
- Model: Markov chain + Surface Elo (WTA data pana in 8 aprilie)
- Fixtures: 9 matches (Stuttgart WTA 500 + Oeiras 125 + Rouen WTA 250)

---

## PRE-ANALYSIS — IMMEDIATE FILTERS

| Match | Tournament | Level | Status |
|-------|-----------|-------|--------|
| Ruzic vs Samsonova | STUTTGART | WTA 500 | ✅ Evaluate |
| Siegemund vs Tomova | STUTTGART | WTA 500 | ✅ Evaluate |
| Montgomery vs Sherif | OEIRAS 125 | **WTA 125** | ❌ HARD PASS |
| Kraus vs Selekhmeteva | OEIRAS 125 | **WTA 125** | ❌ HARD PASS |
| Timofeeva vs Boulter | ROUEN | WTA 250 Clay | ⚠️ Stricter rules |
| Radivojevic vs Sramkova | OEIRAS 125 | **WTA 125** | ❌ HARD PASS |
| Lamens vs Kostovic | OEIRAS 125 | **WTA 125** | ❌ HARD PASS |
| Stephens vs Podrez | ROUEN | WTA 250 Clay | ⚠️ Stricter rules |
| Rakhimova vs Kalieva | ROUEN | WTA 250 Clay | ⚠️ Stricter rules |

**4 HARD PASS (WTA 125). 5 remaining.**

---

## STEP 1 — FILTER RESULTS

### MARKET A: Under 12.5

| Match | Hold A | Hold B | Gap | WTA 250 rules? | Status |
|-------|--------|--------|-----|----------------|--------|
| Ruzic/Samsonova | 0.604 | 0.694 | 0.090 < 0.12 | N/A (500) | **PASS** (gap) |
| Siegemund/Tomova | 0.565 | 0.626 | 0.061 < 0.12 | N/A (500) | **PASS** (gap) |
| Timofeeva/Boulter | 0.474 | 0.629 | 0.155 < 0.18 | Fail gap ≥0.18 | **PASS** |
| Stephens/Podrez | 0.679 | 0.524 | 0.155 < 0.18 | Fail gap + hold >0.48 | **PASS** |
| **Rakhimova/Kalieva** | **0.409** | 0.643 | **0.234** ≥ 0.18 | ✅ Hold <0.48, gap ≥0.18 | **CANDIDATE** |

### MARKET B: Over 7.5

| Match | Hold A | Hold B | Gap | Anti-blowout | Status |
|-------|--------|--------|-----|-------------|--------|
| **Ruzic/Samsonova** | 0.604 | 0.694 | 0.090 | Neither <0.45, gap <0.20 ✅ | **CANDIDATE** |
| Siegemund/Tomova | 0.565 | 0.626 | 0.061 | Tomova LUCKY LOSER (lost 4-6 2-6 yday) | **PASS** (collapse) |
| Timofeeva/Boulter | 0.474 | 0.629 | 0.155 | 0.474 > 0.45 ✅ but borderline | **PASS** (blowout risk) |
| Stephens/Podrez | 0.679 | 0.524 | 0.155 | ✅ but WTA debut for Podrez | **PASS** (unpredictable) |
| Rakhimova/Kalieva | 0.409 | 0.643 | 0.234 | 0.409 < 0.45 ❌ | **PASS** (blowout) |

---

## STEP 2 — RESEARCH

### 🥇 Rakhimova vs Kalieva — Under 12.5 (ROUEN WTA 250 Clay)

**Rakhimova** (#82): Hold **0.409 = ELITE** (< 0.42 🔥). 6-7 in 2026. Lost to Kostyuk at Miami. Uninspiring season.

**Kalieva** (#140, qualifier): **16-6 in 2026** — excellent form. Qualified through 3 rounds at Rouen. Also qualified at Miami (beat Parry, Galfi). Aggressive, in form.

**WTA 250 Clay strict check:**
- ✅ Hold < 0.48 → Rakhimova **0.409** (ELITE!)
- ✅ Gap ≥ 0.18 → **0.234**
- ✅ Score ≥ 9 → see below

**Step F:** Rakhimova 0.409 < 0.55 → Not triggered. ✅

**Final Q:** Can BOTH hold 5-6? Rakhimova 0.409 → P(5+) ≈ **4%**. No. → ✅ VALID

**Score:** Hold 3 + Matchup 2 + Gap 2 + Context 1 + Gut 1 = **9/10** → **capped MODERATE** (WTA 250)

---

### 🥈 Ruzic vs Samsonova — Over 7.5 (STUTTGART WTA 500 Clay)

**Ruzic** (#57): Beat Zheng at Indian Wells (Set 1: 6-4 = 10 games). Lost to Ruse at Miami (Set 1: 7-5 = 12 games). Rising.

**Samsonova** (#18): **4-9 in 2026** — terrible form. Big server (0.694 hold) but lost Set 1 **2-6** to Tagger at Linz.

**⚠️ Bagel risk:** Samsonova 4-9 = could collapse. BUT hold 0.694 is structural (serve-driven). (0.306)^5 × 0.694 ≈ 0.2% for 6-1. Structurally unlikely.

**Final Q:** Bagel likely? Samsonova serve strong even in bad form. **Structurally no, but form uncertain.** → ⚠️

**Score:** Hold 3 + Matchup 1 + Gap 2 + Context 1 + Gut 0 = **7/10** MODERATE

---

### PASSES

- **Siegemund/Tomova:** Tomova = LUCKY LOSER, lost 4-6, 2-6 yesterday. Collapse risk. PASS.
- **Timofeeva/Boulter:** WTA 250 gap 0.155 < 0.18. PASS.
- **Stephens/Podrez:** WTA 250 gap 0.155 < 0.18 + hold 0.524 > 0.48. PASS.

---

## FINAL OUTPUT

| Pick | Model | Research | Score | Confidence | Action | Why | How It Loses |
|------|-------|---------|-------|-----------|--------|-----|-------------|
| **Rakhimova/Kalieva U12.5** | ~93% | — | **9** | MODERATE | ✅ BET | Rakhimova 0.409 ELITE. Kalieva 16-6 in form. Gap 0.234. Blowout 10. | WTA 250 volatility. Rakhimova inspired serving day (4% chance). |
| **Ruzic/Samsonova O7.5** | 85.2% | -2pp = 83.2% | **7** | MODERATE | ⚠️ CAUTION | Both >0.60 Premium. Tight gap. WTA 500. | Samsonova 4-9 form → 2-6 Set 1 possible. |

---

## Sources

- Research: Ruzic beat Zheng at IW, Samsonova 4-9 in 2026, Kalieva 16-6, Tomova LL
- Model: `1.2_WTA_Set1_Over_7_5.csv`
- Template: CoVe 3.2 WTA 250 Clay strict rules

---

*CoVe 3.2 complete. 9 fixtures, 4 HARD PASS (125), 3 PASS (gap/collapse). 2 picks: Rakhimova/Kalieva U12.5 (9/10 MODERATE cap) + Ruzic/Samsonova O7.5 (7/10 MODERATE).*