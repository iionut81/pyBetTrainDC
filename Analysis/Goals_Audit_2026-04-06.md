# Goals Under 3.5 / Under 4.5 Walk-Forward Audit
## Timestamp: 2026-04-06 (post-retrain with Transfermarkt 16 leagues)
## Source: goals_backtest_summary.csv, goals_predictions.csv

---

## Pick Thresholds (based on calibration + sharpness)

### Under 4.5 (better calibrated):

| p_cal Range | Label | Action |
|-------------|-------|--------|
| **≥ 0.88** | Pick premium | BET |
| **0.82–0.88** | Pick bun | BET (with context) |
| **0.75–0.82** | Conditionat de cote | Only with good odds |
| **< 0.72** | Evita | PASS |

### Under 3.5 (more volatile — USE WITH CARE):

| p_cal Range | Label | Action |
|-------------|-------|--------|
| **≥ 0.80** | Pick bun (dar model over-confident!) | BET only in top leagues |
| **0.72–0.80** | Conditionat | Only with strong research |
| **< 0.72** | Evita | PASS |

**WARNING:** Under 3.5 raw model is over-confident at high bands. Calibration corrects it but edge is thinner. **Under 4.5 is the safer market.**

---

## Coverage

- **Data:** historical_matches_transfermarkt.csv
- **Model:** Dixon-Coles per league, walk-forward (lookback=365d, retrain=30d)
- **Date range:** ~2022 → 2026-04-05
- **Leagues:** 16 (D1, D2, DK1, E0, E1, F1, I1, I2, N1, P1, RO1, RS1, SA1, SP1, SP2, SW1)
- **Total evaluated matches:** 21,548
- **Markets:** Under 3.5, Under 4.5

---

## Overall Performance

| Market | n | hit_rate | predicted | gap | brier_cal |
|--------|---|----------|-----------|-----|-----------|
| **Under 3.5** | 21,548 | **70.5%** | 71.0% | +0.5pp | 0.2023 |
| **Under 4.5** | 21,548 | **85.8%** | 85.0% | -0.8pp | 0.1191 |

**Under 4.5 is significantly better calibrated and has higher hit rate.**

---

## Per-League Breakdown — Under 3.5 (sorted by hit_rate)

| League | n | hit_rate | predicted | gap | brier_cal | Tier |
|--------|---|----------|-----------|-----|-----------|------|
| **SP2** | 1,634 | **77.8%** | 78.4% | +0.6pp | 0.1698 | 🔥 PREMIUM |
| **I2** | 1,345 | **76.4%** | 76.1% | -0.2pp | 0.1805 | 🔥 PREMIUM |
| **RO1** | 1,865 | **76.2%** | 76.7% | +0.5pp | 0.1809 | 🔥 PREMIUM |
| **E1** | 2,075 | **75.7%** | 74.8% | -0.8pp | 0.1835 | ✅ BUN |
| **SP1** | 1,398 | **75.0%** | 74.5% | -0.4pp | 0.1820 | ✅ BUN |
| **I1** | 1,409 | **74.8%** | 73.9% | -0.9pp | 0.1877 | ✅ BUN |
| P1 | 1,134 | 71.3% | 73.2% | +2.0pp | 0.2036 | 🟡 CONDITIONAT |
| RS1 | 1,607 | 71.1% | 71.4% | +0.3pp | 0.2010 | 🟡 CONDITIONAT |
| F1 | 1,216 | 68.8% | 68.5% | -0.2pp | 0.2138 | 🟡 CONDITIONAT |
| DK1 | 503 | 66.2% | 66.8% | +0.6pp | 0.2237 | ⚠️ SLAB |
| D2 | 1,139 | 65.8% | 63.7% | -2.1pp | 0.2252 | ⚠️ SLAB |
| SA1 | 1,718 | 65.1% | 69.2% | **+4.1pp** | 0.2193 | ⚠️ SLAB (over-confident!) |
| E0 | 1,415 | 65.0% | 65.7% | +0.7pp | 0.2259 | ⚠️ SLAB |
| SW1 | 802 | 64.1% | 65.2% | +1.1pp | 0.2301 | ⚠️ SLAB |
| N1 | 1,140 | 61.5% | 63.1% | +1.6pp | 0.2335 | ❌ EVITA |
| D1 | 1,148 | 59.6% | 60.8% | +1.3pp | 0.2364 | ❌ EVITA |

### Under 3.5 League Tiers:
- **PREMIUM (hit > 75%):** SP2, I2, RO1 → strongest for Under 3.5
- **BUN (hit 72-75%):** E1, SP1, I1 → reliable
- **CONDITIONAT (hit 68-72%):** P1, RS1, F1 → only with strong odds
- **SLAB (hit 63-68%):** DK1, D2, SA1, E0, SW1 → caution
- **EVITA (hit < 63%):** N1, D1 → Under 3.5 doesn't work here (coin flip!)

**Key warning:** SA1 has +4.1pp over-confidence gap — modelul supraestimeaza Under 3.5 in Saudi Arabia.

---

## Per-League Breakdown — Under 4.5 (sorted by hit_rate)

| League | n | hit_rate | predicted | gap | brier_cal | Tier |
|--------|---|----------|-----------|-----|-----------|------|
| **I2** | 1,345 | **90.6%** | 88.6% | -2.1pp | 0.0848 | 🔥 PREMIUM |
| **SP2** | 1,634 | **90.5%** | 90.0% | -0.5pp | 0.0857 | 🔥 PREMIUM |
| **E1** | 2,075 | **89.8%** | 87.9% | -2.0pp | 0.0911 | 🔥 PREMIUM |
| **I1** | 1,409 | **89.2%** | 87.2% | -2.0pp | 0.0961 | 🔥 PREMIUM |
| **RO1** | 1,865 | **89.1%** | 88.9% | -0.1pp | 0.0971 | 🔥 PREMIUM |
| **P1** | 1,134 | **87.3%** | 86.5% | -0.8pp | 0.1102 | 🔥 PREMIUM |
| **SP1** | 1,398 | **87.1%** | 87.4% | +0.3pp | 0.1091 | 🔥 PREMIUM |
| RS1 | 1,607 | 85.9% | 85.0% | -0.9pp | 0.1176 | ✅ BUN |
| F1 | 1,216 | 84.8% | 83.4% | -1.4pp | 0.1283 | ✅ BUN |
| D2 | 1,139 | 82.7% | 79.8% | -2.9pp | 0.1429 | ✅ BUN |
| E0 | 1,415 | 82.6% | 81.4% | -1.2pp | 0.1430 | ✅ BUN |
| SA1 | 1,718 | 82.2% | 83.5% | +1.3pp | 0.1435 | ✅ BUN |
| DK1 | 503 | 81.3% | 82.1% | +0.8pp | 0.1519 | 🟡 CONDITIONAT |
| SW1 | 802 | 80.5% | 81.4% | +0.8pp | 0.1567 | 🟡 CONDITIONAT |
| N1 | 1,140 | 80.0% | 79.2% | -0.8pp | 0.1581 | 🟡 CONDITIONAT |
| D1 | 1,148 | 78.7% | 77.5% | -1.2pp | 0.1664 | 🟡 CONDITIONAT |

### Under 4.5 League Tiers:
- **PREMIUM (hit > 87%):** I2, SP2, E1, I1, RO1, P1, SP1 → **7 leagues!**
- **BUN (hit 82-87%):** RS1, F1, D2, E0, SA1 → reliable
- **CONDITIONAT (hit 78-82%):** DK1, SW1, N1, D1 → only with good odds

**Under 4.5 has no "EVITA" leagues** — even D1 (worst) is at 78.7%. This is the stronger market.

---

## Calibration Gaps — Under 3.5 (RAW probabilities)

| Bucket | n | pred | obs | gap | Assessment |
|--------|---|------|-----|-----|------------|
| 0.50-0.60 | 2,734 | 0.555 | 0.643 | **-8.8pp** | **Under-confident** |
| 0.60-0.65 | 2,159 | 0.626 | 0.672 | **-4.6pp** | Under-confident |
| 0.65-0.70 | 2,693 | 0.676 | 0.669 | +0.7pp | GOOD |
| 0.70-0.75 | 3,110 | 0.725 | 0.715 | +1.1pp | GOOD |
| 0.75-0.80 | 3,301 | 0.775 | 0.743 | **+3.2pp** | Over-confident |
| 0.80-0.85 | 3,017 | 0.823 | 0.768 | **+5.5pp** | **Over-confident** |
| 0.85-0.90 | 2,001 | 0.873 | 0.774 | **+9.9pp** | **VERY over-confident** |
| 0.90-1.00 | 996 | 0.932 | 0.798 | **+13.3pp** | **EXTREMELY over-confident** |

**CRITICAL FINDING:** Under 3.5 raw model says 93.2% but reality is only 79.8% in the 0.90+ band. The Platt calibration fixes this, but it means:
- **Do NOT trust raw p_raw for Under 3.5** — always use p_cal
- Even after calibration, the model's confidence at high bands is less reliable than Under 4.5

---

## Calibration Gaps — Under 4.5 (RAW probabilities)

| Bucket | n | pred | obs | gap | Assessment |
|--------|---|------|-----|-----|------------|
| 0.50-0.60 | 344 | 0.561 | 0.750 | **-18.9pp** | Under-confident |
| 0.60-0.65 | 391 | 0.628 | 0.752 | **-12.4pp** | Under-confident |
| 0.65-0.70 | 795 | 0.678 | 0.772 | **-9.4pp** | Under-confident |
| 0.70-0.75 | 1,424 | 0.727 | 0.813 | **-8.6pp** | Under-confident |
| 0.75-0.80 | 2,247 | 0.777 | 0.817 | **-4.0pp** | Under-confident |
| 0.80-0.85 | 3,454 | 0.827 | 0.843 | -1.6pp | GOOD |
| **0.85-0.90** | **4,791** | **0.876** | **0.870** | **+0.6pp** | **PERFECT** |
| 0.90-1.00 | 7,930 | 0.937 | 0.900 | **+3.7pp** | Slightly over-confident |

**Under 4.5 is well-calibrated in the 0.80-0.90 band** (where most picks live). The 0.90+ band is slightly over-confident (+3.7pp) but much better than Under 3.5.

**Key insight:** Under 4.5 raw model UNDER-predicts at low bands (0.50-0.75) — it says 72.7% but reality is 81.3%. This means the model is conservative at low probabilities = **hidden edge for cheap Under 4.5 picks**.

---

## League Parameters Summary

| League | Avg Goals/Match | Under 3.5 Rate | Under 4.5 Rate | Best Market |
|--------|----------------|----------------|----------------|-------------|
| SP2 | ~1.9 | 77.8% | 90.5% | 🔥 Both |
| I2 | ~2.1 | 76.4% | 90.6% | 🔥 Both |
| RO1 | ~2.2 | 76.2% | 89.1% | 🔥 Both |
| E1 | ~2.3 | 75.7% | 89.8% | 🔥 Both |
| I1 | ~2.3 | 74.8% | 89.2% | Under 4.5 |
| SP1 | ~2.4 | 75.0% | 87.1% | Under 4.5 |
| D1 | ~3.0 | 59.6% | 78.7% | ❌ Avoid U3.5 |
| N1 | ~2.8 | 61.5% | 80.0% | ❌ Avoid U3.5 |

---

## Actionable Rules for Daily Picks

### Under 4.5 (PRIMARY market — better calibrated):

| p_cal | Action | Expected hit rate |
|-------|--------|-------------------|
| **≥ 0.88** | **BET (premium)** | ~87-91% |
| **0.82-0.88** | **BET (good)** | ~82-87% |
| **0.75-0.82** | Only with odds ≥ 1.20 | ~78-82% |
| **< 0.72** | **PASS** | Model unstable |

### Under 3.5 (SECONDARY — more volatile):

| p_cal | Action | Expected hit rate |
|-------|--------|-------------------|
| **≥ 0.80** | BET only in PREMIUM leagues | ~77-80% |
| **0.72-0.80** | Only with strong research | ~70-77% |
| **< 0.72** | **PASS** | |

### League filter (override):

| League tier (U4.5) | Min p_cal | Why |
|---------------------|----------|-----|
| I2, SP2, E1, I1, RO1, P1, SP1 | ≥ 0.82 | Hit > 87% |
| RS1, F1, D2, E0, SA1 | ≥ 0.84 | Hit 82-86% |
| DK1, SW1, N1, D1 | ≥ 0.86 | Hit 78-82%, needs extra bar |

### League filter (override) for U3.5:

| League tier | Min p_cal | Why |
|-------------|----------|-----|
| SP2, I2, RO1 | ≥ 0.76 | Hit > 76% |
| E1, SP1, I1 | ≥ 0.78 | Hit 74-76% |
| P1, RS1, F1 | ≥ 0.80 | Hit 69-71% |
| **D1, N1, E0, SA1** | **≥ 0.85 or AVOID** | Hit < 65%, model unreliable |

---

## Under 3.5 vs Under 4.5 — When to Use Which?

| Situation | Recommendation |
|-----------|---------------|
| lam_total < 2.0 | **Under 3.5** — 4+ goals nearly impossible |
| lam_total 2.0-2.5 | **Both valid** — Under 3.5 preferred if top league |
| lam_total 2.5-3.0 | **Under 4.5 only** — Under 3.5 too risky |
| lam_total > 3.0 | **Under 4.5 cautiously** or SKIP |
| lam_total > 3.5 | **SKIP both** — too volatile |

---

## Key Differences vs Corners Audit

| Dimension | Goals | Corners |
|-----------|-------|---------|
| Calibration quality | Under 4.5 good, Under 3.5 problematic at extremes | **Near-perfect across all bands** |
| Best market | Under 4.5 (85.8% hit) | Under 12.5 (79.5% hit) |
| Weakest leagues | D1, N1 for U3.5 | N1, E0 |
| Over-confidence risk | Under 3.5 at p>0.85 | None (fixed by calibration) |
| Model reliability | Under 4.5 > Under 3.5 | Uniformly reliable |

**Corners model is better calibrated than Goals model.** Goals Under 3.5 has significant over-confidence at high probability bands that Platt calibration partially but not fully corrects.

---

*Audit generated 2026-04-06. Under 4.5 is the primary market. Under 3.5 restricted to top leagues with lam < 2.0. Next audit after 1000+ new predictions.*
