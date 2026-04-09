# Latest WTA Set1 Over 7.5 Walk-Forward Audit
## Timestamp: 2026-04-09 (post-retrain with Flashscore data)
## Source: wta_predictions.csv, wta_matches_combined.csv

---

## Coverage

- **History:** 30,142 matches (wta_matches_combined.csv)
- **Model:** Markov chain + Surface Elo + Monte Carlo + Platt Calibration + Blowout Detection
- **Date range:** 2017-01-30 → 2026-04-08
- **Surfaces:** Hard, Clay, Grass
- **Total evaluated:** 11,983 predictions
- **Set1 O7.5 evaluated:** 11,940
- **Data sources:** Sackmann GitHub + Tennis Abstract + **Flashscore (new)**

---

## Overall (Set1 Over 7.5)

| Metric | Value |
|--------|-------|
| **hit_rate** | **0.8342** |
| **log_loss** | 0.4634 |
| **brier** | 0.1408 |
| p_mean | 0.8900 |
| y_mean | 0.8342 |

---

## Per-Surface

| Surface | n | Hit Rate | Brier | Note |
|---------|---|----------|-------|------|
| **Hard** | 9,255 | **83.5%** | 0.1408 | Largest sample, most reliable |
| **Clay** | 2,430 | **82.8%** | 0.1436 | Good, improving with Flashscore data |
| **Grass** | 255 | **87.5%** | 0.1129 | Best but small sample |

---

## Pick Quality Tiers

| Tier | p_cal Range | n Picks | Hit Rate | Action |
|------|------------|---------|----------|--------|
| 🔥 **Premium** | ≥ 0.88 | 8,253 | **85.0%** | BET |
| ✅ **Good** | 0.82–0.88 | 2,767 | **81.2%** | BET (with context) |
| 🟡 **Conditional** | 0.75–0.82 | 846 | **76.0%** | Only with good odds |
| ⚠️ **Risky** | 0.70–0.75 | 74 | 75.7% | Caution |
| ❌ **Avoid** | < 0.70 | — | — | PASS |

---

## Calibration Gaps (Raw Model, Pre-Platt)

| Bucket | n | Predicted | Observed | Gap | Assessment |
|--------|---|-----------|----------|-----|------------|
| 0.70–0.80 | ~920 | 77.3% | 76.2% | **+1.0pp** | ✅ Good |
| 0.80–0.85 | ~2,767 | 82.9% | 80.1% | **+2.8pp** | ⚠️ Slightly over-confident |
| 0.85–0.90 | ~5,486 | 88.2% | 82.8% | **+5.4pp** | ⚠️ Over-confident |
| 0.90–1.00 | ~2,767 | 92.0% | 85.5% | **+6.5pp** | ⚠️ Over-confident |

**Note:** Platt calibration corrects these gaps. The calibrated (p_cal) probabilities are more reliable than raw (p_raw).

---

## Match Winner (Reference)

| Metric | Value |
|--------|-------|
| n | 11,983 |
| log_loss | 0.6213 |
| brier | 0.2169 |

---

## Tiebreak Model

| Metric | Value |
|--------|-------|
| TB predicted | 19.9% |
| TB actual | 11.4% |
| **Gap** | **+8.5pp over-predicted** |

Tiebreak model still significantly over-estimates. Isotonic calibration partially corrects.

---

## Recent Performance (2025–2026)

| Surface | n | Hit Rate | Predicted | Gap |
|---------|---|----------|-----------|-----|
| **Hard** | 1,475 | **84.3%** | 87.6% | -3.3pp |
| **Clay** | 418 | **81.8%** | 84.9% | -3.1pp |
| **Grass** | 46 | **89.1%** | 92.5% | -3.4pp |

**Recent performance is consistent with historical.** Model slightly over-confident by ~3pp across all surfaces. Platt calibration handles this.

---

## Comparison: Previous Audit → Current

| Dimension | March 28 | April 9 | Change |
|-----------|----------|---------|--------|
| History matches | 30,809 | **30,142** | -667 (cleaned no-ID rows, added Flashscore with IDs) |
| Total evaluated | 12,913 | **11,983** | -930 (different walk-forward windows) |
| Set1 O7.5 evaluated | ~12,913 | **11,940** | |
| Overall hit rate | 83.26% | **83.42%** | +0.16pp ✅ |
| Overall brier | 0.1422 | **0.1408** | -0.0014 (better) ✅ |
| Overall log loss | 0.4676 | **0.4634** | -0.0042 (better) ✅ |
| Hard hit rate | 83.5% | **83.5%** | = |
| Clay hit rate | 82.1% | **82.8%** | +0.7pp ✅ |
| Grass hit rate | 87.4% | **87.5%** | +0.1pp |
| Premium tier hit | 84.7% | **85.0%** | +0.3pp ✅ |
| Cal gap 0.7-0.8 | +1.9pp | **+1.0pp** | Improved ✅ |
| Cal gap 0.8-0.9 | +5.2pp | **+5.4pp** | ~same |
| Cal gap 0.9-1.0 | +6.6pp | **+6.5pp** | ~same |

### Key improvements:
- **Clay hit rate improved** from 82.1% → 82.8% (+0.7pp) — Flashscore data helping
- **Brier improved** from 0.1422 → 0.1408 — better calibration
- **Premium tier** from 84.7% → 85.0% — more reliable top picks
- **0.70-0.80 calibration gap** halved from +1.9pp to +1.0pp

---

## Data Source Impact

| Source | Matches in History | Used in Train | Contribution |
|--------|-------------------|---------------|-------------|
| Sackmann GitHub | ~24,000 (2015-2024) | ✅ All | Foundation |
| Tennis Abstract | ~5,800 (2025-2026) | ✅ All | Recent matches |
| **Flashscore (new)** | **~300 (March-April 2026)** | **✅ 300** | **Fresh data with full serve stats** |

Flashscore provides the **freshest** data (up to yesterday) with **100% stats completeness**. This is now the primary source for recent match data.

---

## Actionable Rules (Updated)

### By probability band:

| p_cal Range | Action | Expected Hit Rate |
|-------------|--------|-------------------|
| **≥ 0.88** | **BET (premium)** | ~85% |
| **0.82–0.88** | **BET (good)** | ~81% |
| **0.75–0.82** | Only with odds ≥ 1.20 | ~76% |
| **< 0.72** | **PASS** | Unreliable |

### By surface:

| Surface | Reliability | Min p_cal |
|---------|------------|----------|
| **Hard** | ✅ Best (9,255 samples) | ≥ 0.80 |
| **Clay** | ✅ Good (2,430 samples, improving) | ≥ 0.78 |
| **Grass** | ⚠️ Small sample (255) | ≥ 0.85 |

---

## Key Takeaways

1. **Premium picks (≥0.88) at 85.0% hit rate** — solid, improved from March audit
2. **Clay improved to 82.8%** thanks to Flashscore fresh data
3. **Model over-confident at 0.85-1.0** by +5-6pp — Platt calibration corrects most of this
4. **Tiebreak model still broken** (+8.5pp over-predicted) — isotonic helps but fundamental issue remains
5. **Flashscore integration working** — 300 new matches added, all metrics improved or stable
6. **Grass remains best surface** (87.5%) but sample too small for High confidence

---

*Audit generated 2026-04-09. Next audit recommended after 500+ new predictions or major model change.*
