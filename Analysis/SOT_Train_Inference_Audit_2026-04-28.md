# SOT Audit — Train/Inference Parity Report

**Date:** 2026-04-28
**Sprint:** 2 — Code audit (no data refresh needed)
**Scripts compared:**
- `train_sot_per_team.py` (training pipeline)
- `run_sot_daily.py` (daily inference, modified Sprint 1)

---

## 🚨 BUG #1 (CRITICAL): Order of CLIP and BLEND swapped

### Training (`train_sot_per_team.py` lines 273-288):

```python
# Industry form
lam_h = (hrow["h_for"] + arow["a_against"]) / 2.0
lam_a = (arow["a_for"] + hrow["h_against"]) / 2.0
lam_h = lam_h * tempo_h
lam_a = lam_a * tempo_a

# Elo
if elo_ratings is not None:
    lam_h *= _elo_multiplier(...)
    lam_a *= _elo_multiplier(...)

# >>> CLIP FIRST <<<
lam_h = float(np.clip(lam_h, _SC["lambda_min"], _SC["lambda_max"]))
lam_a = float(np.clip(lam_a, _SC["lambda_min"], _SC["lambda_max"]))

# >>> BLEND AFTER CLIP <<<
lam_h = _SC["blend_empirical"] * lam_h + _SC["blend_league_mean"] * mu_h
lam_a = _SC["blend_empirical"] * lam_a + _SC["blend_league_mean"] * mu_a
```

### Inference (`run_sot_daily.py` lines 334-354):

```python
# Tempo
lam_h = ((hrow["h_for"] + arow["a_against"]) / 2.0) * tempo_h
lam_a = ((arow["a_for"] + hrow["h_against"]) / 2.0) * tempo_a

# Elo
lam_h *= elo_mult_h
lam_a *= elo_mult_a

# Depletion (NEW in inference, not in training!)
lam_h *= dep_mult_h
lam_a *= dep_mult_a

# >>> BLEND FIRST <<<
lam_h = _SC["blend_empirical"] * lam_h + _SC["blend_league_mean"] * mu_h
lam_a = _SC["blend_empirical"] * lam_a + _SC["blend_league_mean"] * mu_a

# >>> CLIP AFTER BLEND <<<
lam_h = float(np.clip(lam_h, _SC["lambda_min"], _SC["lambda_max"]))
lam_a = float(np.clip(lam_a, _SC["lambda_min"], _SC["lambda_max"]))
```

### 🔴 Impact

**Numerical example (lambda_max = 3.5, blend_empirical = 0.7, blend_league_mean = 0.3, mu = 1.5):**

For raw lambda = 4.0 (top team):
- **Training:** `clip(4.0) = 3.5` → `blend = 0.7×3.5 + 0.3×1.5 = 2.90`
- **Inference:** `blend = 0.7×4.0 + 0.3×1.5 = 3.25` → `clip(3.25) = 3.25`
- **Discrepancy: +12% on lambda** (inference higher)

For raw lambda = 0.5 (very weak team):
- **Training:** `clip(0.5) = 1.0` → `blend = 0.7×1.0 + 0.3×1.5 = 1.15`
- **Inference:** `blend = 0.7×0.5 + 0.3×1.5 = 0.80` → `clip(0.80) = 1.0`
- **Discrepancy: -15% on lambda** (training higher)

### 🚨 Critical implication for calibration

Platt calibration was **fit on training p_raw** (CLIP-BEFORE-BLEND lambdas).
Inference produces **p_raw with different distribution** (BLEND-BEFORE-CLIP lambdas).

→ **Calibration parameters trained on one distribution applied to a slightly different one.**

This may explain why `run_sot_daily.py` argparse comment mentions:
> "Apply Platt calibration (default ON; corrects 8-12pp overconfidence)"

The overconfidence might be partly caused by this distributional mismatch + actual model bias.

### 🔧 Fix

Swap order in `run_sot_daily.py` to match training:

```python
# AFTER FIX:
lam_h *= elo_mult_h
lam_a *= elo_mult_a
lam_h *= dep_mult_h
lam_a *= dep_mult_a

# CLIP FIRST (matching training)
lam_h = float(np.clip(lam_h, _SC["lambda_min"], _SC["lambda_max"]))
lam_a = float(np.clip(lam_a, _SC["lambda_min"], _SC["lambda_max"]))

# BLEND AFTER (matching training)
lam_h = _SC["blend_empirical"] * lam_h + _SC["blend_league_mean"] * mu_h
lam_a = _SC["blend_empirical"] * lam_a + _SC["blend_league_mean"] * mu_a
```

### Effort: 5 min code change + ~10 min validation testing

---

## ⚠️ BUG #2 (CONDITIONAL): Depletion multiplier in inference but NOT in training

### Training:
- `_predict_lambdas()` does NOT apply depletion
- No absences accounted for during training

### Inference (`run_sot_daily.py` lines 343-349):
```python
# Fix #5: Squad depletion penalty (from optional absences CSV)
absent_h = absences_map.get((league.lower(), home), 0)
absent_a = absences_map.get((league.lower(), away), 0)
dep_mult_h = _depletion_multiplier(absent_h)
dep_mult_a = _depletion_multiplier(absent_a)
lam_h *= dep_mult_h
lam_a *= dep_mult_a
```

### Impact

If `--absences-csv` is empty (default), `dep_mult = 1.0` → **NO impact**.

If user provides absences CSV, inference applies a multiplier the model **was never trained on**.

### Current status: DORMANT BUG

Daily runs without `--absences-csv` are unaffected. ✓

### 🔧 Fix options

**Option A (proper):** Add depletion to training pipeline + retrain.
- Effort: 2-3h
- Requires: re-running `train_sot_per_team.py` (uses existing data, no NEW data)

**Option B (quick):** Document warning + skip depletion until trained version.
- Add warning to `run_sot_daily.py` when absences-csv provided:
  ```
  if absences_map:
      print("⚠️ WARNING: --absences-csv applied but depletion not in training. Disable until retrain.")
  ```
- Effort: 5 min

### Recommendation: **Option B for now (no impact in current usage)**, schedule Option A for next training cycle.

---

## ✅ CONFIRMATIONS — No bugs

### k_dispersion
- **Training:** loaded from `league_params.csv` via `k_map_h/k_map_a`
- **Inference:** loaded from `league_params.csv` via `lp["k_home"]/lp["k_away"]`
- ✅ Same source, same values

### tempo_home/tempo_away
- **Training:** computed in `_league_params()`, saved to CSV
- **Inference:** loaded from same CSV
- ✅ Consistent

### Elo multiplier
- Both call same `_elo_multiplier()` function
- Both use same `team_ratings.pkl`
- ✅ Identical

### scaling_factor (per league)
- Both use `_scale_for(league)` → `_SCALE_PER_LEAGUE` first, `_SCALE` global fallback
- ✅ Same logic

### blend coefficients
- `_SC["blend_empirical"]` and `_SC["blend_league_mean"]` from same CFG
- ✅ Identical values

### Industry form formula
- Training: `lam_h = (h_for + a_against) / 2.0`
- Inference: `lam_h = (h_for + a_against) / 2.0`
- ✅ Identical

---

## 📋 ADDITIONAL OBSERVATIONS

### Recency weights — design choice (not a bug)

`_recency_weights()` exists ONLY in training pipeline (computes `_team_profiles` with anchor).
Profiles are SAVED to CSV with recency-weighted values "baked in".
Inference loads pre-computed profiles → recency is implicit.

**Implication:** If training is run rarely (e.g., weekly), recent matches don't get full recency weight until next retrain.
**Not a bug**, but design choice. Consider documenting retrain cadence.

### `_norm_team` only in inference

Inference normalizes team names via `_norm_team()` before profile lookup.
Training doesn't (data already normalized).

**Status:** ✓ Consistent if profiles are saved with normalized names.

---

## 🎯 SEVERITY SUMMARY

| Bug | Severity | Impact | Effort | Status |
|---|---|---|---|---|
| **#1** CLIP/BLEND order swapped | **CRITICAL** | 5-15% lambda discrepancy on edge cases | 5 min | ⚠️ FIX NEEDED |
| **#2** Depletion in inference only | LOW (dormant) | 0 if absences-csv empty | 5 min (Option B) | ⚠️ Add warning |

---

## 🔧 RECOMMENDED FIXES (this session)

### Fix #1 (critical, 5 min):
Edit `run_sot_daily.py` lines 351-354 → swap clip/blend order.

### Fix #2 (warning, 2 min):
Add warning when absences-csv non-empty.

### Validation step (15 min):
1. Run script before fix → save output
2. Apply fixes
3. Run script after fix → save output
4. Diff lambda_bk values
5. Confirm: only edge cases (max/min hit) differ
6. For Al Hilal example: expected change ~3-8% on lambda for top teams

### After fix DEPLOYED:
**RETRAIN consideration:**
- Bug #1 affects calibration — calibration was fit on slightly different distribution
- Recommendation: **RETRAIN** to refit Platt calibration on consistent distribution
- Retrain command: `python train_sot_per_team.py` (uses existing historical data, no data refresh needed)
- Effort: 5-15 min runtime

---

## 📊 IMPACT ESTIMATE

### If we fix Bug #1 + retrain:

**Expected improvement:**
- Lambda predictions consistent across train/inference (no distributional drift)
- Calibration parameters fit on actual deployed pipeline output
- More accurate probabilities for top teams (where bug #1 impacted most)
- Brier score should improve marginally (~0.5-2% better)

**Expected NO change:**
- League params (k, mu, tempo) — same computation
- Profile values — same computation
- Most middle-of-pack teams (lambda doesn't hit clip)

### Risk of fix
- **LOW** — change is mathematical reordering, well-tested operations
- Backward compatibility: numbers will shift slightly for edge cases
- Existing predictions remain valid for non-edge cases

---

## 🚀 NEXT STEPS

1. **Apply Fix #1** in `run_sot_daily.py` (5 min)
2. **Apply Fix #2** warning (2 min)
3. **Validate** with diff testing (15 min)
4. **Decide on retrain:** recommended after bug #1 fix
5. **Document changes** in commit message

### Sprint 2 conclusion:
**Audit complete. 1 critical bug + 1 dormant bug found.**

No data refresh needed. Code-only fixes + optional retrain (existing data).
