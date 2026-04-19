# CLAUDE.md — Project Brain
# Betting Prediction System — 20 Leagues + WTA Tennis
# Last updated: 2026-04-15

---

## WHO IS THE USER

Romanian BI developer transitioning to sport analyst. Builds betting prediction models. Speaks Romanian casually, templates/analysis in English. Values speed, honesty, data-driven decisions. Daily practice of running models + CoVe verification is the training methodology.

---

## PROJECT ARCHITECTURE

### 5 Models (all active)
1. **DC Double Chance** — Dixon-Coles on goals → `run_dc_daily.py`
2. **Goals Totals** — Under 3.5/4.5, Over 2.5, BTTS → `run_goals_totals_daily.py`
3. **FHG** — First-Half Goals → `run_fhg_daily.py` (11 leagues only)
4. **Corners U12.5** — Negative Binomial → `run_corners_daily.py`
5. **WTA Tennis** — Markov + WElo + Monte Carlo → `run_wta_daily.py`

### 20 Football Leagues
E0, E1, D1, D2, SP1, SP2, I1, I2, F1, N1, P1, RO1, RS1, SA1, SW1, DK1, B1, B2, TR1, TR2

### Data Sources
- **Flashscore** = PRIMARY (free, all 20 leagues)
- **API-Football** = SECONDARY (odds only, free plan limited)
- **Tennis Abstract / Sackmann** = WTA historical data (32,252 matches)

### Key Files
- `config.yaml` — all thresholds
- `team_ids.yaml` — team name aliases (20 leagues)
- `data_loader.py` — Flashscore fixture fetching
- `team_registry.py` — name resolution with fuzzy matching

---

## DAILY WORKFLOW (follow this order)

1. **Run models** sequentially when user says "ruleaza X":
   ```
   PYTHONIOENCODING=utf-8 python run_dc_daily.py --api-key dummy --insecure
   PYTHONIOENCODING=utf-8 python run_goals_totals_daily.py --api-key dummy --insecure
   PYTHONIOENCODING=utf-8 python run_corners_daily.py --api-key dummy --insecure
   PYTHONIOENCODING=utf-8 python run_wta_daily.py --insecure
   ```
2. **Show results briefly** — user decides which to analyze
3. **CoVe analysis** using market-specific templates from `Prompts/`
4. **Save MD file immediately** after each analysis to `Analysis/`
5. **MultiMarket CoVe** at the end if multiple picks survive
6. **Ad-hoc UCL/UEL** — user may ask about European matches outside model

### Weekly Retrain
```
PYTHONIOENCODING=utf-8 python run_weekly_retrain.py --api-key KEY --insecure
# Or skip API fetch: --skip-history
# WTA refresh separately: python import_wta_tennis_abstract.py --top-n 200 --sleep 0.3 --insecure
```

---

## PROMPT LIBRARY (in Prompts/ folder)

| File | Market | Version |
|---|---|---|
| `1.0.0CoVe_DC.md` | Double Chance | v1.2 (research cap +3pp) |
| `1.0.2.0.Goals.md` | Under Goals | v2.0 |
| `1.0.3CoVe_Corners.md` | Corners Under | v1.1 (mismatch > 0.6 = HARD PASS) |
| `1.0.4.WTA_over7.5_Under12.5.md` | WTA Set1 dual market | v3.2 |
| `1.0.5MultiMarket.md` | Multi-Market Accumulator | v1.1 (ROI table + lessons) |
| `3.0.European_Cups.md` | UCL/UEL/UECL | v1.0 |

---

## CRITICAL RULES (NEVER BREAK THESE)

### Date Tracking
- **NEVER mix matches from different days** — always verify `match_date` = today before presenting results
- **WTA: check if match has scheduled time** — no time = projected/unconfirmed, exclude from analysis
- European matches: UCL = Tue/Wed, UEL = Thu, UECL = Thu

### Analysis Quality
- **Always start fresh internet research** for every CoVe — never reuse old research
- **Always include internet sources inline** with clickable URLs in MD files
- **Always analyze BOTH Under 12.5 AND Over 7.5** in every WTA Set 1 CoVe
- **Save analysis MD file immediately** after generating — don't wait
- **Flag when model data contradicts external research**
- **Max research adjustment: +/-10pp** from model probability

### Daily Price Filter
- Research probability >= 82% AND odds >= 1.10 → RECOMMEND
- Research probability < 82% → needs positive edge to recommend
- Odds < 1.10 → PASS (no value)

### Scoring System
- 9-10/10 = HIGH confidence
- 7-8/10 = MODERATE confidence
- < 7/10 = PASS — do not recommend

### Thin Days
- Monday/Tuesday often have few fixtures
- If nothing good exists, say so honestly — don't manufacture picks
- Focus on WTA if football is empty

### Commits & Push
- **NEVER commit or push without explicit user approval**
- Scan ALL untracked files before commit — don't miss new files

---

## MODEL AUDIT RESULTS (2026-04-14)

| Model | Hit Rate | Best Predictions | OOS |
|---|---|---|---|
| DC (20 leagues) | 87.8% | 2,723 matches | 92.4% |
| Goals U3.5 | 71.7% | 9,251 matches | 72.7% |
| Goals U4.5 | 86.7% | 8,026 matches | 88.1% |
| Corners U12.5 | 82.3% | 7,784 matches | 82.5% |
| WTA S1 O7.5 | 83.6% | 12,088 matches | 82.0% |
| WTA Winner | LL=0.6157 | 25,856 matches | — |

---

## EUROPEAN CUPS (manual analysis, no model)

For UCL/UEL/UECL matches:
- Use `Prompts/3.0.European_Cups.md` template
- Auto-search first-leg results + stats (corners, cards, shots)
- Evaluate ALL markets: Goals, Corners, DC, Cards, Shots, BTTS
- Aggregate context determines everything — team losing on aggregate plays differently
- Dead ties (0-3, 1-3) = Under markets. Live ties (1-1, 0-1) = Over/BTTS/Cards markets

---

## WHAT NOT TO DO

- Don't add emojis unless user asks
- Don't give time estimates
- Don't add docstrings/comments to code you didn't change
- Don't create files unless necessary
- Don't use Transfermarkt for daily ops — Flashscore is PRIMARY
- Don't change calibration parameters — they're already optimal
- Don't mix projected WTA R2 matches (no scheduled time) with confirmed R1 matches
- Don't recommend picks from yesterday's evaluations on today's analysis
- Don't run `run_weekly_retrain.py` without `PYTHONIOENCODING=utf-8` on Windows

---

## FILE NAMING CONVENTIONS

### Analysis files (in Analysis/)
- `DC_Analysis_YYYY-MM-DD.md`
- `Goals_U45_Analysis_YYYY-MM-DD.md`
- `Corners_U125_Analysis_YYYY-MM-DD.md`
- `WTA_Set1_DualMarket_CoVe3_YYYY-MM-DD.md`
- `MultiMarket_CoVe_YYYY-MM-DD.md`
- `UCL_Full_Analysis_YYYY-MM-DD.md`
- `UEL_QF_Analysis_YYYY-MM-DD.md`

### Keep all analysis files — they serve as audit trail.

---

## GITHUB
- Repo: `github.com/iionut81/pyBetTrainDC` (private)
- Branch: main