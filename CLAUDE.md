# CLAUDE.md — Project Brain
# Betting Prediction System — 20 Leagues + WTA Tennis
# Last updated: 2026-08-17

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
- **Sofascore** = PRIMARY (free, unauthenticated API, all 20 leagues) — daily fixtures, historical results (FT+HT scores), exact goal minutes via `sofascore_loader.py`. Use this first for any "refresh data" / retrain request.
- **Flashscore** = BACKUP (used if Sofascore is unavailable) — `data_loader.py`
- **API-Football** = SECONDARY (odds + fixture statistics incl. corner kicks via `fixtures/statistics`, free plan limited to 100 req/day)
- **Tennis Abstract / Sackmann** = WTA historical data (32,252 matches)

**IMPORTANT — Sofascore client note:** `api.sofascore.com` blocks Python's native TLS stack (`requests`/`urllib`) with HTTP 403 after a handful of calls, but keeps accepting `curl` indefinitely (TLS/HTTP fingerprinting, confirmed 2026-07-31). `sofascore_loader.py` shells out to `curl` for every request — do not "simplify" this back to `requests`, it will silently start failing.

### Key Files
- `config.yaml` — all thresholds
- `team_ids.yaml` — team name aliases (20 leagues)
- `sofascore_loader.py` — Sofascore fixtures/historical/goal-minutes (PRIMARY data loader)
- `import_sofascore_stats.py` — weekly retrain history refresh (75-col match stats, replaces `import_flashscore_stats.py` in `run_weekly_retrain.py`). 5 advanced metrics (xA, xGOT, errors-to-shot, errors-to-goal, xGOT-faced) are NOT exposed by Sofascore's public stats endpoint — always empty going forward, known accepted gap.
- `data_loader.py` — Flashscore fixture fetching (backup)
- `team_registry.py` — name resolution with fuzzy matching
- `selection_engine/` — generic filter+ranking engine (separate from the tuned production models above): given N candidate matches for a market, eliminates weak ones with a reason, ranks survivors, returns TOP N or NO BET. **FROZEN as of 2026-08-17 pending historical odds data** — do not add categories, CoVe, or new heuristics to it; see `feedback_selection_engine_backtest_before_scoring.md` in memory for why. Only market built so far: `TENNIS_SET1_OVER_7_5` (`selection_engine/markets/tennis_set1_over_7_5.py`). Architecture (final pass 2026-08-17, backtested on 17,081 historical matches): DATA VALIDATION → HARD FILTER → VETO → `rank_signal_fn` returns raw `p_cal_adj` (no rescale) → `selection_engine/classification.py` compares it directly against REAL historical percentile breakpoints (P0/P20/P40/P60/P80/P90/P95/P100, computed from actual outcomes, never guessed, from the **POST_HARD_FILTER / PRE_VETO population, n=17,081** — veto is a selection filter and must NOT also define the statistical universe p_cal_adj is measured against, that was a bug in the first percentile pass) → label (TOP_HISTORICAL_QUINTILE/HIGH/MEDIUM/LOW/VERY_LOW) + bet_eligible (only TOP_HISTORICAL_QUINTILE, i.e. >= P80 = 0.9163, is eligible, via configurable `MarketProfile.bet_threshold_percentile`) → RANK (p_cal_adj desc, tie-break: data_quality, stability, match_id) → TOP N/NO BET. Two earlier designs were replaced in sequence: a 0-100 composite score (beaten by p_cal_adj alone in backtest), then a `minimum_score=80`→"p_cal_adj≥91.8%" cutoff that turned out to be a coincidental rescale artifact, then percentiles computed from the wrong (post-veto) population. FORM/MATCHUP/MARKET_COMPATIBILITY/STABILITY are computed and shown as diagnostics only — they do NOT drive ranking or eligibility (surface bonus proven to be pure noise, removed; `confidence.py`/`contradiction.py` deleted as orphaned). VETO is validated (79.0% hit rate when triggered vs 83.5% when not) and stays. Backtest: band hit rates 80.0/80.9/83.3/85.0/85.4% (VERY_LOW→TOP), Top 1%=86.7%, Top 5%=86.0%, Top 10%=86.5% vs baseline 83.5%; daily simulation (517 historical days) — 408 BET days, 755 total picks, 84.5% pick hit rate; walk-forward (expanding by year, no look-ahead) confirms the pattern out-of-sample in 7/8 years, one inversion in 2020 (small sample, plausibly COVID-disrupted) (`backtest_selection_engine_wta.py`) — a real but modest lift, not a revolution. Cannot yet assess profitability — no historical odds data. A match being the best one available today does NOT make it BET_ELIGIBLE — always state both facts, never say a below-threshold match is "bad" (see `feedback_always_explain_engine_output.md`).
- `run_selection_engine_wta.py` — adapter: reads today's `simulations/WTA/evaluations/1.2_WTA_Set1_Over_7_5.csv` (written by `run_wta_daily.py`), filters to today's confirmed-time matches (excludes `T23:59` no-time placeholders), runs them through `selection_engine/`. Triggered by "Ruleaza WTA NOU" (see Daily Workflow below).
- `wta_recent_form.py` — computes each player's recent-form signal (serve-performance variance over her last 12 matches, any surface) from `data/historical/wta_matches_combined.csv`, feeds `selection_engine`'s FORM category. Normalization constants (`RAW_FLOOR`/`RAW_CEILING`) are calibrated off a 946-player percentile sample — revisit if the historical dataset changes materially.

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

### "Ruleaza WTA NOU" — WTA daily + new selection engine (built 2026-08-16)
When user says **"Ruleaza WTA NOU"**, run:
```
PYTHONIOENCODING=utf-8 python run_wta_daily.py --insecure
PYTHONIOENCODING=utf-8 python run_selection_engine_wta.py
```
This is the lightweight daily version — no historical refresh/retrain, just today's fixtures
through both the production model and the new generic filter+ranking engine
(`selection_engine/`, market `TENNIS_SET1_OVER_7_5` only so far). Output is a
TOP N report (or NO BET) separate from the normal WTA model output — treat it
as a second opinion, not a replacement; it hasn't been backtested like
`wta_set1_filters.py` and picks still need CoVe before recommending.

**Full refresh cycle** (do occasionally, e.g. weekly, or when picks look off vs. current form —
not needed daily, historical data doesn't shift much day to day):
```
PYTHONIOENCODING=utf-8 python import_wta_tennis_abstract.py --top-n 200 --sleep 0.3 --insecure
PYTHONIOENCODING=utf-8 python train_wta.py
PYTHONIOENCODING=utf-8 python train_wta_tiebreak_s2.py
PYTHONIOENCODING=utf-8 python run_wta_daily.py --insecure
PYTHONIOENCODING=utf-8 python run_selection_engine_wta.py
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
| `1.0.6CoVe_SOT.md` | Per-team Shots on Target Over | v1.0 (scaling 1.9x, class-gap HARD PASS) |
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
- Don't use Transfermarkt for daily ops — Sofascore is PRIMARY, Flashscore is backup
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

## WTA U12.5 SET 2 — TRIPLE FILTER WORKFLOW (v1.1, 2026-07-02)

**Regula de bază:** Parcurge cei 3 pași în ordine. Orice SKIP = stop, nu continua.

### PASUL 1 — CSV Model + Market Check (automat, din 1.5_WTA_Under12_5.csv)

**Câmpuri noi în CSV (din 2026-07-11):**
- `min_hold` = hold-ul jucătoarei mai slabe (ex: 0.4043 = ține 40% din servicii)
- `premium_elite` = YES dacă min_hold<0.40 + hold_asym>0.20 + tb_p_cal<0.08 → HR 94.5% clay
- `premium_u125` = YES dacă min_hold<0.50 + hold_asym>0.15 + tb_p_cal<0.10 → HR 93.7% clay
- `danger_zone` = YES dacă min_hold între 0.40–0.45 → HR 88.9% (sub standard, max 7/10)

**Interpretare premium (backtestat 2017-2026, 16.4K meciuri):**
- `premium_elite=YES` + `danger_zone=NO` → pick valid, HR 94.5%
- `premium_u125=YES` + `danger_zone=NO` → pick valid, HR 93.7% clay
- `danger_zone=YES` → scor maxim 7/10, indiferent de premium flag
- min_hold≥0.55 (ambii jucători țin bine) → HR 88-90% — NU e premium, e risc mai mare de TB

```
□ tb_p_cal ≤ 0.10            → semnal U12.5 primar (prag operațional recomandat)
□ Elo/Markov gap > 35pp      → SKIP  |  gap = |p_elo - p_markov| × 100
□ p_elo = 0.0                → SKIP (jucătoare fără date Elo în Sackmann)
□ UNSTABLE flag              → max 7/10 scor final
□ danger_zone = YES          → max 7/10 scor final (min_hold 0.40-0.45 = inconsistenta)

□ Robinhood market check (standard, orice candidat U12.5 S2):
    URL: robinhood.com/us/en/prediction-markets/tennis/events/[p1]-vs-[p2]-[mon-dd-yyyy]/
    P(favorita) < 60%         → SKIP (meci echilibrat, S2 poate fi lung)
    P(favorita) 60-74%        → continuă, notează divergența față de p_markov
    P(favorita) ≥ 75%         → class gap confirmat de piață ✅
    Divergență market vs p_markov > 15pp → investigheaza (injury? form recent?)
      → fără explicație clară → SKIP
```

**De ce triple guard Elo/Markov/Market:**
- `p_markov` = simulare din hold rates pe suprafață → direct relevant pentru TB
- `p_elo` = rezultate reale istorice → validare că hold rates sunt realiste
- `market` = crowd wisdom cu form curent + injuries → tiebreaker când p_elo ≠ p_markov
- Divergență market vs p_markov > 15pp = piața știe ceva ce modelul nu vede → investigheaza
- Validat 02.07.2026: Rybakina p_markov=77% ≈ market=78% (p_elo=67% era outlier)

### PASUL 2 — TennisAbstract (suprafața curentă)
```
□ Meciuri pe suprafață ≥ 10 pentru AMBELE jucătoare → continuă | < 10 → PASS

□ Set 2 TB rate pe suprafață (calculat manual din scoruri brute)
    ≥ 33% S2 TB rate         → risc real, -1pp din scor
    < 15% S2 TB rate         → confirmare, +1pp

□ S1 TB → S2 pattern (factori mental + fizic):
    Pentru fiecare meci cu S1 = 7-6(x):
      → Cine a câștigat S1 TB? Ce scor a fost în S2?
    Calcul: câte S2 TB din totalul meciurilor cu S1 TB?
      > 50% S2 TB după S1 TB → scor maxim 6/10
      > 33% S2 TB după S1 TB → -1pp din scor
      ≤ 20% S2 TB după S1 TB → +1pp confirmare
```

**Motivație S1 TB → S2:**
- Factor mental: pierzătoarea TB Set 1 tinde să se prăbușească în Set 2
- Factor fizic: TB adaugă 15-25 min → serviciu mai slab în primele game-uri Set 2
- Backtest validat (23.06.2026): Navarro 1/7=14% ✅, Bondar 0/3=0% ✅, Lys 0/4=0% ✅, Kalinskaya 2/3=67% 🔴

**ATENȚIE TennisAbstract:** Agentul confundă frecvent TB Set 1 cu Set 2.
Întotdeauna verifică MANUAL scorul complet (ex: "7-6(4) 6-3" → S1=TB, S2=NO TB).

### PASUL 3 — Context manual
```
□ Fatigue: days_rest, had_3sets_7d = True
□ Motivație: miza meciului, home advantage, presiune clasament
□ Condiții: temperatură, tip iarbă
□ UNSTABLE flag din model → max 7/10
```

### Scor final U12.5 Set 2
| Condiție | Scor |
|---|---|
| Toți 3 pași OK, S2 TB ≤15%, S1→S2 ≤20% | 9/10 |
| Pași OK, S2 TB 15-25%, S1→S2 20-33% | 8/10 |
| Sample borderline (8-12) SAU S2 TB 25-35% | 7/10 |
| UNSTABLE flag SAU S1→S2 > 33% | max 6/10 |
| Pasul 1 SKIP SAU Pasul 2 PASS | Nu recomandăm |

### Backtest rezultate (iarbă, 321 meciuri, 2017-2026)
- Baseline fără filtru: **86.0% HR**
- Prag ≤ 0.127: 88.6% HR (+2.6pp)
- **Prag ≤ 0.10: 91.2% HR (+5.3pp) ← OPTIM**
- Valoarea reală a filtrelor: prevenție outlier-e (picks contaminate), nu creștere medie HR

---

## FIX-URI COD APLICATE (2026-06-23)

1. `run_wta_daily.py:294` — `timedelta(days=4)` → `timedelta(days=7)` — Wimbledon qualifying găsit
2. `run_wta_daily.py:513` — First-name guard în `resolve_player_id` — false match Laura≠Liudmila Samsonova eliminat

---

## GITHUB
- Repo: `github.com/iionut81/pyBetTrainDC` (private)
- Branch: main