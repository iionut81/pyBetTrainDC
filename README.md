# Pred — Sports Prediction Pipeline

5 models, 16 leagues, daily predictions + weekly retraining.

---

## Pipeline Entry Points

| Script | When | What it does |
|--------|------|-------------|
| `run_daily_all.py` | Daily | Runs all daily predictions (DC, FHG, Goals, Corners) |
| `run_weekly_retrain.py` | Weekly | Full retrain: data refresh + all models + backtests |

---

## Scripts by Category

### Daily Runners (`run_*_daily.py`)

| Script | Model | Output |
|--------|-------|--------|
| `run_dc_daily.py` | Dixon-Coles Double Chance (1X / X2) | `simulations/DC/` |
| `run_fhg_daily.py` | First-Half Goals | `simulations/FHG/` |
| `run_goals_totals_daily.py` | Goals O/U 2.5, 3.5, 4.5, BTTS | `simulations/Goals/` |
| `run_corners_daily.py` | Corners Under 12.5 | `simulations/Corners U12.5/` |
| `run_wta_daily.py` | WTA Tennis (Winner, Set1, TB) | `simulations/WTA/` |

### Data Fetching (`import_*`)

| Script | Source | What it fetches |
|--------|--------|----------------|
| `import_transfermarkt.py` | Transfermarkt | Match results for DC + Goals (16 leagues) |
| `import_flashscore_stats.py` | Flashscore | xG, shots, corners, possession, cards |
| `import_flashscore_corners.py` | Flashscore | Corner kick data (alternative source) |
| `import_flashscore_fhg_current.py` | Flashscore | Current season FHG stats |
| `import_api_football_history.py` | API-Football | Historical fixtures + stats |
| `import_football_data_co_uk.py` | football-data.co.uk | Historical match data (CSV) |
| `import_wta_history.py` | Sackmann GitHub | WTA match history (2015-2024) |
| `import_wta_tennis_abstract.py` | Tennis Abstract | Recent WTA matches (2025-2026) |

### History Builders (`build_*`)

| Script | What it builds |
|--------|---------------|
| `build_fhg_history.py` | FHG history from API-Football fixtures |
| `build_corners_history.py` | Corners history from API-Football statistics |
| `build_history_fdco.py` | General history from football-data.co.uk |
| `build_corners_history_fdco.py` | Corners history from football-data.co.uk |

### Training (`train_*`)

| Script | Model | Output |
|--------|-------|--------|
| `train_team_ratings.py` | Dixon-Coles team ratings | `team_ratings.pkl` |
| `train_dc_calibration.py` | DC Platt calibration | `dc_calibration.csv` |
| `train_fhg_calibration.py` | FHG walk-forward calibration | `fhg_calibration.csv` |
| `train_fhg_league_bias.py` | FHG league bias factors | `fhg_league_bias.csv` |
| `train_goals_totals.py` | Goals totals calibration | `goals_calibration.csv` |
| `train_corners_under_12_5.py` | Corners Poisson/NB model | team profiles + calibration |
| `train_wta.py` | WTA Elo + Markov + TB | Elo snapshot + calibration |

### Backtesting (`backtest_*`)

| Script | What it tests |
|--------|---------------|
| `backtest_dc.py` | DC Double Chance on historical matches |
| `backtest_under_4_5.py` | Goals Under 4.5 on historical matches |

### Shared Libraries

| Script | Role |
|--------|------|
| `data_loader.py` | Fixture fetching (Flashscore + API-Football), team name resolution |
| `config.py` | Global configuration |
| `team_registry.py` | Team name alias management (uses `team_ids.yaml`) |
| `notify.py` | Success/failure notifications |
| `decision_engine.py` | Recommendation logic |
| `simulation.py` | Monte Carlo match simulation |

### Football Model Core

| Script | Role |
|--------|------|
| `dixon_coles.py` | Dixon-Coles model implementation |
| `dc_double_chance.py` | Double Chance probability calculator |
| `fhg_weibull.py` | FHG Weibull distribution |
| `fhg_calibration.py` | FHG calibration utilities |

### WTA Model Core

| Script | Role |
|--------|------|
| `wta_api.py` | WTA API client |
| `wta_elo.py` | Surface-specific Elo ratings |
| `wta_markov.py` | Markov chain set simulation |
| `wta_ratings.py` | Player rating aggregation |
| `wta_scoring.py` | Score probability calculations |
| `wta_tiebreak.py` | Tiebreak prediction model |
| `wta_set1_filters.py` | Set 1 market filters + blowout detection |

### Utilities / One-off

| Script | Purpose |
|--------|---------|
| `main.py` | Legacy entry point (DC predictions) |
| `audit_fhg.py` | FHG model audit |
| `run_dc_audit.py` | DC model audit |
| `run_goals_audit.py` | Goals model audit |
| `analyze_wta_ablation.py` | WTA feature importance analysis |
| `tune_validation.py` | Hyperparameter tuning |
| `prepare_season_splits.py` | Season data splitting |
| `rebuild_fhg_ratios.py` | Rebuild FHG ratio tables |

---

## Folder Structure

```
Pred/
  Analysis/          CoVe analysis reports (daily)
  Prompts/           CoVe prompt templates (CoVe_WTA.md, etc.)
  data/              Historical data (CSVs, pickles)
    historical/      Combined match history files
  simulations/       Model outputs (evaluations + recommendations)
    DC/              Double Chance
    FHG/             First-Half Goals
    Goals/           Goals Totals
    Corners U12.5/   Corners Under 12.5
    WTA/             WTA Tennis
  automation/        Scheduled tasks / cron
  tests/             Unit tests
```

---

## Quick Commands

```bash
# Daily predictions (all football)
python run_daily_all.py

# Daily WTA predictions
python run_wta_daily.py --insecure

# Weekly full retrain
python run_weekly_retrain.py

# Single model daily run
python run_goals_totals_daily.py --api-key $API_FOOTBALL_TOKEN --insecure
python run_corners_daily.py --api-key $API_FOOTBALL_TOKEN --insecure

# WTA data refresh + retrain
python import_wta_history.py
python import_wta_tennis_abstract.py --insecure
python train_wta.py
```

Note: `--insecure` is needed on corporate networks (LucaNet SSL proxy).
