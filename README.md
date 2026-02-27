# Dixon-Coles 1X/X2 Predictor + Backtest

This project provides a Python script that:

- Collects football match data from:
  - `https://footystats.org/`
  - `https://scores24.live/en/soccer`
  - `https://www.statschecker.com/`
- Fits a Dixon-Coles goal model.
- Produces daily top `5` predictions for double-chance markets (`1X` / `X2`).
- Filters picks around odds `1.20` (default range `1.15` to `1.25`).
- Runs a `30`-day backtest.

Your date `january 20206` is interpreted as **January 2026**.
Default backtest window is **2026-01-01 to 2026-01-30**.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run (default requested setup)

```powershell
python dc_double_chance.py --start-date 2026-01-01 --days 30 --top-n 5 --odds-target 1.20
```

Output:
- `backtest_predictions.csv`

## Notes

- Source websites may change HTML structure or use anti-bot protections.
- The script includes robust parser scaffolding but may require selector updates.
- If no rows are collected, update the scraper selectors in:
  - `FootyStatsScraper`
  - `Scores24Scraper`
  - `StatsCheckerScraper`

## football-data.co.uk Historical Import

Use this helper to build a Dixon-Coles training dataset for:
`E0, E1, SP1, D1, I1, F1, N1, P1` over the latest `5` seasons.

Direct download mode:

```powershell
python import_football_data_co_uk.py --n-seasons 5 --end-season-code 2526 --output-csv historical_matches_football_data.csv
```

Manual files mode (if football-data.co.uk is blocked on your network):

1. Download files manually and save as `2526_E0.csv`, `2526_E1.csv`, ..., `2122_P1.csv`.
2. Run:

```powershell
python import_football_data_co_uk.py --input-dir .\football_data_raw --n-seasons 5 --end-season-code 2526 --output-csv historical_matches_football_data.csv
```

## Transfermarkt Historical Import

If football-data.co.uk is blocked, scrape Transfermarkt fixtures/results for:
`E0, E1, SP1, D1, I1, F1, N1, P1` over `2021/22` to `2025/26`.

```powershell
python import_transfermarkt.py --start-season-year 2021 --n-seasons 5 --output-csv historical_matches_transfermarkt.csv
```

Notes:
- Uses a conservative request delay (`--sleep-seconds`, default `1.2`).
- TLS verification is disabled by default to work in corporate proxy environments.

Daily update command (current season only, merged into master history):

```powershell
python import_transfermarkt.py --start-season-year 2025 --n-seasons 1 --output-csv simulations\transfermarkt_daily_2025_26_snapshot.csv --merge-into historical_matches_transfermarkt.csv --sleep-seconds 0.3
```

Create strict 3+1+1 season splits (Train/Validation/Backtest):

```powershell
python prepare_season_splits.py --input-csv historical_matches_transfermarkt.csv --out-dir simulations\splits
```

Run backtest with time-decay weighting enabled:

```powershell
python dc_double_chance.py --history-store historical_matches_transfermarkt.csv --start-date 2026-01-27 --days 30 --decay-xi 0.0015
```

## Automated Daily Prediction Pipeline

New modular files:

- `data_loader.py`
- `dixon_coles.py`
- `simulation.py`
- `decision_engine.py`
- `main.py`

API mode:

```powershell
python main.py --ratings-pkl team_ratings.pkl --fixtures-api-url "https://YOUR_API_ENDPOINT" --api-key "YOUR_API_KEY" --output-csv simulations\today_recommendations.csv --all-matches-csv simulations\today_all_evaluations.csv
```

Flashscore fallback mode (no API token):

```powershell
python main.py --provider flashscore --target-date 2026-02-28 --ratings-pkl team_ratings.pkl --output-csv simulations\today_recommendations.csv --all-matches-csv simulations\today_all_evaluations.csv --insecure
```

Note:
- Flashscore fallback provides fixtures but not guaranteed 1X/X2 odds fields.
- If odds are missing, matches will still be evaluated, but strict recommendation filter may return zero picks.

Local JSON mode (for testing):

```powershell
python main.py --ratings-pkl simulations\sample_team_ratings.pkl --fixtures-json simulations\sample_fixtures.json --output-csv simulations\today_recommendations.csv --all-matches-csv simulations\today_all_evaluations.csv
```

## Simulation Results Folder

Backtest outputs are stored in `simulations/` by default when `--output-csv`
is passed as a filename (or left at default).

Each backtest run also appends one row to:

- `simulations/simulation_log.csv`

This log keeps run parameters and key metrics (`predictions`, `wins`, `hit_rate`, etc.)
so results are tracked over time.
