# WC 2026 AI Predictor — Technical Summary

---

## What It Does

The WC 2026 AI Predictor is a professional-grade match prediction system for the 2026 FIFA World Cup. It ingests historical match data, live player form, tactical formations, and NLP-derived psychological signals to produce calibrated win/draw/loss probabilities, expected scorelines, and formation recommendations. A Monte Carlo simulator runs 100,000 full bracket simulations to output tournament-progression probabilities for all 48 teams.

---

## Modules Built (Days 1-4)

| Module | File | Description |
|--------|------|-------------|
| Data pipeline orchestrator | `src/pipeline/build_pipeline.py` | 10-step master runner: load, clean, ELO, features, models |
| SQLite database schema | `src/pipeline/database.py` | SQLAlchemy ORM: teams, matches, ELO snapshots, predictions |
| ELO rating engine | `src/pipeline/elo.py` | Dynamic team ratings with K-factors tuned per match type |
| StatsBomb loader | `src/pipeline/statsbomb_loader.py` | WC 2014/18/22 full event data, xG, 360-degree tracking |
| football-data.org loader | `src/pipeline/football_data_loader.py` | 44,000+ historical international results since 1872 |
| FBref scraper | `src/pipeline/fbref_scraper.py` | Per-90 player stats, ratings, squad depth per team |
| Feature engineer | `src/pipeline/feature_engineer.py` | Builds 85-column feature vector per match fixture |
| Injury tracker | `src/pipeline/injury_tracker.py` | Transfermarkt squad availability; injury-risk feature |
| Auto updater | `src/pipeline/auto_updater.py` | Incremental data and ELO refresh without full rebuild |
| XGBoost model | `src/models/xgboost_model.py` | Gradient-boosted classifier for win/draw/loss probabilities |
| Neural network | `src/models/neural_net.py` | PyTorch model predicting xG home and xG away |
| Dixon-Coles Poisson model | `src/models/poisson_model.py` | Exact 6x6 scoreline probability matrix per match |
| Ensemble predictor | `src/models/ensemble.py` | Weighted blend: XGBoost 40% + Neural Net 30% + Dixon-Coles 30% |
| Tournament simulator | `src/simulation/tournament_simulator.py` | 100,000 Monte Carlo full bracket simulations |
| News scraper | `src/psych/news_scraper.py` | BBC/ESPN/FIFA RSS ingestion with entity extraction |
| Sentiment analyzer | `src/psych/sentiment_analyzer.py` | VADER NLP; -1 to +1 psych score per team per match-week |
| Tactical classifier | `src/tactics/tactical_classifier.py` | Formation matchup features and counter-tactic recommendations |
| Backtest engine | `src/evaluation/backtest.py` | Walk-forward accuracy evaluation on historical tournaments |
| CLI interface | `cli.py` | Command-line access to all prediction and simulation functions |
| Streamlit dashboard | `dashboard/app.py` | Interactive dark-themed UI: Match Predictor, Simulator, Team Intel |
| Team name mapper | `src/utils/team_name_map.py` | Cross-source name normalisation (e.g. USA / United States / USMNT) |

---

## Key Technical Decisions

**ELO over FIFA ranking** -- FIFA rankings are a political artifact based on weighted points that reward volume of play over quality. ELO is a pure head-to-head performance measure. K-factors are tuned by match type: higher for knockout rounds, lower for friendlies.

**Dixon-Coles over simple Poisson** -- Standard independent Poisson underestimates low-scoring draws (0-0, 1-0, 0-1). Dixon-Coles adds a correction factor for scores below 2 goals per team, producing better-calibrated scoreline matrices.

**Ensemble approach** -- XGBoost handles tabular/categorical features; the neural net captures non-linear xG interactions; Dixon-Coles anchors the score distribution. Weighted voting (40/30/30) consistently outperforms any single model on held-out World Cup data.

**Semi-automated psych module** -- Scraped sentiment signals are staged for human review before entering the feature vector. The model never ingests unreviewed psych data, preventing RSS noise from corrupting predictions. Psych and injury features are applied as pre-match feature adjustments, not post-hoc overrides.

---

## Data Sources

| Source | Data Provided | Access |
|--------|--------------|--------|
| StatsBomb Open Data | WC 2014/18/22 full event data, xG, 360-degree tracking | `statsbombpy` library |
| football-data.org | 44,000+ international results since 1872; fixtures | Free API key |
| FBref | Per-90 player stats, progressive carries, shot creation | Web scraping |
| BBC / ESPN / FIFA RSS | Team and player news headlines for NLP sentiment scoring | RSS (no key required) |
| Transfermarkt | Current squad availability and injury lists | Web scraping |

---

## How to Run

```powershell
# 1. Install all dependencies (one-time)
.\setup_windows.bat

# 2. Activate environment and build the full pipeline
.\.venv\Scripts\Activate.ps1
python -m src.pipeline.build_pipeline

# 3. Predict a specific match
python cli.py predict --home France --away Brazil --stage semi-final

# 4. Launch the interactive dashboard (opens at http://localhost:8501)
streamlit run dashboard/app.py
```

Other useful commands:
```powershell
python cli.py simulate --n 100000              # Full 100k tournament simulation
python cli.py elo --team Argentina             # Current ELO rating for a team
python cli.py psych --team France --days 7     # Sentiment score, last 7 days
python -m src.models.poisson_model             # Train and test Dixon-Coles model
```

---

## Current Limitations

- **No live in-match data** -- pipeline must be triggered before each match day; no real-time in-game updates.
- **Scraper fragility** -- FBref and Transfermarkt scrapers break when site layouts change; data freshness depends on scrape success.
- **Neural net cold start** -- PyTorch model requires a populated feature table; first run falls back to XGBoost + Dixon-Coles only.
- **English-only NLP** -- VADER is English-language; Spanish, Portuguese, and French press coverage is not processed.
- **No betting market integration** -- market odds are strong predictors but absent here due to API cost.
- **SQLite ceiling** -- adequate for development; concurrent users or real-time writes require migration to PostgreSQL.

## Future Improvements

- Real-time websocket feed during live matches for in-play probability updates.
- Multilingual NLP (Spanish, Portuguese, French) for broader psych signal coverage.
- Betfair or Pinnacle odds as a calibration anchor in the ensemble.
- Auto-approve psych signals above a confidence threshold to reduce manual review burden.
- GPU-accelerated neural net retraining after each match day.
- Docker containerisation for portable, reproducible deployment.

---

## WC 2026 Groups

The 2026 World Cup expands to 48 teams across 12 groups of 4. Top 2 from each group plus the 8 best third-place finishers advance to the Round of 32.

| Group | Teams |
|-------|-------|
| A | Mexico, USA, Canada, New Zealand |
| B | Argentina, Chile, Peru, Australia |
| C | Brazil, Colombia, Paraguay, Japan |
| D | France, Belgium, Morocco, Senegal |
| E | Spain, Portugal, Germany, South Korea |
| F | England, Netherlands, Iran, Saudi Arabia |
| G | Italy, Croatia, Nigeria, Cameroon |
| H | Uruguay, Ecuador, Bolivia, South Africa |
| I | Switzerland, Poland, Serbia, Costa Rica |
| J | Denmark, Austria, Czech Republic, Tunisia |
| K | Turkey, Hungary, Algeria, Ivory Coast |
| L | Qatar, Romania, Ukraine, Honduras |

*Note: Final group draw held in Miami, December 2025.*

---

## Ensemble Architecture

```
Feature vector (85 columns)
        |
        |---> XGBoost classifier  --> Win/Draw/Loss probs  --> x 0.40 --+
        |                                                                 |
        |---> Neural net (PyTorch)--> xG home / xG away   --> x 0.30 --+--> Weighted blend
        |                             (converted via Poisson)            |    --> final probs
        +---> Dixon-Coles Poisson --> 6x6 score matrix    --> x 0.30 --+
                                      (summed to W/D/L)
```

Injury availability and psych sentiment enter as **pre-match feature adjustments** before the vector reaches any model.

---

*Stack: Python 3.11, XGBoost, PyTorch, Streamlit, SQLite, VADER NLP, statsbombpy | Built: Days 1-4*
