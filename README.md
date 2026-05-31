# WC 2026 AI Predictor 🏆

A professional-grade soccer match prediction and tactical recommendation system for the 2026 FIFA World Cup. Trained on historical data, synced with real-time player form, and augmented with psychological/external factor analysis.

---

## Architecture at a glance

```
Data Sources          Processing Modules        ML Core             Output
─────────────         ──────────────────        ───────             ──────
StatsBomb 360    ──►  A. Match stats engine  ──► XGBoost         ──► Win/Draw/Loss %
football-data    ──►  B. Player form tracker ──► Neural net      ──► Expected score
FBref/Sofascore  ──►                          ──► Tactical class  ──► Formation advice
Press conf.      ──►  C. Psych signal NLP    ──► Ensemble        ──► Player risk flags
News headlines   ──►
```

---

## Quick start (Windows)

```bat
:: 1. Clone / download the project, then:
setup_windows.bat

:: 2. Run Day 1 pipeline (downloads data, computes ELO)
.venv\Scripts\activate
python -m src.pipeline.build_pipeline

:: 3. Open notebooks for exploration
jupyter notebook notebooks/
```

---

## Project structure

```
wc2026_predictor/
│
├── config.py                      ← All constants, paths, API keys
├── requirements.txt
├── setup_windows.bat
│
├── data/
│   ├── raw/                       ← Original downloaded files (never edit)
│   ├── processed/                 ← Cleaned, feature-engineered datasets
│   ├── cache/                     ← Parquet cache (auto-generated)
│   └── wc2026.db                  ← SQLite database (auto-generated)
│
├── src/
│   ├── pipeline/
│   │   ├── build_pipeline.py      ← Master Day 1 runner ✅
│   │   ├── database.py            ← SQLAlchemy schema ✅
│   │   ├── elo.py                 ← ELO rating engine ✅
│   │   ├── statsbomb_loader.py    ← StatsBomb free data ✅
│   │   ├── football_data_loader.py← Historical results ✅
│   │   ├── fbref_scraper.py       ← Player ratings (Day 2)
│   │   └── feature_engineer.py    ← Feature vectors (Day 2)
│   │
│   ├── models/
│   │   ├── xgboost_model.py       ← XGBoost classifier (Day 5)
│   │   ├── neural_net.py          ← PyTorch score predictor (Day 6)
│   │   ├── tactical_model.py      ← Formation classifier (Day 7)
│   │   └── ensemble.py            ← Weighted voting (Day 10)
│   │
│   ├── psych/
│   │   ├── scraper.py             ← News + press conf scraper (Day 8)
│   │   ├── nlp_analyser.py        ← Sentiment + NER (Day 8)
│   │   └── review_ui.py           ← Your manual review tool (Day 9)
│   │
│   ├── tactics/
│   │   ├── formation_analyser.py  ← Formation effectiveness (Day 7)
│   │   └── recommender.py         ← Tactic suggestion engine (Day 11)
│   │
│   └── utils/
│       ├── team_name_map.py       ← Name normalisation across sources
│       └── validators.py          ← Data quality checks
│
├── notebooks/
│   ├── 01_elo_exploration.ipynb   ← ELO visualisation (Day 2)
│   ├── 02_feature_analysis.ipynb  ← Feature importance (Day 5)
│   └── 03_model_eval.ipynb        ← Model calibration (Day 7)
│
├── tests/
│   ├── test_elo.py
│   └── test_loaders.py
│
├── dashboard/
│   └── app.py                     ← Streamlit dashboard (Day 12)
│
└── logs/                          ← Auto-generated
```

---

## 14-Day Build Plan

| Day | Goal | Module |
|-----|------|--------|
| 1–2 | Data pipeline + ELO ratings | `src/pipeline/` ✅ |
| 3   | FBref player form scraper | `fbref_scraper.py` |
| 4   | Feature engineering (120 features) | `feature_engineer.py` |
| 5   | XGBoost model + SHAP explainability | `models/xgboost_model.py` |
| 6   | Neural net for goal prediction | `models/neural_net.py` |
| 7   | Tactical formation classifier | `models/tactical_model.py` |
| 8   | NLP psych signal scraper | `psych/scraper.py` |
| 9   | Manual review UI for psych signals | `psych/review_ui.py` |
| 10  | Ensemble + model calibration | `models/ensemble.py` |
| 11  | Tactic recommendation engine | `tactics/recommender.py` |
| 12–14 | Streamlit dashboard + fine-tuning | `dashboard/app.py` |

---

## Free data sources used

| Source | What we get | URL |
|--------|-------------|-----|
| StatsBomb Open | WC 2022/18/14 full event data, xG, 360° tracking | statsbombpy |
| Martj42 Dataset | 44,000+ international results since 1872 | GitHub |
| football-data.org | Recent fixtures, results, basic stats | Free API key |
| FBref | Player ratings, per-90 stats | Scraped |
| Google News RSS | Team/player news headlines | RSS feed |

---

## Key design decisions

**ELO over FIFA ranking** — FIFA ranking is a political artifact. ELO is a pure performance measure. We tune K-factors for international football: higher for knockout rounds, lower for friendlies.

**SQLite first** — zero config, portable, fast enough for 50k rows. Clean migration to Postgres if needed.

**Semi-automated psych module** — you review scraped signals before they affect the model. This prevents noise from affecting predictions. The model never ingests unreviewed psych data.

**Ensemble approach** — XGBoost handles categorical + tabular features well. Neural net captures non-linear interactions in score prediction. Tactical classifier handles formation matchup logic. Weighted voting beats any single model.

---

## Environment variables (.env)

```
FOOTBALL_DATA_API_KEY=your_key_here   # optional, extends rate limits
```

Get a free key at: https://www.football-data.org/client/register

---

## Day 4 — Advanced Models, Simulation & Dashboard

### Added
1. **Dixon-Coles Poisson model** (`src/models/poisson_model.py`) — predicts exact scorelines (0–0 through 5–5 probability matrix) using fitted attack/defence strength parameters per team
2. **Monte Carlo tournament simulator** (`src/simulation/tournament_simulator.py`) — 100,000 full World Cup bracket simulations; outputs champion/finalist/semi-finalist probability for every team
3. **NLP psych module** (`src/psych/news_scraper.py` + `src/psych/sentiment_analyzer.py`) — automated BBC/ESPN/FIFA RSS scraping with VADER sentence-level sentiment; produces a −1 to +1 psych score per team per match-week
4. **Streamlit dashboard** (`dashboard/app.py`) — interactive dark-themed UI: Match Predictor, Tournament Simulator, and Team Intel tabs
5. **Tactical classifier** (`src/tactics/tactical_classifier.py`) — formation matchup features + counter-tactic recommendations (e.g. 4-3-3 → counter with 5-3-2)
6. **Ensemble predictor** (`src/models/ensemble.py`) — blends XGBoost 40% + Neural Network 30% + Dixon-Coles Poisson 30% into a single calibrated probability
7. **Injury tracker** (`src/pipeline/injury_tracker.py`) — scrapes Transfermarkt for squad availability, computes injury-risk feature for the vector
8. **CLI tool** (`cli.py`) — command-line interface for instant queries

### How to run each new component

```powershell
# Install new dependencies (one-time)
.\.venv\Scripts\Activate.ps1
pip install vaderSentiment streamlit plotly rich feedparser

# Poisson model — train and predict
python -m src.models.poisson_model

# Tournament simulation — 100k bracket simulations
python -m src.simulation.tournament_simulator

# NLP sentiment test
python -m src.psych.sentiment_analyzer

# Streamlit dashboard (opens in browser at http://localhost:8501)
streamlit run dashboard/app.py

# Ensemble prediction
python src/models/ensemble.py --home France --away Brazil --stage semi-final

# Full pipeline (all steps including new ones)
python -m src.pipeline.build_pipeline

# CLI quick commands
python cli.py predict --home Germany --away Spain --stage group
python cli.py simulate --n 10000
python cli.py elo --team Argentina
python cli.py psych --team France --days 7
```

### Updated project structure

```
wc2026_predictor/
├── cli.py                              # NEW — command-line interface
├── config.py                           # updated — groups, ensemble weights, RSS feeds
├── requirements.txt                    # updated — vaderSentiment, streamlit, plotly, rich
├── dashboard/
│   └── app.py                          # NEW — Streamlit interactive UI
└── src/
    ├── models/
    │   ├── xgboost_model.py            # existing
    │   ├── neural_net.py               # existing
    │   ├── poisson_model.py            # NEW — Dixon-Coles exact score model
    │   └── ensemble.py                 # NEW — weighted blend of all 3 models
    ├── pipeline/
    │   ├── build_pipeline.py           # updated — 10-step orchestrator
    │   ├── feature_engineer.py         # existing
    │   ├── injury_tracker.py           # NEW — Transfermarkt squad availability
    │   ├── fbref_scraper.py            # existing
    │   ├── football_data_loader.py     # existing
    │   └── statsbomb_loader.py         # existing
    ├── psych/
    │   ├── news_scraper.py             # NEW — BBC/ESPN/FIFA RSS scraper
    │   └── sentiment_analyzer.py       # NEW — VADER + keyword NLP scorer
    ├── simulation/
    │   └── tournament_simulator.py     # NEW — Monte Carlo WC 2026 bracket
    ├── tactics/
    │   └── tactical_classifier.py      # NEW — formation matchup + counter-tactic
    └── utils/
        └── team_name_map.py            # existing
```

### Ensemble architecture

```
Feature vector (85 cols)
        │
        ├──► XGBoost classifier  ──── Win/Draw/Loss probs  ──► × 0.40 ──┐
        │                                                                  │
        ├──► Neural net          ──── xG home / xG away    ──► × 0.30 ──┼──► Weighted blend
        │                              (→ Poisson → W/D/L)               │    → final probs
        └──► Dixon-Coles Poisson ──── 6×6 score matrix     ──► × 0.30 ──┘
                                       (→ sum → W/D/L)
```

Injury tracker and psych sentiment feed in as **pre-match feature adjustments** before the vector reaches the models — not as post-hoc corrections.
