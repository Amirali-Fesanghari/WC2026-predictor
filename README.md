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
