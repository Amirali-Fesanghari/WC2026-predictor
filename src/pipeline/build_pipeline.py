"""
src/pipeline/build_pipeline.py
MASTER runner for the FULL WC2026 predictor system (Day 4+).

Runs in sequence:
  1. DatabaseManager.create_tables()
  2. FootballDataLoader.load_open_international_results() + ELO build
  3. StatsBombLoader.load_wc_matches()
  4. FBrefScraper.scrape_all_teams()  (skippable via --skip-scrape)
  5. FeatureEngineer.build_training_dataset()
  6. XGBoostMatchPredictor.train()
  7. GoalPredictor (neural net) .train()
  8. DixonColesModel.fit()
  9. TacticalClassifier.train()
 10. TournamentSimulator.run(n_simulations=10_000)
 11. Print champion probabilities table + final summary

Usage:
    python -m src.pipeline.build_pipeline [flags]

CLI flags:
    --skip-scrape       skip FBref scraping, use cached squad data
    --skip-train        skip all model training, load saved models
    --simulate-only     jump straight to tournament simulation
    --predict "France vs Brazil"
                        quick single-match prediction then exit
    --n-sims N          number of Monte Carlo simulations (default 10000)
"""

import sys
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on the path ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from loguru import logger

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/pipeline_{time}.log",
    rotation="50 MB",
    level="DEBUG",
    enqueue=True,
)

# ── Config ────────────────────────────────────────────────────────────────────
from config import DB_PATH, WC_2026_TEAMS, CACHE_DIR, MODELS_DIR

METRICS_PATH = CACHE_DIR / "pipeline_metrics.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Step helpers
# ═════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    """Print a section banner."""
    width = 68
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _ok(msg: str) -> None:
    logger.success(msg)


def _warn(msg: str) -> None:
    logger.warning(msg)


def _save_metrics(metrics: dict) -> None:
    """Persist pipeline metrics to disk for later display."""
    existing = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ═════════════════════════════════════════════════════════════════════════════
#  Step 1: Database initialisation
# ═════════════════════════════════════════════════════════════════════════════

def step1_init_db():
    """Create all DB tables via DatabaseManager / init_db."""
    _banner("STEP 1: Database initialisation")
    try:
        from src.pipeline.database import init_db
        Session = init_db()
        _ok(f"Database ready: {DB_PATH}")
        return Session
    except Exception as exc:
        _warn(f"Database init failed: {exc}. Continuing without persistent DB.")
        logger.debug(f"DB error detail: {exc}", exc_info=True)
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  Step 2: Historical data load + ELO
# ═════════════════════════════════════════════════════════════════════════════

def step2_load_data_and_elo():
    """
    Load open international results, compute ELO ratings for all teams,
    cache the results, seed WC 2026 teams into DB.
    Returns (hist_df, elo_engine).
    """
    _banner("STEP 2: Load historical data + build ELO")
    hist_df = None
    elo_engine = None

    try:
        from src.pipeline.football_data_loader import load_open_international_results
        from src.pipeline.elo import EloEngine

        hist_df = load_open_international_results()
        hist_df = hist_df[hist_df["match_date"] >= "1990-01-01"].copy()
        logger.info(f"Loaded {len(hist_df):,} matches (1990 onwards)")

        elo_engine = EloEngine()
        elo_history = elo_engine.process_dataframe(hist_df)

        # Cache ELO history
        elo_cache = CACHE_DIR / "elo_history.parquet"
        elo_history.to_parquet(elo_cache)
        logger.info(f"ELO history cached: {elo_cache}")

        ratings = elo_engine.get_ratings()
        _ok(f"ELO computed for {len(ratings)} teams")
        logger.info("Top 10 teams by ELO:\n" + ratings.head(10).to_string(index=False))

        _save_metrics({
            "n_historical_matches": int(len(hist_df)),
            "n_elo_teams": int(len(ratings)),
        })

    except Exception as exc:
        _warn(f"Step 2 failed: {exc}. Will attempt to continue.")
        logger.debug(f"Step 2 error: {exc}", exc_info=True)

    return hist_df, elo_engine


# ═════════════════════════════════════════════════════════════════════════════
#  Step 3: StatsBomb match data
# ═════════════════════════════════════════════════════════════════════════════

def step3_load_statsbomb() -> dict:
    """
    Load StatsBomb free WC data for 2014, 2018, 2022.
    Returns dict: edition -> DataFrame.
    """
    _banner("STEP 3: StatsBomb WC match data")
    sb_data = {}
    try:
        from src.pipeline.statsbomb_loader import get_wc_matches
        from config import SB_COMPETITIONS

        for edition in SB_COMPETITIONS:
            try:
                df = get_wc_matches(edition)
                sb_data[edition] = df
                _ok(f"  {edition}: {len(df)} matches loaded")
            except Exception as exc:
                _warn(f"  {edition}: failed ({exc})")

    except Exception as exc:
        _warn(f"Step 3 failed: {exc}")
        logger.debug(f"Step 3 error: {exc}", exc_info=True)

    total = sum(len(v) for v in sb_data.values())
    logger.info(f"StatsBomb: {total} total WC matches across {len(sb_data)} editions")
    return sb_data


# ═════════════════════════════════════════════════════════════════════════════
#  Step 4: FBref scraping
# ═════════════════════════════════════════════════════════════════════════════

def step4_fbref_scrape(skip: bool = False) -> int:
    """
    Scrape current squad data from FBref for all WC 2026 teams.
    Returns number of teams successfully scraped / loaded from cache.
    skip=True: load from cache only, no HTTP requests.
    """
    _banner("STEP 4: FBref squad data" + (" (cache only — --skip-scrape)" if skip else ""))
    n_loaded = 0
    try:
        from src.pipeline.fbref_scraper import (
            scrape_team_squad,
            load_cached_squad,
            compute_squad_quality_score,
        )

        if skip:
            logger.info("Skipping FBref scraping. Loading from cache...")
            for team in WC_2026_TEAMS:
                df = load_cached_squad(team)
                if not df.empty:
                    n_loaded += 1
            _ok(f"Cache: {n_loaded}/{len(WC_2026_TEAMS)} teams available")
        else:
            logger.info(f"Scraping FBref for {len(WC_2026_TEAMS)} teams (4s delay each)...")
            for i, team in enumerate(WC_2026_TEAMS, 1):
                try:
                    df = scrape_team_squad(team)
                    if not df.empty:
                        n_loaded += 1
                        logger.debug(f"  [{i}/{len(WC_2026_TEAMS)}] {team}: {len(df)} players")
                    else:
                        _warn(f"  [{i}/{len(WC_2026_TEAMS)}] {team}: empty result")
                except Exception as exc:
                    _warn(f"  [{i}/{len(WC_2026_TEAMS)}] {team}: {exc}")
            _ok(f"Scraped {n_loaded}/{len(WC_2026_TEAMS)} teams")

    except ImportError as exc:
        _warn(f"FBref scraper not available: {exc}. Skipping.")
    except Exception as exc:
        _warn(f"Step 4 failed: {exc}")
        logger.debug(f"Step 4 error: {exc}", exc_info=True)

    _save_metrics({"fbref_teams_loaded": n_loaded})
    return n_loaded


# ═════════════════════════════════════════════════════════════════════════════
#  Step 5: Feature engineering
# ═════════════════════════════════════════════════════════════════════════════

def step5_build_features() -> pd.DataFrame:
    """
    Build the full ML-ready training dataset using FeatureEngineer.
    Returns the training DataFrame (or empty DF on failure).
    """
    _banner("STEP 5: Feature engineering — build training dataset")
    train_df = pd.DataFrame()
    try:
        from src.pipeline.feature_engineer import FeatureEngineer

        fe = FeatureEngineer()
        fe.load_data()

        train_df = fe.build_training_dataset(
            competition_filter="FIFA World Cup",
            start_year=2006,
        )
        n_matches, n_features = train_df.shape
        _ok(f"Training dataset: {n_matches} matches × {n_features} features")

        _save_metrics({
            "n_training_matches": int(n_matches),
            "n_features": int(n_features),
        })

    except Exception as exc:
        _warn(f"Step 5 failed: {exc}")
        logger.debug(f"Step 5 error: {exc}", exc_info=True)
        # Try loading cached dataset if build failed
        cache_path = CACHE_DIR / "training_features.parquet"
        if cache_path.exists():
            try:
                train_df = pd.read_parquet(cache_path)
                n_matches, n_features = train_df.shape
                _warn(f"Loaded cached training data: {n_matches} × {n_features}")
                _save_metrics({
                    "n_training_matches": int(n_matches),
                    "n_features": int(n_features),
                })
            except Exception as e2:
                _warn(f"Cache load also failed: {e2}")

    return train_df


# ═════════════════════════════════════════════════════════════════════════════
#  Step 6: XGBoost outcome model
# ═════════════════════════════════════════════════════════════════════════════

def step6_train_xgboost(skip: bool = False):
    """
    Train (or load saved) XGBoost match outcome classifier.
    Returns WCOutcomeModel instance or None on failure.
    """
    _banner("STEP 6: XGBoost outcome model" + (" (load saved — --skip-train)" if skip else ""))
    model = None
    try:
        from src.models.xgboost_model import WCOutcomeModel

        if skip:
            model = WCOutcomeModel.load_latest()
            _ok(f"XGBoost model loaded (version {model.version})")
        else:
            model = WCOutcomeModel()
            model.load_training_data()
            metrics = model.train(tune_hyperparams=True)
            model.save()
            _ok(
                f"XGBoost trained: CV accuracy={metrics['cv_accuracy_mean']:.1%} "
                f"(±{metrics['cv_accuracy_std']:.1%}), "
                f"log-loss={metrics['cv_logloss_mean']:.4f}"
            )
            _save_metrics({
                "xgb_cv_accuracy": metrics["cv_accuracy_mean"],
                "xgb_cv_logloss": metrics["cv_logloss_mean"],
                "xgb_brier_home": metrics["brier_home"],
                "xgb_brier_draw": metrics["brier_draw"],
                "xgb_brier_away": metrics["brier_away"],
            })

    except Exception as exc:
        _warn(f"Step 6 (XGBoost) failed: {exc}")
        logger.debug(f"Step 6 error: {exc}", exc_info=True)

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Step 7: Neural net goal predictor
# ═════════════════════════════════════════════════════════════════════════════

def step7_train_neural_net(skip: bool = False):
    """
    Train (or load saved) neural net goal predictor.
    Returns WCGoalModel instance or None on failure.
    """
    _banner("STEP 7: Neural net goal predictor" + (" (load saved)" if skip else ""))
    model = None
    try:
        from src.models.neural_net import WCGoalModel

        if skip:
            model = WCGoalModel.load_latest()
            _ok(f"Neural net loaded (version {model.version})")
        else:
            model = WCGoalModel()
            model.load_training_data()
            metrics = model.train(epochs=200, patience=25)
            model.save()
            _ok(
                f"Neural net trained: val_loss={metrics['best_val_loss']:.4f}, "
                f"MAE={metrics['mae_total']:.3f} goals"
            )
            _save_metrics({
                "nn_best_val_loss": metrics["best_val_loss"],
                "nn_mae_total": metrics["mae_total"],
                "nn_mae_home": metrics["mae_home_goals"],
                "nn_mae_away": metrics["mae_away_goals"],
            })

    except Exception as exc:
        _warn(f"Step 7 (Neural net) failed: {exc}")
        logger.debug(f"Step 7 error: {exc}", exc_info=True)

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Step 8: Dixon-Coles Poisson model
# ═════════════════════════════════════════════════════════════════════════════

def step8_train_poisson(skip: bool = False):
    """
    Fit (or load saved) Dixon-Coles Poisson model on historical data.
    Returns DixonColesModel instance or None on failure.
    """
    _banner("STEP 8: Dixon-Coles Poisson model" + (" (load saved)" if skip else ""))
    model = None
    try:
        from src.models.poisson_model import DixonColesModel
        from src.pipeline.football_data_loader import load_open_international_results
        from src.utils.team_name_map import normalize

        if skip:
            model = DixonColesModel.load_latest()
            _ok(f"Dixon-Coles model loaded (version {model.version})")
        else:
            # Build the training DataFrame for Poisson fit
            hist_df = load_open_international_results()
            hist_df = hist_df[hist_df["match_date"] >= "2010-01-01"].copy()
            # Keep only WC + continental tournament matches for higher signal
            wc_mask = hist_df["competition"].str.contains(
                "FIFA World Cup|UEFA Euro|Copa America|African Cup|Asian Cup",
                case=False, na=False,
            )
            fit_df = hist_df[wc_mask].copy()
            fit_df = fit_df.rename(columns={"match_date": "date"})
            fit_df["home_team"] = fit_df["home_team"].apply(normalize)
            fit_df["away_team"] = fit_df["away_team"].apply(normalize)

            logger.info(f"Fitting Dixon-Coles on {len(fit_df)} tournament matches")
            model = DixonColesModel()
            model.fit(fit_df)
            model.save()

            _ok(
                f"Dixon-Coles fitted: home_adv={model.home_adv:.4f}, "
                f"rho={model.rho:.4f}, {len(model.teams)} teams"
            )
            _save_metrics({
                "dc_home_adv": model.home_adv,
                "dc_rho": model.rho,
                "dc_n_teams": len(model.teams),
            })

    except Exception as exc:
        _warn(f"Step 8 (Dixon-Coles) failed: {exc}")
        logger.debug(f"Step 8 error: {exc}", exc_info=True)

    return model


# ═════════════════════════════════════════════════════════════════════════════
#  Step 9: Tactical classifier
# ═════════════════════════════════════════════════════════════════════════════

def step9_train_tactical(skip: bool = False):
    """
    Train (or initialise with rules-based fallback) TacticalClassifier.
    Returns TacticalClassifier instance or None on failure.
    """
    _banner("STEP 9: Tactical formation classifier" + (" (load cached)" if skip else ""))
    classifier = None
    try:
        from src.tactics.tactical_classifier import TacticalClassifier

        classifier = TacticalClassifier()
        # train() gracefully falls back to rule-based if StatsBomb data is absent
        classifier.train()
        mode = "ML" if classifier._trained_ml else "rule-based"
        _ok(f"TacticalClassifier ready ({mode} mode)")
        _save_metrics({"tactical_mode": mode})

    except Exception as exc:
        _warn(f"Step 9 (Tactical) failed: {exc}")
        logger.debug(f"Step 9 error: {exc}", exc_info=True)

    return classifier


# ═════════════════════════════════════════════════════════════════════════════
#  Step 10: Tournament simulation
# ═════════════════════════════════════════════════════════════════════════════

def step10_simulate(
    xgb_model=None,
    goal_model=None,
    dc_model=None,
    n_simulations: int = 10_000,
):
    """
    Run Monte Carlo tournament simulation with the best available predictor.

    Tries to build an ensemble predictor; falls back to the built-in ELO predictor
    when models are unavailable.

    Returns TournamentSimulator after run().
    """
    _banner(f"STEP 10: Tournament simulation ({n_simulations:,} iterations)")

    # ── Build predictor function ──────────────────────────────────────────────
    predictor = None

    if xgb_model is not None or goal_model is not None:
        try:
            from src.pipeline.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()
            fe.load_data()

            def _ensemble_predictor(home: str, away: str, stage: str):
                """
                Ensemble outcome predictor using loaded models.
                XGBoost drives win/draw/loss; goal model adds score signal.
                """
                try:
                    vec = fe.build_prediction_vector(
                        home_team=home,
                        away_team=away,
                        match_date=datetime(2026, 6, 20),
                        stage=stage,
                    )
                except Exception:
                    vec = None

                p_home, p_draw, p_away = 0.40, 0.25, 0.35  # baseline

                if xgb_model is not None and vec is not None:
                    try:
                        pred = xgb_model.predict(vec)
                        p_home = pred["p_home_win"]
                        p_draw = pred["p_draw"]
                        p_away = pred["p_away_win"]
                    except Exception:
                        pass

                # No draws in knockout
                if stage not in ("group",):
                    p_home += p_draw / 2.0
                    p_away += p_draw / 2.0
                    p_draw = 0.0

                total = p_home + p_draw + p_away
                if total <= 0:
                    total = 1.0
                return p_home / total, p_draw / total, p_away / total

            predictor = _ensemble_predictor
            _ok("Using ensemble predictor (XGBoost + Feature Engineer)")

        except Exception as exc:
            _warn(f"Ensemble predictor build failed ({exc}). Using ELO fallback.")
            predictor = None

    try:
        from src.simulation.tournament_simulator import TournamentSimulator

        # If no ensemble predictor, TournamentSimulator uses built-in ELO predictor
        sim = TournamentSimulator(predictor=predictor, seed=42)
        sim.run(n_simulations=n_simulations)

        _ok(
            f"Simulation complete: {n_simulations:,} iterations in "
            f"{sim._run_time_s:.2f}s"
        )

        # Save top-10 champion probs to metrics
        results = sim.get_results()
        top10 = sorted(
            results.items(),
            key=lambda x: x[1]["champion_prob"],
            reverse=True,
        )[:10]
        _save_metrics({
            "champion_top10": [
                {"team": t, "champion_prob": round(s["champion_prob"], 5)}
                for t, s in top10
            ],
            "n_simulations": n_simulations,
        })

        return sim

    except Exception as exc:
        _warn(f"Step 10 (Simulation) failed: {exc}")
        logger.debug(f"Step 10 error: {exc}", exc_info=True)
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  Summary printer
# ═════════════════════════════════════════════════════════════════════════════

def print_final_summary(sim=None) -> None:
    """Print the final pipeline summary: dataset stats, model accuracy, top-5 picks."""
    metrics = _load_metrics()

    width = 68
    print("\n" + "═" * width)
    print(f"  {'WC 2026 PREDICTOR — PIPELINE COMPLETE':^{width - 4}}")
    print("═" * width)

    # ── Dataset stats ──────────────────────────────────────────────────────
    print("\n  DATASET STATISTICS")
    print("  " + "─" * 40)
    n_matches  = metrics.get("n_training_matches", "N/A")
    n_features = metrics.get("n_features", "N/A")
    n_hist     = metrics.get("n_historical_matches", "N/A")
    n_elo      = metrics.get("n_elo_teams", "N/A")
    fbref_ok   = metrics.get("fbref_teams_loaded", "N/A")

    print(f"  Historical matches loaded : {n_hist:>10}")
    print(f"  Teams with ELO ratings    : {n_elo:>10}")
    print(f"  FBref squads available    : {fbref_ok:>10}")
    print(f"  Training dataset matches  : {n_matches:>10}")
    print(f"  Feature vector width      : {n_features:>10}")

    # ── Model accuracy ─────────────────────────────────────────────────────
    print("\n  MODEL PERFORMANCE")
    print("  " + "─" * 40)

    xgb_acc  = metrics.get("xgb_cv_accuracy")
    xgb_ll   = metrics.get("xgb_cv_logloss")
    nn_mae   = metrics.get("nn_mae_total")
    dc_hadv  = metrics.get("dc_home_adv")
    tac_mode = metrics.get("tactical_mode", "N/A")

    if xgb_acc is not None:
        print(f"  XGBoost CV accuracy       : {xgb_acc:>9.1%}")
    if xgb_ll is not None:
        print(f"  XGBoost CV log-loss       : {xgb_ll:>10.4f}")
    if nn_mae is not None:
        print(f"  Neural net MAE (goals)    : {nn_mae:>10.3f}")
    if dc_hadv is not None:
        print(f"  Dixon-Coles home advantage: {dc_hadv:>10.4f}")
    print(f"  Tactical classifier mode  : {tac_mode:>10}")

    # ── Champion probabilities ─────────────────────────────────────────────
    print("\n  TOP 10 WC 2026 CHAMPION PROBABILITIES")
    print("  " + "─" * 50)
    print(f"  {'Rank':<6}{'Team':<28}{'Champion%':>10}{'Final%':>10}{'SF%':>8}")
    print("  " + "─" * 50)

    champion_data = None

    # Prefer live simulator data
    if sim is not None:
        try:
            results = sim.get_results()
            champion_data = sorted(
                results.items(),
                key=lambda x: x[1]["champion_prob"],
                reverse=True,
            )[:10]
            for rank, (team, stats) in enumerate(champion_data, 1):
                print(
                    f"  {rank:<6}{team:<28}"
                    f"{stats['champion_prob']:>9.1%} "
                    f"{stats['final_prob']:>9.1%} "
                    f"{stats['sf_prob']:>7.1%}"
                )
        except Exception as exc:
            _warn(f"Could not display simulation results: {exc}")
            champion_data = None

    if champion_data is None:
        # Fall back to saved metrics
        saved = metrics.get("champion_top10")
        if saved:
            for rank, item in enumerate(saved[:10], 1):
                print(
                    f"  {rank:<6}{item['team']:<28}"
                    f"{item['champion_prob']:>9.1%}"
                )
        else:
            print("  No simulation results available. Run without --skip-train.")

    # ── Highlight top-5 ───────────────────────────────────────────────────
    if sim is not None or metrics.get("champion_top10"):
        print("\n  TOP 5 CHAMPION PREDICTIONS")
        print("  " + "─" * 50)
        if sim is not None:
            try:
                results = sim.get_results()
                top5 = sorted(
                    results.items(),
                    key=lambda x: x[1]["champion_prob"],
                    reverse=True,
                )[:5]
                medals = ["1st", "2nd", "3rd", "4th", "5th"]
                for medal, (team, stats) in zip(medals, top5):
                    bar_len = int(stats["champion_prob"] * 200)
                    bar = "#" * min(bar_len, 35)
                    print(
                        f"  {medal}  {team:<26} {stats['champion_prob']:>5.1%}  {bar}"
                    )
            except Exception:
                pass

    print("\n" + "═" * width)
    print(f"  Pipeline finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * width + "\n")


# ═════════════════════════════════════════════════════════════════════════════
#  Quick single-match prediction
# ═════════════════════════════════════════════════════════════════════════════

def quick_predict(match_str: str) -> None:
    """
    Parse "France vs Brazil" or "France v Brazil" and print a quick prediction
    using the best available saved model (ensemble → XGBoost → ELO).
    """
    # Parse the match string
    sep = re.split(r"\s+v(?:s)?\.?\s+", match_str, flags=re.IGNORECASE)
    if len(sep) != 2:
        logger.error(
            f"Could not parse match string: '{match_str}'. "
            "Expected format: 'TeamA vs TeamB'"
        )
        return

    home, away = sep[0].strip(), sep[1].strip()
    print(f"\n  Quick prediction: {home} vs {away}")
    print("  " + "─" * 50)

    pred_done = False

    # Try ensemble (XGBoost + FeatureEngineer)
    try:
        from src.models.xgboost_model import WCOutcomeModel
        from src.models.neural_net import WCGoalModel
        from src.pipeline.feature_engineer import FeatureEngineer

        xgb   = WCOutcomeModel.load_latest()
        nn    = WCGoalModel.load_latest()
        fe    = FeatureEngineer()
        fe.load_data()

        vec = fe.build_prediction_vector(
            home_team=home,
            away_team=away,
            match_date=datetime(2026, 6, 20),
            stage="group",
        )

        outcome = xgb.predict(vec, home_team=home, away_team=away)
        goals   = nn.predict(vec)

        print(f"  {home:<22} win probability : {outcome['p_home_win']:>7.1%}")
        print(f"  Draw                       : {outcome['p_draw']:>7.1%}")
        print(f"  {away:<22} win probability : {outcome['p_away_win']:>7.1%}")
        print(f"\n  Predicted outcome  : {outcome['predicted']}  ({outcome['confidence']} confidence)")
        print(f"  Expected score     : {home} {goals['predicted_score']} {away}")
        print(f"  xG home / away     : {goals['home_xg']:.2f} / {goals['away_xg']:.2f}")
        pred_done = True

    except Exception as exc:
        logger.debug(f"Ensemble prediction failed: {exc}. Trying ELO fallback.")

    # Fallback: ELO only
    if not pred_done:
        try:
            from src.pipeline.elo import EloEngine
            from src.pipeline.football_data_loader import load_open_international_results
            from src.utils.team_name_map import normalize

            hist_df = load_open_international_results()
            hist_df = hist_df[hist_df["match_date"] >= "1990-01-01"]
            engine = EloEngine()
            engine.process_dataframe(hist_df)
            pred = engine.predict_match(normalize(home), normalize(away))

            print(f"  (ELO-only fallback)")
            print(f"  {home:<22} win probability : {pred['p_home_win']:>7.1%}")
            print(f"  Draw                       : {pred['p_draw']:>7.1%}")
            print(f"  {away:<22} win probability : {pred['p_away_win']:>7.1%}")
            print(f"  ELO ratings: {pred['home_elo']:.0f} vs {pred['away_elo']:.0f}")

        except Exception as exc2:
            logger.error(f"ELO fallback also failed: {exc2}")

    print()


# ═════════════════════════════════════════════════════════════════════════════
#  Main pipeline orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    skip_scrape: bool = False,
    skip_train: bool = False,
    simulate_only: bool = False,
    n_simulations: int = 10_000,
) -> None:
    """
    Master runner for the complete WC 2026 predictor system.

    Parameters
    ----------
    skip_scrape    : do not HTTP-scrape FBref; use cached data only
    skip_train     : skip all model training; load saved models
    simulate_only  : skip all data/training steps; run simulation only
    n_simulations  : number of Monte Carlo iterations for simulation
    """
    start_time = datetime.now()
    _banner(
        f"WC 2026 PREDICTOR — FULL PIPELINE  "
        f"[{start_time.strftime('%Y-%m-%d %H:%M')}]"
    )
    logger.info(
        f"Flags: skip_scrape={skip_scrape}, skip_train={skip_train}, "
        f"simulate_only={simulate_only}, n_sims={n_simulations:,}"
    )

    xgb_model    = None
    goal_model   = None
    dc_model     = None
    tactical_clf = None
    sim          = None

    if simulate_only:
        logger.info("--simulate-only: jumping to tournament simulation.")
        # Try to load all saved models first for a better predictor
        try:
            from src.models.xgboost_model import WCOutcomeModel
            xgb_model = WCOutcomeModel.load_latest()
        except Exception:
            pass
        try:
            from src.models.neural_net import WCGoalModel
            goal_model = WCGoalModel.load_latest()
        except Exception:
            pass
        sim = step10_simulate(xgb_model, goal_model, dc_model, n_simulations)
        print_final_summary(sim)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────
    step1_init_db()
    step2_load_data_and_elo()
    step3_load_statsbomb()
    step4_fbref_scrape(skip=skip_scrape)
    step5_build_features()
    xgb_model    = step6_train_xgboost(skip=skip_train)
    goal_model   = step7_train_neural_net(skip=skip_train)
    dc_model     = step8_train_poisson(skip=skip_train)
    tactical_clf = step9_train_tactical(skip=skip_train)
    sim          = step10_simulate(xgb_model, goal_model, dc_model, n_simulations)

    elapsed = (datetime.now() - start_time).total_seconds()
    _save_metrics({"total_elapsed_s": round(elapsed, 1)})
    logger.info(f"Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Print tournament results table
    if sim is not None:
        print("\n" + sim.get_results_table(sort_by="champion_prob"))

    print_final_summary(sim)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WC 2026 Predictor — Master Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all steps):
  python -m src.pipeline.build_pipeline

  # Skip FBref web scraping, use cached squad data:
  python -m src.pipeline.build_pipeline --skip-scrape

  # Skip training entirely, use saved models:
  python -m src.pipeline.build_pipeline --skip-train

  # Only run the tournament simulation (fastest):
  python -m src.pipeline.build_pipeline --simulate-only --n-sims 50000

  # Quick single-match prediction:
  python -m src.pipeline.build_pipeline --predict "France vs Brazil"
""",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip FBref HTTP scraping; load squad data from cache only.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip all model training; load the most recently saved models.",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Skip data loading and training; run tournament simulation only.",
    )
    parser.add_argument(
        "--predict",
        metavar="'TEAM_A vs TEAM_B'",
        type=str,
        default=None,
        help='Quick single-match prediction, e.g. --predict "France vs Brazil"',
    )
    parser.add_argument(
        "--n-sims",
        metavar="N",
        type=int,
        default=10_000,
        help="Number of Monte Carlo tournament simulations (default: 10000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.predict:
        # Quick prediction mode — just run the single match and exit
        quick_predict(args.predict)
        sys.exit(0)

    run_full_pipeline(
        skip_scrape=args.skip_scrape,
        skip_train=args.skip_train,
        simulate_only=args.simulate_only,
        n_simulations=args.n_sims,
    )
