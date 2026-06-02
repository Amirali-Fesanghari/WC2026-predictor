"""
auto_updater.py
Weekly auto-update module that keeps WC 2026 predictor data fresh.
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import schedule
import time

from loguru import logger

WC_2026_TEAMS = [
    "Argentina", "France", "Brazil", "England", "Spain", "Portugal",
    "Netherlands", "Germany", "Belgium", "Croatia", "Uruguay", "Denmark",
    "Switzerland", "Morocco", "Senegal", "Japan", "South Korea", "Mexico",
    "USA", "Canada", "Australia", "Serbia", "Poland", "Colombia",
    "Ecuador", "Wales", "Ghana", "Cameroon", "Tunisia", "Saudi Arabia",
    "Iran", "Qatar", "Costa Rica", "Panama", "Honduras", "Jamaica",
    "New Zealand", "Egypt", "Algeria", "Nigeria", "Ivory Coast", "Mali",
    "Venezuela", "Chile", "Paraguay", "Peru", "Bolivia", "Indonesia",
]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
UPDATE_LOG_PATH = CACHE_DIR / "last_update.json"


class AutoUpdater:

    def __init__(self):
        self._errors = []
        self._teams_updated = []
        self._run_start = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual update steps
    # ------------------------------------------------------------------

    def update_fbref_players(self):
        """Scrape FBref for all 48 WC teams using the existing fbref_scraper."""
        logger.info("Step 1/7 — update_fbref_players: scraping FBref for all 48 teams")
        try:
            from pipeline.fbref_scraper import FBrefScraper
            scraper = FBrefScraper()
            for team in WC_2026_TEAMS:
                try:
                    scraper.scrape_team(team)
                    if team not in self._teams_updated:
                        self._teams_updated.append(team)
                    logger.debug("FBref updated: {}", team)
                except Exception as team_err:
                    msg = "FBref scrape failed for {}: {}".format(team, team_err)
                    logger.warning(msg)
                    self._errors.append(msg)
            logger.success("update_fbref_players complete")
        except Exception as err:
            msg = "update_fbref_players step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def update_news_sentiment(self):
        """Scrape news and run sentiment analysis for all 48 teams."""
        logger.info("Step 2/7 — update_news_sentiment: fetching news + sentiment")
        try:
            from pipeline.news_scraper import NewsScraper
            from pipeline.sentiment import SentimentAnalyzer
            scraper = NewsScraper()
            analyzer = SentimentAnalyzer()
            for team in WC_2026_TEAMS:
                try:
                    articles = scraper.fetch(team)
                    analyzer.run(team, articles)
                    logger.debug("Sentiment updated: {}", team)
                except Exception as team_err:
                    msg = "Sentiment update failed for {}: {}".format(team, team_err)
                    logger.warning(msg)
                    self._errors.append(msg)
            logger.success("update_news_sentiment complete")
        except Exception as err:
            msg = "update_news_sentiment step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def update_injuries(self):
        """Check injury tracker for all 48 teams."""
        logger.info("Step 3/7 — update_injuries: checking injury tracker")
        try:
            from pipeline.injury_tracker import InjuryTracker
            tracker = InjuryTracker()
            for team in WC_2026_TEAMS:
                try:
                    tracker.update(team)
                    logger.debug("Injury data updated: {}", team)
                except Exception as team_err:
                    msg = "Injury update failed for {}: {}".format(team, team_err)
                    logger.warning(msg)
                    self._errors.append(msg)
            logger.success("update_injuries complete")
        except Exception as err:
            msg = "update_injuries step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def update_elo(self):
        """Recompute ELO ratings with any new results from football-data.org."""
        logger.info("Step 4/7 — update_elo: recomputing ELO ratings")
        try:
            from pipeline.football_data_loader import FootballDataLoader
            from pipeline.elo import EloRater
            loader = FootballDataLoader()
            new_results = loader.fetch_recent_results()
            rater = EloRater()
            rater.update(new_results)
            logger.success("update_elo complete — {} new results processed", len(new_results))
        except Exception as err:
            msg = "update_elo step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def rebuild_features(self):
        """Regenerate feature vectors with fresh data."""
        logger.info("Step 5/7 — rebuild_features: regenerating feature vectors")
        try:
            from pipeline.feature_engineer import FeatureEngineer
            engineer = FeatureEngineer()
            engineer.build_all()
            logger.success("rebuild_features complete")
        except Exception as err:
            msg = "rebuild_features step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def retrain_models(self):
        """Retrain XGBoost and Poisson models with fresh feature vectors (~30 secs)."""
        logger.info("Step 6/7 — retrain_models: retraining XGBoost + Poisson")
        try:
            from models.xgboost_model import XGBoostPredictor
            from models.poisson_model import PoissonPredictor
            xgb = XGBoostPredictor()
            xgb.train()
            logger.debug("XGBoost retrained")
            poisson = PoissonPredictor()
            poisson.train()
            logger.debug("Poisson retrained")
            logger.success("retrain_models complete")
        except Exception as err:
            msg = "retrain_models step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def run_simulation(self):
        """Run 10k tournament simulation with fresh models."""
        logger.info("Step 7/7 — run_simulation: running 10k tournament simulation")
        try:
            from simulation.tournament_simulator import TournamentSimulator
            simulator = TournamentSimulator(n_simulations=10_000)
            results = simulator.run()
            out_path = CACHE_DIR / "simulation_results.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            logger.success("run_simulation complete — results saved to {}", out_path)
        except Exception as err:
            msg = "run_simulation step failed: {}".format(err)
            logger.error(msg)
            self._errors.append(msg)

    def save_update_log(self):
        """Write a JSON log to data/cache/last_update.json."""
        logger.info("Saving update log to {}", UPDATE_LOG_PATH)
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "teams_updated": self._teams_updated,
            "errors": self._errors,
            "duration_seconds": (
                round((datetime.now(timezone.utc) - self._run_start).total_seconds(), 2)
                if self._run_start else None
            ),
        }
        try:
            with open(UPDATE_LOG_PATH, "w", encoding="utf-8") as fh:
                json.dump(log_data, fh, indent=2)
            logger.success("Update log saved — {} errors encountered", len(self._errors))
        except Exception as err:
            logger.error("Failed to save update log: {}", err)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_weekly_update(self):
        """Orchestrate all update steps in order."""
        self._errors = []
        self._teams_updated = []
        self._run_start = datetime.now(timezone.utc)
        logger.info("=== Weekly auto-update started at {} ===", self._run_start.isoformat())

        self.update_fbref_players()
        self.update_news_sentiment()
        self.update_injuries()
        self.update_elo()
        self.rebuild_features()
        self.retrain_models()
        self.run_simulation()
        self.save_update_log()

        logger.info(
            "=== Weekly auto-update finished — {} errors ===",
            len(self._errors),
        )
        if self._errors:
            for err in self._errors:
                logger.warning("  - {}", err)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def schedule_weekly(self):
        """Use the schedule library to run every Monday at 06:00."""
        logger.info("Scheduling weekly update every Monday at 06:00")
        schedule.every().monday.at("06:00").do(self.run_weekly_update)
        logger.info("Scheduler running — waiting for next Monday 06:00 trigger")
        while True:
            schedule.run_pending()
            time.sleep(30)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_last_update_info(self):
        """Return a dict with timestamp, teams_updated, and errors from the last run."""
        if not UPDATE_LOG_PATH.exists():
            return {
                "timestamp": None,
                "teams_updated": [],
                "errors": [],
            }
        try:
            with open(UPDATE_LOG_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {
                "timestamp": data.get("timestamp"),
                "teams_updated": data.get("teams_updated", []),
                "errors": data.get("errors", []),
            }
        except Exception as err:
            logger.error("Could not read update log: {}", err)
            return {
                "timestamp": None,
                "teams_updated": [],
                "errors": [str(err)],
            }


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WC 2026 Predictor — weekly auto-updater"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Start the scheduler loop (runs every Monday at 06:00)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print info about the last completed update and exit",
    )
    args = parser.parse_args()

    updater = AutoUpdater()

    if args.status:
        info = updater.get_last_update_info()
        print(json.dumps(info, indent=2))
    elif args.schedule:
        updater.schedule_weekly()
    else:
        updater.run_weekly_update()
