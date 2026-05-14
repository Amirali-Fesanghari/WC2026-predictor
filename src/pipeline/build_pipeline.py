"""
src/pipeline/build_pipeline.py
Master runner for Day 1–2 data pipeline.
Runs: data download → ELO computation → DB seed → validation report.

Usage:
    python -m src.pipeline.build_pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
from datetime import datetime
from loguru import logger

from config import DB_PATH, WC_2026_TEAMS, CACHE_DIR
from src.pipeline.database import init_db, Team, EloHistory, Match
from src.pipeline.elo import EloEngine
from src.pipeline.football_data_loader import (
    load_open_international_results,
    compute_form_features,
)

# Setup logging
Path("logs").mkdir(exist_ok=True)
logger.add("logs/pipeline_{time}.log", rotation="50 MB", level="DEBUG")


def step1_init_db() -> object:
    """Create all DB tables."""
    logger.info("STEP 1: Initialising database...")
    Session = init_db()
    logger.success(f"DB ready at {DB_PATH}")
    return Session


def step2_load_historical_data() -> pd.DataFrame:
    """Download open international results dataset."""
    logger.info("STEP 2: Loading historical international results...")
    df = load_open_international_results()

    # Filter to 1990+ for ELO (older data is noisier, different football era)
    df = df[df["match_date"] >= "1990-01-01"].copy()
    logger.success(f"Loaded {len(df):,} matches (1990–present)")
    return df


def step3_compute_elo(df: pd.DataFrame) -> EloEngine:
    """Run full ELO computation on historical data."""
    logger.info("STEP 3: Computing ELO ratings for all teams...")
    engine = EloEngine()
    history_df = engine.process_dataframe(df)

    # Save ELO history to cache
    history_df.to_parquet(CACHE_DIR / "elo_history.parquet")

    ratings = engine.get_ratings()
    logger.success(f"ELO computed for {len(ratings)} national teams")
    logger.info("Top 20 teams by ELO:")
    print(ratings.head(20).to_string(index=False))
    return engine


def step4_seed_teams(Session, engine: EloEngine):
    """Insert / update all WC 2026 teams with their current ELO."""
    logger.info("STEP 4: Seeding teams into database...")
    sess = Session()
    ratings = engine.get_ratings().set_index("team")["elo"].to_dict()

    count = 0
    for team_name in WC_2026_TEAMS:
        existing = sess.query(Team).filter_by(name=team_name).first()
        elo = float(ratings.get(team_name, 1500.0))

        if existing:
            existing.elo_current = elo
            existing.elo_peak = max(existing.elo_peak, elo)
            existing.updated_at = datetime.utcnow()
        else:
            sess.add(Team(
                name=team_name,
                elo_current=elo,
                elo_peak=elo,
            ))
        count += 1

    sess.commit()
    sess.close()
    logger.success(f"Seeded/updated {count} teams in DB")


def step5_seed_matches(Session, df: pd.DataFrame, limit: int = None):
    """
    Seed historical WC matches into the matches table.
    For now: only actual World Cup tournament matches (not qualifiers).
    limit: set a number for testing (e.g. limit=64 for WC 2022 only)
    """
    logger.info("STEP 5: Seeding WC matches into database...")
    sess = Session()

    wc_mask = df["competition"].str.contains("FIFA World Cup", case=False, na=False)
    wc_df = df[wc_mask & ~df["competition"].str.contains("qualification|qualifier", case=False, na=False)]

    if limit:
        wc_df = wc_df.tail(limit)

    logger.info(f"Inserting {len(wc_df)} WC matches...")
    count = 0
    for _, row in wc_df.iterrows():
        ext_id = f"fd_{row['home_team']}_{row['away_team']}_{row['match_date'].date()}"
        if sess.query(Match).filter_by(external_id=ext_id).first():
            continue

        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        outcome = "home" if hg > ag else ("draw" if hg == ag else "away")

        # Lookup teams
        home = sess.query(Team).filter_by(name=row["home_team"]).first()
        away = sess.query(Team).filter_by(name=row["away_team"]).first()

        sess.add(Match(
            external_id    = ext_id,
            source         = "football_data_open",
            competition    = row["competition"],
            stage          = row.get("stage", "group"),
            match_date     = row["match_date"],
            neutral_ground = bool(row.get("neutral_ground", True)),
            home_team_id   = home.id if home else None,
            away_team_id   = away.id if away else None,
            home_score     = hg,
            away_score     = ag,
            outcome        = outcome,
        ))
        count += 1

        if count % 100 == 0:
            sess.commit()
            logger.debug(f"  ...{count} matches inserted")

    sess.commit()
    sess.close()
    logger.success(f"Seeded {count} WC matches into DB")


def step6_validation_report(Session, engine: EloEngine):
    """Print a validation report to confirm everything looks right."""
    logger.info("STEP 6: Validation report")
    sess = Session()

    n_teams   = sess.query(Team).count()
    n_matches = sess.query(Match).count()
    sess.close()

    ratings_df = engine.get_ratings()

    print("\n" + "═" * 60)
    print("  WC 2026 PREDICTOR — DAY 1 VALIDATION REPORT")
    print("═" * 60)
    print(f"  Teams in DB:          {n_teams}")
    print(f"  Matches in DB:        {n_matches}")
    print(f"  Teams with ELO:       {len(ratings_df)}")
    print()
    print("  Top 10 teams by ELO (current ratings):")
    print(ratings_df.head(10).to_string(index=False))
    print()

    # Sample prediction
    print("  Sample predictions (ELO-only, no ML yet):")
    test_matchups = [
        ("France", "Morocco"),
        ("Brazil", "Argentina"),
        ("England", "Germany"),
        ("Spain", "Portugal"),
    ]
    for h, a in test_matchups:
        try:
            pred = engine.predict_match(h, a)
            print(f"  {h} vs {a}: "
                  f"H={pred['p_home_win']:.1%} "
                  f"D={pred['p_draw']:.1%} "
                  f"A={pred['p_away_win']:.1%} "
                  f"(ELO: {pred['home_elo']:.0f} vs {pred['away_elo']:.0f})")
        except Exception as e:
            print(f"  {h} vs {a}: skipped ({e})")

    print()
    print("  Form check — Argentina (last 5):")
    form = compute_form_features("Argentina", datetime.now(), n=5)
    for k, v in form.items():
        print(f"    {k}: {v}")

    print()
    print("═" * 60)
    print("  Day 1 complete. Proceed to Day 2: feature engineering.")
    print("═" * 60 + "\n")


def run_pipeline(limit_matches: int = None):
    """Run the full Day 1 pipeline."""
    logger.info("Starting WC 2026 Predictor — Day 1 Pipeline")

    Session = step1_init_db()
    df = step2_load_historical_data()
    engine = step3_compute_elo(df)
    step4_seed_teams(Session, engine)
    step5_seed_matches(Session, df, limit=limit_matches)
    step6_validation_report(Session, engine)

    return engine   # return for interactive use


if __name__ == "__main__":
    # Run with limit_matches=None to seed all WC matches
    # During testing, use limit_matches=128 (WC 2022 + 2018)
    elo_engine = run_pipeline(limit_matches=None)
