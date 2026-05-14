"""
src/pipeline/database.py
Defines and initialises the SQLite schema.
Run this once: python -m src.pipeline.database
"""
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, ForeignKey, Text, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import inspect
from datetime import datetime
from loguru import logger
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
from config import DB_PATH

Base = declarative_base()


# ── Teams ────────────────────────────────────────────────────
class Team(Base):
    __tablename__ = "teams"

    id           = Column(Integer, primary_key=True)
    name         = Column(String(100), unique=True, nullable=False)
    fifa_code    = Column(String(10))           # e.g. "ARG"
    confederation= Column(String(20))           # UEFA, CONMEBOL, etc.
    elo_current  = Column(Float, default=1500.0)
    elo_peak     = Column(Float, default=1500.0)
    updated_at   = Column(DateTime, default=datetime.utcnow)

    elo_history  = relationship("EloHistory", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")

    def __repr__(self):
        return f"<Team {self.name} ELO={self.elo_current:.0f}>"


# ── ELO history (one row per match per team) ─────────────────
class EloHistory(Base):
    __tablename__ = "elo_history"

    id          = Column(Integer, primary_key=True)
    team_id     = Column(Integer, ForeignKey("teams.id"), nullable=False)
    match_id    = Column(Integer, ForeignKey("matches.id"), nullable=False)
    elo_before  = Column(Float, nullable=False)
    elo_after   = Column(Float, nullable=False)
    elo_delta   = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    team  = relationship("Team", back_populates="elo_history")
    match = relationship("Match", back_populates="elo_records")


# ── Matches ──────────────────────────────────────────────────
class Match(Base):
    __tablename__ = "matches"

    id               = Column(Integer, primary_key=True)
    external_id      = Column(String(100), unique=True)   # StatsBomb / football-data ID
    source           = Column(String(30))                  # "statsbomb" | "football_data"
    competition      = Column(String(100))
    season           = Column(String(20))
    stage            = Column(String(50))                  # group / R16 / QF / SF / F
    match_date       = Column(DateTime)
    venue            = Column(String(100))
    neutral_ground   = Column(Boolean, default=True)

    home_team_id     = Column(Integer, ForeignKey("teams.id"))
    away_team_id     = Column(Integer, ForeignKey("teams.id"))
    home_score       = Column(Integer)
    away_score       = Column(Integer)
    outcome          = Column(String(10))                  # "home" | "draw" | "away"

    # Advanced stats (from StatsBomb where available)
    home_xg          = Column(Float)
    away_xg          = Column(Float)
    home_possession  = Column(Float)
    away_possession  = Column(Float)
    home_shots       = Column(Integer)
    away_shots       = Column(Integer)
    home_shots_ot    = Column(Integer)
    away_shots_ot    = Column(Integer)
    home_passes      = Column(Integer)
    away_passes      = Column(Integer)
    home_pass_acc    = Column(Float)
    away_pass_acc    = Column(Float)
    home_pressures   = Column(Integer)
    away_pressures   = Column(Integer)
    home_formation   = Column(String(20))
    away_formation   = Column(String(20))

    created_at       = Column(DateTime, default=datetime.utcnow)

    home_team    = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team    = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    elo_records  = relationship("EloHistory", back_populates="match")
    player_stats = relationship("PlayerMatchStat", back_populates="match")
    psych_signals= relationship("PsychSignal", back_populates="match")
    features     = relationship("MatchFeatures", back_populates="match", uselist=False)


# ── Players ──────────────────────────────────────────────────
class Player(Base):
    __tablename__ = "players"

    id           = Column(Integer, primary_key=True)
    external_id  = Column(String(100), unique=True)
    name         = Column(String(150), nullable=False)
    team_id      = Column(Integer, ForeignKey("teams.id"))
    position     = Column(String(30))    # GK / CB / LB / RB / DM / CM / AM / LW / RW / ST
    club         = Column(String(100))
    birth_date   = Column(DateTime)
    nationality  = Column(String(100))

    match_stats  = relationship("PlayerMatchStat", back_populates="player")
    psych_signals= relationship("PsychSignal", back_populates="player")


# ── Player per-match stats ───────────────────────────────────
class PlayerMatchStat(Base):
    __tablename__ = "player_match_stats"

    id              = Column(Integer, primary_key=True)
    player_id       = Column(Integer, ForeignKey("players.id"))
    match_id        = Column(Integer, ForeignKey("matches.id"))
    team_id         = Column(Integer, ForeignKey("teams.id"))
    minutes_played  = Column(Integer)
    goals           = Column(Integer, default=0)
    assists         = Column(Integer, default=0)
    shots           = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    xg              = Column(Float, default=0.0)     # expected goals
    xa              = Column(Float, default=0.0)     # expected assists
    passes          = Column(Integer, default=0)
    pass_accuracy   = Column(Float)
    key_passes      = Column(Integer, default=0)
    dribbles        = Column(Integer, default=0)
    tackles         = Column(Integer, default=0)
    interceptions   = Column(Integer, default=0)
    pressures       = Column(Integer, default=0)
    fbref_rating    = Column(Float)   # scraped from FBref (0-10)
    sofascore_rating= Column(Float)   # scraped from Sofascore (1-10)

    player = relationship("Player", back_populates="match_stats")
    match  = relationship("Match", back_populates="player_stats")


# ── Psychological signals ────────────────────────────────────
class PsychSignal(Base):
    __tablename__ = "psych_signals"

    id            = Column(Integer, primary_key=True)
    match_id      = Column(Integer, ForeignKey("matches.id"), nullable=True)
    player_id     = Column(Integer, ForeignKey("players.id"), nullable=True)
    team_id       = Column(Integer, ForeignKey("teams.id"))
    source_url    = Column(Text)
    source_type   = Column(String(30))   # "press_conf" | "news" | "manual"
    headline      = Column(Text)
    raw_text      = Column(Text)
    # After your review, you set these:
    sentiment_score   = Column(Float)    # -1.0 (very negative) to +1.0 (very positive)
    risk_category     = Column(String(50))  # "injury" | "family" | "political" | "form" | etc.
    severity          = Column(Integer)  # 1 (minor) to 5 (critical)
    affects_performance = Column(Boolean, default=True)
    reviewer_notes    = Column(Text)     # your manual notes
    reviewed          = Column(Boolean, default=False)
    recorded_at       = Column(DateTime, default=datetime.utcnow)
    applies_to_match_date = Column(DateTime)  # which upcoming match does this affect?

    match  = relationship("Match", back_populates="psych_signals")
    player = relationship("Player", back_populates="psych_signals")


# ── Feature vectors (one row per match, ML-ready) ────────────
class MatchFeatures(Base):
    __tablename__ = "match_features"

    id       = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True)

    # Team-level features (home perspective)
    home_elo              = Column(Float)
    away_elo              = Column(Float)
    elo_diff              = Column(Float)

    # Rolling form (last 5 matches)
    home_form_pts         = Column(Float)   # points/15 in last 5
    away_form_pts         = Column(Float)
    home_form_gf          = Column(Float)   # goals for avg last 5
    away_form_gf          = Column(Float)
    home_form_ga          = Column(Float)   # goals against avg last 5
    away_form_ga          = Column(Float)
    home_form_xg          = Column(Float)
    away_form_xg          = Column(Float)

    # H2H (head to head)
    h2h_home_wins         = Column(Float)   # win rate in last 10 H2H
    h2h_draws             = Column(Float)
    h2h_away_wins         = Column(Float)
    h2h_home_xg_avg       = Column(Float)
    h2h_away_xg_avg       = Column(Float)

    # Player quality
    home_avg_player_rating= Column(Float)
    away_avg_player_rating= Column(Float)
    home_star_player_fit  = Column(Float)   # key player available? 0-1
    away_star_player_fit  = Column(Float)

    # Psych signals
    home_psych_score      = Column(Float)   # aggregate, -1 to +1
    away_psych_score      = Column(Float)
    home_psych_risk_flags = Column(Integer) # count of flagged players
    away_psych_risk_flags = Column(Integer)

    # Contextual
    days_since_last_match_home = Column(Float)
    days_since_last_match_away = Column(Float)
    tournament_stage_weight    = Column(Float)   # group=1, R16=1.5, QF=2, SF=2.5, F=3
    is_neutral                 = Column(Boolean)

    # Tactical fit (filled after tactics module)
    home_formation_enc    = Column(Integer)  # label-encoded
    away_formation_enc    = Column(Integer)
    tactical_matchup_score= Column(Float)    # how well formations clash

    # Target (for training)
    outcome               = Column(String(10))  # "home" | "draw" | "away"
    home_goals_actual     = Column(Integer)
    away_goals_actual     = Column(Integer)

    # Predictions (filled at inference time)
    pred_home_win_prob    = Column(Float)
    pred_draw_prob        = Column(Float)
    pred_away_win_prob    = Column(Float)
    pred_home_goals       = Column(Float)
    pred_away_goals       = Column(Float)
    model_version         = Column(String(50))

    match = relationship("Match", back_populates="features")


# ── Init ─────────────────────────────────────────────────────
def init_db(echo: bool = False) -> sessionmaker:
    """Create all tables and return a session factory."""
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=echo)
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    logger.success(f"Database ready at {DB_PATH}")
    logger.info(f"Tables: {tables}")

    Session = sessionmaker(bind=engine)
    return Session


def get_session():
    """Quick helper to get a session anywhere in the codebase."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    from loguru import logger
    logger.add("logs/db_init.log", rotation="10 MB")
    Session = init_db(echo=True)
    sess = Session()
    logger.success(f"Session opened. Ready.")
    sess.close()
