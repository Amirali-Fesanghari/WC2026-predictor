"""
src/pipeline/elo.py
International football ELO rating engine.

Key differences from club ELO:
- Higher K-factor for knockout rounds (more meaningful)
- Decay toward 1500 for inactive teams (qualifiers stretch over years)
- Goal-difference multiplier (same as World Football ELO system)
- Neutral ground: no home advantage added
"""
import math
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
from config import (
    ELO_INITIAL, ELO_K_FRIENDLY, ELO_K_QUALIFIER,
    ELO_K_TOURNAMENT, ELO_K_KNOCKOUT, ELO_HOME_ADVANTAGE,
    ELO_DECAY_RATE
)


# ── Stage → K-factor mapping ─────────────────────────────────
STAGE_K = {
    "friendly":       ELO_K_FRIENDLY,
    "qualifier":      ELO_K_QUALIFIER,
    "group":          ELO_K_TOURNAMENT,
    "round_of_32":    ELO_K_KNOCKOUT,
    "round_of_16":    ELO_K_KNOCKOUT,
    "quarter_final":  ELO_K_KNOCKOUT,
    "semi_final":     ELO_K_KNOCKOUT,
    "third_place":    ELO_K_KNOCKOUT,
    "final":          ELO_K_KNOCKOUT,
}


def _expected_score(rating_a: float, rating_b: float, home_adv: float = 0) -> float:
    """
    Expected score for team A vs team B using standard ELO formula.
    home_adv: added to team A's rating (0 for neutral ground WC matches).
    Returns probability 0-1 of team A winning.
    """
    return 1.0 / (1.0 + 10 ** (-(rating_a + home_adv - rating_b) / 400))


def _goal_diff_multiplier(goal_diff: int) -> float:
    """
    World Football ELO multiplier based on goal difference.
    Makes a 3-0 win worth more than a 1-0 win.
    Formula from: https://www.eloratings.net/about
    """
    gd = abs(goal_diff)
    if gd == 0 or gd == 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8.0


def _actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """
    Convert match result to ELO score (1=win, 0.5=draw, 0=loss).
    Returns (home_score, away_score).
    """
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals == away_goals:
        return 0.5, 0.5
    else:
        return 0.0, 1.0


def update_elo(
    home_elo: float,
    away_elo: float,
    home_goals: int,
    away_goals: int,
    stage: str = "group",
    neutral: bool = True,
) -> tuple[float, float, float, float]:
    """
    Core ELO update function.

    Returns:
        (new_home_elo, new_away_elo, home_delta, away_delta)
    """
    k = STAGE_K.get(stage.lower().replace(" ", "_"), ELO_K_TOURNAMENT)
    home_adv = 0 if neutral else ELO_HOME_ADVANTAGE

    # Expected
    exp_home = _expected_score(home_elo, away_elo, home_adv)
    exp_away = 1.0 - exp_home

    # Actual
    act_home, act_away = _actual_score(home_goals, away_goals)

    # Goal diff multiplier
    gdm = _goal_diff_multiplier(home_goals - away_goals)

    # Deltas
    home_delta = k * gdm * (act_home - exp_home)
    away_delta = k * gdm * (act_away - exp_away)

    return (
        home_elo + home_delta,
        away_elo + away_delta,
        home_delta,
        away_delta,
    )


def decay_elo(elo: float, months_inactive: int) -> float:
    """
    Decay rating toward 1500 for teams that haven't played.
    Used to handle long qualifier gaps.
    """
    return ELO_INITIAL + (elo - ELO_INITIAL) * (ELO_DECAY_RATE ** months_inactive)


class EloEngine:
    """
    Stateful ELO engine.
    Processes a chronological sequence of matches and maintains
    current ratings for all teams.

    Usage:
        engine = EloEngine()
        engine.process_dataframe(matches_df)
        ratings = engine.get_ratings()
    """

    def __init__(self):
        self.ratings: dict[str, float] = {}        # team_name → current ELO
        self.history: list[dict] = []              # audit trail
        self.last_played: dict[str, datetime] = {} # team → last match date

    def _get_rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = float(ELO_INITIAL)
            logger.debug(f"New team initialised: {team} @ {ELO_INITIAL}")
        return self.ratings[team]

    def _apply_decay(self, team: str, match_date: datetime):
        """Apply inactivity decay before processing a new match."""
        if team not in self.last_played:
            return
        months = max(0, (match_date - self.last_played[team]).days // 30)
        if months > 2:  # only decay after 2+ months inactivity
            old = self.ratings[team]
            self.ratings[team] = decay_elo(old, months)
            logger.debug(f"{team}: decay {old:.1f}→{self.ratings[team]:.1f} ({months}mo inactive)")

    def process_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        match_date: datetime,
        stage: str = "group",
        neutral: bool = True,
        match_id: Optional[str] = None,
    ) -> dict:
        """Process a single match and update ratings."""

        # Apply decay
        self._apply_decay(home_team, match_date)
        self._apply_decay(away_team, match_date)

        elo_home_before = self._get_rating(home_team)
        elo_away_before = self._get_rating(away_team)

        new_home, new_away, delta_h, delta_a = update_elo(
            elo_home_before, elo_away_before,
            home_goals, away_goals,
            stage=stage, neutral=neutral,
        )

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away
        self.last_played[home_team] = match_date
        self.last_played[away_team] = match_date

        record = {
            "match_id":       match_id,
            "date":           match_date,
            "home_team":      home_team,
            "away_team":      away_team,
            "home_goals":     home_goals,
            "away_goals":     away_goals,
            "stage":          stage,
            "elo_home_before":elo_home_before,
            "elo_away_before":elo_away_before,
            "elo_home_after": new_home,
            "elo_away_after": new_away,
            "delta_home":     delta_h,
            "delta_away":     delta_a,
            "exp_home_win":   _expected_score(elo_home_before, elo_away_before),
        }
        self.history.append(record)
        return record

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of matches in chronological order.

        Expected columns:
            home_team, away_team, home_goals, away_goals,
            match_date, stage, neutral_ground, match_id (optional)

        Returns history as a DataFrame.
        """
        required = ["home_team", "away_team", "home_goals", "away_goals", "match_date"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df = df.sort_values("match_date").reset_index(drop=True)
        logger.info(f"Processing {len(df)} matches chronologically...")

        for _, row in df.iterrows():
            self.process_match(
                home_team  = row["home_team"],
                away_team  = row["away_team"],
                home_goals = int(row["home_goals"]),
                away_goals = int(row["away_goals"]),
                match_date = pd.to_datetime(row["match_date"]),
                stage      = row.get("stage", "group"),
                neutral    = bool(row.get("neutral_ground", True)),
                match_id   = row.get("match_id"),
            )

        logger.success(f"ELO processing complete. {len(self.ratings)} teams rated.")
        return self.get_history_df()

    def get_ratings(self) -> pd.DataFrame:
        """Return current ratings sorted descending."""
        return (
            pd.DataFrame(
                [(team, elo) for team, elo in self.ratings.items()],
                columns=["team", "elo"]
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> dict:
        """
        Predict win/draw/loss probabilities for an upcoming match.
        Draw probability estimated using the Bradley-Terry-Luce model
        adjustment (a common approximation for soccer).
        """
        elo_h = self._get_rating(home_team)
        elo_a = self._get_rating(away_team)
        home_adv = 0 if neutral else ELO_HOME_ADVANTAGE

        p_home_win = _expected_score(elo_h, elo_a, home_adv)
        p_away_win = _expected_score(elo_a, elo_h, -home_adv)

        # Draw estimation: subtract from the "gap" between the two win probs
        # Empirically calibrated for international football (~25% draw rate)
        raw_gap = abs(p_home_win - p_away_win)
        p_draw = max(0.10, 0.30 - 0.25 * raw_gap)
        scale = (1 - p_draw)
        p_home_win = p_home_win * scale
        p_away_win = p_away_win * scale

        # Normalise
        total = p_home_win + p_draw + p_away_win
        p_home_win /= total
        p_draw     /= total
        p_away_win /= total

        return {
            "home_team":   home_team,
            "away_team":   away_team,
            "home_elo":    round(elo_h, 1),
            "away_elo":    round(elo_a, 1),
            "p_home_win":  round(p_home_win, 4),
            "p_draw":      round(p_draw, 4),
            "p_away_win":  round(p_away_win, 4),
        }


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    engine = EloEngine()

    # Simulate a small WC 2022 group stage
    sample_matches = [
        ("Qatar",    "Ecuador",   0, 2, "2022-11-20", "group"),
        ("England",  "Iran",      6, 2, "2022-11-21", "group"),
        ("Senegal",  "Netherlands",0, 2, "2022-11-21", "group"),
        ("Argentina","Saudi Arabia",1,2,"2022-11-22", "group"),
        ("France",   "Australia", 4, 1, "2022-11-22", "group"),
        ("Argentina","France",    3, 3, "2022-12-18", "final"),
    ]
    for h, a, hg, ag, date, stage in sample_matches:
        r = engine.process_match(h, a, hg, ag, datetime.fromisoformat(date), stage)
        outcome = "DRAW" if hg == ag else (h if hg > ag else a)
        print(f"{h} {hg}-{ag} {a} | Δ {r['delta_home']:+.1f} / {r['delta_away']:+.1f} | Winner: {outcome}")

    print("\n─── Current ELO ratings ───")
    print(engine.get_ratings().to_string(index=False))

    print("\n─── Prediction: Argentina vs France (neutral) ───")
    print(engine.predict_match("Argentina", "France"))
