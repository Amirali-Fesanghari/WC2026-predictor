"""
src/pipeline/statsbomb_loader.py
Loads StatsBomb free data for World Cup competitions.
StatsBomb releases full event-level data for select competitions for free —
including WC 2022, 2018, 2014 with 360-degree tracking for 2022.

No API key needed. Data is fetched from their public GitHub.
"""
import pandas as pd
from statsbombpy import sb
from loguru import logger
from typing import Optional
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
from config import SB_COMPETITIONS, CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_competitions() -> pd.DataFrame:
    """List all free competitions available from StatsBomb."""
    comps = sb.competitions()
    logger.info(f"StatsBomb: {len(comps)} free competitions available")
    return comps


def get_wc_matches(edition: str = "WC_2022") -> pd.DataFrame:
    """
    Fetch all matches for a World Cup edition.
    edition: one of "WC_2022", "WC_2018", "WC_2014"
    """
    cache_file = CACHE_DIR / f"sb_matches_{edition}.parquet"
    if cache_file.exists():
        logger.info(f"Loading cached matches for {edition}")
        return pd.read_parquet(cache_file)

    cfg = SB_COMPETITIONS.get(edition)
    if not cfg:
        raise ValueError(f"Unknown edition: {edition}. Choose from {list(SB_COMPETITIONS)}")

    logger.info(f"Fetching {edition} matches from StatsBomb...")
    matches = sb.matches(
        competition_id=cfg["competition_id"],
        season_id=cfg["season_id"],
    )
    matches.to_parquet(cache_file)
    logger.success(f"Fetched {len(matches)} matches for {edition}")
    return matches


def get_match_events(match_id: int) -> pd.DataFrame:
    """
    Fetch all events for a single match.
    Each event = one action (pass, shot, tackle, dribble, etc.)
    This gives us: xG, pressure events, pass accuracy, etc.
    """
    cache_file = CACHE_DIR / f"sb_events_{match_id}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    logger.debug(f"Fetching events for match {match_id}...")
    events = sb.events(match_id=match_id)
    events.to_parquet(cache_file)
    return events


def extract_match_stats(match_id: int, match_row: pd.Series) -> dict:
    """
    Aggregate event-level data into per-match team statistics.
    Returns a flat dict ready to insert into the matches table.
    """
    try:
        events = get_match_events(match_id)
    except Exception as e:
        logger.warning(f"Could not load events for match {match_id}: {e}")
        return {}

    home_team = match_row.get("home_team", "")
    away_team = match_row.get("away_team", "")

    def team_events(team_name: str) -> pd.DataFrame:
        return events[events["team"] == team_name]

    def count_type(team_name: str, event_type: str) -> int:
        return len(team_events(team_name)[team_events(team_name)["type"] == event_type])

    # Shots and xG
    home_shots_df = team_events(home_team)[team_events(home_team)["type"] == "Shot"]
    away_shots_df = team_events(away_team)[team_events(away_team)["type"] == "Shot"]

    home_xg = home_shots_df["shot_statsbomb_xg"].sum() if "shot_statsbomb_xg" in home_shots_df.columns else None
    away_xg = away_shots_df["shot_statsbomb_xg"].sum() if "shot_statsbomb_xg" in away_shots_df.columns else None

    home_sot = home_shots_df[home_shots_df.get("shot_outcome", pd.Series(dtype=str)).isin(["Goal","Saved","Saved to Post"])].shape[0] if len(home_shots_df) else 0
    away_sot = away_shots_df[away_shots_df.get("shot_outcome", pd.Series(dtype=str)).isin(["Goal","Saved","Saved to Post"])].shape[0] if len(away_shots_df) else 0

    # Passes
    home_passes_df = team_events(home_team)[team_events(home_team)["type"] == "Pass"]
    away_passes_df = team_events(away_team)[team_events(away_team)["type"] == "Pass"]

    def pass_accuracy(passes_df: pd.DataFrame) -> Optional[float]:
        if len(passes_df) == 0:
            return None
        if "pass_outcome" not in passes_df.columns:
            return None
        completed = passes_df["pass_outcome"].isna().sum()  # NaN outcome = completed in SB
        return round(completed / len(passes_df) * 100, 1)

    # Pressure events (pressing)
    home_pressures = count_type(home_team, "Pressure")
    away_pressures = count_type(away_team, "Pressure")

    # Possession (derived from touch counts)
    total_events = len(events[events["type"].isin(["Pass", "Carry", "Shot", "Dribble"])])
    home_touch = len(team_events(home_team)[team_events(home_team)["type"].isin(["Pass","Carry","Shot","Dribble"])])
    home_poss = round(home_touch / total_events * 100, 1) if total_events > 0 else None
    away_poss = round(100 - home_poss, 1) if home_poss is not None else None

    return {
        "home_xg":          round(float(home_xg), 3) if home_xg is not None else None,
        "away_xg":          round(float(away_xg), 3) if away_xg is not None else None,
        "home_possession":  home_poss,
        "away_possession":  away_poss,
        "home_shots":       len(home_shots_df),
        "away_shots":       len(away_shots_df),
        "home_shots_ot":    home_sot,
        "away_shots_ot":    away_sot,
        "home_passes":      len(home_passes_df),
        "away_passes":      len(away_passes_df),
        "home_pass_acc":    pass_accuracy(home_passes_df),
        "away_pass_acc":    pass_accuracy(away_passes_df),
        "home_pressures":   home_pressures,
        "away_pressures":   away_pressures,
    }


def extract_player_stats(match_id: int) -> pd.DataFrame:
    """
    Build per-player stats from events in a single match.
    Returns a DataFrame with one row per player.
    """
    try:
        events = get_match_events(match_id)
    except Exception as e:
        logger.warning(f"Events unavailable for match {match_id}: {e}")
        return pd.DataFrame()

    players = events[["player_id","player","team","position"]].dropna(subset=["player_id"]).drop_duplicates("player_id")
    stats = []

    for _, prow in players.iterrows():
        pid   = prow["player_id"]
        pname = prow["player"]
        team  = prow["team"]
        pe    = events[events["player_id"] == pid]

        # Minutes played (last event timestamp)
        minutes = int(pe["minute"].max()) if "minute" in pe.columns else 0

        shots_df = pe[pe["type"] == "Shot"]
        xg = shots_df["shot_statsbomb_xg"].sum() if "shot_statsbomb_xg" in shots_df.columns else 0.0
        goals = shots_df[shots_df.get("shot_outcome", pd.Series(dtype=str)) == "Goal"].shape[0]

        passes_df = pe[pe["type"] == "Pass"]
        key_passes = passes_df[passes_df.get("pass_goal_assist", pd.Series()).fillna(False)].shape[0] if "pass_goal_assist" in passes_df.columns else 0

        stats.append({
            "player_id":       pid,
            "player_name":     pname,
            "team":            team,
            "position":        prow.get("position", ""),
            "minutes_played":  minutes,
            "goals":           goals,
            "shots":           len(shots_df),
            "xg":              round(float(xg), 3),
            "passes":          len(passes_df),
            "key_passes":      key_passes,
            "dribbles":        len(pe[pe["type"] == "Dribble"]),
            "tackles":         len(pe[pe["type"] == "Tackle"]),
            "interceptions":   len(pe[pe["type"] == "Interception"]),
            "pressures":       len(pe[pe["type"] == "Pressure"]),
        })

    return pd.DataFrame(stats)


def load_full_tournament(edition: str = "WC_2022") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ALL matches and ALL player stats for a tournament.
    Returns (matches_df, player_stats_df).
    This is the main entry point for building the training dataset.
    """
    matches = get_wc_matches(edition)
    all_player_stats = []

    logger.info(f"Extracting stats from {len(matches)} matches...")
    for _, match_row in matches.iterrows():
        mid = match_row["match_id"]

        # Inject event-level stats into the match row
        event_stats = extract_match_stats(mid, match_row)
        for key, val in event_stats.items():
            matches.loc[matches["match_id"] == mid, key] = val

        # Player stats
        pstats = extract_player_stats(mid)
        if not pstats.empty:
            pstats["match_id"] = mid
            all_player_stats.append(pstats)

    player_stats_df = pd.concat(all_player_stats, ignore_index=True) if all_player_stats else pd.DataFrame()
    logger.success(f"Done. {len(matches)} matches, {len(player_stats_df)} player-match records.")
    return matches, player_stats_df


if __name__ == "__main__":
    # Quick test: list available competitions
    comps = get_competitions()
    wc_comps = comps[comps["competition_name"] == "FIFA World Cup"]
    print("Available World Cup data from StatsBomb:")
    print(wc_comps[["competition_id","season_id","season_name"]].to_string(index=False))

    # Load WC 2022 matches (no events yet — just match list)
    matches_2022 = get_wc_matches("WC_2022")
    print(f"\nWC 2022: {len(matches_2022)} matches loaded")
    print(matches_2022[["match_id","home_team","away_team","home_score","away_score","match_week"]].head(10))
