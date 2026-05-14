"""
src/pipeline/football_data_loader.py
Loads historical international results from football-data.org.
Used to build ELO ratings going back to 2000+.

Free API key: https://www.football-data.org/client/register
Without a key: 10 req/min, only current season.
With free key:  10 req/min, multiple seasons.
"""
import time
import requests
import pandas as pd
from datetime import datetime
from loguru import logger
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
from config import FOOTBALL_DATA_API_KEY, CACHE_DIR, COMPETITIONS

CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": FOOTBALL_DATA_API_KEY} if FOOTBALL_DATA_API_KEY else {}

# Fallback: martini-style open dataset from GitHub
# Bart Megemoet's international soccer results dataset: 44,000+ matches back to 1872
OPEN_DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)


def load_open_international_results() -> pd.DataFrame:
    """
    Load Mart Jürisoo's open international soccer results dataset.
    44,000+ matches from 1872 to present.
    This is the backbone for our ELO bootstrapping — no API key needed.
    URL: https://github.com/martj42/international_results
    """
    cache_file = CACHE_DIR / "international_results.parquet"
    if cache_file.exists():
        age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info("Loading cached international results (< 24h old)")
            return pd.read_parquet(cache_file)

    logger.info("Downloading international results dataset (~5MB)...")
    df = pd.read_csv(OPEN_DATASET_URL)

    # Standardise columns
    df.rename(columns={
        "date":        "match_date",
        "home_team":   "home_team",
        "away_team":   "away_team",
        "home_score":  "home_goals",
        "away_score":  "away_goals",
        "tournament":  "competition",
        "neutral":     "neutral_ground",
    }, inplace=True)

    df["match_date"] = pd.to_datetime(df["match_date"])
    df["neutral_ground"] = df["neutral_ground"].astype(bool)
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df.dropna(subset=["home_goals", "away_goals"], inplace=True)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Compute outcome from scores
    df["outcome"] = df.apply(
        lambda r: "home" if r["home_goals"] > r["away_goals"]
        else ("draw" if r["home_goals"] == r["away_goals"] else "away"),
        axis=1
    )

    # Stage classification
    df["stage"] = df["competition"].apply(_classify_stage)

    df.to_parquet(cache_file)
    logger.success(f"Loaded {len(df):,} international matches ({df['match_date'].min().year}–{df['match_date'].max().year})")
    return df


def _classify_stage(competition: str) -> str:
    """Map competition name → ELO stage for K-factor selection."""
    comp = competition.lower()
    if any(x in comp for x in ["friendly", "kirin", "four nations"]):
        return "friendly"
    if any(x in comp for x in ["qualification", "qualifier", "qualifying"]):
        return "qualifier"
    if "final" in comp and "quarter" not in comp and "semi" not in comp:
        return "final"
    if "quarter" in comp:
        return "quarter_final"
    if "semi" in comp:
        return "semi_final"
    if any(x in comp for x in ["world cup", "euro", "copa", "afcon", "afc", "concacaf", "gold cup", "nations league"]):
        return "group"
    return "qualifier"


def get_wc_history(start_year: int = 2002) -> pd.DataFrame:
    """
    Filter international results to World Cup matches only.
    Used for targeted ELO training on WC-specific data.
    """
    df = load_open_international_results()
    wc_mask = df["competition"].str.contains("FIFA World Cup", case=False, na=False)
    wc = df[wc_mask & (df["match_date"].dt.year >= start_year)].copy()

    # Map to stage names
    wc["stage"] = wc["competition"].apply(_map_wc_stage)
    logger.info(f"WC matches {start_year}+: {len(wc)}")
    return wc


def _map_wc_stage(competition: str) -> str:
    comp = competition.lower()
    if "qualification" in comp or "qualifier" in comp:
        return "qualifier"
    if "group" in comp:
        return "group"
    if "round of 16" in comp:
        return "round_of_16"
    if "quarter" in comp:
        return "quarter_final"
    if "semi" in comp:
        return "semi_final"
    if "third" in comp or "3rd" in comp:
        return "third_place"
    if "final" in comp:
        return "final"
    return "group"


def get_recent_national_matches(team_name: str, n: int = 20) -> pd.DataFrame:
    """
    Get the last N international matches for a national team.
    Used for the current-form module.
    """
    df = load_open_international_results()
    team_mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    team_matches = df[team_mask].sort_values("match_date", ascending=False).head(n).copy()

    # Normalise perspective to "team" vs "opponent"
    rows = []
    for _, row in team_matches.iterrows():
        if row["home_team"] == team_name:
            gf, ga = row["home_goals"], row["away_goals"]
            opponent = row["away_team"]
        else:
            gf, ga = row["away_goals"], row["home_goals"]
            opponent = row["home_team"]

        result = "W" if gf > ga else ("D" if gf == ga else "L")
        rows.append({
            "date":       row["match_date"],
            "team":       team_name,
            "opponent":   opponent,
            "gf":         gf,
            "ga":         ga,
            "result":     result,
            "competition":row["competition"],
            "stage":      row["stage"],
            "neutral":    row["neutral_ground"],
        })

    return pd.DataFrame(rows)


def compute_form_features(team_name: str, as_of_date: datetime, n: int = 5) -> dict:
    """
    Compute rolling form features for a team in the last N matches
    before a given date. Used in feature engineering.

    Returns a flat dict of form features ready for MatchFeatures.
    """
    df = load_open_international_results()
    team_mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    past = df[team_mask & (df["match_date"] < as_of_date)]
    recent = past.sort_values("match_date", ascending=False).head(n)

    if recent.empty:
        return {
            "form_pts": 0.0, "form_gf": 0.0, "form_ga": 0.0,
            "form_gd": 0.0, "form_wins": 0, "form_draws": 0, "form_losses": 0,
            "days_since_last": 999,
        }

    pts, gf_list, ga_list = [], [], []
    for _, row in recent.iterrows():
        if row["home_team"] == team_name:
            gf, ga = row["home_goals"], row["away_goals"]
        else:
            gf, ga = row["away_goals"], row["home_goals"]
        gf_list.append(gf)
        ga_list.append(ga)
        if gf > ga:   pts.append(3)
        elif gf == ga: pts.append(1)
        else:          pts.append(0)

    last_date = recent["match_date"].max()
    days_since = (as_of_date - last_date).days

    return {
        "form_pts":        round(sum(pts) / (len(pts) * 3), 4),  # normalised 0-1
        "form_gf":         round(sum(gf_list) / len(gf_list), 3),
        "form_ga":         round(sum(ga_list) / len(ga_list), 3),
        "form_gd":         round((sum(gf_list) - sum(ga_list)) / len(gf_list), 3),
        "form_wins":       pts.count(3),
        "form_draws":      pts.count(1),
        "form_losses":     pts.count(0),
        "days_since_last": days_since,
    }


def compute_h2h_features(home_team: str, away_team: str, as_of_date: datetime, n: int = 10) -> dict:
    """
    Head-to-head stats for two teams in the last N meetings.
    """
    df = load_open_international_results()
    h2h_mask = (
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    )
    h2h = df[h2h_mask & (df["match_date"] < as_of_date)].sort_values("match_date", ascending=False).head(n)

    if h2h.empty:
        return {"h2h_home_win_rate": 0.5, "h2h_draw_rate": 0.2, "h2h_away_win_rate": 0.3,
                "h2h_home_gf_avg": 1.2, "h2h_away_gf_avg": 1.2, "h2h_matches": 0}

    home_wins, draws, away_wins = 0, 0, 0
    home_gf_total, away_gf_total = 0, 0

    for _, row in h2h.iterrows():
        if row["home_team"] == home_team:
            hg, ag = row["home_goals"], row["away_goals"]
        else:
            hg, ag = row["away_goals"], row["home_goals"]  # flip perspective
        home_gf_total += hg
        away_gf_total += ag
        if hg > ag:   home_wins += 1
        elif hg == ag: draws += 1
        else:          away_wins += 1

    n_matches = len(h2h)
    return {
        "h2h_home_win_rate": round(home_wins / n_matches, 4),
        "h2h_draw_rate":     round(draws / n_matches, 4),
        "h2h_away_win_rate": round(away_wins / n_matches, 4),
        "h2h_home_gf_avg":   round(home_gf_total / n_matches, 3),
        "h2h_away_gf_avg":   round(away_gf_total / n_matches, 3),
        "h2h_matches":       n_matches,
    }


if __name__ == "__main__":
    df = load_open_international_results()
    print(f"Total matches: {len(df):,}")
    print(f"Date range: {df['match_date'].min().date()} → {df['match_date'].max().date()}")
    print(f"\nStage breakdown:\n{df['stage'].value_counts()}")

    print("\n─── Argentina last 5 matches ───")
    print(get_recent_national_matches("Argentina", n=5)[["date","opponent","gf","ga","result","competition"]])

    print("\n─── Argentina form features (today) ───")
    print(compute_form_features("Argentina", datetime.now()))

    print("\n─── Argentina vs France H2H ───")
    print(compute_h2h_features("Argentina", "France", datetime.now()))
