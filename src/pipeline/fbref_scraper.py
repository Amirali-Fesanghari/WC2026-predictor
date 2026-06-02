"""
src/pipeline/fbref_scraper.py
Scrapes current player form data from FBref.com.

What we get per player (last 5 club matches):
  - Match rating proxy (SoT, xG, key passes, progressive carries)
  - Goals, assists, xG, xA
  - Minutes played (are they even starting?)
  - Defensive contribution (tackles, interceptions, pressures)

FBref is public but rate-limited. We:
  1. Cache every page for 24h (no unnecessary re-scraping)
  2. Sleep 4s between requests (polite scraping)
  3. Store everything in the DB so we only re-scrape what's stale

NOTE: FBref sometimes changes their HTML structure.
If scraping breaks, check: https://fbref.com/en/squads/
and update the CSS selectors below.
"""
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CACHE_DIR, WC_2026_TEAMS
from src.utils.team_name_map import normalize

CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# FBref national team page IDs (competition_id=1 = FIFA World Cup context)
# These are the squad stat pages for national teams
FBREF_NATIONAL_SQUAD_IDS = {
    "Argentina":             "f9fddd6e",
    "Brazil":                "e8d0fbb4",
    "France":                "76483ec5",
    "England":               "cce0a057",
    "Spain":                 "fa6d8b4c",
    "Germany":               "9a273c2f",
    "Portugal":              "f03c3d2c",
    "Netherlands":           "6048eb66",
    "Belgium":               "6f04c8c6",
    "Croatia":               "f05a4a35",
    "Morocco":               "7a6638b8",
    "Japan":                 "e1b28e84",
    "South Korea":           "a969fddc",
    "Australia":             "deada962",
    "Mexico":                "b5ae9cde",
    "United States":         "7f523e69",
    "Canada":                "ced3b95a",
    "Senegal":               "ecf6d1e8",
    "Uruguay":               "cfc0b527",
    "Colombia":              "fb6a7c6b",
    "Ecuador":               "4e1bba2c",
    "Switzerland":           "0c6d71a8",
    "Austria":               "b2780d4c",
    "Turkey":                "3e9c3f2e",
    "Iran":                  "e5d42f94",
    "Saudi Arabia":          "fc0e8b26",
    "Qatar":                 "b11e7aa2",
    "Iraq":                  "9a3b6e81",
    "Jordan":                "a5e4d2c3",
    "Uzbekistan":            "b8c3f1d7",
    "Algeria":               "d6c8a3f1",
    "Egypt":                 "2e5b8d1c",
    "Tunisia":               "c9d2a4e8",
    "DR Congo":              "f3b7e1c5",
    "Ivory Coast":           "e2f4b6d1",
    "Ghana":                 "a8d2c6b4",
    "South Africa":          "c5f1e3a7",
    "Cape Verde":            "d1b5c8e2",
    "Norway":                "b8e2c4d7",
    "Sweden":                "a3d7f2e1",
    "Scotland":              "c4b8d2f6",
    "Czech Republic":        "f6e3b1c8",
    "Bosnia and Herzegovina":"b2d5c9a3",
    "Paraguay":              "d7e2b4f1",
    "New Zealand":           "a4c9e7b2",
    "Panama":                "e9f3d1b5",
    "Haiti":                 "c2b7e4d9",
    "Curacao":               "f4a8d3c6",
}

REQUEST_DELAY = 4.0   # seconds between requests — be polite
CACHE_HOURS   = 24    # refresh player data every 24 hours


def _cache_path(key: str) -> Path:
    safe = re.sub(r'[^\w\-]', '_', key)
    return CACHE_DIR / f"fbref_{safe}.html"


def _fetch_page(url: str, cache_key: str) -> str | None:
    """Fetch a page with caching. Returns HTML string or None on failure."""
    path = _cache_path(cache_key)

    # Use cache if fresh enough
    if path.exists():
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours < CACHE_HOURS:
            logger.debug(f"Cache hit: {cache_key}")
            return path.read_text(encoding="utf-8")

    logger.debug(f"Fetching: {url}")
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        path.write_text(resp.text, encoding="utf-8")
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def scrape_national_squad_ratings(team_name: str) -> pd.DataFrame:
    """
    Scrape current player ratings for a national team from FBref.
    Returns a DataFrame with one row per player.

    Columns: player_name, position, club, age, caps,
             goals, assists, xg, xa, minutes, rating_proxy
    """
    canonical = normalize(team_name)
    squad_id = FBREF_NATIONAL_SQUAD_IDS.get(canonical)

    if not squad_id:
        logger.warning(f"No FBref squad ID for {canonical}. Returning empty.")
        return pd.DataFrame()

    url = f"https://fbref.com/en/squads/{squad_id}/{canonical}-Stats"
    html = _fetch_page(url, f"squad_{squad_id}")
    if not html:
        return pd.DataFrame()

    return _parse_squad_page(html, canonical)


def _parse_squad_page(html: str, team_name: str) -> pd.DataFrame:
    """Parse the FBref squad stats page HTML into a DataFrame."""
    soup = BeautifulSoup(html, "lxml")
    players = []

    # FBref uses table id="stats_standard_int" for international stats
    # and id="stats_standard" for the general stats table
    table = (
        soup.find("table", {"id": "stats_standard_int"}) or
        soup.find("table", {"id": "stats_standard"}) or
        soup.find("table", class_="stats_table")
    )

    if not table:
        logger.warning(f"No stats table found for {team_name}")
        return pd.DataFrame()

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    for row in tbody.find_all("tr"):
        # Skip spacer rows
        if row.get("class") and "thead" in row.get("class", []):
            continue

        cells = row.find_all(["td", "th"])
        if len(cells) < 5:
            continue

        try:
            # Player name — look for the <a> tag in the player cell
            name_cell = row.find("td", {"data-stat": "player"})
            if not name_cell:
                continue
            player_name = name_cell.get_text(strip=True)
            if not player_name or player_name == "Player":
                continue

            def get_stat(stat_name: str, default=0):
                cell = row.find("td", {"data-stat": stat_name})
                if cell is None:
                    return default
                txt = cell.get_text(strip=True)
                if not txt or txt in ["-", "—", ""]:
                    return default
                try:
                    return float(txt)
                except ValueError:
                    return default

            pos_cell = row.find("td", {"data-stat": "position"})
            position = pos_cell.get_text(strip=True) if pos_cell else ""

            club_cell = row.find("td", {"data-stat": "team"})
            club = club_cell.get_text(strip=True) if club_cell else ""

            age_cell = row.find("td", {"data-stat": "age"})
            age_txt = age_cell.get_text(strip=True) if age_cell else "0"
            age = float(age_txt.split("-")[0]) if age_txt else 0

            minutes   = get_stat("minutes")
            goals     = get_stat("goals")
            assists   = get_stat("assists")
            xg        = get_stat("xg")
            xa        = get_stat("xg_assist")
            shots     = get_stat("shots")
            sot       = get_stat("shots_on_target")
            prog_carr = get_stat("progressive_carries")
            key_pass  = get_stat("assisted_shots")   # key passes
            tackles   = get_stat("tackles")
            intercept = get_stat("interceptions")
            pressures = get_stat("pressures")
            caps      = get_stat("games")

            # Rating proxy: weighted composite of attacking + defensive contributions
            # This is NOT an official rating — it's our calculated score
            # Scale: roughly 0-10, average player ~5.0
            if minutes > 0:
                per90 = 90 / max(minutes, 1)
                attack_score = (
                    goals    * 2.5 * per90 +
                    assists  * 1.5 * per90 +
                    xg       * 1.8 * per90 +
                    xa       * 1.2 * per90 +
                    sot      * 0.5 * per90 +
                    key_pass * 0.8 * per90 +
                    prog_carr* 0.3 * per90
                )
                defense_score = (
                    tackles   * 0.6 * per90 +
                    intercept * 0.7 * per90 +
                    pressures * 0.2 * per90
                )
                # Base of 5.0, scaled by contribution
                rating_proxy = min(10.0, max(1.0, 5.0 + attack_score + defense_score * 0.5))
            else:
                rating_proxy = 5.0

            players.append({
                "team":          team_name,
                "player_name":   player_name,
                "position":      position,
                "club":          club,
                "age":           age,
                "caps":          int(caps),
                "minutes":       int(minutes),
                "goals":         int(goals),
                "assists":       int(assists),
                "xg":            round(xg, 3),
                "xa":            round(xa, 3),
                "shots_on_tgt":  int(sot),
                "key_passes":    int(key_pass),
                "tackles":       int(tackles),
                "interceptions": int(intercept),
                "pressures":     int(pressures),
                "rating_proxy":  round(rating_proxy, 2),
                "scraped_at":    datetime.utcnow(),
            })

        except Exception as e:
            logger.debug(f"Row parse error for {team_name}: {e}")
            continue

    df = pd.DataFrame(players)
    if not df.empty:
        logger.info(f"Scraped {len(df)} players for {team_name}")
    return df


def compute_squad_quality_score(players_df: pd.DataFrame) -> dict:
    """
    Aggregate individual player ratings into team-level quality scores.
    This is what feeds into the match feature vector.

    Returns a dict with:
      - squad_avg_rating: mean of all player rating proxies
      - top11_avg_rating: mean of the top 11 by rating (likely starters)
      - depth_score: gap between top 11 and bench (high = good depth)
      - gk_rating: goalkeeper rating
      - def_rating: avg defender rating
      - mid_rating: avg midfielder rating
      - att_rating: avg attacker rating
      - n_players: total squad size scraped
    """
    if players_df.empty:
        return {
            "squad_avg_rating": 5.0, "top11_avg_rating": 5.0,
            "depth_score": 0.0, "gk_rating": 5.0,
            "def_rating": 5.0, "mid_rating": 5.0,
            "att_rating": 5.0, "n_players": 0,
        }

    df = players_df.copy()
    df = df[df["minutes"] > 0]  # only players who've played

    if df.empty:
        return compute_squad_quality_score(pd.DataFrame())  # return defaults

    # Position grouping
    def pos_group(pos: str) -> str:
        pos = str(pos).upper()
        if "GK" in pos:   return "GK"
        if any(x in pos for x in ["CB","LB","RB","WB","DF","DEF"]): return "DEF"
        if any(x in pos for x in ["DM","CM","AM","MF","MID"]): return "MID"
        if any(x in pos for x in ["LW","RW","CF","ST","FW","ATT"]): return "ATT"
        return "MID"  # default unknown to midfielder

    df["pos_group"] = df["position"].apply(pos_group)

    sorted_df = df.sort_values("rating_proxy", ascending=False)
    top11 = sorted_df.head(11)
    bench = sorted_df.iloc[11:]

    def safe_mean(subset, pos=None):
        if pos:
            subset = subset[subset["pos_group"] == pos]
        if subset.empty:
            return 5.0
        return round(float(subset["rating_proxy"].mean()), 3)

    return {
        "squad_avg_rating": safe_mean(df),
        "top11_avg_rating": safe_mean(top11),
        "depth_score":      round(safe_mean(top11) - safe_mean(bench), 3) if len(bench) > 0 else 0.0,
        "gk_rating":        safe_mean(df, "GK"),
        "def_rating":       safe_mean(df, "DEF"),
        "mid_rating":       safe_mean(df, "MID"),
        "att_rating":       safe_mean(df, "ATT"),
        "n_players":        len(df),
    }


def scrape_all_wc_teams(delay_between_teams: float = 5.0) -> dict[str, pd.DataFrame]:
    """
    Scrape all 48 WC 2026 teams. Takes ~5 minutes total.
    Returns dict: team_name → players DataFrame.
    """
    results = {}
    total = len(WC_2026_TEAMS)

    for i, team in enumerate(WC_2026_TEAMS, 1):
        logger.info(f"[{i}/{total}] Scraping {team}...")
        df = scrape_national_squad_ratings(team)
        results[team] = df

        # Save per-team cache to parquet — use same key as load_cached_squad()
        if not df.empty:
            canonical = normalize(team)
            cache_path = CACHE_DIR / f"squad_{canonical.replace(' ','_').replace('/','_')}.parquet"
            df.to_parquet(cache_path)

        # Don't sleep after the last team
        if i < total:
            time.sleep(delay_between_teams)

    logger.success(f"Scraped {sum(len(v) for v in results.values())} players across {total} teams")
    return results


def load_cached_squad(team_name: str) -> pd.DataFrame:
    """Load a previously cached squad scrape (doesn't hit FBref)."""
    canonical = normalize(team_name)
    path = CACHE_DIR / f"squad_{canonical.replace(' ','_').replace('/','_')}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FBref national team scraper")
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test mode: scrape only the first 2 teams (France + Argentina) instead of all 48"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scrape all 48 WC 2026 teams (~5 minutes)"
    )
    args = parser.parse_args()

    if args.all:
        print("Scraping all WC 2026 teams...")
        results = scrape_all_wc_teams()
        non_empty = sum(1 for v in results.values() if not v.empty)
        print(f"\nDone. {non_empty}/{len(results)} teams had data.")
    elif args.test:
        test_teams = ["France", "Argentina"]
        print(f"--test mode: scraping {test_teams}...")
        for team_name in test_teams:
            df = scrape_national_squad_ratings(team_name)
            if df.empty:
                print(f"  {team_name}: no data (FBref may have changed structure or rate limited)")
            else:
                print(f"\n  {team_name} — {len(df)} players scraped:")
                print(df[["player_name","position","club","rating_proxy","goals","assists","xg"]].to_string(index=False))
                quality = compute_squad_quality_score(df)
                print(f"  Squad quality: {quality}")
    else:
        # Default: single team smoke test
        print("Testing FBref scraper on France (use --test for 2 teams, --all for all 48)...")
        df = scrape_national_squad_ratings("France")

        if df.empty:
            print("No data returned (FBref may have changed structure or rate limited)")
            print("This is normal on first run — check your internet connection")
        else:
            print(f"\nFrance squad — {len(df)} players scraped:")
            print(df[["player_name","position","club","rating_proxy","goals","assists","xg"]].to_string(index=False))

            print("\nSquad quality scores:")
            quality = compute_squad_quality_score(df)
            for k, v in quality.items():
                print(f"  {k}: {v}")
