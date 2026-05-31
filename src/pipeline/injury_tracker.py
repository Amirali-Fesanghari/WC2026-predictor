"""
src/pipeline/injury_tracker.py
Tracks injury and availability status for players and computes squad impact scores.

Data flow:
  1. scrape_injuries(team_name)
       -> tries Transfermarkt injuries page first
       -> falls back to BBC Sport player search
       -> caches results for 24h in data/cache/injuries/<slug>.json
       -> saves each entry to psych_signals DB table with risk_category='injury'

  2. compute_squad_impact(team_name, squad_quality_scores)
       -> returns available_quality, injury_risk_score,
          key_players_out, depth_score

  3. get_availability_feature(team_name) -> float 0-1
       returns 0.5 on any failure
"""

import re
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

# ── Configuration import with graceful fallback ───────────────────────────────
try:
    from config import DB_PATH, CACHE_DIR  # type: ignore
    logger.debug("config.DB_PATH / CACHE_DIR loaded.")
except Exception as _cfg_exc:
    logger.warning("config not importable ({}), using defaults.", _cfg_exc)
    DB_PATH   = Path(__file__).parents[2] / "data" / "wc2026.db"
    CACHE_DIR = Path(__file__).parents[2] / "data" / "cache"

INJURY_CACHE_DIR = CACHE_DIR / "injuries"
INJURY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_HOURS   = 24
REQUEST_DELAY = 3.0   # polite scraping pause in seconds

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Transfermarkt national team slugs: team_name → (slug, team_id) ───────────
TRANSFERMARKT_SLUGS = {
    "Argentina":             ("argentinien",                "3437"),
    "Australia":             ("australien",                 "3401"),
    "Austria":               ("osterreich",                 "3438"),
    "Belgium":               ("belgien",                    "3382"),
    "Bosnia and Herzegovina":("bosnien-herzegowina",        "3555"),
    "Brazil":                ("brasilien",                  "3439"),
    "Canada":                ("kanada",                     "3468"),
    "Colombia":              ("kolumbien",                  "3474"),
    "Croatia":               ("kroatien",                   "3553"),
    "Czech Republic":        ("tschechien",                 "3448"),
    "Ecuador":               ("ecuador",                    "3452"),
    "Egypt":                 ("agypten",                    "3456"),
    "England":               ("england",                    "3-england"),
    "France":                ("frankreich",                 "3377"),
    "Germany":               ("deutschland",                "3378"),
    "Ghana":                 ("ghana",                      "3457"),
    "Iran":                  ("iran",                       "3408"),
    "Japan":                 ("japan",                      "3413"),
    "Mexico":                ("mexiko",                     "3479"),
    "Morocco":               ("marokko",                    "3461"),
    "Netherlands":           ("niederlande",                "3379"),
    "Norway":                ("norwegen",                   "3380"),
    "Panama":                ("panama",                     "3481"),
    "Portugal":              ("portugal",                   "3476"),
    "Qatar":                 ("katar",                      "3412"),
    "Saudi Arabia":          ("saudi-arabien",              "3414"),
    "Senegal":               ("senegal",                    "3464"),
    "Serbia":                ("serbien",                    "3447"),
    "South Korea":           ("sudkorea",                   "3416"),
    "Spain":                 ("spanien",                    "3375"),
    "Switzerland":           ("schweiz",                    "3385"),
    "Tunisia":               ("tunesien",                   "3463"),
    "United States":         ("usa",                        "3482"),
    "Uruguay":               ("uruguay",                    "3477"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _team_slug(team_name):
    """Return a filesystem-safe slug for the team name."""
    return re.sub(r"[^\w\-]", "_", team_name.strip().lower())


def _normalise_status(raw_text):
    """
    Map scrape text to one of: injured | doubt | suspended | available.
    """
    lo = raw_text.lower()
    if any(w in lo for w in [
        "injur", "out", "unavailab", "surgery", "fracture",
        "torn", "strain", "sprain", "muscle", "ligament",
    ]):
        return "injured"
    if any(w in lo for w in [
        "doubt", "50/50", "knock", "minor concern",
        "uncertain", "fitness test",
    ]):
        return "doubt"
    if any(w in lo for w in ["suspend", "ban", "red card", "accumulation"]):
        return "suspended"
    return "available"


def _cache_path(slug):
    return INJURY_CACHE_DIR / "{}.json".format(slug)


def _cache_is_fresh(path):
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
    return age_hours < CACHE_HOURS


def _load_cache(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Cache read failed ({}): {}", path, exc)
        return []


def _save_cache(path, data):
    try:
        path.write_text(
            json.dumps(data, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Cache write failed ({}): {}", path, exc)


def _fetch_html(url):
    """Single-page polite HTTP fetch. Returns HTML string or None."""
    time.sleep(REQUEST_DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.warning("HTTP fetch failed [{}]: {}", url, exc)
        return None


# ── Transfermarkt scraper ─────────────────────────────────────────────────────

def _scrape_transfermarkt(team_name):
    """
    Scrape the Transfermarkt injuries/absences page for a national team.
    Returns list of injury dicts or [] on failure.
    """
    info = TRANSFERMARKT_SLUGS.get(team_name)
    if not info:
        logger.debug("No Transfermarkt slug for {!r}", team_name)
        return []

    slug, team_id = info
    url = "https://www.transfermarkt.com/{}/verletzungen/verein/{}".format(
        slug, team_id
    )

    html = _fetch_html(url)
    if not html:
        return []

    soup  = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="items")
    if not table:
        logger.debug("Transfermarkt: no .items table for {}", team_name)
        return []

    results = []
    date_re = re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")

    for row in table.find_all("tr", class_=["odd", "even"]):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        try:
            name_tag = (
                row.find("a", {"class": "spielprofil_tooltip"})
                or row.find("td", class_="hauptlink")
            )
            if name_tag:
                player = name_tag.get_text(strip=True)
            else:
                player = cols[1].get_text(strip=True) if len(cols) > 1 else "Unknown"

            all_text   = " ".join(c.get_text(strip=True) for c in cols)
            status     = _normalise_status(all_text)
            dates      = date_re.findall(all_text)
            exp_return = dates[-1] if dates else "unknown"

            if player and player not in ("", "Player"):
                results.append({
                    "player":          player,
                    "status":          status,
                    "expected_return": exp_return,
                    "source_url":      url,
                })
        except Exception as exc:
            logger.debug("Transfermarkt row parse error for {}: {}", team_name, exc)

    logger.info("Transfermarkt: {} records for {}", len(results), team_name)
    return results


# ── BBC Sport fallback ────────────────────────────────────────────────────────

def _scrape_bbc_sport(team_name):
    """
    Fallback: scrape BBC Sport team page for injury mentions.
    Best-effort — BBC Sport layout changes frequently.
    Returns list of injury dicts or [].
    """
    slug      = team_name.lower().replace(" ", "-")
    url       = "https://www.bbc.co.uk/sport/football/teams/{}".format(slug)
    html      = _fetch_html(url)
    if not html:
        # Try underscore variant
        url  = "https://www.bbc.co.uk/sport/football/teams/{}".format(
            team_name.lower().replace(" ", "_")
        )
        html = _fetch_html(url)
    if not html:
        logger.debug("BBC Sport: no page for {}", team_name)
        return []

    soup    = BeautifulSoup(html, "lxml")
    results = []
    seen    = set()
    date_re = re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")
    kws     = [
        "injured", "out", "doubt", "suspended", "fitness",
        "unavailable", "ruled out", "miss", "strain", "fracture",
    ]

    for tag in soup.find_all(["p", "li", "span"]):
        text = tag.get_text(strip=True)
        if not any(kw in text.lower() for kw in kws):
            continue

        name_m = re.search(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\s+(?:is|has|was|will)", text
        )
        if not name_m:
            continue

        player = name_m.group(1).strip()
        if player in seen:
            continue
        seen.add(player)

        dates      = date_re.findall(text)
        exp_return = dates[-1] if dates else "unknown"

        results.append({
            "player":          player,
            "status":          _normalise_status(text),
            "expected_return": exp_return,
            "source_url":      url,
        })

    logger.info("BBC Sport: {} records for {}", len(results), team_name)
    return results


# ── SQLite persistence ────────────────────────────────────────────────────────

def _save_to_psych_signals(team_name, injuries):
    """
    Persist injury records to psych_signals table.
    Silently skips on any DB error to remain non-fatal.
    """
    if not injuries:
        return
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur  = conn.cursor()

        cur.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
        row     = cur.fetchone()
        team_id = row[0] if row else None

        now = datetime.utcnow().isoformat()
        for inj in injuries:
            headline = "{} -- {} (return: {})".format(
                inj["player"], inj["status"], inj["expected_return"]
            )
            cur.execute(
                """
                SELECT id FROM psych_signals
                WHERE team_id IS ?
                  AND headline = ?
                  AND risk_category = 'injury'
                  AND recorded_at >= datetime('now', '-24 hours')
                """,
                (team_id, headline),
            )
            if cur.fetchone():
                continue

            sentiment = -0.5 if inj["status"] in ("injured", "suspended") else -0.2
            severity  = (
                3 if inj["status"] == "injured"   else
                2 if inj["status"] == "doubt"     else
                4 if inj["status"] == "suspended" else 1
            )
            cur.execute(
                """
                INSERT INTO psych_signals
                    (team_id, source_url, source_type, headline, raw_text,
                     sentiment_score, risk_category, severity,
                     affects_performance, reviewed, recorded_at)
                VALUES (?, ?, 'injury_scrape', ?, ?, ?, 'injury', ?, 1, 0, ?)
                """,
                (
                    team_id,
                    inj.get("source_url", ""),
                    headline,
                    json.dumps(inj),
                    sentiment,
                    severity,
                    now,
                ),
            )

        conn.commit()
        conn.close()
        logger.debug("Saved {} injury records to psych_signals for {}", len(injuries), team_name)
    except Exception as exc:
        logger.warning("DB save failed for {} injuries: {}", team_name, exc)


# ══════════════════════════════════════════════════════════════════════════════
#  InjuryTracker
# ══════════════════════════════════════════════════════════════════════════════

class InjuryTracker:
    """
    Tracks injury and availability status for national football teams.

    Usage:
        tracker  = InjuryTracker()
        injuries = tracker.scrape_injuries("France")
        impact   = tracker.compute_squad_impact("France", squad_quality_scores)
        feature  = tracker.get_availability_feature("France")
    """

    def __init__(self):
        # In-memory cache to avoid duplicate scrapes in one session
        self._session_cache = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def scrape_injuries(self, team_name):
        """
        Scrape current injury / suspension status for a team.

        Strategy:
          1. In-memory session cache
          2. 24h on-disk file cache  (data/cache/injuries/<slug>.json)
          3. Transfermarkt injuries page
          4. BBC Sport fallback (if Transfermarkt returns nothing)
          5. Save to disk cache + psych_signals DB table

        Returns
        -------
        list of dicts:
          {
            "player":          str,
            "status":          "injured" | "doubt" | "suspended" | "available",
            "expected_return": str,
            "source_url":      str,
          }
        Falls back to [] on any scraping failure.
        """
        slug = _team_slug(team_name)

        # 1. Session cache
        if slug in self._session_cache:
            logger.debug("Session cache hit for {}", team_name)
            return self._session_cache[slug]

        # 2. Disk cache
        cache_file = _cache_path(slug)
        if _cache_is_fresh(cache_file):
            data = _load_cache(cache_file)
            if data:
                logger.debug("Disk cache hit for {} ({} records)", team_name, len(data))
                self._session_cache[slug] = data
                return data

        # 3. Transfermarkt
        try:
            injuries = _scrape_transfermarkt(team_name)
        except Exception as exc:
            logger.warning("Transfermarkt scrape error for {}: {}", team_name, exc)
            injuries = []

        # 4. BBC Sport fallback
        if not injuries:
            logger.info("Falling back to BBC Sport for {}", team_name)
            try:
                injuries = _scrape_bbc_sport(team_name)
            except Exception as exc:
                logger.warning("BBC Sport scrape error for {}: {}", team_name, exc)
                injuries = []

        # 5. Persist
        _save_cache(cache_file, injuries)
        _save_to_psych_signals(team_name, injuries)

        self._session_cache[slug] = injuries
        logger.info("scrape_injuries({}): {} records total", team_name, len(injuries))
        return injuries

    def compute_squad_impact(self, team_name, squad_quality_scores):
        """
        Compute the impact of current injuries on squad strength.

        Parameters
        ----------
        team_name            : str
        squad_quality_scores : dict  player_name -> float 0-10

        Returns
        -------
        dict with keys:
            available_quality  float  weighted mean quality of fit players
            injury_risk_score  float  0-1, higher = worse injury situation
            key_players_out    list   names of players rated > 7.5 who are out
            depth_score        float  0-1, how well backups cover the gaps
        """
        _NEUTRAL = {
            "available_quality": 5.0,
            "injury_risk_score": 0.0,
            "key_players_out":   [],
            "depth_score":       0.5,
        }

        if not squad_quality_scores:
            logger.debug("compute_squad_impact: no quality scores for {}", team_name)
            return _NEUTRAL

        try:
            injuries = self.scrape_injuries(team_name)

            if not injuries:
                ratings = list(squad_quality_scores.values())
                avg_q   = sum(ratings) / len(ratings) if ratings else 5.0
                return {
                    "available_quality": round(avg_q, 3),
                    "injury_risk_score": 0.0,
                    "key_players_out":   [],
                    "depth_score":       0.5,
                }

            out_statuses  = {"injured", "suspended", "doubt"}
            unavail_lower = {
                inj["player"].lower()
                for inj in injuries
                if inj.get("status") in out_statuses
            }

            available_ratings   = []
            unavailable_ratings = []
            key_players_out     = []

            for player, rating in squad_quality_scores.items():
                player_lo = player.lower()
                is_out = any(
                    un in player_lo or player_lo in un
                    for un in unavail_lower
                )
                if is_out:
                    unavailable_ratings.append(rating)
                    if rating > 7.5:
                        key_players_out.append(player)
                else:
                    available_ratings.append(rating)

            total_players = len(squad_quality_scores)
            n_out         = len(unavailable_ratings)
            n_avail       = len(available_ratings)

            available_quality = (
                round(sum(available_ratings) / n_avail, 3) if n_avail else 0.0
            )

            fraction_out  = n_out / total_players if total_players else 0.0
            all_ratings   = list(squad_quality_scores.values())
            total_quality = sum(all_ratings) if all_ratings else 1.0
            quality_loss  = sum(unavailable_ratings) / total_quality if total_quality > 0 else 0.0

            injury_risk_score = round(
                min(1.0, 0.4 * fraction_out + 0.6 * quality_loss), 4
            )

            all_avg     = sum(all_ratings) / len(all_ratings) if all_ratings else 5.0
            depth_score = round(
                max(0.0, min(1.0, available_quality / all_avg if all_avg > 0 else 0.5)), 4
            )

            return {
                "available_quality": available_quality,
                "injury_risk_score": injury_risk_score,
                "key_players_out":   key_players_out,
                "depth_score":       depth_score,
            }

        except Exception as exc:
            logger.error("compute_squad_impact failed for {}: {}", team_name, exc)
            return _NEUTRAL

    def get_availability_feature(self, team_name):
        """
        Return a single 0-1 availability feature for the ML feature vector.

        1.0 = fully fit squad
        0.0 = catastrophic injury crisis
        0.5 = fallback value on any failure
        """
        try:
            injuries = self.scrape_injuries(team_name)

            if not injuries:
                return 1.0

            status_weight = {
                "injured":   1.0,
                "suspended": 0.8,
                "doubt":     0.4,
                "available": 0.0,
            }
            weight_sum = sum(
                status_weight.get(inj.get("status", "available"), 0.0)
                for inj in injuries
            )

            assumed_squad_size = 26
            penalty  = min(1.0, weight_sum / assumed_squad_size)
            feature  = round(1.0 - penalty, 4)

            logger.debug(
                "get_availability_feature({}): {} (penalty={:.4f}, n={})",
                team_name, feature, penalty, len(injuries),
            )
            return feature

        except Exception as exc:
            logger.warning("get_availability_feature failed for {}: {}", team_name, exc)
            return 0.5

    def clear_session_cache(self):
        """Clear in-memory session cache (does not remove disk cache)."""
        self._session_cache.clear()
        logger.debug("InjuryTracker session cache cleared.")


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.add("logs/injury_tracker_{time}.log", rotation="10 MB", level="DEBUG")

    tracker    = InjuryTracker()
    test_teams = ["France", "England", "Brazil"]

    for team in test_teams:
        print("\n" + "=" * 60)
        print("Team: {}".format(team))

        injuries = tracker.scrape_injuries(team)
        print("Injuries scraped: {}".format(len(injuries)))
        for inj in injuries[:5]:
            print("  {:<25s} | {:<10s} | return: {}".format(
                inj["player"], inj["status"], inj["expected_return"]
            ))

        dummy_scores = {
            "Kylian Mbappe":      9.2,
            "Antoine Griezmann":  8.1,
            "Aurelien Tchouameni":7.8,
            "Marcus Rashford":    8.0,
            "Jude Bellingham":    9.0,
            "Phil Foden":         8.5,
            "Vinicius Junior":    9.1,
            "Rodrygo":            8.3,
        }

        impact = tracker.compute_squad_impact(team, dummy_scores)
        print("Squad impact:")
        print("  available_quality : {}".format(impact["available_quality"]))
        print("  injury_risk_score : {}".format(impact["injury_risk_score"]))
        print("  key_players_out   : {}".format(impact["key_players_out"]))
        print("  depth_score       : {}".format(impact["depth_score"]))

        feat = tracker.get_availability_feature(team)
        print("Availability feature: {}".format(feat))
