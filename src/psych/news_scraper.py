"""
src/psych/news_scraper.py
Scrapes soccer news from RSS feeds (BBC Sport, ESPN FC, FIFA) and fetches
full article body text. Results are cached as JSON files under
data/cache/news/ to avoid re-scraping within a 6-hour window.

Dependencies (pip-installable):
    feedparser, requests, beautifulsoup4, loguru
"""

import json
import time
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))
try:
    from config import NEWS_CACHE_DIR
except ImportError:
    NEWS_CACHE_DIR = Path(__file__).parents[2] / "data" / "cache" / "news"


# ── Constants ─────────────────────────────────────────────────────────────────

CACHE_TTL_HOURS = 6
RATE_LIMIT_SECONDS = 2

RSS_FEEDS = {
    "bbc_sport": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "espn_fc":   "https://www.espn.com/espn/rss/soccer/news",
    "fifa":      "https://www.fifa.com/fifa-world-ranking/news.rss",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(text):
    """Turn an arbitrary string into a filesystem-safe slug."""
    slug = re.sub(r"[^\w\-]", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80]


def _now_utc():
    return datetime.now(timezone.utc)


def _cache_path(key):
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return NEWS_CACHE_DIR / f"{_safe_filename(key)}.json"


def _load_cache(key):
    """Return cached list of articles if cache is fresh, else None."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        cached_at = datetime.fromisoformat(payload["cached_at"])
        age_hours = (_now_utc() - cached_at).total_seconds() / 3600
        if age_hours < CACHE_TTL_HOURS:
            logger.debug(f"Cache hit for '{key}' ({age_hours:.1f}h old)")
            return payload["articles"]
        logger.debug(f"Cache stale for '{key}' ({age_hours:.1f}h old)")
    except Exception as exc:
        logger.warning(f"Cache read error for '{key}': {exc}")
    return None


def _save_cache(key, articles):
    path = _cache_path(key)
    payload = {
        "cached_at": _now_utc().isoformat(),
        "articles": articles,
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        logger.debug(f"Cached {len(articles)} articles for '{key}'")
    except Exception as exc:
        logger.warning(f"Cache write error for '{key}': {exc}")


def _fetch_article_text(url):
    """Fetch the full body text of a single article URL. Returns '' on failure."""
    try:
        time.sleep(RATE_LIMIT_SECONDS)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Try common article containers first
        article_body = (
            soup.find("article")
            or soup.find("div", {"class": re.compile(r"article[_-]?body|story[_-]?body|content[_-]?body", re.I)})
            or soup.find("div", {"id": re.compile(r"article|story|content", re.I)})
            or soup.find("main")
            or soup.body
        )

        if article_body:
            paragraphs = article_body.find_all("p")
            text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        else:
            text = soup.get_text(separator=" ", strip=True)

        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    except requests.exceptions.RequestException as exc:
        logger.warning(f"Failed to fetch article body from {url}: {exc}")
        return ""
    except Exception as exc:
        logger.warning(f"Unexpected error parsing {url}: {exc}")
        return ""


def _parse_entry_date(entry):
    """Return a datetime (UTC-aware) from a feedparser entry, or epoch on failure."""
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _query_matches(text, query_terms):
    """Return True if any query term appears in text (case-insensitive)."""
    lower = text.lower()
    return any(term.lower() in lower for term in query_terms)


def _scrape_feeds(query_terms, days_back):
    """
    Pull all RSS feeds, filter entries that mention any query term within
    days_back days, then fetch their full body text.

    Returns a list of article dicts.
    """
    cutoff = _now_utc() - timedelta(days=days_back)
    articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        logger.info(f"Parsing RSS feed: {source_name}")
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            feed = feedparser.parse(feed_url)
        except Exception as exc:
            logger.warning(f"RSS parse error for {source_name}: {exc}")
            continue

        if feed.bozo and feed.bozo_exception:
            # bozo = malformed feed; often still parseable
            logger.debug(f"Feed {source_name} bozo flag: {feed.bozo_exception}")

        for entry in feed.entries:
            pub_date = _parse_entry_date(entry)
            if pub_date < cutoff:
                continue

            title = entry.get("title", "")
            summary = entry.get("summary", "")
            url = entry.get("link", "")

            # Quick relevance check on title + summary before full fetch
            combined_preview = f"{title} {summary}"
            if not _query_matches(combined_preview, query_terms):
                continue

            logger.debug(f"  Fetching: {title[:60]}")
            body_text = _fetch_article_text(url)

            # If body is very short, fall back to summary
            if len(body_text) < 100:
                body_text = summary

            articles.append({
                "title":  title,
                "text":   body_text,
                "url":    url,
                "date":   pub_date.isoformat(),
                "source": source_name,
            })

    logger.info(f"Scraped {len(articles)} relevant articles for query {query_terms}")
    return articles


# ── Main class ────────────────────────────────────────────────────────────────

class NewsScraper:
    """
    Scrapes soccer news from BBC Sport, ESPN FC, and FIFA RSS feeds.

    Usage
    -----
    scraper = NewsScraper()
    articles = scraper.scrape_team("Brazil", days_back=7)
    articles = scraper.scrape_player("Vinicius", "Brazil", days_back=5)

    Each article is a dict with keys: title, text, url, date, source.
    """

    def __init__(self):
        NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("NewsScraper initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    def scrape_team(self, team_name, days_back=7):
        """
        Scrape news articles that mention team_name from all RSS sources.

        Parameters
        ----------
        team_name : str
            The team name to search for (e.g. "Brazil", "France").
        days_back : int
            How many days back to look in the feed (default 7).

        Returns
        -------
        list of dict  [{title, text, url, date, source}, ...]
        Empty list on network failure or no matching articles.
        """
        cache_key = f"team_{team_name}_{days_back}"
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Scraping news for team: {team_name} (days_back={days_back})")
        try:
            query_terms = [team_name]
            articles = _scrape_feeds(query_terms, days_back)
        except Exception as exc:
            logger.error(f"scrape_team failed for '{team_name}': {exc}")
            return []

        _save_cache(cache_key, articles)
        return articles

    def scrape_player(self, player_name, team_name, days_back=7):
        """
        Scrape news articles that mention player_name (optionally also team_name).

        Parameters
        ----------
        player_name : str
            Player's name (e.g. "Kylian Mbappe").
        team_name : str
            Player's team — used as an additional relevance hint.
        days_back : int
            How many days back to look (default 7).

        Returns
        -------
        list of dict  [{title, text, url, date, source}, ...]
        Empty list on network failure or no matching articles.
        """
        cache_key = f"player_{player_name}_{team_name}_{days_back}"
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Scraping news for player: {player_name} ({team_name}, days_back={days_back})")
        try:
            # First name, last name, and full name all increase recall
            name_parts = player_name.strip().split()
            query_terms = [player_name]
            if len(name_parts) > 1:
                query_terms += name_parts          # "Kylian", "Mbappe"
            articles = _scrape_feeds(query_terms, days_back)

            # Secondary filter: keep only articles that genuinely mention the
            # player name (avoid false positives caught by first/last name alone)
            def _relevant(article):
                full_text = (article["title"] + " " + article["text"]).lower()
                if player_name.lower() in full_text:
                    return True
                if team_name.lower() in full_text and any(
                    p.lower() in full_text for p in name_parts
                ):
                    return True
                return False

            articles = [a for a in articles if _relevant(a)]
            logger.info(f"After player filter: {len(articles)} articles for {player_name}")

        except Exception as exc:
            logger.error(f"scrape_player failed for '{player_name}': {exc}")
            return []

        _save_cache(cache_key, articles)
        return articles


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scraper = NewsScraper()
    articles = scraper.scrape_team("France", days_back=7)
    print(f"\nFound {len(articles)} articles about France:\n")
    for i, article in enumerate(articles, 1):
        print(f"  {i:>2}. [{article['source']}] {article['title']}")
    if not articles:
        print("  (no articles found — network may be unavailable or no recent news)")
