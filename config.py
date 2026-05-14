"""
config.py — central configuration for WC2026 Predictor
All paths, constants, and API keys live here.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR= DATA_DIR / "processed"
CACHE_DIR    = DATA_DIR / "cache"
DB_PATH      = DATA_DIR / "wc2026.db"

# ── API keys (set in .env, never hardcode) ───────────────────
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
# Free key at: https://www.football-data.org/client/register
# StatsBomb free data needs no key.

# ── ELO tuning constants ─────────────────────────────────────
ELO_INITIAL        = 1500     # starting rating for any team
ELO_K_FRIENDLY     = 20       # low weight — friendlies are noisy
ELO_K_QUALIFIER    = 40       # WC qualifiers
ELO_K_TOURNAMENT   = 60       # group stage
ELO_K_KNOCKOUT     = 80       # knockout rounds (higher stakes)
ELO_HOME_ADVANTAGE = 100      # for neutral-ground WC: set to 0
ELO_DECAY_RATE     = 0.99     # per-month decay toward 1500 (inactivity)

# ── football-data.org competition IDs ────────────────────────
COMPETITIONS = {
    "WC_2022":        2000,
    "WC_2018":        2000,   # same endpoint, filter by season
    "UEFA_EURO_2024": 2018,
    "COPA_2024":      2152,
    "AFC_ASIAN_2023": 2344,
    "AFCON_2023":     2005,
}

# ── StatsBomb competition IDs ────────────────────────────────
SB_COMPETITIONS = {
    "WC_2022": {"competition_id": 43, "season_id": 106},
    "WC_2018": {"competition_id": 43, "season_id": 3},
    "WC_2014": {"competition_id": 43, "season_id": 9},
}

# ── 2026 WC qualified teams (as of early 2026) ───────────────
WC_2026_TEAMS = [
    # AFC (9)
    "Australia", "Iran", "Iraq", "Japan", "Jordan",
    "Qatar", "Saudi Arabia", "South Korea", "Uzbekistan",
    # CAF (10)
    "Algeria", "Cape Verde", "DR Congo", "Egypt", "Ghana",
    "Ivory Coast", "Morocco", "Senegal", "South Africa", "Tunisia",
    # CONCACAF (6)
    "Canada", "Curacao", "Haiti", "Mexico", "Panama", "United States",
    # CONMEBOL (6)
    "Argentina", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay",
    # OFC (1)
    "New Zealand",
    # UEFA (16)
    "Austria", "Belgium", "Bosnia and Herzegovina", "Croatia",
    "Czech Republic", "England", "France", "Germany", "Netherlands",
    "Norway", "Portugal", "Scotland", "Spain", "Sweden",
    "Switzerland", "Turkey",
]

# Name aliases: how different data sources refer to the same team
TEAM_NAME_ALIASES = {
    "USA":                       "United States",
    "US":                        "United States",
    "United States of America":  "United States",
    "Korea Republic":            "South Korea",
    "Republic of Korea":         "South Korea",
    "Korea":                     "South Korea",
    "Cote d'Ivoire":           "Ivory Coast",
    "Ivory Coast":               "Ivory Coast",
    "Czechia":                   "Czech Republic",
    "Czech Rep.":                "Czech Republic",
    "Congo DR":                  "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia-Herzegovina":        "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":      "Bosnia and Herzegovina",
    "Holland":                   "Netherlands",
    "Curacao":                   "Curacao",
    "Curacoa":                   "Curacao",
    "IR Iran":                   "Iran",
}

# ── Tactical formations library ──────────────────────────────
FORMATIONS = [
    "4-3-3", "4-2-3-1", "4-4-2", "3-5-2",
    "3-4-3", "5-3-2", "4-1-4-1", "4-3-2-1",
]

# ── Psych risk keywords (seed list, expandable) ──────────────
PSYCH_NEGATIVE_KEYWORDS = [
    "injury", "injured", "doubt", "suspended", "ban", "banned",
    "controversy", "arrested", "family", "funeral", "mourning",
    "illness", "ill", "sick", "crisis", "conflict", "war",
    "protest", "political", "divorce", "scandal", "fired",
    "dropped", "benched", "rift", "fallout", "argument",
]
PSYCH_POSITIVE_KEYWORDS = [
    "confident", "fit", "recovered", "motivated", "united",
    "captain", "leader", "record", "milestone", "awarded",
]

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = "DEBUG"
