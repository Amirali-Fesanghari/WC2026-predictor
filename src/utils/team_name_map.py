"""
src/utils/team_name_map.py
Normalises team names across all data sources.

Problem: StatsBomb calls them "United States", football-data calls them "USA",
FBref calls them "USMNT", the open dataset calls them "United States".
This module is the single resolver for all of that.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import TEAM_NAME_ALIASES, WC_2026_TEAMS

# Build reverse lookup: canonical → canonical (identity) + all aliases → canonical
_LOOKUP: dict[str, str] = {}

# Identity entries (canonical names map to themselves)
for team in WC_2026_TEAMS:
    _LOOKUP[team.lower()] = team

# Alias entries
for alias, canonical in TEAM_NAME_ALIASES.items():
    _LOOKUP[alias.lower()] = canonical


def normalize(name: str) -> str:
    """
    Resolve any team name variant to our canonical name.
    Returns the canonical name, or the original string if not found.

    Examples:
        normalize("USA")          → "United States"
        normalize("Korea Republic") → "South Korea"
        normalize("Czechia")      → "Czech Republic"
        normalize("Argentina")    → "Argentina"
    """
    if not name:
        return name
    result = _LOOKUP.get(name.lower().strip())
    if result:
        return result
    # Try partial match for edge cases (e.g. "Iran (Islamic Republic of)")
    for key, canonical in _LOOKUP.items():
        if key in name.lower() or name.lower() in key:
            return canonical
    return name  # unknown team — return as-is, log downstream


def is_wc2026_team(name: str) -> bool:
    """Check if a team (any variant) is in the 2026 WC."""
    return normalize(name) in WC_2026_TEAMS


def get_confederation(team: str) -> str:
    """Return the confederation for a WC 2026 team."""
    canonical = normalize(team)
    afc  = ["Australia","Iran","Iraq","Japan","Jordan","Qatar",
            "Saudi Arabia","South Korea","Uzbekistan"]
    caf  = ["Algeria","Cape Verde","DR Congo","Egypt","Ghana",
            "Ivory Coast","Morocco","Senegal","South Africa","Tunisia"]
    conc = ["Canada","Curacao","Haiti","Mexico","Panama","United States"]
    csbl = ["Argentina","Brazil","Colombia","Ecuador","Paraguay","Uruguay"]
    ofc  = ["New Zealand"]
    uefa = ["Austria","Belgium","Bosnia and Herzegovina","Croatia",
            "Czech Republic","England","France","Germany","Netherlands",
            "Norway","Portugal","Scotland","Spain","Sweden","Switzerland","Turkey"]
    for conf, teams in [("AFC",afc),("CAF",caf),("CONCACAF",conc),
                        ("CONMEBOL",csbl),("OFC",ofc),("UEFA",uefa)]:
        if canonical in teams:
            return conf
    return "UNKNOWN"


if __name__ == "__main__":
    tests = [
        "USA", "Korea Republic", "Czechia", "Cote d'Ivoire",
        "IR Iran", "Bosnia-Herzegovina", "Holland", "Argentina",
        "Curacao", "DR Congo", "New Zealand", "Uzbekistan",
    ]
    print("Team name normalisation tests:")
    for t in tests:
        canonical = normalize(t)
        wc = "✓ WC2026" if is_wc2026_team(t) else "✗ not in WC"
        conf = get_confederation(t)
        print(f"  {t:35s} → {canonical:30s} [{conf}] {wc}")
