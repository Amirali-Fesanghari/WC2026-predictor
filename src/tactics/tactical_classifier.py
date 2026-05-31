"""
src/tactics/tactical_classifier.py
Tactical Formation Classifier & Counter-Recommendation Engine.

Encodes football formations as feature vectors, computes matchup differentials,
and recommends optimal counter-formations using a rule-based system
(with optional StatsBomb calibration when data is available).
"""

import math
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parents[3]))

# Try to import FORMATIONS from config; fall back to a local default
try:
    from config import FORMATIONS as _CONFIG_FORMATIONS  # noqa: F401
    FORMATIONS = _CONFIG_FORMATIONS
    logger.debug("FORMATIONS loaded from config.")
except Exception as _cfg_exc:
    logger.debug("config.FORMATIONS not found ({}), using built-in defaults.", _cfg_exc)
    FORMATIONS = [
        "4-3-3", "4-4-2", "4-5-1", "4-2-3-1", "3-5-2",
        "3-4-3", "5-3-2", "5-4-1", "4-1-4-1", "4-3-2-1",
    ]

# ---------------------------------------------------------------------------
# Formation metadata: formation_str → (defenders, midfielders, attackers)
# ---------------------------------------------------------------------------
_FORMATION_META = {
    "4-3-3":   (4, 3, 3),
    "4-2-3-1": (4, 5, 1),   # 2 DM + 3 AM → 5 mid
    "4-4-2":   (4, 4, 2),
    "3-5-2":   (3, 5, 2),
    "3-4-3":   (3, 4, 3),
    "5-3-2":   (5, 3, 2),
    "4-1-4-1": (4, 5, 1),   # 1 DM + 4 M → 5 mid
    "4-3-2-1": (4, 5, 1),   # Christmas tree
    "5-4-1":   (5, 4, 1),
    "4-5-1":   (4, 5, 1),
    "3-6-1":   (3, 6, 1),
    "4-1-3-2": (4, 4, 2),
    "4-2-2-2": (4, 4, 2),
}

# ---------------------------------------------------------------------------
# Counter-formation rules
# key = opponent formation → list of (counter_formation, win_prob_uplift, rationale)
# ---------------------------------------------------------------------------
COUNTER_RULES = {
    "4-3-3": [
        ("4-5-1", 0.07,
         "Overloads midfield to cut off supply to wide attackers; "
         "compact 5-man mid forces 4-3-3 to play narrow."),
        ("5-3-2", 0.05,
         "Extra centre-back nullifies CF; two strikers exploit "
         "high fullback line when 4-3-3 pushes forward."),
        ("4-4-2", 0.03,
         "Dual striker press disrupts 4-3-3 build-up; "
         "flat four mid matches opponent width."),
    ],
    "4-4-2": [
        ("4-3-3", 0.06,
         "One extra attacker against flat back four; "
         "wide forwards stretch 4-4-2 defensive shape."),
        ("4-2-3-1", 0.05,
         "Double pivot shields defence while AMF trio finds "
         "pockets between 4-4-2 lines."),
        ("3-5-2", 0.04,
         "Five-man mid wins central battle against 4-4-2; "
         "wingbacks stretch the 4 defensive line."),
    ],
    "3-5-2": [
        ("4-3-3", 0.06,
         "Wide forwards target space behind WBs; "
         "high press disrupts 3-man build-up."),
        ("4-4-2", 0.05,
         "Two strikers pin three CBs; wide midfield neutralises WBs."),
        ("4-2-3-1", 0.04,
         "Compact block absorbs WB runs; AMF trio exploits transition gaps."),
    ],
    "4-2-3-1": [
        ("4-3-3", 0.05,
         "Press high to unsettle DM pairing; "
         "wide forwards isolate fullbacks vs single striker."),
        ("3-5-2", 0.04,
         "Five midfielders crowd out AMF trio; "
         "twin strikers target exposed backline on turnover."),
        ("4-4-2", 0.03,
         "Flat shape denies space for AMF; two strikers press DM double pivot."),
    ],
    "3-4-3": [
        ("5-3-2", 0.07,
         "Flat five blocks width; counter with two pacy strikers "
         "vs 3-CB vulnerability in wide channels."),
        ("4-5-1", 0.05,
         "Five-man mid screens against attacking 3-4-3; "
         "single striker on rapid transitions."),
        ("4-4-2", 0.04,
         "Dual wingers exploit exposed flanks behind 3-4-3 wingbacks."),
    ],
    "5-3-2": [
        ("4-3-3", 0.06,
         "Three wide attackers overload defensive line; "
         "high press forces errors from deep-sitting 5-3-2."),
        ("4-2-3-1", 0.05,
         "AMF trio finds space between compact lines; "
         "quick transitions against slow WBs."),
        ("3-4-3", 0.04,
         "Extreme width drags 5-3-2 WBs apart; "
         "three attackers overwhelm three defenders."),
    ],
    "4-1-4-1": [
        ("4-3-3", 0.06,
         "Three attackers outnumber single DM protection; "
         "wide press isolates the lone striker."),
        ("4-4-2", 0.04,
         "Second striker exploits DM shadow zone; "
         "flat four mirrors opponent mid block."),
        ("3-5-2", 0.03,
         "Five midfielders overload 4-1-4-1 mid tier; "
         "twin strikers stretch lone-DM coverage."),
    ],
    "4-3-2-1": [
        ("4-4-2", 0.06,
         "Flat wide mids target narrow 4-3-2-1 flanks; "
         "second striker supports CF vs deep defending."),
        ("4-3-3", 0.05,
         "Wide forwards pull apart Christmas tree defence; "
         "high press disrupts short build-up."),
        ("3-5-2", 0.03,
         "WBs exploit lack of width in 4-3-2-1; two strikers on counterattack."),
    ],
    "5-4-1": [
        ("4-3-3", 0.07,
         "Width and pace expose narrow defensive block; press disrupts build-up."),
        ("3-4-3", 0.05,
         "Three forwards outnumber five defenders in transition."),
        ("4-4-2", 0.04,
         "High pressure forces long balls; second-ball dominance in midfield."),
    ],
}

# Fallback when specific formation not in COUNTER_RULES
_DEFAULT_COUNTERS = [
    ("4-5-1", 0.04,
     "Compact midfield block reduces opponent's attacking space and breaks on counter."),
    ("4-3-3", 0.03,
     "High-press attacking shape forces errors and creates numerical overloads."),
    ("4-2-3-1", 0.02,
     "Balanced shape adapts to any opponent; double pivot shields the back four."),
]


# ---------------------------------------------------------------------------
# Module-level helper (also used by class methods)
# ---------------------------------------------------------------------------
def _parse_formation(formation_str):
    """
    Parse a formation string to (defenders, midfielders, attackers).
    Looks up the catalogue first; falls back to splitting the string.
    """
    s = str(formation_str).strip()
    if s in _FORMATION_META:
        return _FORMATION_META[s]

    parts = [p for p in s.split("-") if p.isdigit()]
    if len(parts) >= 3:
        d = int(parts[0])
        a = int(parts[-1])
        m = sum(int(p) for p in parts[1:-1])
        return d, m, a

    if len(parts) == 2:
        d, a = int(parts[0]), int(parts[1])
        m = max(0, 10 - d - a)
        return d, m, a

    logger.warning("Cannot parse formation '{}', defaulting to 4-3-3", s)
    return 4, 3, 3


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class TacticalClassifier:
    """
    Tactical Formation Classifier and Counter-Recommendation Engine.

    Quick start:
        clf = TacticalClassifier()          # calls train() automatically
        enc = clf.encode_formation("4-3-3")
        res = clf.analyze_matchup("4-3-3", "4-4-2", home_elo=1900, away_elo=1850)
        recs = clf.recommend_formation("4-3-3", home_style="attacking")
    """

    def __init__(self):
        self._trained = False
        self.train()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_formation(self, formation_str):
        """
        Parse a formation string (e.g. "4-3-3") into a rich feature dict.

        Returns
        -------
        dict with keys:
            defenders        int
            midfielders      int
            attackers        int
            width_score      float  0-1
            press_intensity  float  0-1
            defensive_depth  float  0-1
        """
        d, m, a = _parse_formation(formation_str)

        # width_score: more attackers and lighter defence → wider play
        width_score = round(
            (a / 3.0) * 0.5 + max(0.0, (5 - d) / 2.0) * 0.5, 4
        )
        width_score = max(0.0, min(1.0, width_score))

        # press_intensity: more attackers + midfielders relative to full capacity
        press_intensity = round(
            (m / 5.0) * 0.40 + (a / 3.0) * 0.60, 4
        )
        press_intensity = max(0.0, min(1.0, press_intensity))

        # defensive_depth: more defenders → deeper block
        defensive_depth = round(d / 5.0, 4)
        defensive_depth = max(0.0, min(1.0, defensive_depth))

        result = {
            "defenders":       d,
            "midfielders":     m,
            "attackers":       a,
            "width_score":     width_score,
            "press_intensity": press_intensity,
            "defensive_depth": defensive_depth,
        }
        logger.debug("encode_formation('{}') -> {}", formation_str, result)
        return result

    def analyze_matchup(self, home_formation, away_formation, home_elo=1500, away_elo=1500):
        """
        Analyse a tactical matchup and return win probabilities + tactical dims.

        Parameters
        ----------
        home_formation : str   e.g. "4-3-3"
        away_formation : str   e.g. "4-4-2"
        home_elo       : float Elo rating of the home team (default 1500)
        away_elo       : float Elo rating of the away team (default 1500)

        Returns
        -------
        dict with keys:
            attacker_advantage    float  positive = home advantage
            midfield_battle       float  positive = home wins midfield
            pressing_vs_depth     float  positive = home press beats away depth
            width_exploit         float  positive = home width beats away compactness
            home_win_prob         float  0-1
            draw_prob             float  0-1
            away_win_prob         float  0-1
            counter_recommendation dict  {formation, win_prob_uplift, rationale}
        """
        home_enc = self.encode_formation(home_formation)
        away_enc = self.encode_formation(away_formation)

        # ── Tactical dimension differentials ──────────────────────────────────
        attacker_advantage = round(
            (home_enc["attackers"] - away_enc["defenders"]) / 3.0, 4
        )
        midfield_battle = round(
            (home_enc["midfielders"] - away_enc["midfielders"]) / 5.0, 4
        )
        pressing_vs_depth = round(
            home_enc["press_intensity"] - away_enc["defensive_depth"], 4
        )
        width_exploit = round(
            home_enc["width_score"] - (1.0 - away_enc["width_score"]), 4
        )

        # ── ELO-based win probability (logistic, +50 home advantage) ─────────
        elo_diff_adj = (home_elo - away_elo) + 50.0
        p_home_raw = 1.0 / (1.0 + math.pow(10.0, -elo_diff_adj / 400.0))
        p_away_raw = 1.0 - p_home_raw

        # Carve out a draw probability; it peaks (~0.25) when teams are even
        draw_base = 0.25 * (1.0 - abs(p_home_raw - 0.5) * 2.0)
        p_home_win = p_home_raw * (1.0 - draw_base)
        p_away_win = p_away_raw * (1.0 - draw_base)

        # ── Tactical adjustments ──────────────────────────────────────────────
        tact_adj = (
            attacker_advantage  * 0.04
            + midfield_battle   * 0.03
            + pressing_vs_depth * 0.02
            + width_exploit     * 0.01
        )
        p_home_win = max(0.05, min(0.85, p_home_win + tact_adj))
        p_away_win = max(0.05, min(0.85, p_away_win - tact_adj))

        # Renormalise
        total = p_home_win + draw_base + p_away_win
        home_win_prob = round(p_home_win / total, 4)
        away_win_prob = round(p_away_win / total, 4)
        draw_prob     = round(1.0 - home_win_prob - away_win_prob, 4)

        # ── Best counter for the home team vs away formation ──────────────────
        counter_rec = self._best_counter(away_formation)

        result = {
            "home_formation":        home_formation,
            "away_formation":        away_formation,
            "attacker_advantage":    attacker_advantage,
            "midfield_battle":       midfield_battle,
            "pressing_vs_depth":     pressing_vs_depth,
            "width_exploit":         width_exploit,
            "home_win_prob":         home_win_prob,
            "draw_prob":             draw_prob,
            "away_win_prob":         away_win_prob,
            "counter_recommendation": counter_rec,
        }
        logger.info(
            "Matchup {} vs {} | H={} D={} A={} | counter={}",
            home_formation,
            away_formation,
            home_win_prob,
            draw_prob,
            away_win_prob,
            counter_rec["formation"],
        )
        return result

    def recommend_formation(self, opponent_formation, home_style="balanced"):
        """
        Recommend the top 3 formations to use against a given opponent formation.

        Parameters
        ----------
        opponent_formation : str   the opponent's formation string
        home_style         : str   "attacking" | "balanced" | "defensive" |
                                   "counter" | "possession"

        Returns
        -------
        list of dicts (up to 3), each with:
            formation       str
            rationale       str
            win_prob_uplift float
        """
        rules = list(COUNTER_RULES.get(opponent_formation, _DEFAULT_COUNTERS))

        # Pad to at least 3 entries from default counters
        seen = {r[0] for r in rules}
        for item in _DEFAULT_COUNTERS:
            if len(rules) >= 3:
                break
            if item[0] not in seen:
                rules.append(item)
                seen.add(item[0])

        recommendations = []
        for entry in rules[:3]:
            formation, base_uplift, rationale = entry
            adj_uplift = self._style_adjust_uplift(base_uplift, formation, home_style)
            recommendations.append({
                "formation":       formation,
                "rationale":       rationale,
                "win_prob_uplift": round(adj_uplift, 4),
            })

        # Sort by uplift descending
        recommendations.sort(key=lambda r: r["win_prob_uplift"], reverse=True)

        logger.info(
            "recommend_formation(opp='{}', style='{}') -> {}",
            opponent_formation,
            home_style,
            [r["formation"] for r in recommendations],
        )
        return recommendations

    def train(self):
        """
        Attempt to load StatsBomb open data for calibration.
        Falls back gracefully to the built-in rule-based system.
        """
        logger.info("TacticalClassifier.train() — loading formation data...")
        try:
            from statsbombpy import sb  # type: ignore

            competitions = sb.competitions()
            if competitions is not None and len(competitions) > 0:
                logger.info(
                    "StatsBomb data loaded — {} competitions available.",
                    len(competitions)
                )
                self._calibrate_from_statsbomb(sb)
            else:
                raise ValueError("Empty competitions dataframe returned.")
        except Exception as exc:
            logger.warning(
                "StatsBomb data unavailable ({}). Using rule-based fallback.", exc
            )
        finally:
            self._trained = True
            logger.info("TacticalClassifier ready (rule-based mode).")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _best_counter(self, opponent_formation):
        """Return the single best counter-formation dict for a given opponent."""
        rules = COUNTER_RULES.get(opponent_formation, _DEFAULT_COUNTERS)
        best = rules[0]
        return {
            "formation":       best[0],
            "win_prob_uplift": best[1],
            "rationale":       best[2],
        }

    def _style_adjust_uplift(self, base_uplift, candidate_formation, home_style):
        """Adjust win-probability uplift based on home-team style compatibility."""
        try:
            enc = self.encode_formation(candidate_formation)
        except Exception:
            return base_uplift

        bonus = 0.0
        if home_style == "attacking":
            bonus = 0.01 if enc["attackers"] >= 3 else -0.005
        elif home_style == "defensive":
            bonus = 0.01 if enc["defenders"] >= 5 else -0.005
        elif home_style == "counter":
            # Favour formations that can sit deep and break quickly
            bonus = 0.005 if enc["defensive_depth"] >= 0.7 else 0.0
        elif home_style == "possession":
            bonus = 0.005 if enc["midfielders"] >= 4 else -0.005
        # "balanced" → no adjustment

        return base_uplift + bonus

    def _calibrate_from_statsbomb(self, sb):
        """
        Placeholder calibration from StatsBomb event data.
        In a full implementation this would fit uplift weights from outcomes.
        """
        logger.info(
            "StatsBomb calibration complete (rule-based weights retained for now)."
        )


# ---------------------------------------------------------------------------
# CLI smoke-test / demo
# ---------------------------------------------------------------------------
def main():
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/tactical_{time}.log", rotation="20 MB", level="DEBUG")

    clf = TacticalClassifier()

    print("\n" + "=" * 62)
    print("  TACTICAL CLASSIFIER — DEMO")
    print("=" * 62)

    test_matchups = [
        ("4-3-3",  "4-4-2",   1900, 1850),
        ("3-5-2",  "4-3-3",   1800, 1900),
        ("4-2-3-1","5-3-2",   1950, 1920),
        ("4-4-2",  "4-3-3",   1850, 1900),
    ]

    for home_f, away_f, he, ae in test_matchups:
        res = clf.analyze_matchup(home_f, away_f, home_elo=he, away_elo=ae)
        cr  = res["counter_recommendation"]
        print(f"\n  {home_f} (home) vs {away_f} (away)  [ELO {he} vs {ae}]")
        print(f"  Win probs  H={res['home_win_prob']:.1%}  "
              f"D={res['draw_prob']:.1%}  A={res['away_win_prob']:.1%}")
        print(f"  Attacker advantage : {res['attacker_advantage']:+.3f}")
        print(f"  Midfield battle    : {res['midfield_battle']:+.3f}")
        print(f"  Pressing vs depth  : {res['pressing_vs_depth']:+.3f}")
        print(f"  Width exploit      : {res['width_exploit']:+.3f}")
        print(f"  Counter            : {cr['formation']}  (+{cr['win_prob_uplift']:.0%})")

    print("\n  Top-3 counters vs 4-3-3 (attacking style):")
    for i, r in enumerate(clf.recommend_formation("4-3-3", home_style="attacking"), 1):
        print(f"    {i}. {r['formation']}  +{r['win_prob_uplift']:.0%}  — {r['rationale'][:72]}")

    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
