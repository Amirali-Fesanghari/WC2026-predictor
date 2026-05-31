"""
src/simulation/tournament_simulator.py

Monte Carlo simulation of the FIFA World Cup 2026.

WC 2026: 48 teams, 12 groups of 4.
  Group stage  -> top-2 per group (24 teams) + best 8 third-placed teams = 32 advance.
  Knockout     -> Round of 32 -> Round of 16 -> Quarter-finals -> Semi-finals -> Final.

Predictor contract
  The caller injects a predictor callable:
      win_prob, draw_prob, away_win_prob = predictor(home_team, away_team, stage)
  where all three float values sum to 1.0.

  If no predictor is provided a built-in ELO predictor backed by DEFAULT_ELO is used.
"""

import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

# allow running as __main__ from repo root
sys.path.insert(0, str(Path(__file__).parents[3]))

# ---------------------------------------------------------------------------
# WC 2026 group definitions
# ---------------------------------------------------------------------------

WC2026_GROUPS = {
    "A": ["United States", "Panama", "Uruguay", "Bosnia and Herzegovina"],
    "B": ["Mexico", "Ecuador", "Germany", "New Zealand"],
    "C": ["Argentina", "Colombia", "Morocco", "Ukraine"],
    "D": ["England", "Senegal", "Serbia", "Netherlands"],
    "E": ["Spain", "Brazil", "Japan", "South Africa"],
    "F": ["France", "Algeria", "South Korea", "Belgium"],
    "G": ["Portugal", "Ivory Coast", "Turkey", "Paraguay"],
    "H": ["Croatia", "DR Congo", "Colombia", "Canada"],
    "I": ["Australia", "Norway", "Egypt", "Switzerland"],
    "J": ["Saudi Arabia", "Czech Republic", "Ghana", "Scotland"],
    "K": ["Qatar", "Iran", "Austria", "Haiti"],
    "L": ["Iraq", "Uzbekistan", "Curacao", "Cape Verde"],
}

# Round-robin fixture pairs within a 4-team group (indices into team list)
_GROUP_FIXTURES = [
    (0, 1), (2, 3),
    (0, 2), (1, 3),
    (0, 3), (1, 2),
]

# ---------------------------------------------------------------------------
# Default ELO ratings (approximate, circa 2026)
# ---------------------------------------------------------------------------

DEFAULT_ELO = {
    # Top contenders
    "Argentina":              1850,
    "France":                 1820,
    "Brazil":                 1800,
    "England":                1760,
    "Spain":                  1750,
    "Portugal":               1740,
    "Germany":                1730,
    "Netherlands":            1720,
    # Strong contenders
    "Belgium":                1710,
    "Croatia":                1700,
    "Uruguay":                1690,
    "Morocco":                1685,
    "Japan":                  1680,
    "Colombia":               1675,
    "Switzerland":            1670,
    "Senegal":                1660,
    "United States":          1655,
    "Mexico":                 1650,
    "Denmark":                1645,
    "Serbia":                 1640,
    "South Korea":            1635,
    "Ecuador":                1625,
    "Turkey":                 1620,
    "Norway":                 1615,
    "Austria":                1610,
    "Ukraine":                1605,
    "Czech Republic":         1600,
    "Algeria":                1590,
    "Ivory Coast":            1585,
    "Australia":              1580,
    "Iran":                   1575,
    "Scotland":               1570,
    "Canada":                 1565,
    "Egypt":                  1555,
    "Paraguay":               1550,
    "Ghana":                  1540,
    "South Africa":           1530,
    "Panama":                 1520,
    "Saudi Arabia":           1515,
    "DR Congo":               1510,
    "Bosnia and Herzegovina": 1505,
    "Qatar":                  1495,
    "Uzbekistan":             1475,
    "Iraq":                   1465,
    "New Zealand":            1450,
    "Scotland":               1570,
    "Ghana":                  1540,
    "Haiti":                  1430,
    "Curacao":                1420,
    "Cape Verde":             1415,
}

# ---------------------------------------------------------------------------
# ELO-based fallback predictor
# ---------------------------------------------------------------------------

def _elo_predictor_factory(elo_ratings):
    """
    Build a simple ELO-based predictor callable.

    Draw probability is calibrated from the Elo gap:
        p_draw = max(0.10, 0.30 - 0.25 * |p_win_raw - p_loss_raw|)
    win/loss probs are then scaled by (1 - p_draw).
    Knockout stages use p_draw = 0 (no draws).
    """
    def predictor(home, away, stage):
        elo_h = elo_ratings.get(home, 1500.0)
        elo_a = elo_ratings.get(away, 1500.0)
        diff = (elo_h - elo_a) / 400.0
        raw_win = 1.0 / (1.0 + 10.0 ** (-diff))
        raw_loss = 1.0 - raw_win
        raw_gap = abs(raw_win - raw_loss)

        if stage == "group":
            p_draw = max(0.10, 0.30 - 0.25 * raw_gap)
        else:
            p_draw = 0.0

        scale = 1.0 - p_draw
        p_win = raw_win * scale
        p_loss = raw_loss * scale
        total = p_win + p_draw + p_loss
        return p_win / total, p_draw / total, p_loss / total

    return predictor


# ---------------------------------------------------------------------------
# TournamentSimulator
# ---------------------------------------------------------------------------

class TournamentSimulator:
    """
    Monte Carlo simulator for the FIFA World Cup 2026.

    Parameters
    ----------
    predictor : callable or None
        Signature: (home_team, away_team, stage) -> (win_prob, draw_prob, away_win_prob)
        Probabilities must sum to 1.0.  If None a built-in ELO predictor is used.
    n_simulations : int
        Default number of simulations used when run() is called without an argument.
    groups : dict or None
        Group definitions.  Defaults to WC2026_GROUPS.
    elo_ratings : dict or None
        Used by the built-in predictor and as a deterministic final tiebreaker.
        Defaults to DEFAULT_ELO.
    seed : int or None
        NumPy random seed for reproducibility.

    Usage
    -----
    sim = TournamentSimulator()
    sim.run()
    print(sim.get_results_table())
    """

    _STATS_KEYS = [
        "group_exit_prob",
        "r32_prob",
        "r16_prob",
        "qf_prob",
        "sf_prob",
        "final_prob",
        "champion_prob",
    ]

    def __init__(self, predictor=None, n_simulations=100000, groups=None,
                 elo_ratings=None, seed=None):
        self.groups = groups or WC2026_GROUPS
        self.elo_ratings = elo_ratings or DEFAULT_ELO
        self.predictor = predictor or _elo_predictor_factory(self.elo_ratings)
        self._default_n = n_simulations
        self.rng = np.random.default_rng(seed)

        self._all_teams = [t for teams in self.groups.values() for t in teams]
        self._n_groups = len(self.groups)

        # Counts accumulated during run(); divided by N to get probabilities
        self._counts = {t: {k: 0 for k in self._STATS_KEYS} for t in self._all_teams}
        self._n_simulations = 0
        self._run_time_s = 0.0

        logger.info(
            "TournamentSimulator initialised: {} groups, {} teams",
            self._n_groups, len(self._all_teams),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, n_simulations=None):
        """
        Execute Monte Carlo simulations.

        Parameters
        ----------
        n_simulations : int or None
            Number of iterations.  Defaults to the value passed to __init__.
        """
        N = n_simulations if n_simulations is not None else self._default_n
        logger.info("Starting {:,} Monte Carlo simulations...", N)
        t0 = time.perf_counter()

        group_names = list(self.groups.keys())
        n_g = self._n_groups

        # Reset counts so run() is idempotent
        self._counts = {t: {k: 0 for k in self._STATS_KEYS} for t in self._all_teams}

        # ------------------------------------------------------------------
        # Pre-compute group-stage match probabilities
        # ------------------------------------------------------------------
        group_match_probs = {}   # g_name -> (6, 3) float64
        for g_name, teams in self.groups.items():
            probs = np.empty((6, 3), dtype=np.float64)
            for m_idx, (hi, ai) in enumerate(_GROUP_FIXTURES):
                hw, d, aw = self.predictor(teams[hi], teams[ai], "group")
                probs[m_idx] = [hw, d, aw]
            group_match_probs[g_name] = probs

        # ------------------------------------------------------------------
        # Simulate group-stage outcomes — vectorised per match
        # outcome 0 = home win, 1 = draw, 2 = away win
        # ------------------------------------------------------------------
        group_outcomes = {}   # g_name -> (6, N) int8
        for g_name, probs in group_match_probs.items():
            outcomes = np.empty((6, N), dtype=np.int8)
            for m_idx in range(6):
                outcomes[m_idx] = self.rng.choice(3, size=N, p=probs[m_idx])
            group_outcomes[g_name] = outcomes

        # ------------------------------------------------------------------
        # Simulate scorelines (Poisson) for GD/GF tiebreaking
        # ------------------------------------------------------------------
        group_scores = {}   # g_name -> (6, N, 2) int8
        for g_name, outcomes in group_outcomes.items():
            scores = np.zeros((6, N, 2), dtype=np.int8)
            for m_idx in range(6):
                out = outcomes[m_idx]

                home_goals = np.zeros(N, dtype=np.int8)
                away_goals = np.zeros(N, dtype=np.int8)

                mask_hw = out == 0
                n_hw = int(mask_hw.sum())
                if n_hw:
                    hg = np.clip(self.rng.poisson(1.9, n_hw), 1, 7).astype(np.int8)
                    ag = np.clip(self.rng.poisson(0.6, n_hw), 0, hg - 1).astype(np.int8)
                    home_goals[mask_hw] = hg
                    away_goals[mask_hw] = ag

                mask_d = out == 1
                n_d = int(mask_d.sum())
                if n_d:
                    g_both = np.clip(self.rng.poisson(1.1, n_d), 0, 5).astype(np.int8)
                    home_goals[mask_d] = g_both
                    away_goals[mask_d] = g_both

                mask_aw = out == 2
                n_aw = int(mask_aw.sum())
                if n_aw:
                    ag = np.clip(self.rng.poisson(1.9, n_aw), 1, 7).astype(np.int8)
                    hg = np.clip(self.rng.poisson(0.6, n_aw), 0, ag - 1).astype(np.int8)
                    home_goals[mask_aw] = hg
                    away_goals[mask_aw] = ag

                scores[m_idx, :, 0] = home_goals
                scores[m_idx, :, 1] = away_goals
            group_scores[g_name] = scores

        # ------------------------------------------------------------------
        # Build standings and determine group qualifiers
        # ------------------------------------------------------------------
        team_global_idx = {t: i for i, t in enumerate(self._all_teams)}
        teams_arr = np.array(self._all_teams)

        all_group_winners = np.empty((n_g, N), dtype=np.int16)
        all_group_runners = np.empty((n_g, N), dtype=np.int16)
        all_group_thirds  = np.empty((n_g, N), dtype=np.int16)

        thirds_pts = np.empty((n_g, N), dtype=np.int16)
        thirds_gd  = np.empty((n_g, N), dtype=np.int16)
        thirds_gf  = np.empty((n_g, N), dtype=np.int16)
        thirds_elo = np.empty((n_g, N), dtype=np.float32)

        for g_idx, g_name in enumerate(group_names):
            teams = self.groups[g_name]
            n_teams = len(teams)
            outcomes = group_outcomes[g_name]   # (6, N)
            scores   = group_scores[g_name]     # (6, N, 2)

            pts = np.zeros((n_teams, N), dtype=np.int16)
            gd  = np.zeros((n_teams, N), dtype=np.int16)
            gf  = np.zeros((n_teams, N), dtype=np.int16)

            for m_idx, (hi, ai) in enumerate(_GROUP_FIXTURES):
                out = outcomes[m_idx]
                hg  = scores[m_idx, :, 0].astype(np.int16)
                ag  = scores[m_idx, :, 1].astype(np.int16)

                pts[hi] += np.where(out == 0, 3, np.where(out == 1, 1, 0))
                pts[ai] += np.where(out == 2, 3, np.where(out == 1, 1, 0))
                gd[hi]  += (hg - ag)
                gd[ai]  += (ag - hg)
                gf[hi]  += hg
                gf[ai]  += ag

            # Sort key (higher = better): pts >> gd >> gf >> elo (deterministic)
            elo_vec = np.array(
                [self.elo_ratings.get(t, 1500.0) for t in teams], dtype=np.float64
            )  # (n_teams,)
            sort_score = (
                pts.astype(np.float64) * 1e8
                + gd.astype(np.float64) * 1e4
                + gf.astype(np.float64) * 1e2
                + elo_vec[:, None] * 0.1
            )  # (n_teams, N)

            ranks = np.argsort(-sort_score, axis=0)   # (n_teams, N)

            global_ids = np.array([team_global_idx[t] for t in teams], dtype=np.int16)

            all_group_winners[g_idx] = global_ids[ranks[0]]
            all_group_runners[g_idx] = global_ids[ranks[1]]
            all_group_thirds[g_idx]  = global_ids[ranks[2]]

            # Record group-exit (4th place) counts
            fourth_local = ranks[3]   # (N,) local team index
            fourth_global = global_ids[fourth_local]
            fourth_names = teams_arr[fourth_global]
            for name in fourth_names:
                self._counts[name]["group_exit_prob"] += 1

            # Save 3rd-place records for best-8 selection
            third_local = ranks[2]
            col_idx = np.arange(N)
            thirds_pts[g_idx] = pts[third_local, col_idx]
            thirds_gd[g_idx]  = gd[third_local, col_idx]
            thirds_gf[g_idx]  = gf[third_local, col_idx]
            thirds_elo[g_idx] = elo_vec[third_local].astype(np.float32)

        logger.debug("Group stage simulated.")

        # ------------------------------------------------------------------
        # Select best 8 third-place teams
        # Tiebreaker: (points, gd, gf, elo) — same as group standings
        # ------------------------------------------------------------------
        thirds_score = (
            thirds_pts.astype(np.float64) * 1e8
            + thirds_gd.astype(np.float64) * 1e4
            + thirds_gf.astype(np.float64) * 1e2
            + thirds_elo.astype(np.float64) * 0.1
        )  # (12, N)

        thirds_ranks = np.argsort(-thirds_score, axis=0)   # (12, N)
        best_third_group_idx = thirds_ranks[:8, :]          # (8, N)

        best_thirds_global = np.take_along_axis(
            all_group_thirds, best_third_group_idx, axis=0
        )  # (8, N)

        logger.debug("Best-8 third-place selection done.")

        # ------------------------------------------------------------------
        # Accumulate r32 counts (everyone who advances to Round of 32)
        # ------------------------------------------------------------------
        for g_idx in range(n_g):
            for global_arr in (all_group_winners[g_idx], all_group_runners[g_idx]):
                for name in teams_arr[global_arr]:
                    self._counts[name]["r32_prob"] += 1

        for i in range(8):
            for name in teams_arr[best_thirds_global[i]]:
                self._counts[name]["r32_prob"] += 1

        # ------------------------------------------------------------------
        # Build the 32-team bracket — classic seeded draw
        # Seeds 0-11  : group winners
        # Seeds 12-23 : group runners-up
        # Seeds 24-31 : best 8 third-placed teams
        # Bracket: seed 0 vs seed 31, 1 vs 30, ..., 15 vs 16
        # ------------------------------------------------------------------
        bracket = np.empty((N, 32), dtype=np.int16)
        for g_idx in range(n_g):
            bracket[:, g_idx]      = all_group_winners[g_idx]
            bracket[:, g_idx + 12] = all_group_runners[g_idx]
        for i in range(8):
            bracket[:, 24 + i] = best_thirds_global[i]

        # ELO lookup array indexed by global team id
        elo_lookup = np.array(
            [self.elo_ratings.get(t, 1500.0) for t in self._all_teams],
            dtype=np.float64,
        )  # (n_teams,)

        # ------------------------------------------------------------------
        # Simulate knockout rounds (R32 -> R16 -> QF -> SF -> Final)
        # ------------------------------------------------------------------
        # round_names[i] = stat key for WINNERS of round i
        round_names = ["r16_prob", "qf_prob", "sf_prob", "final_prob", "champion_prob"]

        current_round = bracket.copy()   # (N, 32)
        n_teams_in_round = 32

        for round_idx in range(5):
            n_matches = n_teams_in_round // 2
            next_round = np.empty((N, n_matches), dtype=np.int16)

            for m in range(n_matches):
                home_ids = current_round[:, m]
                away_ids = current_round[:, n_teams_in_round - 1 - m]

                elo_h = elo_lookup[home_ids]
                elo_a = elo_lookup[away_ids]
                diff = (elo_h - elo_a) / 400.0
                p_home = 1.0 / (1.0 + 10.0 ** (-diff))   # (N,) vectorised

                # No draws in knockout — use win_prob / (win_prob + away_win_prob) ratio
                rand = self.rng.random(N)
                winner_ids = np.where(rand < p_home, home_ids, away_ids)
                next_round[:, m] = winner_ids

            # Record who advanced to the NEXT round
            stat_key = round_names[round_idx]
            for m in range(n_matches):
                for name in teams_arr[next_round[:, m]]:
                    self._counts[name][stat_key] += 1

            current_round = next_round
            n_teams_in_round = n_matches

        self._n_simulations = N
        self._run_time_s = time.perf_counter() - t0
        logger.success(
            "Simulation complete: {:,} iterations in {:.2f}s ({:,.0f} sims/sec)",
            N, self._run_time_s, N / self._run_time_s,
        )

    # ------------------------------------------------------------------
    # Results API
    # ------------------------------------------------------------------

    def get_results(self):
        """
        Return simulation results as a nested dict.

        Returns
        -------
        dict[team_name, dict[stat_key, probability]]
            Probabilities are in [0, 1].

        Raises
        ------
        RuntimeError
            If run() has not been called yet.
        """
        if self._n_simulations == 0:
            raise RuntimeError("Call run() before get_results().")

        N = self._n_simulations
        return {
            team: {k: round(self._counts[team][k] / N, 6) for k in self._STATS_KEYS}
            for team in self._all_teams
        }

    def get_results_table(self, sort_by="champion_prob"):
        """
        Return a formatted ASCII table of simulation results.

        Parameters
        ----------
        sort_by : str
            Column to sort by.  Must be one of the _STATS_KEYS.

        Returns
        -------
        str
            Printable ASCII table.
        """
        if sort_by not in self._STATS_KEYS:
            raise ValueError("sort_by must be one of " + str(self._STATS_KEYS))

        results = self.get_results()
        teams_sorted = sorted(
            self._all_teams,
            key=lambda t: results[t][sort_by],
            reverse=True,
        )

        header_keys = [
            ("group_exit_prob", "GrpExit%"),
            ("r32_prob",        "R32%"),
            ("r16_prob",        "R16%"),
            ("qf_prob",         "QF%"),
            ("sf_prob",         "SF%"),
            ("final_prob",      "Final%"),
            ("champion_prob",   "Champ%"),
        ]

        col_w  = 9
        team_w = 28
        total_w = team_w + col_w * len(header_keys) + 3
        sep = "-" * total_w

        lines = []
        lines.append(sep)
        lines.append(
            "  " + "WC 2026 Monte Carlo Simulation".center(total_w - 2)
        )
        lines.append(
            "  " + ("n = {:,} iterations".format(self._n_simulations)).center(total_w - 2)
        )
        lines.append(sep)
        header_str = "  {:<{w}}".format("Team", w=team_w) + "".join(
            "{:>{cw}}".format(h, cw=col_w) for _, h in header_keys
        )
        lines.append(header_str)
        lines.append(sep)

        for team in teams_sorted:
            row = "  {:<{w}}".format(team, w=team_w)
            for k, _ in header_keys:
                pct = results[team][k] * 100
                row += "{:>{cw}.1f}%".format(pct, cw=col_w - 1)
            lines.append(row)

        lines.append(sep)
        lines.append(
            "  Elapsed: {:.2f}s  |  Throughput: {:,.0f} sims/sec".format(
                self._run_time_s, self._n_simulations / self._run_time_s
            )
        )
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
    )

    logger.info("WC 2026 Monte Carlo Simulation — standalone run")
    logger.info("Using built-in ELO predictor (DEFAULT_ELO)")

    sim = TournamentSimulator(seed=42)
    sim.run()

    print("\n" + sim.get_results_table(sort_by="champion_prob"))
