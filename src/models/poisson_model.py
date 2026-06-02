"""
src/models/poisson_model.py
Dixon-Coles Poisson model for exact score prediction in WC 2026.

Mathematical foundations
------------------------
Goals scored in a football match are modelled as independent Poisson
random variables, corrected for the well-known under-representation of
high-scoring draws and the over-representation of 0-0 results.

For a match between a home team h and an away team a:

    lambda  = exp(attack_h + defence_a + home_advantage)   [home expected goals]
    mu      = exp(attack_a + defence_h)                     [away expected goals]

Parameters:
    attack_i   – log attack strength of team i  (positive = prolific)
    defence_i  – log defensive weakness of team i  (positive = leaky)
    home_adv   – scalar log home advantage (typically ~0.3)
    rho        – Dixon-Coles low-score correction (~-0.13 for soccer)

Identifiability constraint:
    The sum of all attack parameters is fixed to zero in log space, which
    pins the scale and ensures a unique solution.  Equivalently we can
    remove one attack parameter and reconstruct it from the rest.

Dixon-Coles tau correction:
    tau(0,0, λ, μ, ρ) = 1 − λ·μ·ρ
    tau(1,0, λ, μ, ρ) = 1 + μ·ρ
    tau(0,1, λ, μ, ρ) = 1 + λ·ρ
    tau(1,1, λ, μ, ρ) = 1 − ρ
    tau(x,y, …)       = 1   for x ≥ 2 or y ≥ 2

Full joint probability:
    P(X=x, Y=y) = tau(x,y,λ,μ,ρ) · Poisson(x;λ) · Poisson(y;μ)

Fitting:
    We maximise the (weighted) log-likelihood over all historical matches
    using scipy.optimize.minimize with L-BFGS-B.  Recent matches receive
    higher weight via exponential time decay.

Reference:
    Dixon, M. J., & Coles, S. G. (1997).
    "Modelling Association Football Scores and Inefficiencies in the
    Football Betting Market."
    Journal of the Royal Statistical Society: Series C, 46(2), 265-280.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CACHE_DIR

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Score matrix dimension: 0-0 through (MAX_GOALS-1)-(MAX_GOALS-1)
MAX_GOALS = 6

# Default rho from the original Dixon-Coles paper (soccer calibration)
DEFAULT_RHO = -0.13

# Exponential time-decay half-life in days for weighting recent matches
TIME_DECAY_HALFLIFE_DAYS = 365.0


# ---------------------------------------------------------------------------
# Low-level helper functions (module-level, pure)
# ---------------------------------------------------------------------------

def _dixon_coles_tau(x, y, lam, mu, rho):
    """
    Return the Dixon-Coles tau correction scalar for a single scoreline (x, y).

    Corrects the four low-scoring outcomes; all others return 1.0.
    Note: rho is typically negative for football (~-0.13).

    Parameters
    ----------
    x   : int   home goals
    y   : int   away goals
    lam : float home expected goals (lambda)
    mu  : float away expected goals
    rho : float Dixon-Coles correction parameter
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _compute_lambdas(home_idx, away_idx, attack, defence, home_adv):
    """
    Compute (lambda, mu) for a single match from parameter arrays.

    attack  : 1-D array of log attack strengths, length n_teams
    defence : 1-D array of log defensive weaknesses, length n_teams
    home_adv: scalar log home advantage
    """
    lam = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    mu  = np.exp(attack[away_idx] + defence[home_idx])
    return lam, mu


def _dc_log_likelihood_vectorised(params, home_goals, away_goals,
                                   home_idx, away_idx, weights, n_teams):
    """
    Negative weighted log-likelihood of the Dixon-Coles model.

    params layout:
        [attack_0, …, attack_{n-1}, defence_0, …, defence_{n-1},
         home_adv, rho]
    Total length = 2*n_teams + 2.

    The first attack parameter is NOT optimised (held fixed to enforce the
    identifiability constraint attack_0 = 0).  The optimiser only sees
    params[1:], so we prepend a 0 before unpacking.

    This implementation is fully vectorised with numpy for performance.
    """
    # params arrives without attack[0] (it is fixed at 0 by the caller)
    full = np.concatenate([[0.0], params])

    attack   = full[:n_teams]
    defence  = full[n_teams : 2 * n_teams]
    home_adv = full[2 * n_teams]
    rho      = full[2 * n_teams + 1]

    # Vectorised lambda/mu for all matches at once
    lam = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    mu  = np.exp(attack[away_idx] + defence[home_idx])

    # Vectorised Dixon-Coles tau correction
    # tau = 1.0 for all matches by default; override low-score cases
    tau = np.ones(len(home_goals))
    h = home_goals
    a = away_goals

    mask_00 = (h == 0) & (a == 0)
    mask_10 = (h == 1) & (a == 0)
    mask_01 = (h == 0) & (a == 1)
    mask_11 = (h == 1) & (a == 1)

    tau[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
    tau[mask_10] = 1.0 + mu[mask_10] * rho
    tau[mask_01] = 1.0 + lam[mask_01] * rho
    tau[mask_11] = 1.0 - rho

    # Guard against tau <= 0 (numerical edge case)
    tau = np.maximum(tau, 1e-10)

    # Vectorised log-likelihood using scipy.stats.poisson.logpmf
    ll = (
        np.log(tau)
        + poisson.logpmf(h, lam)
        + poisson.logpmf(a, mu)
    )
    total_ll = np.dot(weights, ll)

    # Return negative because scipy.minimize minimises
    return -total_ll


def _time_decay_weights(match_dates, reference_date=None,
                        halflife_days=TIME_DECAY_HALFLIFE_DAYS):
    """
    Compute exponential time-decay weights from match dates.

    Matches closer to reference_date receive weight 1.0; older matches
    decay exponentially with the given half-life.

    Parameters
    ----------
    match_dates    : pd.Series of datetime-like values
    reference_date : cutoff date (default: today)
    halflife_days  : half-life in days (default: 365)

    Returns
    -------
    np.ndarray of weights in (0, 1]
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    else:
        reference_date = pd.Timestamp(reference_date)

    dates   = pd.to_datetime(match_dates)
    days_ago = (reference_date - dates).dt.total_seconds() / 86400.0
    days_ago = days_ago.clip(lower=0).values

    decay = np.exp(-np.log(2) * days_ago / halflife_days)
    return decay


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DixonColesModel:
    """
    Dixon-Coles Poisson model for predicting exact football scores.

    Usage
    -----
        model = DixonColesModel()
        model.fit(df)                                   # df has required columns
        matrix = model.predict_score_matrix("France", "Morocco")
        result = model.predict_match("France", "Morocco")
        model.save()

        loaded = DixonColesModel.load_latest()

    Required DataFrame columns for fit()
    -------------------------------------
        home_team  : str
        away_team  : str
        home_goals : int
        away_goals : int
        weight     : float  (optional; if absent, time-decay is applied
                             automatically using a 'date' column if present,
                             otherwise uniform weights are used)
    """

    def __init__(self, rho=DEFAULT_RHO, max_goals=MAX_GOALS):
        self.rho = rho                  # Dixon-Coles correction (overwritten by fit)
        self.max_goals = max_goals
        self.home_adv = None            # scalar, set by fit()
        self.attack = {}                # team -> log attack
        self.defence = {}               # team -> log defence
        self.teams = []                 # ordered list of teams known to the model
        self.avg_attack = 0.0           # league-average fallback
        self.avg_defence = 0.0          # league-average fallback
        self.is_fitted = False
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        self._train_df_info = {}        # lightweight metadata stored with the model

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df, reference_date=None):
        """
        Fit the Dixon-Coles model to historical match data.

        Parameters
        ----------
        df             : pd.DataFrame
            Must contain: home_team, away_team, home_goals, away_goals.
            Optional:  weight  (pre-computed weights override time decay),
                       date    (used for time-decay if weight is absent).
        reference_date : str or datetime-like, optional
            The date from which time-decay is measured (default: today).
            Set to the last match date for reproducible back-tests.
        """
        df = df.copy()
        df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        if len(df) == 0:
            raise ValueError("No valid rows in training DataFrame.")

        # ── Weights ───────────────────────────────────────────────────
        if "weight" in df.columns:
            weights = df["weight"].astype(float).values
            logger.debug("Using pre-computed weights from 'weight' column.")
        elif "date" in df.columns:
            weights = _time_decay_weights(df["date"], reference_date=reference_date)
            logger.debug(
                f"Time-decay weights applied (halflife={TIME_DECAY_HALFLIFE_DAYS} days). "
                f"Min weight: {weights.min():.4f}"
            )
        else:
            weights = np.ones(len(df))
            logger.debug("No date/weight column found — using uniform weights.")

        # ── Team indexing ──────────────────────────────────────────────
        all_teams = sorted(
            set(df["home_team"].tolist()) | set(df["away_team"].tolist())
        )
        team_to_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)
        logger.info(
            f"Fitting Dixon-Coles model on {len(df)} matches, "
            f"{n_teams} teams."
        )

        home_idx = df["home_team"].map(team_to_idx).values
        away_idx = df["away_team"].map(team_to_idx).values
        home_goals = df["home_goals"].values
        away_goals = df["away_goals"].values

        # ── Initial parameter guess ────────────────────────────────────
        # attack[0] is held fixed at 0; remaining attack[1..] are free
        # Layout (optimiser sees):  [attack_1,..,attack_{n-1},
        #                            defence_0,..,defence_{n-1},
        #                            home_adv, rho]
        n_free = (n_teams - 1) + n_teams + 1 + 1

        x0 = np.zeros(n_free)
        # home_adv near log(1.3) ≈ 0.26 is a reasonable starting point
        x0[-(2)] = 0.26
        # rho near the paper's empirical estimate
        x0[-1] = DEFAULT_RHO

        # ── Bounds ────────────────────────────────────────────────────
        # attack/defence unconstrained; rho in (-1, 0]; home_adv unconstrained
        bounds = (
            [(None, None)] * (n_teams - 1)      # free attacks
            + [(None, None)] * n_teams           # defences
            + [(None, None)]                     # home_adv
            + [(-1.0, 0.5)]                      # rho (must keep tau > 0)
        )

        logger.info("Optimising log-likelihood (L-BFGS-B)…")
        result = minimize(
            _dc_log_likelihood_vectorised,
            x0,
            args=(home_goals, away_goals, home_idx, away_idx, weights, n_teams),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-7},
        )

        if not result.success:
            logger.warning(
                f"Optimiser did not fully converge: {result.message}. "
                "Results may still be usable."
            )
        else:
            logger.success(
                f"Optimisation converged. Iterations: {result.nit}, "
                f"Log-likelihood: {-result.fun:.4f}"
            )

        # ── Unpack parameters ──────────────────────────────────────────
        full_params = np.concatenate([[0.0], result.x])
        attack_arr  = full_params[:n_teams]
        defence_arr = full_params[n_teams : 2 * n_teams]
        home_adv    = full_params[2 * n_teams]
        rho         = float(full_params[2 * n_teams + 1])

        # ── Apply sum-to-zero constraint on attack (re-centre) ────────
        # Even though we fixed attack[0]=0, numerical drift can occur in
        # the effective scale.  Re-centre to ensure the league average
        # attack is exactly 0 in log space.
        attack_arr  -= attack_arr.mean()

        self.teams    = all_teams
        self.attack   = dict(zip(all_teams, attack_arr.tolist()))
        self.defence  = dict(zip(all_teams, defence_arr.tolist()))
        self.home_adv = float(home_adv)
        self.rho      = rho

        # League-average fallback (for unseen teams)
        self.avg_attack  = float(attack_arr.mean())
        self.avg_defence = float(defence_arr.mean())

        self.is_fitted = True
        self._train_df_info = {
            "n_matches": int(len(df)),
            "n_teams":   int(n_teams),
            "fitted_at": datetime.utcnow().isoformat(),
        }

        logger.success(
            f"Model fitted.  home_adv={self.home_adv:.4f}  rho={self.rho:.4f}"
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_params(self, team):
        """
        Return (attack, defence) for a team, using league-average fallback
        for teams not seen during training.
        """
        if team in self.attack:
            return self.attack[team], self.defence[team]
        logger.warning(
            f"Team '{team}' not seen in training data. "
            "Using league-average attack/defence."
        )
        return self.avg_attack, self.avg_defence

    def _lambdas(self, home_team, away_team):
        """
        Compute expected goals (lambda, mu) for a match.

        Returns
        -------
        (lam, mu) : (float, float)
            lam = home expected goals, mu = away expected goals
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predicting.")

        att_h, def_h = self._get_params(home_team)
        att_a, def_a = self._get_params(away_team)

        lam = np.exp(att_h + def_a + self.home_adv)
        mu  = np.exp(att_a + def_h)
        return float(lam), float(mu)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_score_matrix(self, home_team, away_team):
        """
        Return a (max_goals × max_goals) numpy array of exact-score probabilities.

        Row index = home goals (0 … max_goals-1)
        Column index = away goals (0 … max_goals-1)

        Example: matrix[2, 1] = P(home 2 – 1 away)

        Parameters
        ----------
        home_team : str
        away_team : str

        Returns
        -------
        np.ndarray of shape (max_goals, max_goals), values sum to ≈1
        """
        lam, mu = self._lambdas(home_team, away_team)
        matrix  = np.zeros((self.max_goals, self.max_goals))

        for i in range(self.max_goals):
            for j in range(self.max_goals):
                tau = _dixon_coles_tau(i, j, lam, mu, self.rho)
                matrix[i, j] = (
                    tau
                    * poisson.pmf(i, lam)
                    * poisson.pmf(j, mu)
                )

        return matrix

    def predict_match(self, home_team, away_team):
        """
        Predict match outcome probabilities and most likely score.

        Parameters
        ----------
        home_team : str
        away_team : str

        Returns
        -------
        dict with keys:
            home_win_prob         : float  P(home goals > away goals)
            draw_prob             : float  P(home goals == away goals)
            away_win_prob         : float  P(away goals > home goals)
            expected_home_goals   : float  lambda
            expected_away_goals   : float  mu
            most_likely_score     : str    e.g. "1-0"
            score_matrix          : np.ndarray  (max_goals × max_goals)
        """
        lam, mu = self._lambdas(home_team, away_team)
        matrix  = self.predict_score_matrix(home_team, away_team)

        # Win / draw / loss from the triangular structure of the matrix
        # tril(k=-1): rows > cols  →  home goals > away goals  →  home win
        # diag       : home goals == away goals                →  draw
        # triu(k=+1): cols > rows  →  away goals > home goals  →  away win
        home_win_prob = float(np.sum(np.tril(matrix, -1)))
        draw_prob     = float(np.sum(np.diag(matrix)))
        away_win_prob = float(np.sum(np.triu(matrix, 1)))

        # Most likely exact score
        flat_idx  = np.argmax(matrix)
        best_home, best_away = divmod(flat_idx, self.max_goals)
        most_likely_score = f"{best_home}-{best_away}"

        result = {
            "home_team":           home_team,
            "away_team":           away_team,
            "home_win_prob":       round(home_win_prob, 4),
            "draw_prob":           round(draw_prob, 4),
            "away_win_prob":       round(away_win_prob, 4),
            "expected_home_goals": round(lam, 4),
            "expected_away_goals": round(mu, 4),
            "most_likely_score":   most_likely_score,
            "score_matrix":        matrix,
        }

        logger.debug(
            f"{home_team} vs {away_team}: "
            f"H={home_win_prob:.1%} D={draw_prob:.1%} A={away_win_prob:.1%}  "
            f"EG={lam:.2f}-{mu:.2f}  MLS={most_likely_score}"
        )
        return result

    # ------------------------------------------------------------------
    # Team parameter introspection
    # ------------------------------------------------------------------

    def team_ratings(self):
        """
        Return a DataFrame of all team attack / defence ratings, sorted by
        net strength (attack - defence, descending).

        A high attack score means prolific; a high defence score means leaky.
        net_strength = attack - defence: higher is better.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        rows = []
        for team in self.teams:
            att = self.attack[team]
            def_ = self.defence[team]
            rows.append({
                "team":         team,
                "attack":       round(att, 4),
                "defence":      round(def_, 4),
                "net_strength": round(att - def_, 4),
            })

        df = pd.DataFrame(rows).sort_values("net_strength", ascending=False)
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, name="poisson_dc"):
        """
        Serialise model parameters to disk using joblib.

        Returns
        -------
        Path to the saved file.
        """
        payload = {
            "attack":         self.attack,
            "defence":        self.defence,
            "home_adv":       self.home_adv,
            "rho":            self.rho,
            "avg_attack":     self.avg_attack,
            "avg_defence":    self.avg_defence,
            "teams":          self.teams,
            "max_goals":      self.max_goals,
            "version":        self.version,
            "train_df_info":  self._train_df_info,
        }
        path = MODELS_DIR / f"{name}_{self.version}.joblib"
        joblib.dump(payload, path)
        logger.success(f"Dixon-Coles model saved: {path}")
        return path

    @classmethod
    def load(cls, path):
        """
        Load a previously saved model from path.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        DixonColesModel instance (is_fitted=True)
        """
        payload = joblib.load(path)
        instance = cls(
            rho=payload["rho"],
            max_goals=payload["max_goals"],
        )
        instance.attack        = payload["attack"]
        instance.defence       = payload["defence"]
        instance.home_adv      = payload["home_adv"]
        instance.avg_attack    = payload["avg_attack"]
        instance.avg_defence   = payload["avg_defence"]
        instance.teams         = payload["teams"]
        instance.version       = payload["version"]
        instance._train_df_info = payload.get("train_df_info", {})
        instance.is_fitted     = True
        logger.success(
            f"Dixon-Coles model loaded: {path}  "
            f"(v{instance.version}, "
            f"{len(instance.teams)} teams)"
        )
        return instance

    @classmethod
    def load_latest(cls):
        """
        Load the most recently saved Dixon-Coles model from the default
        models directory.

        Returns
        -------
        DixonColesModel instance
        """
        saved = sorted(MODELS_DIR.glob("poisson_dc_*.joblib"))
        if not saved:
            raise FileNotFoundError(
                f"No saved Dixon-Coles models found in {MODELS_DIR}. "
                "Run fit() and save() first."
            )
        return cls.load(saved[-1])


# ---------------------------------------------------------------------------
# Standalone runner — quick smoke-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  WC 2026 Predictor — Dixon-Coles Poisson Model")
    print("=" * 60)

    # ── Build a tiny synthetic dataset so the module is self-contained ──
    rng = np.random.default_rng(42)
    teams = [
        "France", "Brazil", "England", "Germany",
        "Argentina", "Spain", "Morocco", "Japan",
    ]
    rows = []
    for _ in range(200):
        h, a = rng.choice(teams, size=2, replace=False)
        hg = int(rng.poisson(1.5))
        ag = int(rng.poisson(1.1))
        rows.append({
            "home_team":  h,
            "away_team":  a,
            "home_goals": hg,
            "away_goals": ag,
            "date":       pd.Timestamp("2024-01-01")
                          + pd.Timedelta(days=int(rng.integers(0, 730))),
        })
    df = pd.DataFrame(rows)

    # ── Fit ──────────────────────────────────────────────────────────
    model = DixonColesModel()
    model.fit(df)

    # ── Team ratings ──────────────────────────────────────────────────
    print("\n  Team ratings (attack / defence / net):")
    print(model.team_ratings().to_string(index=False))

    # ── Predict some matches ──────────────────────────────────────────
    matchups = [
        ("France",    "Morocco"),
        ("Brazil",    "Argentina"),
        ("England",   "Germany"),
    ]

    print(f"\n  {'Match':30s} {'H%':>7s} {'D%':>7s} {'A%':>7s}  {'MLS':>5s}  EG")
    print("  " + "-" * 70)
    for home, away in matchups:
        r = model.predict_match(home, away)
        print(
            f"  {home+' vs '+away:30s} "
            f"{r['home_win_prob']:>6.1%} "
            f"{r['draw_prob']:>7.1%} "
            f"{r['away_win_prob']:>7.1%}  "
            f"{r['most_likely_score']:>5s}  "
            f"{r['expected_home_goals']:.2f}-{r['expected_away_goals']:.2f}"
        )

    # ── Score matrix for one match ─────────────────────────────────────
    print("\n  Score matrix — France vs Morocco (rows=home, cols=away):")
    matrix = model.predict_score_matrix("France", "Morocco")
    header = "      " + "  ".join(f"A{j}" for j in range(MAX_GOALS))
    print("  " + header)
    for i, row in enumerate(matrix):
        cells = "  ".join(f"{v:.3f}" for v in row)
        print(f"  H{i}  {cells}")

    # ── Fallback for an unknown team ────────────────────────────────────
    print("\n  Predicting with an unknown team (should use league-average):")
    r = model.predict_match("France", "Atlantis FC")
    print(
        f"  France vs Atlantis FC: "
        f"H={r['home_win_prob']:.1%}  D={r['draw_prob']:.1%}  "
        f"A={r['away_win_prob']:.1%}  MLS={r['most_likely_score']}"
    )

    # ── Save / reload round-trip ────────────────────────────────────────
    saved_path = model.save()
    reloaded   = DixonColesModel.load(saved_path)
    r_orig = model.predict_match("France", "Morocco")
    r2     = reloaded.predict_match("France", "Morocco")
    assert abs(r2["home_win_prob"] - r_orig["home_win_prob"]) < 1e-6, "Round-trip mismatch!"
    print(f"\n  Save/load round-trip OK.  Saved at: {saved_path}")
    print("\n  Done.")
