"""
src/models/ensemble.py
Weighted ensemble: XGBoost + Neural Net (goal predictor) + Dixon-Coles Poisson.

Combines three complementary models:
  - WCOutcomeModel  (xgboost)     40%: win/draw/loss classifier
  - GoalPredictor   (neural_net)  30%: neural net expected goals
  - DixonColesModel (poisson)     30%: bivariate Poisson with low-score correction

Each model can fail independently; the ensemble degrades gracefully by
redistributing weight to the available models.

Usage:
    predictor = EnsemblePredictor()
    result = predictor.predict("France", "Morocco", "group")
    print(result["most_likely_score"])   # e.g. "2-1"

CLI:
    python -m src.models.ensemble --home France --away Morocco --stage group
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

MODELS_DIR = Path(__file__).parent / "saved"

# ── ELO baseline ratings (fallback when models are unavailable) ───────────────
_BASELINE_ELO = {
    "France":        2055,
    "Brazil":        2040,
    "Argentina":     2080,
    "England":       1990,
    "Spain":         2000,
    "Germany":       1970,
    "Portugal":      1985,
    "Netherlands":   1975,
    "Belgium":       1960,
    "Uruguay":       1930,
    "Croatia":       1920,
    "Morocco":       1880,
    "United States": 1860,
    "Mexico":        1850,
    "Japan":         1840,
    "Senegal":       1830,
    "Australia":     1780,
    "South Korea":   1810,
    "Canada":        1820,
    "Colombia":      1900,
    "Ecuador":       1850,
    "Poland":        1840,
    "Switzerland":   1900,
    "Denmark":       1880,
    "Serbia":        1840,
    "Iran":          1810,
    "Qatar":         1760,
    "Saudi Arabia":  1790,
    "Ghana":         1790,
    "Cameroon":      1800,
    "Tunisia":       1780,
    "Costa Rica":    1770,
}
_DEFAULT_ELO = 1750.0

# Average international goals per game
_HOME_GOALS_AVG = 1.55
_AWAY_GOALS_AVG = 1.10
_DC_RHO = -0.13   # Dixon-Coles low-score correction


# ══════════════════════════════════════════════════════════════════════════════
#  Internal Poisson helpers
# ══════════════════════════════════════════════════════════════════════════════

def _elo_to_strength(team):
    elo = _BASELINE_ELO.get(team, _DEFAULT_ELO)
    strength = 0.75 + (elo - 1500) / 2000.0
    strength = max(0.40, min(strength, 2.00))
    attack  = math.sqrt(strength)
    defence = 1.0 / math.sqrt(strength)
    return attack, defence


def _poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _tau(x, y, lam, mu):
    """Dixon-Coles correction factor for low-score cells."""
    rho = _DC_RHO
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _goals_to_probs(home_xg, away_xg, max_goals=8):
    """Convert (home_xg, away_xg) to outcome probs via Poisson integration."""
    p_home_win = p_draw = p_away_win = 0.0
    best_p = -1.0
    best_score = "1-1"

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ph = _poisson_pmf(h, home_xg)
            pa = _poisson_pmf(a, away_xg)
            p  = ph * pa
            if h > a:
                p_home_win += p
            elif h == a:
                p_draw += p
            else:
                p_away_win += p
            if p > best_p:
                best_p = p
                best_score = "{}-{}".format(h, a)

    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw     /= total
        p_away_win /= total

    return round(p_home_win, 4), round(p_draw, 4), round(p_away_win, 4), best_score


# ══════════════════════════════════════════════════════════════════════════════
#  Lazy model loaders (thin wrappers)
# ══════════════════════════════════════════════════════════════════════════════

class _XGBWrapper:
    """Lazily loads WCOutcomeModel from xgboost_model.py."""

    def __init__(self):
        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from src.models.xgboost_model import WCOutcomeModel  # type: ignore
            self._model = WCOutcomeModel.load_latest()
            logger.info("XGBoost model loaded successfully.")
        except Exception as exc:
            logger.warning("XGBoost model unavailable: {}", exc)
            self._model = None

    def predict(self, home_team, away_team, stage, feature_vector=None):
        self._ensure_loaded()
        if self._model is None or feature_vector is None:
            return None
        try:
            raw = self._model.predict(feature_vector, home_team=home_team, away_team=away_team)
            return {
                "p_home_win": raw["p_home_win"],
                "p_draw":     raw["p_draw"],
                "p_away_win": raw["p_away_win"],
            }
        except Exception as exc:
            logger.warning("XGBoost predict() failed: {}", exc)
            return None

    @property
    def available(self):
        self._ensure_loaded()
        return self._model is not None


class _NNWrapper:
    """Lazily loads GoalPredictor from neural_net.py."""

    def __init__(self):
        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            from src.models.neural_net import WCGoalModel, GoalPredictorNet  # type: ignore
            import torch  # type: ignore

            saved_files = sorted(
                (Path(__file__).parent / "saved").glob("neural_goal_*.pt")
            )
            if not saved_files:
                raise FileNotFoundError("No saved neural_goal_*.pt files found.")

            path = saved_files[-1]
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(path, map_location="cpu")  # type: ignore

            instance = WCGoalModel(device="cpu")
            instance.feature_names = data["feature_names"]
            instance.feature_mean  = data["feature_mean"]
            instance.feature_std   = data["feature_std"]
            instance.version       = data["version"]
            instance.net = GoalPredictorNet(data["input_dim"])
            instance.net.load_state_dict(data["state_dict"])
            instance.net.eval()
            self._model = instance
            logger.info("Neural net goal predictor loaded: {}", path)
        except Exception as exc:
            logger.warning("Neural net model unavailable: {}", exc)
            self._model = None

    def predict(self, home_team, away_team, stage, feature_vector=None):
        self._ensure_loaded()
        if self._model is None or feature_vector is None:
            return None
        try:
            raw = self._model.predict(feature_vector)
            home_xg = raw["home_xg"]
            away_xg = raw["away_xg"]
            hw, d, aw, mls = _goals_to_probs(home_xg, away_xg)
            return {
                "p_home_win":          hw,
                "p_draw":              d,
                "p_away_win":          aw,
                "expected_home_goals": round(home_xg, 3),
                "expected_away_goals": round(away_xg, 3),
                "most_likely_score":   mls,
            }
        except Exception as exc:
            logger.warning("Neural net predict() failed: {}", exc)
            return None

    @property
    def available(self):
        self._ensure_loaded()
        return self._model is not None


class _PoissonWrapper:
    """
    Dixon-Coles bivariate Poisson model.
    Always available — parameters are derived from ELO ratings.
    """

    def predict(self, home_team, away_team, max_goals=8):
        ha, hd = _elo_to_strength(home_team)
        aa, ad = _elo_to_strength(away_team)
        lam = max(_HOME_GOALS_AVG * ha * ad, 0.10)
        mu  = max(_AWAY_GOALS_AVG * aa * hd, 0.10)

        p_home_win = p_draw = p_away_win = 0.0
        score_grid = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = (
                    _poisson_pmf(h, lam)
                    * _poisson_pmf(a, mu)
                    * _tau(h, a, lam, mu)
                )
                p = max(p, 0.0)
                score_grid["{}-{}".format(h, a)] = p
                if h > a:
                    p_home_win += p
                elif h == a:
                    p_draw += p
                else:
                    p_away_win += p

        total = p_home_win + p_draw + p_away_win
        if total > 0:
            p_home_win /= total
            p_draw     /= total
            p_away_win /= total

        most_likely = max(score_grid, key=score_grid.__getitem__)

        return {
            "p_home_win":          round(p_home_win, 4),
            "p_draw":              round(p_draw, 4),
            "p_away_win":          round(p_away_win, 4),
            "expected_home_goals": round(lam, 3),
            "expected_away_goals": round(mu, 3),
            "most_likely_score":   most_likely,
        }

    @property
    def available(self):
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  EnsemblePredictor
# ══════════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Weighted ensemble combining XGBoost, Neural Net, and Dixon-Coles Poisson.

    Default weights:
        xgboost      40%
        neural_net   30%
        poisson      30%

    When a model is unavailable its weight is redistributed proportionally
    among the remaining models.  The Poisson model is always available
    (ELO-derived), so there is always at least one contributing model.

    Quick start:
        ep = EnsemblePredictor()
        result = ep.predict("France", "Morocco", "group")
    """

    _DEFAULT_WEIGHTS = {"xgboost": 0.40, "neural_net": 0.30, "poisson": 0.30}

    def __init__(self, weights=None):
        self.weights = dict(weights if weights is not None else self._DEFAULT_WEIGHTS)

        # Validate keys — accept both old-style ("xgb", "nn") and new-style
        self.weights = self._normalise_weight_keys(self.weights)

        # Lazy-loaded sub-models
        self._xgb_model     = None
        self._nn_model      = None
        self._poisson_model = None

        logger.info(
            "EnsemblePredictor initialised with weights: xgboost={xgboost:.0%} "
            "neural_net={neural_net:.0%} poisson={poisson:.0%}",
            **self.weights,
        )

    # ── Lazy model access ─────────────────────────────────────────────────────

    @property
    def xgb(self):
        if self._xgb_model is None:
            self._xgb_model = _XGBWrapper()
        return self._xgb_model

    @property
    def nn(self):
        if self._nn_model is None:
            self._nn_model = _NNWrapper()
        return self._nn_model

    @property
    def poisson(self):
        if self._poisson_model is None:
            self._poisson_model = _PoissonWrapper()
        return self._poisson_model

    # ── Weight management ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise_weight_keys(w):
        """Accept legacy keys like 'xgb'/'nn' and normalise to canonical names."""
        mapping = {"xgb": "xgboost", "nn": "neural_net"}
        result = {}
        for k, v in w.items():
            result[mapping.get(k, k)] = v
        # Ensure all three keys present
        for key in ("xgboost", "neural_net", "poisson"):
            result.setdefault(key, EnsemblePredictor._DEFAULT_WEIGHTS[key])
        # Renormalise
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        return result

    def _effective_weights(self, xgb_result, nn_result):
        """
        Redistribute weights when a model is unavailable (returns None).
        Poisson is always present.
        """
        active = {"poisson": self.weights["poisson"]}
        if xgb_result is not None:
            active["xgboost"] = self.weights["xgboost"]
        if nn_result is not None:
            active["neural_net"] = self.weights["neural_net"]

        total = sum(active.values())
        return {k: v / total for k, v in active.items()}

    # ── Blending ──────────────────────────────────────────────────────────────

    def _blend(self, xgb_result, nn_result, poisson_result):
        ew = self._effective_weights(xgb_result, nn_result)
        home_win = draw = away_win = 0.0

        if "xgboost" in ew and xgb_result:
            home_win += ew["xgboost"] * xgb_result["p_home_win"]
            draw     += ew["xgboost"] * xgb_result["p_draw"]
            away_win += ew["xgboost"] * xgb_result["p_away_win"]

        if "neural_net" in ew and nn_result:
            home_win += ew["neural_net"] * nn_result["p_home_win"]
            draw     += ew["neural_net"] * nn_result["p_draw"]
            away_win += ew["neural_net"] * nn_result["p_away_win"]

        home_win += ew["poisson"] * poisson_result["p_home_win"]
        draw     += ew["poisson"] * poisson_result["p_draw"]
        away_win += ew["poisson"] * poisson_result["p_away_win"]

        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw     /= total
            away_win /= total

        return round(home_win, 4), round(draw, 4), round(away_win, 4)

    # ── Model agreement ───────────────────────────────────────────────────────

    @staticmethod
    def _model_agreement(xgb_result, nn_result, poisson_result):
        """
        Mean pairwise L1 distance, normalised 0-1, then inverted.
        1.0 = full agreement, 0.0 = maximum divergence.
        """
        def to_vec(r):
            return np.array([r["p_home_win"], r["p_draw"], r["p_away_win"]], dtype=float)

        results = [poisson_result]
        if xgb_result is not None:
            results.append(xgb_result)
        if nn_result is not None:
            results.append(nn_result)

        if len(results) == 1:
            return 1.0

        vecs = [to_vec(r) for r in results]
        distances = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                distances.append(float(np.abs(vecs[i] - vecs[j]).sum()))

        mean_dist = sum(distances) / len(distances)
        agreement = 1.0 - (mean_dist / 2.0)
        return round(max(0.0, min(1.0, agreement)), 4)

    # ── Confidence label ──────────────────────────────────────────────────────

    @staticmethod
    def _confidence(max_prob):
        if max_prob > 0.55:
            return "HIGH"
        if max_prob > 0.45:
            return "MEDIUM"
        if max_prob > 0.35:
            return "LOW"
        return "VERY LOW"

    # ── Goals & score ─────────────────────────────────────────────────────────

    def _best_goals_estimate(self, nn_result, poisson_result):
        """Neural net takes priority; falls back to Poisson."""
        if nn_result is not None:
            return nn_result["expected_home_goals"], nn_result["expected_away_goals"]
        return poisson_result["expected_home_goals"], poisson_result["expected_away_goals"]

    def _most_likely_score(self, home_xg, away_xg, max_goals=7):
        _, _, _, mls = _goals_to_probs(home_xg, away_xg, max_goals=max_goals)
        return mls

    # ── Public prediction API ─────────────────────────────────────────────────

    def predict(self, home_team, away_team, stage="group", feature_vector=None):
        """
        Full ensemble prediction.

        Parameters
        ----------
        home_team      : str
        away_team      : str
        stage          : str  "group" | "round_of_16" | "quarter_final" |
                              "semi_final" | "final"
        feature_vector : pd.DataFrame or None
            Pre-built feature vector. Required for XGBoost and Neural Net.
            Pass None to use Poisson-only (ELO fallback).

        Returns
        -------
        dict with keys:
            home_win_prob, draw_prob, away_win_prob,
            expected_home_goals, expected_away_goals,
            most_likely_score, confidence, model_agreement,
            xgb_probs, nn_goals, poisson_probs
        """
        logger.debug(
            "EnsemblePredictor.predict: {} vs {} [{}]", home_team, away_team, stage
        )

        xgb_result     = self.xgb.predict(home_team, away_team, stage, feature_vector)
        nn_result      = self.nn.predict(home_team, away_team, stage, feature_vector)
        poisson_result = self.poisson.predict(home_team, away_team)

        home_win_prob, draw_prob, away_win_prob = self._blend(
            xgb_result, nn_result, poisson_result
        )

        home_xg, away_xg = self._best_goals_estimate(nn_result, poisson_result)
        most_likely_score = self._most_likely_score(home_xg, away_xg)

        max_prob   = max(home_win_prob, draw_prob, away_win_prob)
        confidence = self._confidence(max_prob)
        agreement  = self._model_agreement(xgb_result, nn_result, poisson_result)

        return {
            "home_win_prob":       home_win_prob,
            "draw_prob":           draw_prob,
            "away_win_prob":       away_win_prob,
            "expected_home_goals": round(home_xg, 2),
            "expected_away_goals": round(away_xg, 2),
            "most_likely_score":   most_likely_score,
            "confidence":          confidence,
            "model_agreement":     agreement,
            "xgb_probs":           xgb_result,
            "nn_goals":            nn_result,
            "poisson_probs": {
                "p_home_win":          poisson_result["p_home_win"],
                "p_draw":              poisson_result["p_draw"],
                "p_away_win":          poisson_result["p_away_win"],
                "expected_home_goals": poisson_result["expected_home_goals"],
                "expected_away_goals": poisson_result["expected_away_goals"],
                "most_likely_score":   poisson_result["most_likely_score"],
            },
        }

    def predict_tournament_match(self, home_team, away_team, stage):
        """
        Predict a tournament match without a pre-built feature vector.

        Uses ELO/Poisson as primary source; XGBoost and Neural Net are invoked
        but will return None and have their weight redistributed.
        Suitable for rapid bracket simulations.
        """
        logger.info(
            "predict_tournament_match: {} vs {} [{}] — ELO/Poisson fallback",
            home_team, away_team, stage,
        )
        return self.predict(home_team, away_team, stage, feature_vector=None)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — formatted prediction card
# ══════════════════════════════════════════════════════════════════════════════

def _bar(prob, width=20):
    filled = round(prob * width)
    return "#" * filled + "." * (width - filled)


def _print_prediction_card(home, away, stage, result):
    sep  = "=" * 64
    thin = "-" * 64

    lines = [
        "",
        sep,
        "  WC 2026 PREDICTION CARD",
        thin,
        "  {:<28s}  vs  {:<28s}".format(home, away),
        "  Stage: {}".format(stage),
        thin,
        "  HOME WIN  {}  {:>5.1%}".format(_bar(result["home_win_prob"]), result["home_win_prob"]),
        "  DRAW      {}  {:>5.1%}".format(_bar(result["draw_prob"]),     result["draw_prob"]),
        "  AWAY WIN  {}  {:>5.1%}".format(_bar(result["away_win_prob"]), result["away_win_prob"]),
        thin,
        "  Expected goals : {:.2f} - {:.2f}".format(
            result["expected_home_goals"], result["expected_away_goals"]
        ),
        "  Most likely    : {}".format(result["most_likely_score"]),
        "  Confidence     : {}".format(result["confidence"]),
        "  Model agreement: {:.0%}".format(result["model_agreement"]),
        thin,
    ]

    if result["xgb_probs"]:
        xp = result["xgb_probs"]
        lines.append(
            "  XGBoost   (40%): {:.1%} / {:.1%} / {:.1%}".format(
                xp["p_home_win"], xp["p_draw"], xp["p_away_win"]
            )
        )
    else:
        lines.append("  XGBoost   (40%): unavailable")

    if result["nn_goals"]:
        ng = result["nn_goals"]
        lines.append(
            "  NeuralNet (30%): {:.1%} / {:.1%} / {:.1%}".format(
                ng["p_home_win"], ng["p_draw"], ng["p_away_win"]
            )
        )
    else:
        lines.append("  NeuralNet (30%): unavailable")

    pp = result["poisson_probs"]
    lines.append(
        "  DixonColes(30%): {:.1%} / {:.1%} / {:.1%}".format(
            pp["p_home_win"], pp["p_draw"], pp["p_away_win"]
        )
    )
    lines += [sep, ""]

    for line in lines:
        sys.stdout.write(line + "\n")


def _cli():
    parser = argparse.ArgumentParser(
        prog="ensemble",
        description="WC 2026 ensemble predictor — prints a formatted prediction card.",
    )
    parser.add_argument("--home",  required=True, help="Home team name, e.g. 'France'")
    parser.add_argument("--away",  required=True, help="Away team name, e.g. 'Morocco'")
    parser.add_argument(
        "--stage",
        default="group",
        help="Match stage: group / round_of_16 / quarter_final / semi_final / final",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="XGB,NN,POISSON",
        help="Custom weights as comma-separated floats, e.g. '0.5,0.25,0.25'",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: WARNING)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    weights = None
    if args.weights:
        try:
            parts = [float(x) for x in args.weights.split(",")]
            if len(parts) != 3:
                parser.error("--weights must have exactly 3 comma-separated values.")
            total = sum(parts)
            weights = {
                "xgboost":    parts[0] / total,
                "neural_net": parts[1] / total,
                "poisson":    parts[2] / total,
            }
        except ValueError:
            parser.error("--weights values must be numeric floats.")

    predictor = EnsemblePredictor(weights=weights)
    result    = predictor.predict_tournament_match(args.home, args.away, args.stage)
    _print_prediction_card(args.home, args.away, args.stage, result)


if __name__ == "__main__":
    _cli()
