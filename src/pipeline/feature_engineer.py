"""
src/pipeline/feature_engineer.py
Assembles the full feature vector for each match.

This is the bridge between raw data and the ML models.
Every match becomes a ~120-number vector. The model never
sees raw text, dates, or team names — only numbers.

Feature groups:
  A. ELO (2 features)
  B. Rolling form — last 5 matches (14 features)
  C. Head-to-head history (6 features)
  D. StatsBomb advanced stats — last 3 WC matches (16 features)
  E. Player quality from FBref (16 features)
  F. Psychological signals (4 features)
  G. Contextual / situational (8 features)
  H. Tactical (4 features)
  ─────────────────────────────────────────
  Total: ~70 core features (expandable to 120 with interactions)

Each feature is computed from the perspective of the MATCH,
not a single team. Features come in pairs: home_X / away_X.
The label is: 0=away win, 1=draw, 2=home win.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from config import CACHE_DIR, WC_2026_TEAMS, FORMATIONS
from src.pipeline.elo import EloEngine
from src.pipeline.football_data_loader import (
    load_open_international_results,
    compute_form_features,
    compute_h2h_features,
)
from src.pipeline.fbref_scraper import (
    load_cached_squad,
    compute_squad_quality_score,
)
from src.utils.team_name_map import normalize

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FORMATION_ENC = {f: i for i, f in enumerate(FORMATIONS)}   # label encoding


# ── Helpers ──────────────────────────────────────────────────

def _safe(val, default=0.0) -> float:
    """Return float or default if None/NaN."""
    if val is None:
        return float(default)
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return float(default)


def _stage_weight(stage: str) -> float:
    """Tournament stage importance multiplier."""
    weights = {
        "group":        1.0,
        "round_of_32":  1.3,
        "round_of_16":  1.5,
        "quarter_final":2.0,
        "semi_final":   2.5,
        "third_place":  2.0,
        "final":        3.0,
        "friendly":     0.5,
        "qualifier":    0.8,
    }
    return weights.get(stage.lower().replace(" ", "_"), 1.0)


def _get_statsbomb_team_stats(
    team_name: str,
    as_of_date: datetime,
    historical_df: pd.DataFrame,
    n: int = 3,
) -> dict:
    """
    Get average StatsBomb advanced stats for a team's last N matches
    that have xG data available.
    Returns per-match averages: xg, possession, pass_acc, shots, pressures.
    """
    canonical = normalize(team_name)
    mask = (
        (
            (historical_df.get("home_team", pd.Series(dtype=str)) == canonical) |
            (historical_df.get("away_team", pd.Series(dtype=str)) == canonical)
        ) &
        (historical_df["match_date"] < as_of_date) &
        (historical_df.get("home_xg", pd.Series(dtype=float)).notna())
    )
    recent = historical_df[mask].sort_values("match_date", ascending=False).head(n)

    if recent.empty:
        return {"xg_avg": 1.2, "xga_avg": 1.2, "poss_avg": 50.0,
                "pass_acc_avg": 80.0, "shots_avg": 12.0, "pressures_avg": 150.0}

    xg_vals, xga_vals, poss_vals, pass_vals, shot_vals, press_vals = [], [], [], [], [], []

    for _, row in recent.iterrows():
        is_home = normalize(str(row.get("home_team", ""))) == canonical
        if is_home:
            xg_vals.append(_safe(row.get("home_xg"), 1.2))
            xga_vals.append(_safe(row.get("away_xg"), 1.2))
            poss_vals.append(_safe(row.get("home_possession"), 50.0))
            pass_vals.append(_safe(row.get("home_pass_acc"), 80.0))
            shot_vals.append(_safe(row.get("home_shots"), 12.0))
            press_vals.append(_safe(row.get("home_pressures"), 150.0))
        else:
            xg_vals.append(_safe(row.get("away_xg"), 1.2))
            xga_vals.append(_safe(row.get("home_xg"), 1.2))
            poss_vals.append(_safe(row.get("away_possession"), 50.0))
            pass_vals.append(_safe(row.get("away_pass_acc"), 80.0))
            shot_vals.append(_safe(row.get("away_shots"), 12.0))
            press_vals.append(_safe(row.get("away_pressures"), 150.0))

    def avg(lst, default): return round(float(np.mean(lst)), 3) if lst else default

    return {
        "xg_avg":        avg(xg_vals, 1.2),
        "xga_avg":       avg(xga_vals, 1.2),
        "poss_avg":      avg(poss_vals, 50.0),
        "pass_acc_avg":  avg(pass_vals, 80.0),
        "shots_avg":     avg(shot_vals, 12.0),
        "pressures_avg": avg(press_vals, 150.0),
    }


def _get_psych_scores(
    team_name: str,
    match_date: datetime,
    psych_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Aggregate reviewed psychological signals for a team into numeric scores.
    psych_df: DataFrame from psych_signals table (reviewed=True rows only).
    If None: returns neutral defaults.
    """
    if psych_df is None or psych_df.empty:
        return {"psych_score": 0.0, "psych_risk_flags": 0, "psych_severity_max": 0}

    canonical = normalize(team_name)
    window_start = match_date - pd.Timedelta(days=14)  # last 2 weeks

    relevant = psych_df[
        (psych_df["team_name"] == canonical) &
        (psych_df["reviewed"] == True) &
        (psych_df["applies_to_match_date"] >= window_start) &
        (psych_df["applies_to_match_date"] <= match_date)
    ]

    if relevant.empty:
        return {"psych_score": 0.0, "psych_risk_flags": 0, "psych_severity_max": 0}

    # Aggregate: negative signals pull down, positive pull up
    scores = []
    for _, sig in relevant.iterrows():
        base = _safe(sig.get("sentiment_score"), 0.0)
        sev  = _safe(sig.get("severity"), 1.0)
        scores.append(base * (sev / 5.0))  # normalise severity to 0-1

    return {
        "psych_score":       round(float(np.mean(scores)), 4),
        "psych_risk_flags":  int((relevant["severity"] >= 3).sum()),
        "psych_severity_max":int(relevant["severity"].max()),
    }


# ── Core feature builder ─────────────────────────────────────

class FeatureEngineer:
    """
    Builds ML-ready feature vectors for matches.

    Usage:
        fe = FeatureEngineer()
        fe.load_data()              # loads historical data into memory
        vector = fe.build_features_for_match(
            home_team="France",
            away_team="Morocco",
            match_date=datetime(2026, 6, 20),
            stage="group",
        )
        # vector is a flat dict of ~70 features
    """

    def __init__(self):
        self.hist_df: Optional[pd.DataFrame] = None
        self.elo_engine: Optional[EloEngine] = None
        self.squad_quality: dict[str, dict] = {}   # team → quality scores
        self._loaded = False

    def load_data(self, rebuild_elo: bool = False):
        """
        Load all data sources into memory.
        Call this once before building features.
        rebuild_elo=True forces re-running ELO from scratch (slow, ~30s).
        """
        logger.info("Loading data sources...")

        # Historical match results
        self.hist_df = load_open_international_results()
        self.hist_df["home_team"] = self.hist_df["home_team"].apply(normalize)
        self.hist_df["away_team"] = self.hist_df["away_team"].apply(normalize)

        # ELO engine — try to load from cache first
        elo_cache = CACHE_DIR / "elo_history.parquet"
        if elo_cache.exists() and not rebuild_elo:
            logger.info("Loading ELO from cache...")
            self.elo_engine = EloEngine()
            # Re-process only to get current ratings (fast since data is cached)
            df_1990 = self.hist_df[self.hist_df["match_date"] >= "1990-01-01"].copy()
            self.elo_engine.process_dataframe(df_1990)
        else:
            logger.info("Building ELO from scratch (this takes ~30s)...")
            self.elo_engine = EloEngine()
            df_1990 = self.hist_df[self.hist_df["match_date"] >= "1990-01-01"].copy()
            self.elo_engine.process_dataframe(df_1990)

        # Squad quality from FBref cache (offline — only if cached parquets exist)
        logger.info("Loading squad quality scores from FBref cache...")
        loaded = 0
        for team in WC_2026_TEAMS:
            squad_df = load_cached_squad(team)
            if not squad_df.empty:
                self.squad_quality[team] = compute_squad_quality_score(squad_df)
                loaded += 1
        logger.info(f"Squad quality loaded for {loaded}/{len(WC_2026_TEAMS)} teams")

        self._loaded = True
        logger.success("FeatureEngineer ready.")

    def build_features_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        stage: str = "group",
        neutral: bool = True,
        home_formation: Optional[str] = None,
        away_formation: Optional[str] = None,
        psych_df: Optional[pd.DataFrame] = None,
        outcome: Optional[str] = None,            # "home"/"draw"/"away" (for training)
        home_goals: Optional[int] = None,
        away_goals: Optional[int] = None,
    ) -> dict:
        """
        Build the complete feature vector for one match.
        Returns a flat dict ready to insert into match_features table
        or pass directly to the ML model.
        """
        if not self._loaded:
            raise RuntimeError("Call load_data() first.")

        home = normalize(home_team)
        away = normalize(away_team)
        features = {}

        # ── A. ELO features ──────────────────────────────────
        elo_pred = self.elo_engine.predict_match(home, away, neutral=neutral)
        features["home_elo"]       = _safe(elo_pred["home_elo"], 1500)
        features["away_elo"]       = _safe(elo_pred["away_elo"], 1500)
        features["elo_diff"]       = features["home_elo"] - features["away_elo"]
        features["elo_home_win_prob"] = _safe(elo_pred["p_home_win"])
        features["elo_draw_prob"]     = _safe(elo_pred["p_draw"])
        features["elo_away_win_prob"] = _safe(elo_pred["p_away_win"])

        # ── B. Rolling form — last 5 matches ─────────────────
        home_form = compute_form_features(home, match_date, n=5)
        away_form = compute_form_features(away, match_date, n=5)

        for team_label, form in [("home", home_form), ("away", away_form)]:
            features[f"{team_label}_form_pts"]        = _safe(form["form_pts"])
            features[f"{team_label}_form_gf"]         = _safe(form["form_gf"])
            features[f"{team_label}_form_ga"]         = _safe(form["form_ga"])
            features[f"{team_label}_form_gd"]         = _safe(form["form_gd"])
            features[f"{team_label}_form_wins"]       = _safe(form["form_wins"])
            features[f"{team_label}_form_draws"]      = _safe(form["form_draws"])
            features[f"{team_label}_form_losses"]     = _safe(form["form_losses"])
            features[f"{team_label}_days_since_last"] = _safe(form["days_since_last"], 30)

        # Derived form differential
        features["form_pts_diff"] = features["home_form_pts"] - features["away_form_pts"]
        features["form_gd_diff"]  = features["home_form_gd"]  - features["away_form_gd"]

        # ── C. Head-to-head ───────────────────────────────────
        h2h = compute_h2h_features(home, away, match_date, n=10)
        features["h2h_home_win_rate"] = _safe(h2h["h2h_home_win_rate"], 0.33)
        features["h2h_draw_rate"]     = _safe(h2h["h2h_draw_rate"],     0.25)
        features["h2h_away_win_rate"] = _safe(h2h["h2h_away_win_rate"], 0.33)
        features["h2h_home_gf_avg"]   = _safe(h2h["h2h_home_gf_avg"],   1.2)
        features["h2h_away_gf_avg"]   = _safe(h2h["h2h_away_gf_avg"],   1.2)
        features["h2h_n_matches"]     = _safe(h2h["h2h_matches"],        0)

        # ── D. StatsBomb advanced stats ───────────────────────
        if self.hist_df is not None:
            home_sb = _get_statsbomb_team_stats(home, match_date, self.hist_df)
            away_sb = _get_statsbomb_team_stats(away, match_date, self.hist_df)
        else:
            home_sb = away_sb = {}

        for team_label, sb in [("home", home_sb), ("away", away_sb)]:
            features[f"{team_label}_xg_avg"]        = _safe(sb.get("xg_avg"), 1.2)
            features[f"{team_label}_xga_avg"]       = _safe(sb.get("xga_avg"), 1.2)
            features[f"{team_label}_poss_avg"]      = _safe(sb.get("poss_avg"), 50.0)
            features[f"{team_label}_pass_acc_avg"]  = _safe(sb.get("pass_acc_avg"), 80.0)
            features[f"{team_label}_shots_avg"]     = _safe(sb.get("shots_avg"), 12.0)
            features[f"{team_label}_pressures_avg"] = _safe(sb.get("pressures_avg"), 150.0)

        features["xg_diff"]   = features["home_xg_avg"]   - features["away_xg_avg"]
        features["poss_diff"] = features["home_poss_avg"]  - features["away_poss_avg"]

        # ── E. Player quality (FBref) ─────────────────────────
        default_quality = {
            "squad_avg_rating": 5.0, "top11_avg_rating": 5.0,
            "depth_score": 0.0, "gk_rating": 5.0,
            "def_rating": 5.0, "mid_rating": 5.0,
            "att_rating": 5.0,
        }
        home_q = self.squad_quality.get(home, default_quality)
        away_q = self.squad_quality.get(away, default_quality)

        for team_label, q in [("home", home_q), ("away", away_q)]:
            features[f"{team_label}_squad_avg_rating"]  = _safe(q.get("squad_avg_rating"), 5.0)
            features[f"{team_label}_top11_avg_rating"]  = _safe(q.get("top11_avg_rating"), 5.0)
            features[f"{team_label}_depth_score"]       = _safe(q.get("depth_score"), 0.0)
            features[f"{team_label}_gk_rating"]         = _safe(q.get("gk_rating"), 5.0)
            features[f"{team_label}_def_rating"]        = _safe(q.get("def_rating"), 5.0)
            features[f"{team_label}_mid_rating"]        = _safe(q.get("mid_rating"), 5.0)
            features[f"{team_label}_att_rating"]        = _safe(q.get("att_rating"), 5.0)

        features["player_quality_diff"] = (
            features["home_top11_avg_rating"] - features["away_top11_avg_rating"]
        )

        # ── F. Psychological signals ──────────────────────────
        home_psych = _get_psych_scores(home, match_date, psych_df)
        away_psych = _get_psych_scores(away, match_date, psych_df)

        features["home_psych_score"]       = _safe(home_psych["psych_score"])
        features["away_psych_score"]       = _safe(away_psych["psych_score"])
        features["home_psych_risk_flags"]  = _safe(home_psych["psych_risk_flags"])
        features["away_psych_risk_flags"]  = _safe(away_psych["psych_risk_flags"])
        features["psych_score_diff"]       = (
            features["home_psych_score"] - features["away_psych_score"]
        )

        # ── G. Contextual features ────────────────────────────
        features["stage_weight"]   = _stage_weight(stage)
        features["is_neutral"]     = float(neutral)
        features["match_month"]    = float(match_date.month)
        features["is_knockout"]    = float(stage not in ["group", "qualifier", "friendly"])

        # Confederation encoding — models pick up confederation-level patterns
        # (e.g. CONMEBOL teams systematically differ from AFC teams at WC)
        conf_map = {
            "AFC": 0, "CAF": 1, "CONCACAF": 2,
            "CONMEBOL": 3, "OFC": 4, "UEFA": 5, "UNKNOWN": -1
        }
        from src.utils.team_name_map import get_confederation
        features["home_confederation"] = float(conf_map.get(get_confederation(home), -1))
        features["away_confederation"] = float(conf_map.get(get_confederation(away), -1))

        # Rest days (proxy — days since last match, capped at 30)
        features["home_rest_days"] = min(float(features["home_days_since_last"]), 30.0)
        features["away_rest_days"] = min(float(features["away_days_since_last"]), 30.0)
        features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

        # ── H. Tactical features ──────────────────────────────
        features["home_formation_enc"] = float(FORMATION_ENC.get(home_formation or "", -1))
        features["away_formation_enc"] = float(FORMATION_ENC.get(away_formation or "", -1))

        # Tactical matchup score: simple heuristic (will be replaced by ML in Day 7)
        # Attacking formations (4-3-3, 3-4-3) vs defensive (5-3-2) create matchup dynamics
        attacking = {"4-3-3", "3-4-3", "4-1-4-1"}
        defensive = {"5-3-2", "4-4-2"}
        home_att = float(home_formation in attacking) if home_formation else 0.5
        away_def = float(away_formation in defensive) if away_formation else 0.5
        features["tactical_matchup_score"] = round(home_att - away_def, 2)

        # ── Interaction features (derived combinations) ───────
        # These often help tree models capture non-linear relationships
        features["elo_x_form"]         = features["elo_diff"] * features["form_pts_diff"]
        features["quality_x_psych"]    = (
            features["player_quality_diff"] + features["psych_score_diff"]
        )
        features["xg_x_quality"]       = features["xg_diff"] * features["player_quality_diff"]
        features["momentum_score_home"] = (
            features["home_form_pts"] * 0.4 +
            features["home_xg_avg"]  * 0.3 +
            features["home_psych_score"] * 0.3
        )
        features["momentum_score_away"] = (
            features["away_form_pts"] * 0.4 +
            features["away_xg_avg"]  * 0.3 +
            features["away_psych_score"] * 0.3
        )
        features["momentum_diff"] = (
            features["momentum_score_home"] - features["momentum_score_away"]
        )

        # ── Target labels (for training only) ─────────────────
        if outcome:
            label_map = {"home": 2, "draw": 1, "away": 0}
            features["target_outcome"]    = label_map.get(outcome, -1)
            features["target_home_goals"] = float(home_goals) if home_goals is not None else np.nan
            features["target_away_goals"] = float(away_goals) if away_goals is not None else np.nan

        # ── Metadata (not used in training, for tracing) ──────
        features["_home_team"]   = home
        features["_away_team"]   = away
        features["_match_date"]  = str(match_date.date())
        features["_stage"]       = stage

        return features

    def build_training_dataset(
        self,
        competition_filter: str = "FIFA World Cup",
        start_year: int = 2006,
    ) -> pd.DataFrame:
        """
        Build the full training dataset from all historical WC matches.
        Each row = one match = one feature vector with the outcome label.

        start_year: 2006+ recommended — earlier WC data has less rich stats.
        """
        if not self._loaded:
            raise RuntimeError("Call load_data() first.")

        wc_mask = (
            self.hist_df["competition"].str.contains(competition_filter, case=False, na=False) &
            ~self.hist_df["competition"].str.contains("qualification|qualifier", case=False, na=False) &
            (self.hist_df["match_date"].dt.year >= start_year)
        )
        wc_matches = self.hist_df[wc_mask].copy()
        logger.info(f"Building features for {len(wc_matches)} WC matches ({start_year}+)...")

        rows = []
        for i, (_, match) in enumerate(wc_matches.iterrows()):
            try:
                feat = self.build_features_for_match(
                    home_team   = match["home_team"],
                    away_team   = match["away_team"],
                    match_date  = pd.to_datetime(match["match_date"]),
                    stage       = match.get("stage", "group"),
                    neutral     = bool(match.get("neutral_ground", True)),
                    outcome     = match.get("outcome"),
                    home_goals  = int(match["home_goals"]),
                    away_goals  = int(match["away_goals"]),
                )
                rows.append(feat)
            except Exception as e:
                logger.debug(f"Skipping match {i}: {e}")
                continue

            if (i + 1) % 50 == 0:
                logger.info(f"  ...{i+1}/{len(wc_matches)} matches processed")

        df = pd.DataFrame(rows)
        # Drop metadata columns for pure ML use
        meta_cols = [c for c in df.columns if c.startswith("_")]
        feature_cols = [c for c in df.columns if not c.startswith("_")]
        df_features = df[feature_cols]

        logger.success(
            f"Training dataset: {len(df_features)} matches × {len(feature_cols)} features\n"
            f"  Outcome distribution:\n{df_features['target_outcome'].value_counts().to_string()}"
        )

        # Cache it
        out_path = CACHE_DIR / "training_features.parquet"
        df_features.to_parquet(out_path)
        logger.info(f"Saved to {out_path}")
        return df_features

    def build_prediction_vector(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        stage: str = "group",
        home_formation: Optional[str] = None,
        away_formation: Optional[str] = None,
        psych_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a single-row DataFrame ready for model.predict().
        This is what you'll call for real WC 2026 predictions.
        """
        feat = self.build_features_for_match(
            home_team      = home_team,
            away_team      = away_team,
            match_date     = match_date,
            stage          = stage,
            neutral        = True,   # WC is always neutral ground
            home_formation = home_formation,
            away_formation = away_formation,
            psych_df       = psych_df,
        )
        # Drop metadata and target columns
        drop_cols = [c for c in feat if c.startswith("_") or c.startswith("target_")]
        for col in drop_cols:
            feat.pop(col, None)

        return pd.DataFrame([feat])


# ── Feature names list (for model column ordering) ────────────
def get_feature_names(include_targets: bool = False) -> list[str]:
    """
    Return the canonical ordered list of feature names.
    Models must always see features in this exact order.
    """
    fe = FeatureEngineer()
    # Build a dummy vector to get column names
    fe._loaded = True
    fe.hist_df = pd.DataFrame(columns=[
        "home_team","away_team","home_goals","away_goals",
        "match_date","stage","neutral_ground","competition",
        "home_xg","away_xg","home_possession","away_possession",
        "home_pass_acc","away_pass_acc","home_shots","away_shots",
        "home_pressures","away_pressures",
    ])
    fe.elo_engine = EloEngine()

    dummy = fe.build_features_for_match(
        "France", "Morocco", datetime(2026, 6, 20), "group"
    )
    cols = [k for k in dummy if not k.startswith("_") and
            (include_targets or not k.startswith("target_"))]
    return cols


if __name__ == "__main__":
    logger.info("Testing FeatureEngineer...")
    fe = FeatureEngineer()
    fe.load_data()

    # Build one prediction vector
    print("\n── Prediction vector: France vs Morocco (Group stage) ──")
    vec = fe.build_prediction_vector(
        home_team = "France",
        away_team = "Morocco",
        match_date = datetime(2026, 6, 20),
        stage = "group",
    )
    print(f"Feature vector shape: {vec.shape}")
    print("\nKey features:")
    key_cols = [
        "home_elo","away_elo","elo_diff",
        "home_form_pts","away_form_pts","form_pts_diff",
        "h2h_home_win_rate","h2h_away_win_rate",
        "home_xg_avg","away_xg_avg","xg_diff",
        "home_psych_score","away_psych_score",
        "stage_weight","momentum_diff",
    ]
    for col in key_cols:
        if col in vec.columns:
            print(f"  {col:35s}: {vec[col].values[0]:.4f}")

    # ELO-only prediction
    print("\n── ELO prediction ──")
    pred = fe.elo_engine.predict_match("France", "Morocco")
    print(f"  France win:  {pred['p_home_win']:.1%}")
    print(f"  Draw:        {pred['p_draw']:.1%}")
    print(f"  Morocco win: {pred['p_away_win']:.1%}")

    # Build training dataset (WC matches only)
    print("\n── Building training dataset ──")
    train_df = fe.build_training_dataset(start_year=2006)
    print(f"\nFinal dataset: {train_df.shape}")
    print(f"Features: {[c for c in train_df.columns if not c.startswith('target_')][:10]}...")
    print(f"\nOutcome distribution:")
    outcome_map = {0: "Away win", 1: "Draw", 2: "Home win"}
    for k, v in train_df["target_outcome"].value_counts().items():
        pct = v / len(train_df) * 100
        print(f"  {outcome_map.get(int(k), k)}: {v} ({pct:.1f}%)")
