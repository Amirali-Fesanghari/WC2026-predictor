"""
src/models/xgboost_model.py
XGBoost match outcome classifier.

Predicts: 0=Away win, 1=Draw, 2=Home win

Why XGBoost for this problem:
  - Handles mixed feature types well (ELO floats, formation ints, psych scores)
  - Built-in feature importance
  - Robust to missing values (important: some teams lack FBref data)
  - Trains in seconds on our 320-match dataset
  - Works great with SHAP for explainability

Training strategy:
  - TimeSeriesSplit cross-validation (CRITICAL for sports data —
    you must NEVER train on future matches to predict past ones)
  - Hyperparameter tuning via RandomizedSearchCV
  - Probability calibration (Platt scaling) so 60% really means 60%
  - Class weighting to handle draw imbalance (draws are hardest to predict)
"""
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    log_loss, confusion_matrix, brier_score_loss
)
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CACHE_DIR, DB_PATH

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Columns that must be dropped before training (metadata, not features)
META_COLS = [
    "_home_team", "_away_team", "_match_date", "_stage",
    "target_home_goals", "target_away_goals",
]
TARGET_COL = "target_outcome"

# Features to always exclude from the model
# (leaky: formation is unknown before match, psych only if unreviewed)
EXCLUDE_COLS = [
    "home_formation_enc", "away_formation_enc",   # unknown pre-match
    "tactical_matchup_score",                      # depends on formation
]


class WCOutcomeModel:
    """
    XGBoost classifier for World Cup match outcome prediction.

    Workflow:
        model = WCOutcomeModel()
        model.load_training_data()          # loads cached feature parquet
        model.train()                       # trains + evaluates
        model.save()                        # saves to disk
        probs = model.predict("France", "Morocco", feature_vector)
        model.explain(feature_vector)       # SHAP waterfall plot
    """

    def __init__(self):
        self.model: Optional[CalibratedClassifierCV] = None
        self.feature_names: list[str] = []
        self.train_df: Optional[pd.DataFrame] = None
        self.shap_explainer = None
        self.version = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Data loading ─────────────────────────────────────────

    def load_training_data(self, parquet_path: Optional[Path] = None) -> pd.DataFrame:
        """Load the cached training features built by feature_engineer.py"""
        path = parquet_path or (CACHE_DIR / "training_features.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"Training data not found at {path}.\n"
                f"Run: python -m src.pipeline.feature_engineer first."
            )
        df = pd.read_parquet(path)
        logger.info(f"Loaded training data: {df.shape[0]} matches × {df.shape[1]} columns")

        # Drop rows with missing target
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

        self.train_df = df
        logger.success(
            f"Training data ready: {len(df)} matches\n"
            f"  Home wins: {(df[TARGET_COL]==2).sum()}  "
            f"Draws: {(df[TARGET_COL]==1).sum()}  "
            f"Away wins: {(df[TARGET_COL]==0).sum()}"
        )
        return df

    def _prepare_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Extract X (features) and y (target) from the DataFrame."""
        drop = META_COLS + EXCLUDE_COLS + [TARGET_COL]
        drop_existing = [c for c in drop if c in df.columns]

        X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
        y = df[TARGET_COL].astype(int)

        # Fill any remaining NaN with column median (robust imputation)
        X = X.fillna(X.median())

        self.feature_names = list(X.columns)
        return X, y

    # ── Training ─────────────────────────────────────────────

    def train(
        self,
        tune_hyperparams: bool = True,
        n_splits: int = 5,
        calibrate: bool = True,
    ) -> dict:
        """
        Train the XGBoost model with time-series cross-validation.

        tune_hyperparams: run RandomizedSearchCV (adds ~30s, improves results)
        n_splits: number of CV folds (5 = good balance for 320 matches)
        calibrate: apply Platt scaling for better probability estimates

        Returns: dict of evaluation metrics
        """
        if self.train_df is None:
            raise RuntimeError("Call load_training_data() first.")

        X, y = self._prepare_xy(self.train_df)
        logger.info(f"Training on {len(X)} matches, {len(self.feature_names)} features")

        # ── Class weights (draws are hardest — upweight them) ──
        class_counts = y.value_counts()
        total = len(y)
        # Inverse frequency weighting
        weights = {
            cls: total / (3 * count)
            for cls, count in class_counts.items()
        }
        sample_weights = y.map(weights).values
        logger.info(f"Class weights: {weights}")

        # ── Base XGBoost config ────────────────────────────────
        base_params = {
            "objective":        "multi:softprob",
            "num_class":        3,
            "eval_metric":      "mlogloss",
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
            # Regularisation — important for small dataset (320 matches)
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "max_depth":        4,      # shallow — prevents overfitting on small data
            "learning_rate":    0.05,
            "n_estimators":     300,
        }

        if tune_hyperparams:
            logger.info("Tuning hyperparameters (RandomizedSearchCV)...")
            param_grid = {
                "max_depth":        [3, 4, 5, 6],
                "learning_rate":    [0.01, 0.05, 0.1, 0.15],
                "n_estimators":     [100, 200, 300, 500],
                "subsample":        [0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5, 7],
                "reg_alpha":        [0, 0.1, 0.5, 1.0],   # L1
                "reg_lambda":       [1, 1.5, 2.0, 3.0],   # L2
            }
            # TimeSeriesSplit: CRITICAL — fold N only trains on matches before fold N
            # This prevents data leakage (model can't "see the future")
            tscv = TimeSeriesSplit(n_splits=n_splits)

            search = RandomizedSearchCV(
                XGBClassifier(**base_params),
                param_distributions=param_grid,
                n_iter=40,
                cv=tscv,
                scoring="neg_log_loss",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X, y, sample_weight=sample_weights)
            best_params = {**base_params, **search.best_params_}
            logger.info(f"Best params: {search.best_params_}")
            xgb = XGBClassifier(**best_params)
        else:
            xgb = XGBClassifier(**base_params)

        # ── Probability calibration ────────────────────────────
        # Raw XGBoost probabilities are often overconfident.
        # CalibratedClassifierCV wraps it with isotonic regression
        # so predicted probabilities are better calibrated.
        if calibrate:
            tscv_cal = TimeSeriesSplit(n_splits=3)
            self.model = CalibratedClassifierCV(xgb, cv=tscv_cal, method="isotonic")
        else:
            self.model = xgb

        # ── Final fit on all training data ────────────────────
        logger.info("Fitting final model on full training set...")
        if calibrate:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, sample_weight=sample_weights)

        # ── Evaluation ────────────────────────────────────────
        metrics = self._evaluate(X, y, n_splits=n_splits)

        # ── SHAP explainer ────────────────────────────────────
        logger.info("Building SHAP explainer...")
        try:
            # For calibrated model, use the underlying XGBoost estimator
            if calibrate:
                base_xgb = self.model.calibrated_classifiers_[0].estimator
            else:
                base_xgb = self.model
            self.shap_explainer = shap.TreeExplainer(base_xgb)
            logger.success("SHAP explainer ready.")
        except Exception as e:
            logger.warning(f"SHAP explainer failed: {e}")

        return metrics

    def _evaluate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """Run cross-validated evaluation and print a full report."""
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Cross-validated log loss (main metric — lower is better)
        cv_logloss = cross_val_score(
            self.model, X, y,
            cv=tscv, scoring="neg_log_loss", n_jobs=-1
        )
        cv_acc = cross_val_score(
            self.model, X, y,
            cv=tscv, scoring="accuracy", n_jobs=-1
        )

        # In-sample predictions for confusion matrix
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        acc    = accuracy_score(y, y_pred)
        ll     = log_loss(y, y_proba)

        # Brier score per class (lower = better calibration)
        brier_home = brier_score_loss((y == 2).astype(int), y_proba[:, 2])
        brier_draw = brier_score_loss((y == 1).astype(int), y_proba[:, 1])
        brier_away = brier_score_loss((y == 0).astype(int), y_proba[:, 0])

        label_names = ["Away win", "Draw", "Home win"]
        report = classification_report(y, y_pred, target_names=label_names)
        cm = confusion_matrix(y, y_pred)

        metrics = {
            "cv_logloss_mean":  round(-cv_logloss.mean(), 4),
            "cv_logloss_std":   round(cv_logloss.std(), 4),
            "cv_accuracy_mean": round(cv_acc.mean(), 4),
            "cv_accuracy_std":  round(cv_acc.std(), 4),
            "train_accuracy":   round(acc, 4),
            "train_logloss":    round(ll, 4),
            "brier_home":       round(brier_home, 4),
            "brier_draw":       round(brier_draw, 4),
            "brier_away":       round(brier_away, 4),
        }

        print("\n" + "═" * 60)
        print("  MODEL EVALUATION REPORT")
        print("═" * 60)
        print(f"\n  CV Log Loss:  {metrics['cv_logloss_mean']:.4f} ± {metrics['cv_logloss_std']:.4f}")
        print(f"  CV Accuracy:  {metrics['cv_accuracy_mean']:.1%} ± {metrics['cv_accuracy_std']:.1%}")
        print(f"  Train Acc:    {metrics['train_accuracy']:.1%}  (in-sample, ignore if too high)")
        print(f"\n  Brier scores (lower = better calibrated):")
        print(f"    Home win: {metrics['brier_home']:.4f}")
        print(f"    Draw:     {metrics['brier_draw']:.4f}")
        print(f"    Away win: {metrics['brier_away']:.4f}")
        print(f"\n  Classification report (in-sample):")
        for line in report.split("\n"):
            print(f"    {line}")
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"    {'':12s} Away  Draw  Home")
        for i, row in enumerate(cm):
            print(f"    {label_names[i]:12s} {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
        print("═" * 60 + "\n")

        return metrics

    # ── Feature importance ───────────────────────────────────

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top N most important features by SHAP mean absolute value."""
        if self.shap_explainer is None:
            raise RuntimeError("Train the model first.")

        X, _ = self._prepare_xy(self.train_df)
        shap_values = self.shap_explainer.shap_values(X)

        # shap_values shape: (n_classes, n_samples, n_features)
        # Average absolute SHAP across all classes and samples
        mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))

        importance_df = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": mean_abs_shap,
        }).sort_values("importance", ascending=False).head(top_n)

        print(f"\n  Top {top_n} features by SHAP importance:")
        for _, row in importance_df.iterrows():
            bar = "█" * int(row["importance"] / importance_df["importance"].max() * 30)
            print(f"    {row['feature']:35s} {bar} {row['importance']:.4f}")

        return importance_df

    # ── Prediction ───────────────────────────────────────────

    def predict(
        self,
        feature_vector: pd.DataFrame,
        home_team: str = "",
        away_team: str = "",
    ) -> dict:
        """
        Predict match outcome probabilities from a feature vector.

        feature_vector: single-row DataFrame from feature_engineer.build_prediction_vector()
        Returns: dict with probabilities and recommended bet/outcome.
        """
        if self.model is None:
            raise RuntimeError("Train or load the model first.")

        # Align columns to training feature order
        X = feature_vector.reindex(columns=self.feature_names, fill_value=0.0)
        X = X.fillna(X.median() if not X.empty else 0.0)

        proba = self.model.predict_proba(X)[0]  # [away_win, draw, home_win]
        pred_class = int(np.argmax(proba))

        outcome_labels = {0: "Away win", 1: "Draw", 2: "Home win"}
        confidence_label = (
            "HIGH"   if max(proba) > 0.55 else
            "MEDIUM" if max(proba) > 0.40 else
            "LOW"
        )

        result = {
            "home_team":      home_team,
            "away_team":      away_team,
            "p_home_win":     round(float(proba[2]), 4),
            "p_draw":         round(float(proba[1]), 4),
            "p_away_win":     round(float(proba[0]), 4),
            "predicted":      outcome_labels[pred_class],
            "confidence":     confidence_label,
            "max_prob":       round(float(max(proba)), 4),
        }
        return result

    def explain_prediction(
        self,
        feature_vector: pd.DataFrame,
        home_team: str = "",
        away_team: str = "",
        outcome_idx: int = 2,   # 0=away, 1=draw, 2=home (default: explain home win)
    ) -> dict:
        """
        Explain WHY the model made a prediction using SHAP.
        Returns the top drivers pushing toward / against the predicted outcome.

        outcome_idx: which class to explain (2=home win is most natural)
        """
        if self.shap_explainer is None:
            raise RuntimeError("SHAP explainer not available.")

        X = feature_vector.reindex(columns=self.feature_names, fill_value=0.0)
        X = X.fillna(0.0)

        shap_vals = self.shap_explainer.shap_values(X)
        # shap_vals[outcome_idx]: shape (1, n_features)
        sv = shap_vals[outcome_idx][0]

        explanation = pd.DataFrame({
            "feature": self.feature_names,
            "value":   X.values[0],
            "shap":    sv,
        }).sort_values("shap", key=abs, ascending=False).head(10)

        outcome_name = {0: "Away win", 1: "Draw", 2: "Home win"}[outcome_idx]
        team_label = home_team or "Home"

        print(f"\n  SHAP explanation — Why {team_label} {'wins' if outcome_idx==2 else 'draws/loses'}?")
        print(f"  (explaining: {outcome_name})")
        print(f"  {'Feature':35s} {'Value':>10s}  {'SHAP impact':>12s}  Direction")
        print("  " + "─" * 75)
        for _, row in explanation.iterrows():
            direction = "▲ HELPS " if row["shap"] > 0 else "▼ HURTS "
            bar = ("+" if row["shap"] > 0 else "-") * min(15, int(abs(row["shap"]) * 100))
            print(f"  {row['feature']:35s} {row['value']:>10.3f}  {row['shap']:>+12.4f}  {direction} {bar}")

        return explanation

    # ── Save / Load ──────────────────────────────────────────

    def save(self, name: str = "xgboost_outcome") -> Path:
        """Save model and metadata to disk."""
        path = MODELS_DIR / f"{name}_{self.version}.joblib"
        joblib.dump({
            "model":         self.model,
            "feature_names": self.feature_names,
            "version":       self.version,
            "trained_at":    datetime.utcnow().isoformat(),
            "n_training":    len(self.train_df) if self.train_df is not None else 0,
        }, path)
        logger.success(f"Model saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "WCOutcomeModel":
        """Load a saved model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model         = data["model"]
        instance.feature_names = data["feature_names"]
        instance.version       = data["version"]
        logger.success(f"Model loaded: {path} (trained {data['trained_at']})")
        return instance

    @classmethod
    def load_latest(cls) -> "WCOutcomeModel":
        """Load the most recently saved model."""
        saved = sorted(MODELS_DIR.glob("xgboost_outcome_*.joblib"))
        if not saved:
            raise FileNotFoundError(f"No saved models in {MODELS_DIR}")
        return cls.load(saved[-1])


# ── Standalone runner ────────────────────────────────────────
if __name__ == "__main__":
    from src.pipeline.feature_engineer import FeatureEngineer

    print("═" * 60)
    print("  WC 2026 Predictor — Day 3: XGBoost Model Training")
    print("═" * 60)

    # ── Step 1: Train ─────────────────────────────────────
    model = WCOutcomeModel()
    model.load_training_data()
    metrics = model.train(tune_hyperparams=True)

    # ── Step 2: Feature importance ─────────────────────────
    try:
        model.get_feature_importance(top_n=15)
    except Exception as e:
        print(f"  Feature importance skipped: {e}")

    # ── Step 3: Save ──────────────────────────────────────
    saved_path = model.save()

    # ── Step 4: Live prediction ───────────────────────────
    print("\n  Building live predictions for WC 2026 group stage...")
    fe = FeatureEngineer()
    fe.load_data()

    test_matches = [
        ("France",        "Belgium",       "group"),
        ("Brazil",        "Argentina",     "group"),
        ("England",       "Germany",       "group"),
        ("Morocco",       "Spain",         "group"),
        ("Japan",         "South Korea",   "group"),
        ("United States", "Mexico",        "group"),
    ]

    print(f"\n  {'Match':35s} {'Home%':>7s} {'Draw%':>7s} {'Away%':>7s}  {'Predicted':12s}  Conf")
    print("  " + "─" * 80)

    for home, away, stage in test_matches:
        try:
            vec = fe.build_prediction_vector(
                home_team=home, away_team=away,
                match_date=datetime(2026, 6, 20), stage=stage,
            )
            pred = model.predict(vec, home_team=home, away_team=away)
            match_str = f"{home} vs {away}"
            print(
                f"  {match_str:35s} "
                f"{pred['p_home_win']:>6.1%} "
                f"{pred['p_draw']:>7.1%} "
                f"{pred['p_away_win']:>7.1%}  "
                f"{pred['predicted']:12s}  "
                f"{pred['confidence']}"
            )
        except Exception as e:
            print(f"  {home} vs {away}: {e}")

    # ── Step 5: SHAP explanation for one match ─────────────
    print("\n  SHAP explanation for France vs Belgium:")
    try:
        vec = fe.build_prediction_vector(
            "France", "Belgium", datetime(2026, 6, 20), "group"
        )
        model.explain_prediction(vec, home_team="France", away_team="Belgium", outcome_idx=2)
    except Exception as e:
        print(f"  Explanation skipped: {e}")

    print("\n  Day 3 complete. Model saved to:", saved_path)
