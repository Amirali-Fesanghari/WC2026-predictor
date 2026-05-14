"""
src/models/neural_net.py
PyTorch neural network for expected score prediction.

Predicts: home_goals_expected, away_goals_expected (floats)
This complements the XGBoost classifier which predicts win/draw/loss.
Together they answer: "France wins, probably 2-1"

Architecture:
  Input → BatchNorm → Dense(128) → ReLU → Dropout(0.3)
                    → Dense(64)  → ReLU → Dropout(0.2)
                    → Dense(32)  → ReLU
                    → Output(2)  → Softplus  (ensures positive goal predictions)

Why a separate model for goals?
  XGBoost is great at classification (win/draw/loss).
  But "how many goals?" is a regression problem with a different loss function.
  Predicting 2.1 vs 0.8 goals is more informative than just "home win".
  The neural net output also feeds into the ensemble as an additional signal.

Why Softplus instead of ReLU on output?
  Goals must be ≥ 0. Softplus is smooth and always positive,
  unlike ReLU which can produce flat gradients at 0.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CACHE_DIR

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["_home_team","_away_team","_match_date","_stage","target_outcome"]
TARGET_COLS = ["target_home_goals","target_away_goals"]
EXCLUDE_COLS = ["home_formation_enc","away_formation_enc","tactical_matchup_score"]


class GoalPredictorNet(nn.Module):
    """
    Feed-forward neural network for predicting expected goals.
    Takes the same feature vector as XGBoost.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),

            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
            nn.Softplus(),      # ensures output is always > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WCGoalModel:
    """
    Wrapper for training, saving, and predicting with GoalPredictorNet.

    Usage:
        model = WCGoalModel()
        model.load_training_data()
        model.train()
        model.save()
        home_xg, away_xg = model.predict(feature_vector)
    """

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Neural net device: {self.device}")

        self.net: Optional[GoalPredictorNet] = None
        self.feature_names: list[str] = []
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std:  Optional[np.ndarray] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.version = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Data loading ─────────────────────────────────────────

    def load_training_data(self, parquet_path: Optional[Path] = None) -> pd.DataFrame:
        path = parquet_path or (CACHE_DIR / "training_features.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"Training data not found. Run feature_engineer.py first."
            )
        df = pd.read_parquet(path)

        # Need both goal columns
        df = df.dropna(subset=TARGET_COLS)
        df[TARGET_COLS[0]] = df[TARGET_COLS[0]].astype(float)
        df[TARGET_COLS[1]] = df[TARGET_COLS[1]].astype(float)

        self.train_df = df
        logger.success(
            f"Goal model training data: {len(df)} matches\n"
            f"  Avg home goals: {df[TARGET_COLS[0]].mean():.2f}  "
            f"Avg away goals: {df[TARGET_COLS[1]].mean():.2f}"
        )
        return df

    def _prepare_tensors(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare X and y tensors from DataFrame."""
        drop = [c for c in (META_COLS + EXCLUDE_COLS + TARGET_COLS) if c in df.columns]
        # Also drop XGBoost's target column if present
        if "target_outcome" in df.columns:
            drop.append("target_outcome")

        X = df.drop(columns=drop).select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        self.feature_names = list(X.columns)
        X_np = X.values.astype(np.float32)

        # Standardise features (neural nets need this, unlike tree models)
        if fit_scaler:
            self.feature_mean = X_np.mean(axis=0)
            self.feature_std  = X_np.std(axis=0) + 1e-8
        X_np = (X_np - self.feature_mean) / self.feature_std

        y_np = df[TARGET_COLS].values.astype(np.float32)

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        return X_tensor, y_tensor

    # ── Training ─────────────────────────────────────────────

    def train(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 20,
    ) -> dict:
        """
        Train the goal prediction network.

        epochs: max training epochs
        batch_size: mini-batch size (32 is fine for 320 samples)
        lr: learning rate (Adam optimizer)
        patience: early stopping — stop if val loss doesn't improve for N epochs
        """
        if self.train_df is None:
            raise RuntimeError("Call load_training_data() first.")

        # Chronological train/val split (80/20) — no random shuffle!
        n = len(self.train_df)
        split = int(n * 0.8)
        train_df = self.train_df.iloc[:split]
        val_df   = self.train_df.iloc[split:]

        X_train, y_train = self._prepare_tensors(train_df, fit_scaler=True)
        X_val,   y_val   = self._prepare_tensors(val_df,   fit_scaler=False)

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val   = X_val.to(self.device)
        y_val   = y_val.to(self.device)

        # ── Model init ────────────────────────────────────────
        input_dim = X_train.shape[1]
        self.net  = GoalPredictorNet(input_dim).to(self.device)

        # Huber loss: combines MSE (for small errors) + MAE (for large errors)
        # Better than pure MSE for goal counts (reduces penalty for outlier scores)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=False,   # keep chronological order
        )

        # ── Training loop ─────────────────────────────────────
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        history       = {"train_loss": [], "val_loss": []}

        logger.info(f"Training GoalPredictorNet: {input_dim} features, {epochs} epochs max")

        for epoch in range(epochs):
            # Train
            self.net.train()
            train_losses = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val)
                val_loss = criterion(val_pred, y_val).item()

            train_loss = np.mean(train_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.net.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1:3d} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f}"
                )

        # Restore best weights
        if best_state:
            self.net.load_state_dict(best_state)

        # ── Final evaluation ──────────────────────────────────
        self.net.eval()
        with torch.no_grad():
            X_all, y_all = self._prepare_tensors(self.train_df, fit_scaler=False)
            X_all = X_all.to(self.device)
            y_all = y_all.to(self.device)
            preds = self.net(X_all).cpu().numpy()

        y_np = self.train_df[TARGET_COLS].values
        mae_home = float(np.abs(preds[:, 0] - y_np[:, 0]).mean())
        mae_away = float(np.abs(preds[:, 1] - y_np[:, 1]).mean())
        mae_total = (mae_home + mae_away) / 2

        metrics = {
            "best_val_loss": round(best_val_loss, 4),
            "mae_home_goals": round(mae_home, 4),
            "mae_away_goals": round(mae_away, 4),
            "mae_total":      round(mae_total, 4),
        }

        print("\n" + "═" * 60)
        print("  GOAL MODEL EVALUATION")
        print("═" * 60)
        print(f"  Best val loss (Huber): {metrics['best_val_loss']:.4f}")
        print(f"  MAE home goals:        {metrics['mae_home_goals']:.3f} goals")
        print(f"  MAE away goals:        {metrics['mae_away_goals']:.3f} goals")
        print(f"  MAE total:             {metrics['mae_total']:.3f} goals")
        print(f"\n  Sample predictions vs actual:")
        print(f"  {'Actual':20s} {'Predicted':20s}")
        for i in range(min(8, len(preds))):
            actual = f"{y_np[i,0]:.0f} - {y_np[i,1]:.0f}"
            pred_s = f"{preds[i,0]:.1f} - {preds[i,1]:.1f}"
            print(f"  {actual:20s} {pred_s:20s}")
        print("═" * 60 + "\n")

        return metrics

    # ── Prediction ───────────────────────────────────────────

    def predict(self, feature_vector: pd.DataFrame) -> dict:
        """
        Predict expected goals for a match.
        Returns dict with home_xg, away_xg, predicted_score.
        """
        if self.net is None:
            raise RuntimeError("Train or load the model first.")

        X = feature_vector.reindex(columns=self.feature_names, fill_value=0.0)
        X = X.fillna(0.0)
        X_np = X.values.astype(np.float32)
        X_np = (X_np - self.feature_mean) / self.feature_std

        self.net.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
            pred   = self.net(tensor).cpu().numpy()[0]

        home_xg = round(float(pred[0]), 2)
        away_xg = round(float(pred[1]), 2)

        # Round to nearest 0.5 for human-readable score suggestion
        home_score = round(home_xg * 2) / 2
        away_score = round(away_xg * 2) / 2

        return {
            "home_xg":         home_xg,
            "away_xg":         away_xg,
            "predicted_score": f"{home_score:.0f}-{away_score:.0f}",
            "goal_diff_pred":  round(home_xg - away_xg, 2),
        }

    # ── Save / Load ──────────────────────────────────────────

    def save(self, name: str = "neural_goal") -> Path:
        path = MODELS_DIR / f"{name}_{self.version}.pt"
        torch.save({
            "state_dict":    self.net.state_dict(),
            "feature_names": self.feature_names,
            "feature_mean":  self.feature_mean,
            "feature_std":   self.feature_std,
            "input_dim":     len(self.feature_names),
            "version":       self.version,
            "trained_at":    datetime.utcnow().isoformat(),
        }, path)
        logger.success(f"Neural net saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "WCGoalModel":
        data = torch.load(path, map_location="cpu")
        instance = cls(device="cpu")
        instance.feature_names = data["feature_names"]
        instance.feature_mean  = data["feature_mean"]
        instance.feature_std   = data["feature_std"]
        instance.version       = data["version"]
        instance.net = GoalPredictorNet(data["input_dim"])
        instance.net.load_state_dict(data["state_dict"])
        instance.net.eval()
        logger.success(f"Neural net loaded: {path}")
        return instance

    @classmethod
    def load_latest(cls) -> "WCGoalModel":
        saved = sorted(MODELS_DIR.glob("neural_goal_*.pt"))
        if not saved:
            raise FileNotFoundError(f"No saved models in {MODELS_DIR}")
        return cls.load(saved[-1])


if __name__ == "__main__":
    from src.pipeline.feature_engineer import FeatureEngineer

    print("═" * 60)
    print("  WC 2026 Predictor — Neural Net: Goal Prediction")
    print("═" * 60)

    model = WCGoalModel()
    model.load_training_data()
    metrics = model.train(epochs=200, patience=25)
    model.save()

    # Test predictions
    fe = FeatureEngineer()
    fe.load_data()

    print("  Sample goal predictions:")
    test_matches = [
        ("France",    "Morocco"),
        ("Brazil",    "Argentina"),
        ("England",   "Germany"),
        ("Spain",     "Portugal"),
    ]
    print(f"\n  {'Match':35s} {'xG Home':>8s}  {'xG Away':>8s}  Predicted")
    print("  " + "─" * 65)
    for home, away in test_matches:
        try:
            vec  = fe.build_prediction_vector(home, away, datetime(2026, 6, 20))
            pred = model.predict(vec)
            match_str = f"{home} vs {away}"
            print(
                f"  {match_str:35s} "
                f"{pred['home_xg']:>8.2f}  "
                f"{pred['away_xg']:>8.2f}  "
                f"{pred['predicted_score']}"
            )
        except Exception as e:
            print(f"  {home} vs {away}: {e}")
