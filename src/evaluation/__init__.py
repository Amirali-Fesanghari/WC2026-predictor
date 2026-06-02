"""
src/evaluation
Backtesting and evaluation utilities for the WC 2026 predictor.

Modules
-------
backtest
    BacktestEngine: walk-forward backtest of XGBoost, Poisson, and
    Ensemble models on historical World Cup data (2010–2022).
    Computes accuracy, log_loss, brier_score, and calibration error
    per tournament, plus rolling accuracy across all predictions.
"""
