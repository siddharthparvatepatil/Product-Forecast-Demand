"""Gradient-boosted quantile regression + residual-interval baselines (scikit-learn only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL

SEED = 42
DEFAULT_PARAMS = dict(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
    l2_regularization=1.0, random_state=SEED,
)


def _make(loss: str, quantile: float | None, params: dict, cols: list[str]):
    cat = [c for c in CATEGORICAL if c in cols]
    kw = dict(params, loss=loss, categorical_features=cat or None)
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def fit_quantile_models(X: pd.DataFrame, y: np.ndarray, taus, params: dict | None = None) -> dict:
    params = params or DEFAULT_PARAMS
    return {tau: _make("quantile", tau, params, list(X.columns)).fit(X, y) for tau in taus}


def predict_quantiles(models: dict, X: pd.DataFrame) -> tuple[dict, float]:
    """Predict every quantile, clip at 0, and repair quantile crossing by sorting per row.

    Returns ({tau: array}, crossing_rate) where crossing_rate is the share of rows that
    needed repair (reported, because it signals under-fitted / inconsistent models).
    """
    taus = sorted(models)
    raw = np.column_stack([np.clip(models[t].predict(X), 0, None) for t in taus])
    crossing_rate = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
    fixed = np.sort(raw, axis=1)
    return {t: fixed[:, i] for i, t in enumerate(taus)}, crossing_rate


def fit_mean_model(X: pd.DataFrame, y: np.ndarray, params: dict | None = None):
    return _make("squared_error", None, params or DEFAULT_PARAMS, list(X.columns)).fit(X, y)


# ---- Conformalized quantile regression (CQR) -------------------------------------------
def cqr_offset(y_cal, lo_cal, hi_cal, alpha: float) -> float:
    """Split-conformal correction for a [lo, hi] interval with target coverage 1-alpha."""
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


# ---- Residual-based baselines (bonus) --------------------------------------------------
class ResidualIntervals:
    """Point forecast + empirical quantiles of validation residuals.

    bins=1  -> one global residual distribution (homoscedastic assumption)
    bins>1  -> residual quantiles per bin of the predicted level (a stronger baseline)
    """

    def __init__(self, bins: int = 1):
        self.bins = bins

    def fit(self, pred_cal: np.ndarray, y_cal: np.ndarray):
        self.resid = y_cal - pred_cal
        self.edges = np.quantile(pred_cal, np.linspace(0, 1, self.bins + 1)[1:-1]) if self.bins > 1 else np.array([])
        b = np.digitize(pred_cal, self.edges)
        self.by_bin = {i: self.resid[b == i] for i in range(self.bins)}
        return self

    def quantile(self, pred: np.ndarray, tau: float) -> np.ndarray:
        b = np.digitize(pred, self.edges)
        qs = np.array([np.quantile(self.by_bin[i], tau) if len(self.by_bin[i]) else 0.0 for i in range(self.bins)])
        return np.clip(pred + qs[b], 0, None)
