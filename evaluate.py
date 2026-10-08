"""Point-accuracy and interval-calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(y, pred) -> dict:
    err = y - pred
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def pinball(y, q, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_summary(y, lo, hi, alpha: float) -> dict:
    """Coverage should be ~ (1 - alpha). Interval score (Winkler) = width + penalty for misses; lower is better."""
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    penalty = (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return {
        "coverage": float(inside.mean()),
        "nominal": 1 - alpha,
        "gap": float(inside.mean() - (1 - alpha)),
        "below_lo": float((y < lo).mean()),
        "above_hi": float((y > hi).mean()),
        "mean_width": float(width.mean()),
        "interval_score": float((width + penalty).mean()),
    }


def reliability(y, quantile_preds: dict) -> pd.DataFrame:
    """Empirical P(y <= q_tau) vs nominal tau, for every predicted quantile."""
    return pd.DataFrame({"nominal": list(quantile_preds),
                         "empirical": [float((y <= q).mean()) for q in quantile_preds.values()]})


def coverage_by(groups: pd.Series, y, lo, hi, min_n: int = 30) -> pd.DataFrame:
    df = pd.DataFrame({"g": groups.to_numpy(), "inside": (y >= lo) & (y <= hi),
                       "below": y < lo, "above": y > hi})
    out = df.groupby("g", observed=True).agg(n=("inside", "size"), coverage=("inside", "mean"),
                                             below_lo=("below", "mean"), above_hi=("above", "mean"))
    return out[out["n"] >= min_n].reset_index()
