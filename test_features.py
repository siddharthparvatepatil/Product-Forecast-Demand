"""Leakage test: a feature at day t must not change if sales on days > t-28 change."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from demandfc import data as D, features as F, models as M  # noqa: E402


def test_no_future_leakage():
    raw = D.make_synthetic(n_stores=1, n_items=2, start="2014-01-01", end="2015-06-30")
    base = F.build_features(raw)
    cutoff = raw["date"].iloc[500]                      # perturb all sales strictly after this date
    pert = raw.copy()
    pert.loc[pert["date"] > cutoff, "sales"] += 1000
    changed = F.build_features(pert)
    # features for dates <= cutoff + 28 days may only depend on sales up to cutoff
    safe = base["date"] <= cutoff + np.timedelta64(F.HORIZON, "D")
    for c in F.feature_columns(base):
        a, b = base.loc[safe, c].to_numpy(float), changed.loc[safe, c].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), f"feature {c} leaks future sales"


def test_cqr_offset_restores_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full(5000, -0.3), np.full(5000, 0.3)    # deliberately too narrow
    off = M.cqr_offset(y[:2500], lo[:2500], hi[:2500], alpha=0.2)
    cov = np.mean((y[2500:] >= lo[2500:] - off) & (y[2500:] <= hi[2500:] + off))
    assert 0.77 < cov < 0.83


def test_quantiles_not_crossing_after_repair():
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    raw = D.make_synthetic(n_stores=1, n_items=2, start="2014-01-01", end="2015-06-30")
    df = F.build_features(raw).dropna(subset=["lag_28"])
    cols = F.feature_columns(df)
    models = M.fit_quantile_models(df[cols], df["sales"].to_numpy(), [0.1, 0.5, 0.9], dict(M.DEFAULT_PARAMS, max_iter=30))
    q, _ = M.predict_quantiles(models, df[cols])
    assert np.all(q[0.1] <= q[0.5]) and np.all(q[0.5] <= q[0.9])
