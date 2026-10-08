"""Data loading: the real Kaggle file, or a synthetic stand-in with the same schema."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_kaggle(path: str) -> pd.DataFrame:
    """Load the Store Item Demand Forecasting `train.csv` (columns: date, store, item, sales)."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", "store", "item", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return _finalize(df)


def make_synthetic(n_stores: int = 5, n_items: int = 10, start: str = "2013-01-01",
                   end: str = "2017-12-31", seed: int = 42) -> pd.DataFrame:
    """Synthetic daily demand with weekly + yearly seasonality, trend, over-dispersed
    (negative-binomial-like) noise and random, *unannounced* demand spikes.

    Only for smoke-testing / demos when the Kaggle file is unavailable. Spikes are
    deliberately not exposed as a feature, so the model should be overconfident on them.
    The returned frame carries a ground-truth `is_spike` column used only for analysis.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    T, S = len(dates), n_stores * n_items
    t = np.arange(T)

    store_ids = np.repeat(np.arange(1, n_stores + 1), n_items)
    item_ids = np.tile(np.arange(1, n_items + 1), n_stores)

    base = rng.lognormal(mean=2.8, sigma=0.5, size=S)[:, None]
    weekly_profile = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.3, 1.2])  # Mon..Sun
    weekly = weekly_profile[dates.dayofweek.to_numpy()][None, :]
    amp = rng.uniform(0.15, 0.4, size=S)[:, None]
    phase = rng.uniform(0, 2 * np.pi, size=S)[:, None]
    yearly = 1 + amp * np.sin(2 * np.pi * t[None, :] / 365.25 + phase)
    trend = 1 + 0.05 * t[None, :] / 365.25
    mu = base * weekly * yearly * trend

    is_spike = np.zeros((S, T), dtype=bool)
    starts = rng.random((S, T)) < 0.012
    for lag in range(3):  # each event lasts up to 3 days
        shifted = np.zeros_like(starts)
        shifted[:, lag:] = starts[:, : T - lag]
        is_spike |= shifted & (rng.random((S, T)) < (1.0 if lag == 0 else 0.5))
    mu = mu * np.where(is_spike, rng.uniform(1.8, 3.0, size=(S, T)), 1.0)

    k = 20.0  # variance = mu + mu^2 / k
    sales = rng.poisson(mu * rng.gamma(k, 1.0 / k, size=(S, T)))

    df = pd.DataFrame({
        "date": np.tile(dates.to_numpy(), S),
        "store": np.repeat(store_ids, T),
        "item": np.repeat(item_ids, T),
        "sales": sales.ravel(),
        "is_spike": is_spike.ravel(),
    })
    return _finalize(df)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["series_id"] = df.groupby(["store", "item"]).ngroup()
    df["sales"] = df["sales"].astype(float)
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)
