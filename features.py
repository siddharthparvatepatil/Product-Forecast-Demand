"""Leakage-safe feature engineering.

Forecast setting: at the origin we know sales up to day t0 and must forecast days
t0+1 .. t0+H (H = 28). Every history-based feature therefore uses a lag >= H, so a
feature for day t only depends on sales up to day t-28. Calendar features are known
in advance and are safe at any horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HORIZON = 28
LAGS = (28, 35, 42, 364)
ROLL_WINDOWS = (7, 28, 90)

CATEGORICAL = ["store", "item"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = d.dt.month
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    for k in (1, 2, 3):  # Fourier terms for yearly seasonality
        df[f"sin_y{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"cos_y{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)

    # US federal holidays (assumption: the Kaggle data has no stated country).
    hol = USFederalHolidayCalendar().holidays(
        start=d.min() - pd.Timedelta(days=30), end=d.max() + pd.Timedelta(days=30)
    ).to_numpy()
    dv = d.to_numpy()
    nxt = hol[np.clip(np.searchsorted(hol, dv, side="left"), 0, len(hol) - 1)]
    prv = hol[np.clip(np.searchsorted(hol, dv, side="right") - 1, 0, len(hol) - 1)]
    one_day = np.timedelta64(1, "D")
    df["days_to_holiday"] = np.clip((nxt - dv) / one_day, 0, 14)
    df["days_since_holiday"] = np.clip((dv - prv) / one_day, 0, 14)
    df["is_holiday"] = (df["days_to_holiday"] == 0).astype(int)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling stats, all computed on sales shifted by >= HORIZON days."""
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    g = df.groupby("series_id")["sales"]
    new = {}
    for lag in LAGS:
        new[f"lag_{lag}"] = g.shift(lag)
    for w in ROLL_WINDOWS:
        mp = max(2, w // 2)
        new[f"roll_mean_{w}"] = g.transform(lambda s, w=w, mp=mp: s.shift(HORIZON).rolling(w, min_periods=mp).mean())
        new[f"roll_std_{w}"] = g.transform(lambda s, w=w, mp=mp: s.shift(HORIZON).rolling(w, min_periods=mp).std())
    out = pd.concat([df, pd.DataFrame(new)], axis=1)
    out["cv_28"] = out["roll_std_28"] / (out["roll_mean_28"] + 1e-6)  # recent volatility
    out["yoy_ratio"] = out["lag_364"] / (out["roll_mean_28"] + 1e-6)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_history_features(add_calendar_features(df))


NON_FEATURES = {"date", "sales", "series_id", "is_spike"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]
