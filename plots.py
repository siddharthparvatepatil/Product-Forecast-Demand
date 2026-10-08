"""Figures (matplotlib, saved to disk)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_reliability(rel_by_method: dict, path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, rel in rel_by_method.items():
        ax.plot(rel["nominal"], rel["empirical"], "o-", label=name)
    ax.set_xlabel("nominal quantile level")
    ax.set_ylabel("empirical share of actuals below prediction")
    ax.set_title("Quantile reliability (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_fan(test: pd.DataFrame, lo, hi, med, path: str, n_series: int = 3, days: int = 90):
    ids = test["series_id"].drop_duplicates().iloc[:: max(1, test["series_id"].nunique() // n_series)][:n_series]
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 2.8 * len(ids)), sharex=True)
    axes = np.atleast_1d(axes)
    t = test.reset_index(drop=True)
    for ax, sid in zip(axes, ids):
        m = (t["series_id"] == sid).to_numpy()
        d = t.loc[m, "date"].iloc[-days:]
        sl = slice(-days, None)
        ax.fill_between(d, lo[m][sl], hi[m][sl], alpha=0.3, label="80% interval")
        ax.plot(d, med[m][sl], lw=1, label="median forecast")
        ax.plot(d, t.loc[m, "sales"].iloc[-days:], "k.", ms=3, label="actual")
        ax.set_ylabel(f"series {sid}")
    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle("Forecast fan chart (last test days)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cost_curves(curve: pd.DataFrame, path: str):
    scen = curve["scenario"].unique()
    fig, axes = plt.subplots(1, len(scen), figsize=(4 * len(scen), 3.4))
    for ax, s in zip(np.atleast_1d(axes), scen):
        c = curve[curve["scenario"] == s]
        ax.plot(c["tau"], c["total_cost"], "o-")
        best = c.loc[c["total_cost"].idxmin()]
        ax.axvline(best["tau"], color="r", ls="--", lw=1)
        ax.set_title(s, fontsize=8); ax.set_xlabel("stock at quantile")
    np.atleast_1d(axes)[0].set_ylabel("total cost")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_segment_coverage(seg: pd.DataFrame, nominal: float, path: str):
    """seg columns: segment_type, g, coverage, method."""
    types = seg["segment_type"].unique()
    fig, axes = plt.subplots(1, len(types), figsize=(4.2 * len(types), 3.6), sharey=True)
    for ax, ty in zip(np.atleast_1d(axes), types):
        s = seg[seg["segment_type"] == ty]
        piv = s.pivot(index="g", columns="method", values="coverage")
        piv.plot.bar(ax=ax, legend=False, rot=30)
        ax.axhline(nominal, color="k", ls="--", lw=1)
        ax.set_title(ty, fontsize=9); ax.set_xlabel("")
    np.atleast_1d(axes)[0].set_ylabel("empirical coverage")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
