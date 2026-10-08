"""Turn forecast quantiles into a stocking decision (newsvendor logic).

Per day and product we choose a stock level S. Cost = cu * max(y - S, 0) + co * max(S - y, 0),
where cu = cost per unit of unmet demand and co = cost per unit of leftover stock.
The cost-minimising S is the tau* = cu / (cu + co) quantile of demand, so the right
"safety stock" depends on the cost ratio -- the 90th percentile is only optimal when cu = 9 * co.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (cu, co); tau* = cu / (cu + co) must be on the quantile grid.
SCENARIOS = {
    "balanced (cu:co = 1:1)": (1.0, 1.0),
    "stockout 3x worse (3:1)": (3.0, 1.0),
    "stockout 9x worse (9:1)": (9.0, 1.0),
    "stockout 19x worse (19:1)": (19.0, 1.0),
}


def tau_star(cu: float, co: float) -> float:
    return round(cu / (cu + co), 4)


def evaluate_stock(y: np.ndarray, stock: np.ndarray, cu: float, co: float) -> dict:
    stock = np.round(np.clip(stock, 0, None))
    under = np.maximum(y - stock, 0)
    over = np.maximum(stock - y, 0)
    return {
        "total_cost": float(cu * under.sum() + co * over.sum()),
        "fill_rate": float(1 - under.sum() / y.sum()),
        "stockout_days": float((under > 0).mean()),
        "avg_stock": float(stock.mean()),
        "avg_leftover": float(over.mean()),
    }


def compare_strategies(y, q: dict, mean_pred, resid_global, resid_binned) -> pd.DataFrame:
    """Compare stocking rules in every cost scenario. `q` maps tau -> predicted quantile array."""
    rows = []
    for name, (cu, co) in SCENARIOS.items():
        t = tau_star(cu, co)
        if t not in q:
            raise ValueError(f"tau*={t} for scenario '{name}' is not on the quantile grid {sorted(q)}")
        strategies = {
            "point forecast (mean model)": mean_pred,
            "point forecast (median)": q[0.5],
            "fixed upper quantile (q90)": q[0.9],
            "cost-optimal quantile": q[t],
            "residual interval, global": resid_global.quantile(mean_pred, t),
            "residual interval, binned": resid_binned.quantile(mean_pred, t),
        }
        base = evaluate_stock(y, strategies["point forecast (mean model)"], cu, co)["total_cost"]
        for sname, stock in strategies.items():
            m = evaluate_stock(y, stock, cu, co)
            rows.append({"scenario": name, "tau_star": t, "strategy": sname, **m,
                         "cost_vs_point_mean_%": 100 * (m["total_cost"] / base - 1)})
    return pd.DataFrame(rows)


def cost_curve(y, q: dict) -> pd.DataFrame:
    """Total cost of stocking at each quantile level, per scenario (minimum should sit at tau*)."""
    rows = []
    for name, (cu, co) in SCENARIOS.items():
        for tau, pred in q.items():
            rows.append({"scenario": name, "tau": tau, **evaluate_stock(y, pred, cu, co)})
    return pd.DataFrame(rows)
