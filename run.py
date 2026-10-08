"""End-to-end pipeline: features -> quantile models -> calibration -> inventory decision.

    python run.py --data data/train.csv          # real Kaggle data
    python run.py --synthetic                    # synthetic stand-in (no download needed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from demandfc import data as D, evaluate as E, features as F, inventory as I, models as M, plots as P  # noqa: E402

TAUS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
ALPHA = 0.2                     # nominal 80% interval = [q10, q90]
LO, HI = 0.1, 0.9
TEST_DAYS, VAL_DAYS = 90, 90


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="path to Kaggle train.csv")
    g.add_argument("--synthetic", action="store_true", help="use generated demo data")
    ap.add_argument("--n-stores", type=int, default=5, help="synthetic only")
    ap.add_argument("--n-items", type=int, default=10, help="synthetic only")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-iter", type=int, default=M.DEFAULT_PARAMS["max_iter"])
    args = ap.parse_args()

    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    params = dict(M.DEFAULT_PARAMS, max_iter=args.max_iter)

    # 1. data + features -------------------------------------------------------------------
    raw = D.make_synthetic(args.n_stores, args.n_items) if args.synthetic else D.load_kaggle(args.data)
    df = F.build_features(raw)
    df = df[df["lag_28"].notna()].reset_index(drop=True)
    cols = F.feature_columns(df)

    # 2. chronological split: train | validation (calibration) | test --------------------
    last = df["date"].max()
    test_start = last - pd.Timedelta(days=TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=VAL_DAYS)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start].reset_index(drop=True)
    print(f"rows  train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(cols)}")
    ytr, yva, yte = (x["sales"].to_numpy() for x in (train, val, test))

    # 3. models --------------------------------------------------------------------------
    qmodels = M.fit_quantile_models(train[cols], ytr, TAUS, params)
    q_val, _ = M.predict_quantiles(qmodels, val[cols])
    q_te, crossing = M.predict_quantiles(qmodels, test[cols])
    mean_model = M.fit_mean_model(train[cols], ytr, params)
    mu_val = np.clip(mean_model.predict(val[cols]), 0, None)
    mu_te = np.clip(mean_model.predict(test[cols]), 0, None)
    print(f"models fitted in {time.time() - t0:.0f}s; quantile-crossing rows repaired: {crossing:.2%}")

    # 4. intervals: raw quantiles, conformalised (CQR), residual baselines ---------------------
    off = M.cqr_offset(yva, q_val[LO], q_val[HI], ALPHA)
    res_g = M.ResidualIntervals(bins=1).fit(mu_val, yva)
    res_b = M.ResidualIntervals(bins=5).fit(mu_val, yva)
    intervals = {
        "quantile (raw)": (q_te[LO], q_te[HI]),
        "quantile + CQR": (np.clip(q_te[LO] - off, 0, None), q_te[HI] + off),
        "residual (global)": (res_g.quantile(mu_te, LO), res_g.quantile(mu_te, HI)),
        "residual (binned)": (res_b.quantile(mu_te, LO), res_b.quantile(mu_te, HI)),
    }
    interval_tbl = pd.DataFrame({k: E.interval_summary(yte, lo, hi, ALPHA) for k, (lo, hi) in intervals.items()}).T

    point_tbl = pd.DataFrame({
        "quantile model (median)": E.point_metrics(yte, q_te[0.5]),
        "mean model": E.point_metrics(yte, mu_te),
        "seasonal-naive (lag 364 / lag 28)": E.point_metrics(yte, test["lag_364"].fillna(test["lag_28"]).to_numpy()),
    }).T
    pinball_tbl = pd.Series({f"q{int(t * 100):02d}": E.pinball(yte, q_te[t], t) for t in TAUS}, name="pinball_loss")

    # 5. where is the model overconfident? -------------------------------------------------
    cap = lambda s, lab: pd.qcut(s.rank(method="first"), 3, labels=lab)  # noqa: E731
    seg_defs = {
        "volume (recent mean)": cap(test["roll_mean_28"], ["low", "mid", "high"]).astype(str),
        "recent volatility": cap(test["cv_28"].fillna(test["cv_28"].median()), ["calm", "mid", "volatile"]).astype(str),
        "holiday window": np.where((test["days_to_holiday"] <= 2) | (test["days_since_holiday"] <= 1), "holiday window", "normal"),
        "day of week": test["dow"].map(dict(enumerate("Mon Tue Wed Thu Fri Sat Sun".split()))).astype(str),
        "surge (hindsight: y > 1.5x median)": np.where(yte > 1.5 * q_te[0.5], "surge", "normal"),
    }
    if "is_spike" in test:
        seg_defs["true spike (synthetic only)"] = np.where(test["is_spike"], "spike", "normal")
    seg_rows = []
    for ty, grp in seg_defs.items():
        for method in ("quantile (raw)", "quantile + CQR", "residual (binned)"):
            lo, hi = intervals[method]
            c = E.coverage_by(pd.Series(np.asarray(grp)), yte, lo, hi)
            c["segment_type"], c["method"] = ty, method
            seg_rows.append(c)
    seg = pd.concat(seg_rows, ignore_index=True)

    # 6. inventory decision -----------------------------------------------------------------
    decision = I.compare_strategies(yte, q_te, mu_te, res_g, res_b)
    curve = I.cost_curve(yte, q_te)

    # 7. save ----------------------------------------------------------------------------
    interval_tbl.to_csv(out / "interval_calibration.csv")
    point_tbl.to_csv(out / "point_accuracy.csv")
    pinball_tbl.to_csv(out / "pinball_loss.csv")
    seg.to_csv(out / "coverage_by_segment.csv", index=False)
    decision.to_csv(out / "inventory_decision.csv", index=False)
    meta = {"data": "synthetic" if args.synthetic else args.data, "train_rows": len(train), "val_rows": len(val),
            "test_rows": len(test), "test_period": [str(test_start.date()), str(last.date())],
            "cqr_offset": off, "quantile_crossing_rate": crossing, "seed": M.SEED, "max_iter": args.max_iter}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    P.plot_reliability({"quantile (raw)": E.reliability(yte, q_te)}, str(out / "figures/reliability.png"))
    P.plot_fan(test, *(intervals["quantile (raw)"]), q_te[0.5], str(out / "figures/fan_chart.png"))
    P.plot_cost_curves(curve, str(out / "figures/cost_curves.png"))
    focus = seg[seg["segment_type"].isin(["volume (recent mean)", "recent volatility", "holiday window", "surge (hindsight: y > 1.5x median)"])]
    P.plot_segment_coverage(focus, 1 - ALPHA, str(out / "figures/coverage_by_segment.png"))

    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print("\n== Point accuracy (test) ==\n", point_tbl)
    print("\n== 80% interval calibration (test; nominal coverage 0.80) ==\n", interval_tbl)
    print("\n== Coverage by segment (raw quantile) ==\n",
          seg[seg["method"] == "quantile (raw)"].drop(columns="method").to_string(index=False))
    print("\n== Inventory decision (cost, lower is better) ==\n",
          decision[["scenario", "strategy", "total_cost", "fill_rate", "avg_stock", "cost_vs_point_mean_%"]].to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {out}/")


if __name__ == "__main__":
    main()
