# Write-up: Demand forecasting with prediction intervals

> **Numbers below come from the synthetic demo data** (50 series, `python run.py --synthetic --max-iter 150`),
> because the Kaggle file could not be downloaded when this was built. **Re-run on the real data and update
> the tables before submitting** (`python run.py --data data/train.csv`). The method and code are unchanged.

## Approach
- **Problem framing:** forecast daily demand per store-item 28 days ahead. Every history feature uses a lag of
  at least 28 days, so the model never sees information that would be unavailable at forecast time.
  A unit test perturbs future sales and verifies no feature changes.
- **Model:** one gradient-boosting model per quantile (5, 10, 25, 50, 75, 90, 95%) with pinball loss.
  q50 is the point forecast; [q10, q90] is the 80% interval. Predictions are clipped at 0 and sorted
  per row to prevent quantile crossing (0.84% of test rows needed repair).
- **Split:** chronological. Train -> validation (90 days, used only for calibration) -> test (last 90 days).
- **Calibration:** raw quantiles are compared with conformalized quantile regression (CQR), which widens
  or narrows the interval by a constant fitted on the validation window.

## Results (synthetic data)
**Point accuracy (test):** median model MAE 6.73 / RMSE 10.55; seasonal-naive baseline MAE 9.02 / RMSE 13.61.

**80% interval calibration (nominal 0.80):**

| Method | Coverage | Mean width | Interval score (lower = better) |
|---|---|---|---|
| Quantile (raw) | 0.781 | 19.5 | 32.33 |
| Quantile + CQR | 0.807 | 20.5 | 32.31 |
| Residual (global) | 0.802 | 20.3 | 34.32 |
| Residual (binned by level) | 0.793 | 20.7 | 32.76 |

Raw quantile intervals are slightly too narrow (78.1% vs 80%). CQR fixes the average coverage at the price of
about 5% wider intervals.

**Bonus: does quantile regression beat residual intervals?** Only modestly here. A global residual interval has
good average coverage but the worst interval score (34.3 vs 32.3), because it uses one width for every
product. Once residuals are binned by predicted level the gap nearly closes (32.8). On this synthetic data
the noise is fairly regular, so the simple baseline is competitive; real retail data, with more varied
volatility and promotions, is where quantile regression should pull ahead. Check this on the real data
instead of assuming it.

## Where the model is overconfident
- **Average coverage hides the problem.** Coverage is about 78% in every volume, volatility, holiday and
  weekday segment, but on the days that were true spikes it is **11.5%**: the interval catches only about one
  spike in nine. All misses are on the upside (above the upper bound), which is the costly direction for stockouts.
- Cause: spikes are unannounced, so no feature can anticipate them; quantile loss at q90 cannot learn
  a tail event it never sees coming. On the real data, expect the same pattern around promotions and
  unusual events.
- Mitigations: add promo/event features if available, widen the upper quantile for volatile items, or hold
  extra buffer stock for items with spike history.

## Inventory decision
Cost per day-product = `cu * unmet demand + co * leftover stock`. The cost-minimising stock level is the
`cu / (cu + co)` quantile, so the "right" safety stock depends on the cost ratio (cost vs the mean-model point forecast):

| Cost ratio (cu:co) | Median forecast | Fixed q90 | Cost-optimal quantile | Fill rate (optimal) |
|---|---|---|---|---|
| 1:1 | -0.9% | +71.0% | **-0.9%** (q50) | 83.3% |
| 3:1 | +7.3% | -1.6% | **-12.9%** (q75) | 91.5% |
| 9:1 | +12.0% | -43.7% | **-43.7%** (q90) | 95.8% |
| 19:1 | +13.6% | -57.5% | **-61.2%** (q95) | 97.3% |

Takeaways: (1) a point forecast is only appropriate when over- and under-stocking cost the same; (2) a fixed
q90 is wasteful at 1:1 (+71% cost) and not optimal at 3:1 or 19:1, so the quantile should come from the cost
ratio; (3) when stockouts are expensive, uncertainty-aware stocking cuts cost by roughly half or more.
Residual-based safety stock performs similarly on this data (within a few points of the optimal quantile).

## Key decisions and caveats
- 28-day lag floor trades a little accuracy for a leakage-free, realistic forecasting setup.
- scikit-learn's `HistGradientBoostingRegressor` instead of LightGBM keeps installation simple; LightGBM with
  `objective="quantile"` is a drop-in swap in `models.py`.
- CQR assumes exchangeability between the validation and test windows; with real seasonality/drift, re-check
  coverage on a rolling basis.
- Stock decisions are evaluated per day-product with ideal re-ordering; there are no lead-time or
  batch-size constraints.
