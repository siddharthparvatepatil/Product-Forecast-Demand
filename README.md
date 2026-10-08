# Demand Forecasting with Uncertainty Quantification

Probabilistic demand forecasts (10th / 50th / 90th percentile) for the Store Item Demand
Forecasting Challenge, with calibration checks and an inventory decision built on top.
Only `numpy`, `pandas`, `scikit-learn` and `matplotlib` are required.

## What it does
1. **Features** (`src/demandfc/features.py`): lags (28/35/42/364 days), rolling mean/std (7/28/90), recent
   volatility, calendar + Fourier yearly seasonality, US-holiday proximity.
   All history features use lag >= 28 days, so the model is a valid **28-day-ahead** forecaster (no leakage;
   enforced by `tests/test_features.py`).
2. **Models** (`models.py`): gradient boosting with quantile loss (`HistGradientBoostingRegressor`, 7 quantiles
   from 5% to 95%), quantile-crossing repair, plus a mean-loss model.
3. **Calibration**: raw quantile interval, **conformalized quantile regression (CQR)** fitted on a validation
   window, and **residual-based baselines** (global and level-binned) for the bonus comparison.
4. **Evaluation** (`evaluate.py`): MAE/RMSE, pinball loss, empirical coverage vs nominal, interval score,
   reliability curve, coverage by segment (volume, volatility, holiday window, day of week, surges).
5. **Decision** (`inventory.py`): newsvendor-style stocking. Compares point forecast vs fixed q90 vs
   cost-optimal quantile (tau* = cu / (cu + co)) vs residual-interval safety stock, under four cost ratios.

## Setup
```bash
git clone <your-repo-url> && cd demand-forecast-uq
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
Download `train.csv` from the Kaggle competition
[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)
(500 series: 10 stores x 50 items, 2013-2017) and place it at `data/train.csv`.
CLI alternative: `kaggle competitions download -c demand-forecasting-kernels-only -p data/ && unzip data/*.zip -d data/`

## Reproduce results
```bash
python run.py --data data/train.csv          # real data (a few minutes on a laptop)
python run.py --synthetic                    # no download: generated demo data, ~30 s
python -m pytest tests                       # leakage / calibration / non-crossing tests
```
Everything is seeded (`seed=42`). Outputs land in `results/`:
`interval_calibration.csv`, `point_accuracy.csv`, `pinball_loss.csv`, `coverage_by_segment.csv`,
`inventory_decision.csv`, `run_meta.json`, and `figures/` (reliability, fan chart, cost curves, segment coverage).

## Re-running the evaluation only
Splits are chronological: last 90 days = test, the 90 days before = validation/calibration, the rest = train.
Change `TEST_DAYS`, `VAL_DAYS`, `TAUS`, or `ALPHA` at the top of `run.py`, or the cost ratios in
`SCENARIOS` in `src/demandfc/inventory.py`, then re-run `python run.py ...`.

## Layout
```
run.py                  pipeline entry point
src/demandfc/           data, features, models, evaluate, inventory, plots
tests/test_features.py  leakage test, CQR coverage test, quantile non-crossing test
WRITEUP.md              approach, decisions, results
```

## Limitations
- The Kaggle dataset has no promotion column, so promo effects are not modelled; "surge" days stand in for them.
- Holidays assume the US federal calendar (the dataset does not state a country).
- Not implemented: M5 (hierarchical, with prices/SNAP). The code is series-agnostic, so it extends naturally.
- Conformal guarantees assume exchangeability, which time series only approximate.
