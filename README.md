# BTC Forecast — Regime-Aware Bitcoin Forecasting

Round 1 submission project for the Glimpse Trading Hackathon 2026.

## What this project does

This project builds a reproducible Bitcoin time-series forecasting pipeline:

1. Downloads historical BTC-USD OHLCV data.
2. Creates leakage-safe return, trend, momentum, volatility and lag features.
3. Predicts the **next-day log return**, then converts it back to a BTC price forecast.
4. Compares a naive baseline, ARIMA baseline and gradient-boosting model.
5. Uses walk-forward / expanding-window evaluation instead of random train/test splitting.
6. Reports MAE, RMSE, MAPE and directional accuracy.
7. Produces forecast CSVs and plots.
8. Provides an optional lightweight Streamlit dashboard.

The primary model is `HistGradientBoostingRegressor` from scikit-learn so the project remains easy to install and practical on CPU-only laptops. An XGBoost implementation can be added later if resources permit.

## Why this design?

The supplied Round 1 statement asks for a working Bitcoin time-series forecasting model, historical-data testing/backtesting, a README and MIT License. The research material suggests ARIMA as a classical baseline, volatility-aware features, and careful time-series validation.

The model does **not** claim to know the future perfectly. Bitcoin is noisy and non-stationary. The goal is to test whether engineered historical signals improve on simple baselines under an honest temporal evaluation.

## Target

For day t:

`log_return(t+1) = log(Close(t+1)) - log(Close(t))`

The model predicts the next-day return. The predicted price is reconstructed as:

`predicted_price(t+1) = Close(t) * exp(predicted_return)`

## Features

- OHLCV-derived returns
- 1/3/7/14/30-day returns
- SMA and EMA ratios
- RSI
- ATR percentage
- rolling volatility
- volatility change
- volume change
- lagged returns

All features are calculated from information available at or before the forecast origin.

## Backtesting

The backtester uses expanding windows:

```text
Train ───── Test
Train ───────── Test
Train ───────────── Test
Train ───────────────── Test
```

There is no random shuffle.

## Quick start — Windows PowerShell

```powershell
cd btc_forecast_round1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline --start 2023-01-01 --end 2026-09-19
```

Results will be written to `results/`.

To launch the optional dashboard:

```powershell
streamlit run app.py
```

## Project structure

```text
btc_forecast_round1/
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
├── data/
├── models/
├── results/
│   └── plots/
└── src/
    ├── data.py
    ├── features.py
    ├── models.py
    ├── backtest.py
    ├── pipeline.py
    └── predict.py
```

## Submission note

Do not put API keys, credentials or private data into this repository.

The repository should contain the code and reproducible methodology. Generated datasets can be omitted if GitHub size limits are a concern; the pipeline can download them again.

## Important limitation

This is a forecasting research project, not a financial-advice or automated-trading system. Backtest performance is historical and does not guarantee future performance.
