# Round 1 Scope

This project is intentionally limited to the Round 1 Bitcoin forecasting task.

## Included
- Historical Bitcoin OHLCV data
- Time-series feature engineering
- Next-day return forecasting
- Walk-forward backtesting
- Naive, momentum, Ridge, HistGradientBoosting, ML ensemble and ARIMA benchmarks
- Streamlit frontend

## Explicitly excluded
- Glimpse crowd data/API
- Round 2 features
- Trading execution
- Portfolio management
- Paid proprietary market-data services

## Target
The model forecasts the next daily BTC-USD closing price after the latest completed daily candle. The live quote is displayed only as an observed current reference.
