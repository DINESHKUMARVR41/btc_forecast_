# Bitcoin Forecast Engine — Round 1

A Python time-series forecasting system for the Round 1 Bitcoin forecasting challenge.

## What is predicted?

The system predicts the **next daily BTC-USD closing price after the latest completed daily candle**.

It does **not** predict the current/live Bitcoin price. The live quote is an observed market value shown separately for context.

The supervised target is the next-day log return:

```text
target_return(t) = log(close(t+1) / close(t))
```

The deployed ML forecast is converted back to price using the latest completed daily close:

```text
forecast(t+1) = latest_daily_close(t) * exp(predicted_return(t))
```

## Forecast model vs benchmark

The project deliberately separates the **deployed forecast model** from the **benchmark**:

- **ML Ensemble:** transparent 50/50 average of HistGradientBoosting and Ridge; this is the displayed/deployed forecast.
- **Naive/Persistence:** assumes the next daily close equals the latest completed daily close; this is the primary benchmark.
- **Momentum:** simple momentum baseline.
- **ARIMA(5,1,2):** benchmark aligned with the supplied research direction.

The ML model is not declared a winner merely because it is the deployed forecast. Walk-forward metrics for every model are shown in the dashboard and saved to `results/metrics.csv`.

## Features

Daily returns, SMA/EMA ratios, momentum, RSI, ATR, volatility, volatility change, volume change and lagged returns.

## Validation

Backtesting uses chronological walk-forward evaluation: models only train on data available before each test block. No random train/test shuffle is used.

Metrics include:

- Price MAE
- Price RMSE
- Price MAPE
- Next-day return MAE
- Directional accuracy
- Directional coverage

## Run

```powershell
python -m src.pipeline --start 2023-01-01
streamlit run app.py
```

The pipeline writes backtest metrics, forecasts, charts and model artifacts to `results/` and `models/`.

## Dashboard

The frontend clearly separates:

- **Live BTC price:** observed current reference
- **Latest daily close:** model anchor
- **Next daily close forecast:** predicted value
- **Predicted move:** predicted return
- **Naive benchmark:** persistence reference
- **Model signals:** HGB, Ridge, ensemble and momentum
- **Walk-forward backtest:** historical out-of-sample performance
- **Volatility range:** approximate movement range, not a calibrated confidence interval

## Scope

This repository is strictly for the Round 1 Bitcoin forecasting task. It does not use Glimpse crowd data, Glimpse APIs, Round 2 features, trading execution or paid data services.

## License

MIT
