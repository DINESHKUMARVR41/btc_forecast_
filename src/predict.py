import json, math
from pathlib import Path
import joblib, numpy as np
from .data import download_btc, latest_btc_price
from .features import make_features
from .models import arima_price_forecast, momentum_return


def predict():
    required = ["models/feature_columns.joblib", "models/selected_model.joblib"]
    if not all(Path(x).exists() for x in required):
        raise FileNotFoundError("Run `python -m src.pipeline` first.")
    raw = download_btc()
    feat = make_features(raw, include_target=False)
    cols = joblib.load(required[0])
    selected = joblib.load(required[1])
    row = feat.iloc[-1]
    live, ts = latest_btc_price()
    current = float(live) if live is not None else float(row.close)
    x = row[cols].to_frame().T

    hgb_r = float(joblib.load("models/hgb_model.joblib").predict(x)[0])
    ridge_r = float(joblib.load("models/ridge_model.joblib").predict(x)[0])
    ensemble_r = 0.5 * hgb_r + 0.5 * ridge_r
    momentum_r = 0.08 * momentum_return(row)

    if selected == "HistGradientBoosting": r = hgb_r
    elif selected == "Ridge": r = ridge_r
    elif selected == "ML_Ensemble": r = ensemble_r
    elif selected == "MomentumBaseline": r = momentum_r
    elif selected == "ARIMA_5_1_2":
        p = float(arima_price_forecast(raw["close"].to_numpy(), (5, 1, 2), 1)[0])
        r = math.log(p / float(row.close))
    else: r = 0.0

    # Forecast the next daily close from the latest completed daily close.
    # The live quote is only the current observed reference.
    anchor = float(row.close)
    price = anchor * np.exp(r)
    vol = float(row.volatility_30)
    move = 1.2816 * vol / math.sqrt(365)
    result = {
        "forecast_horizon": "Next daily BTC-USD close",
        "as_of_daily_data": str(row.date),
        "live_quote_timestamp": str(ts) if ts is not None else None,
        "current_price": current,
        "model_anchor_close": anchor,
        "next_day_forecast": float(price),
        "naive_benchmark_forecast": anchor,
        "forecast_vs_naive_usd": float(price-anchor),
        "forecast_vs_live_usd": float(price-current),
        "predicted_return_pct": r * 100,
        "hgb_return_pct": hgb_r * 100,
        "ridge_return_pct": ridge_r * 100,
        "ml_ensemble_return_pct": ensemble_r * 100,
        "selected_model": selected,
        "benchmark_model": "Naive_Persistence",
        "annualized_volatility_pct": vol * 100,
        "volatility_based_80pct_range": {"lower": float(price*np.exp(-move)), "upper": float(price*np.exp(move))},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predict()
