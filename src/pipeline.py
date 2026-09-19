import argparse, json, math, warnings
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .backtest import (naive_predictions, summarize, walk_forward_arima,
                       walk_forward_ml, walk_forward_ml_ensemble, walk_forward_momentum)
from .data import download_btc, latest_btc_price
from .features import FEATURE_COLUMNS, make_features
from .models import fit_model, arima_price_forecast, momentum_return


def classify_regime(feat):
    row = feat.iloc[-1]
    mx = float(feat["close"].tail(90).max())
    dd = float(row.close / mx - 1) if mx else 0
    vr = float(row.volatility_7 / row.volatility_30) if row.volatility_30 else 1
    regime = "CRASH_RISK" if dd <= -.20 and vr >= 1.15 else "ELEVATED_RISK" if dd <= -.10 or vr >= 1.25 else "NORMAL"
    return regime, dd, vr


def volatility_range(price, annualized_vol):
    daily_sigma = max(float(annualized_vol), 0) / math.sqrt(365)
    move = 1.2816 * daily_sigma
    return price * math.exp(-move), price * math.exp(move)


def _model_name(selected):
    return selected


def run(start, end, min_train, skip_arima=False):
    for p in [Path("results/plots"), Path("models"), Path("results"), Path("data")]:
        p.mkdir(parents=True, exist_ok=True)

    raw = download_btc(start=start, end=end)
    labeled = make_features(raw, include_target=True)
    inference = make_features(raw, include_target=False)
    split = max(min_train, int(len(labeled) * .75))
    if split >= len(labeled) - 30:
        raise ValueError("Not enough post-training observations for a meaningful backtest.")

    frames = [
        naive_predictions(labeled.iloc[split:]),
        walk_forward_ml(labeled, FEATURE_COLUMNS, split, 7, "HistGradientBoosting"),
        walk_forward_ml(labeled, FEATURE_COLUMNS, split, 7, "Ridge"),
        walk_forward_ml_ensemble(labeled, FEATURE_COLUMNS, split, 7),
        walk_forward_momentum(labeled, split, 7),
    ]
    if not skip_arima:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = walk_forward_arima(labeled, split, 7, (5, 1, 2))
        if not a.empty:
            frames.append(a)

    preds = pd.concat(frames, ignore_index=True)
    summary = summarize(preds)
    summary.to_csv("results/metrics.csv", index=False)
    preds.to_csv("results/forecasts.csv", index=False)

    # The Naive model is a benchmark, not the deployed forecast.
    # The deployed forecast is the transparent ML ensemble so the dashboard
    # represents an actual learned next-day return forecast while keeping
    # the persistence baseline visible for an honest comparison.
    selected = "ML_Ensemble"

    hgb_model = fit_model(labeled, FEATURE_COLUMNS, "HistGradientBoosting")
    ridge_model = fit_model(labeled, FEATURE_COLUMNS, "Ridge")
    joblib.dump(hgb_model, "models/hgb_model.joblib")
    joblib.dump(ridge_model, "models/ridge_model.joblib")
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.joblib")
    joblib.dump(selected, "models/selected_model.joblib")

    row = inference.iloc[-1]
    xrow = row[FEATURE_COLUMNS].to_frame().T
    hgb_raw = float(hgb_model.predict(xrow)[0])
    ridge_raw = float(ridge_model.predict(xrow)[0])
    ensemble_r = 0.5 * hgb_raw + 0.5 * ridge_raw
    momentum_r = 0.08 * momentum_return(row)

    live, live_ts = latest_btc_price()
    current = float(live) if live is not None else float(row.close)

    if selected == "HistGradientBoosting":
        selected_r = hgb_raw
    elif selected == "Ridge":
        selected_r = ridge_raw
    elif selected == "ML_Ensemble":
        selected_r = ensemble_r
    elif selected == "MomentumBaseline":
        selected_r = momentum_r
    elif selected == "ARIMA_5_1_2" and not skip_arima:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(arima_price_forecast(raw["close"].to_numpy(), (5, 1, 2), 1)[0])
        selected_r = math.log(p / float(row.close))
    else:
        selected_r = 0.0

    # The forecast is for the next DAILY close after the latest completed daily candle.
    # IMPORTANT: the prediction is anchored to the latest completed daily close,
    # not the intraday live quote. The live quote is only a current reference.
    anchor = float(row.close)
    forecast = anchor * math.exp(selected_r)
    vol = float(row.volatility_30)
    lower, upper = volatility_range(forecast, vol)
    regime, dd, vr = classify_regime(inference)

    result = {
        "forecast_horizon": "Next daily BTC-USD close",
        "as_of_daily_data": str(row.date),
        "live_quote_timestamp": str(live_ts) if live_ts is not None else None,
        "current_price": current,
        "model_anchor_close": anchor,
        "next_day_forecast": forecast,
        "naive_benchmark_forecast": anchor,
        "forecast_vs_naive_usd": forecast - anchor,
        "forecast_vs_live_usd": forecast - current,
        "predicted_return_pct": selected_r * 100,
        "hgb_return_pct": hgb_raw * 100,
        "ridge_return_pct": ridge_raw * 100,
        "ml_ensemble_return_pct": ensemble_r * 100,
        "momentum_return_pct": momentum_r * 100,
        "selected_model": selected,
        "benchmark_model": "Naive_Persistence",
        "annualized_volatility_pct": vol * 100,
        "volatility_based_80pct_range": {"lower": lower, "upper": upper},
        "market_regime": regime,
        "90d_drawdown_pct": dd * 100,
        "volatility_ratio_7d_to_30d": vr,
        "note": "This system does not predict the current BTC price. It forecasts the next daily BTC-USD close from the latest completed daily candle. The live quote is an observed current reference. The Naive model is retained as a benchmark, while the deployed forecast is the transparent ML ensemble. The 80% range is volatility-based, not a calibrated confidence interval."
    }
    Path("results/latest_forecast.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump({"selected_model": selected, "feature_columns": FEATURE_COLUMNS}, "models/model_metadata.joblib")

    g = preds[preds.model == selected]
    if not g.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(g.date, g.actual_price, label="Actual")
        plt.plot(g.date, g.predicted_price, label=selected)
        plt.title("BTC Walk-Forward Forecast")
        plt.xlabel("Date"); plt.ylabel("Price (USD)"); plt.legend(); plt.tight_layout()
        plt.savefig("results/plots/walk_forward_forecast.png", dpi=160); plt.close()

    print("\n=== BACKTEST METRICS ===")
    print(summary.to_string(index=False))
    print("\n=== LATEST FORECAST ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-train", type=int, default=750)
    ap.add_argument("--skip-arima", action="store_true")
    args = ap.parse_args()
    run(args.start, args.end, args.min_train, args.skip_arima)
