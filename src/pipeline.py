import argparse
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .data import download_btc, latest_btc_price
from .features import make_features, FEATURE_COLUMNS
from .backtest import walk_forward_ml, naive_predictions, summarize
from .models import fit_calibrated_model

def run(start, end, min_train):
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    df = download_btc(start=start, end=end)
    feat = make_features(df)

    split = max(min_train, int(len(feat) * 0.75))
    backtest = walk_forward_ml(feat, FEATURE_COLUMNS, min_train=split, step=30)
    naive = naive_predictions(feat.iloc[split:])

    all_preds = pd.concat([naive, backtest], ignore_index=True)
    summary = summarize(all_preds)
    ml_mae = float(summary.loc[summary["model"] == "ExtraTreesCalibrated", "MAE"].iloc[0])
    naive_mae = float(summary.loc[summary["model"] == "Naive", "MAE"].iloc[0])
    selected_model = "ExtraTreesCalibrated" if ml_mae < naive_mae else "Naive"

    summary.to_csv("results/metrics.csv", index=False)
    all_preds.to_csv("results/forecasts.csv", index=False)

    # Fit final model on all currently available supervised data.
    model, scale = fit_calibrated_model(feat, FEATURE_COLUMNS)
    joblib.dump(model, "models/final_model.joblib")
    joblib.dump(FEATURE_COLUMNS, "models/feature_columns.joblib")

    latest = feat.iloc[-1]
    pred_return = float(model.predict(latest[FEATURE_COLUMNS].to_frame().T)[0] * scale)
    if selected_model == "Naive":
        pred_return = 0.0
    live_price, live_timestamp = latest_btc_price()
    current_close = live_price if live_price is not None else float(latest["close"])
    pred_price = float(current_close * __import__("math").exp(pred_return))

    result = {
        "as_of": str(live_timestamp if live_timestamp is not None else latest["date"]),
        "current_close": current_close,
        "next_day_forecast": pred_price,
        "predicted_return_pct": pred_return * 100,
        "model_return_scale": scale,
        "selected_model": selected_model,
        "latest_annualized_volatility_pct": float(latest["volatility_30"] * 100),
    }
    with open("results/latest_forecast.json", "w") as f:
        json.dump(result, f, indent=2)

    # Forecast plot.
    g = backtest.copy()
    plt.figure(figsize=(12, 5))
    plt.plot(g["date"], g["actual_price"], label="Actual")
    plt.plot(g["date"], g["predicted_price"], label="Model")
    plt.title("BTC Walk-Forward Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/walk_forward_forecast.png", dpi=160)
    plt.close()

    print("\n=== BACKTEST METRICS ===")
    print(summary.to_string(index=False))
    print("\n=== LATEST FORECAST ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-train", type=int, default=150)
    args = parser.parse_args()
    run(args.start, args.end, args.min_train)
