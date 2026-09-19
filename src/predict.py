import json
import joblib
import numpy as np
import pandas as pd
from .data import download_btc
from .features import make_features

def predict():
    df = download_btc()
    feat = make_features(df)
    model = joblib.load("models/final_model.joblib")
    cols = joblib.load("models/feature_columns.joblib")
    row = feat.iloc[-1]
    r = float(model.predict(row[cols].to_frame().T)[0])
    price = float(row["close"] * np.exp(r))
    result = {
        "as_of": str(row["date"].date()),
        "current_price": float(row["close"]),
        "next_day_forecast": price,
        "expected_move_pct": r * 100,
        "annualized_volatility_pct": float(row["volatility_30"] * 100),
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    predict()
