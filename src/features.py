import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1", "return_3", "return_7", "return_14", "return_30",
    "sma_7_ratio", "sma_21_ratio", "sma_50_ratio",
    "ema_12_ratio", "ema_26_ratio", "momentum_7", "momentum_14",
    "rsi_14", "atr_pct", "volatility_7", "volatility_14",
    "volatility_30", "volatility_change", "volume_change",
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_7",
    "lag_return_14", "lag_return_30"
]

def make_features(df):
    x = df.copy().sort_values("date").reset_index(drop=True)

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    x["log_return"] = np.log(close).diff()

    for n in [1, 3, 7, 14, 30]:
        x[f"return_{n}"] = np.log(close / close.shift(n))

    for n in [7, 21, 50]:
        sma = close.rolling(n).mean()
        x[f"sma_{n}_ratio"] = close / sma - 1

    for n in [12, 26]:
        ema = close.ewm(span=n, adjust=False).mean()
        x[f"ema_{n}_ratio"] = close / ema - 1

    x["momentum_7"] = close.pct_change(7)
    x["momentum_14"] = close.pct_change(14)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / close

    for n in [7, 14, 30]:
        x[f"volatility_{n}"] = x["log_return"].rolling(n).std() * np.sqrt(365)

    x["volatility_change"] = x["volatility_7"] / x["volatility_30"] - 1
    x["volume_change"] = volume.pct_change(7)

    for n in [1, 2, 3, 7, 14, 30]:
        x[f"lag_return_{n}"] = x["log_return"].shift(n)

    # Forecast target: next-day log return.
    x["target_return"] = x["log_return"].shift(-1)
    x["target_price"] = close.shift(-1)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x.dropna(subset=FEATURE_COLUMNS + ["target_return", "target_price"]).reset_index(drop=True)
