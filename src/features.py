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


def make_features(df, include_target=True):
    """Create daily forecasting features.

    If include_target=True, rows without a known next-day target are removed for
    supervised training/backtesting. For live inference, set include_target=False
    so the latest completed candle is retained.
    """
    x = df.copy().sort_values("date").reset_index(drop=True)
    close, high, low, volume = x["close"], x["high"], x["low"], x["volume"]

    x["log_return"] = np.log(close).diff()
    for n in [1, 3, 7, 14, 30]:
        x[f"return_{n}"] = np.log(close / close.shift(n))

    for n in [7, 21, 50]:
        x[f"sma_{n}_ratio"] = close / close.rolling(n).mean() - 1
    for n in [12, 26]:
        x[f"ema_{n}_ratio"] = close / close.ewm(span=n, adjust=False).mean() - 1

    x["momentum_7"] = close.pct_change(7)
    x["momentum_14"] = close.pct_change(14)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / close

    for n in [7, 14, 30]:
        x[f"volatility_{n}"] = x["log_return"].rolling(n).std() * np.sqrt(365)

    x["volatility_change"] = x["volatility_7"] / x["volatility_30"] - 1
    x["volume_change"] = volume.pct_change(7)
    for n in [1, 2, 3, 7, 14, 30]:
        x[f"lag_return_{n}"] = x["log_return"].shift(n)

    x["target_return"] = x["log_return"].shift(-1)
    x["target_price"] = close.shift(-1)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if include_target:
        x = x.dropna(subset=["target_return", "target_price"]).reset_index(drop=True)
    return x
