import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, make_features


def sample_df(n=120):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = 50000 * np.exp(np.cumsum(np.full(n, 0.001)))
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def test_feature_output_has_expected_columns_and_no_nan_features():
    out = make_features(sample_df())
    assert set(FEATURE_COLUMNS).issubset(out.columns)
    assert not out[FEATURE_COLUMNS + ["target_return", "target_price"]].isna().any().any()
