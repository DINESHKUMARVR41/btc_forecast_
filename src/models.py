import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


def make_ml_model(random_state=42):
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.035, max_leaf_nodes=15,
        max_depth=6, min_samples_leaf=30, l2_regularization=1.0,
        random_state=random_state,
    )


def make_ridge_model(alpha=8.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_model(df, feature_columns, kind="HistGradientBoosting"):
    if len(df) < 100:
        raise ValueError("Not enough rows to fit the forecasting model.")
    model = make_ridge_model() if kind == "Ridge" else make_ml_model()
    model.fit(df[feature_columns], df["target_return"])
    return model


def momentum_return(row):
    values = [float(row.get("return_3", 0)), float(row.get("return_7", 0)), float(row.get("return_14", 0))]
    return 0.50 * values[0] + 0.30 * values[1] + 0.20 * values[2]


def arima_price_forecast(close_prices, order=(5, 1, 2), steps=1):
    prices = np.asarray(close_prices, dtype=float)
    if len(prices) < 80:
        raise ValueError("Not enough observations for ARIMA forecasting.")
    fitted = ARIMA(np.log(prices), order=order, trend=None).fit()
    return np.exp(np.asarray(fitted.forecast(steps=steps), dtype=float))
