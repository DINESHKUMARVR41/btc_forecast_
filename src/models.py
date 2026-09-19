import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from statsmodels.tsa.arima.model import ARIMA

class NaiveModel:
    def fit(self, X, y):
        self.last = float(y.iloc[-1] if hasattr(y, "iloc") else y[-1])
        return self
    def predict(self, X):
        return np.full(len(X), self.last)

def make_ml_model(random_state=42):
    return ExtraTreesRegressor(
        n_estimators=160,
        max_depth=5,
        min_samples_leaf=8,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

def fit_calibrated_model(df, feature_columns, calibration_size=90):
    """Fit the ML model and calibrate return strength on past data only."""
    size = min(calibration_size, max(30, len(df) // 5))
    fit_data = df.iloc[:-size]
    calibration_data = df.iloc[-size:]

    calibration_model = make_ml_model()
    calibration_model.fit(fit_data[feature_columns], fit_data["target_return"])
    raw_return = calibration_model.predict(calibration_data[feature_columns])
    base_price = calibration_data["target_price"].to_numpy() / np.exp(
        calibration_data["target_return"].to_numpy()
    )
    actual_price = calibration_data["target_price"].to_numpy()
    scales = np.arange(0.0, 1.01, 0.05)
    scale = min(
        scales,
        key=lambda value: mean_absolute_error(
            actual_price, base_price * np.exp(value * raw_return)
        ),
    )

    model = make_ml_model()
    model.fit(df[feature_columns], df["target_return"])
    return model, float(scale)

def arima_forecast(returns, order=(5, 1, 2)):
    series = np.asarray(returns, dtype=float)
    model = ARIMA(series, order=order, trend=None)
    fitted = model.fit()
    return float(fitted.forecast(steps=1)[0])
