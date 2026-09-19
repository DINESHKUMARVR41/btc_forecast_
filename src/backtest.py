import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from .models import fit_model, arima_price_forecast, momentum_return


def metrics(actual_price, predicted_price, actual_return, predicted_return):
    actual_price = np.asarray(actual_price, float)
    predicted_price = np.asarray(predicted_price, float)
    actual_return = np.asarray(actual_return, float)
    predicted_return = np.asarray(predicted_return, float)
    mask = np.abs(predicted_return) > 1e-10
    direction = float(np.mean(np.sign(actual_return[mask]) == np.sign(predicted_return[mask])) * 100) if mask.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(actual_price, predicted_price)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_price, predicted_price))),
        "MAPE": float(mean_absolute_percentage_error(actual_price, predicted_price) * 100),
        "Return_MAE": float(mean_absolute_error(actual_return, predicted_return) * 100),
        "Directional_Accuracy": direction,
        "Directional_Coverage": float(mask.mean() * 100),
    }


def _rows(test, pred_r, name):
    rows = []
    for i, (_, row) in enumerate(test.iterrows()):
        r = float(pred_r[i])
        rows.append({"date": row.date, "model": name,
                     "actual_price": float(row.target_price),
                     "predicted_price": float(row.close) * np.exp(r),
                     "actual_return": float(row.target_return),
                     "predicted_return": r})
    return rows


def walk_forward_ml(df, feature_columns, min_train=1000, step=7, kind="HistGradientBoosting"):
    rows = []
    name = f"{kind}"
    for end in range(min_train, len(df), step):
        test = df.iloc[end:min(end + step, len(df))]
        train = df.iloc[:end]
        model = fit_model(train, feature_columns, kind=kind)
        pred_r = model.predict(test[feature_columns])
        rows.extend(_rows(test, pred_r, name))
    return pd.DataFrame(rows)


def walk_forward_ml_ensemble(df, feature_columns, min_train=1000, step=7):
    rows = []
    for end in range(min_train, len(df), step):
        test = df.iloc[end:min(end + step, len(df))]
        train = df.iloc[:end]
        hgb = fit_model(train, feature_columns, "HistGradientBoosting")
        ridge = fit_model(train, feature_columns, "Ridge")
        # Equal-weight ensemble keeps the method transparent and prevents either
        # model from dominating due to a single noisy validation period.
        pred_r = 0.5 * hgb.predict(test[feature_columns]) + 0.5 * ridge.predict(test[feature_columns])
        rows.extend(_rows(test, pred_r, "ML_Ensemble"))
    return pd.DataFrame(rows)


def walk_forward_momentum(df, min_train=1000, step=7):
    rows = []
    for end in range(min_train, len(df), step):
        test = df.iloc[end:min(end + step, len(df))]
        for _, row in test.iterrows():
            r = 0.08 * momentum_return(row)
            rows.extend(_rows(pd.DataFrame([row]), [r], "MomentumBaseline"))
    return pd.DataFrame(rows)


def naive_predictions(df):
    return pd.DataFrame({"date": df["date"], "model": "Naive", "actual_price": df["target_price"],
                         "predicted_price": df["close"], "actual_return": df["target_return"],
                         "predicted_return": np.zeros(len(df))})


def walk_forward_arima(df, min_train=1000, step=7, order=(5, 1, 2)):
    rows = []
    for end in range(min_train, len(df), step):
        test = df.iloc[end:min(end + step, len(df))]
        try:
            forecast = arima_price_forecast(df.iloc[:end]["close"].to_numpy(), order=order, steps=len(test))
        except Exception:
            continue
        for i, (_, row) in enumerate(test.iterrows()):
            p = float(forecast[i])
            rows.append({"date": row.date, "model": "ARIMA_5_1_2", "actual_price": float(row.target_price),
                         "predicted_price": p, "actual_return": float(row.target_return),
                         "predicted_return": float(np.log(p / row.close))})
    return pd.DataFrame(rows)


def summarize(preds):
    out = []
    for model, g in preds.groupby("model"):
        m = metrics(g.actual_price, g.predicted_price, g.actual_return, g.predicted_return)
        m["model"] = model
        out.append(m)
    cols = ["model", "MAE", "RMSE", "MAPE", "Return_MAE", "Directional_Accuracy", "Directional_Coverage"]
    return pd.DataFrame(out)[cols].sort_values("MAE").reset_index(drop=True)
