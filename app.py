import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BTC Forecast Engine", page_icon="₿", layout="wide")

st.markdown("# ₿ Bitcoin Forecast Engine")
st.caption("Round 1 • next-daily-close forecasting • walk-forward validation")

latest_path = Path("results/latest_forecast.json")
metrics_path = Path("results/metrics.csv")
forecast_path = Path("results/forecasts.csv")

if not latest_path.exists():
    st.warning("No trained forecast is available yet.")
    st.code("python -m src.pipeline --start 2023-01-01", language="powershell")
    st.stop()

latest = json.loads(latest_path.read_text(encoding="utf-8"))

st.info("**Prediction target:** the **next daily BTC-USD closing price after the latest completed daily candle**. The live BTC quote is an observed current market reference — it is **not** the prediction target.")

# Top-level market/forecast cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Live BTC price (observed)", f"${latest['current_price']:,.2f}")
c2.metric("Next daily close forecast", f"${latest['next_day_forecast']:,.2f}")
c3.metric("Predicted move", f"{latest['predicted_return_pct']:+.2f}%")
c4.metric("Annualized volatility", f"{latest['annualized_volatility_pct']:.1f}%")

st.divider()

left, mid, right = st.columns(3)
left.metric("Forecast model", latest["selected_model"])
mid.metric("Naive benchmark", f"${latest['naive_benchmark_forecast']:,.2f}")
right.metric("Market regime", latest["market_regime"])

st.subheader("Forecast interpretation")
a, b, c = st.columns(3)
a.metric("Latest daily close", f"${latest['model_anchor_close']:,.2f}")
b.metric("Forecast vs daily-close baseline", f"${latest['forecast_vs_naive_usd']:+,.2f}")
c.metric("Forecast vs live price", f"${latest['forecast_vs_live_usd']:+,.2f}")
st.caption("The Naive/Persistence benchmark assumes the next daily close equals the latest completed daily close. The deployed ML forecast is independent of the live intraday quote.")

st.subheader("Model signals")
signals = pd.DataFrame({
    "Model": ["HistGradientBoosting", "Ridge", "ML Ensemble", "Momentum"],
    "Predicted next-day return": [
        latest["hgb_return_pct"], latest["ridge_return_pct"],
        latest["ml_ensemble_return_pct"], latest["momentum_return_pct"]
    ]
})
signals["Predicted next-day return"] = signals["Predicted next-day return"].map(lambda x: f"{x:+.3f}%")
st.dataframe(signals, use_container_width=True, hide_index=True)

st.subheader("Volatility-based range")
r = latest["volatility_based_80pct_range"]
st.write(f"**${r['lower']:,.0f} — ${r['upper']:,.0f}**")
st.caption("Approximate volatility-based 80% movement range. This is not a calibrated confidence interval or probability guarantee.")

if metrics_path.exists():
    st.subheader("Walk-forward backtest")
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption("All reported metrics are out-of-sample walk-forward results. Lower MAE/RMSE/MAPE/Return MAE is better. Directional accuracy measures the sign of predicted vs actual next-day return when the model makes a non-zero call.")

    if not metrics.empty:
        st.markdown("**How to read this:** Naive/Persistence is the benchmark. The ML Ensemble is the deployed forecast model. The table is deliberately not hiding weaker results.")

if forecast_path.exists():
    fc = pd.read_csv(forecast_path, parse_dates=["date"])
    st.subheader("Historical walk-forward forecast vs actual")
    chart = fc[fc["model"] == latest["selected_model"]]
    if not chart.empty:
        st.line_chart(chart.set_index("date")[["actual_price", "predicted_price"]])

    st.subheader("Recent BTC daily closes")
    recent = fc[["date", "actual_price"]].drop_duplicates("date").tail(120).set_index("date")
    st.line_chart(recent.rename(columns={"actual_price": "BTC close"}))

with st.expander("How the system works"):
    st.markdown("""
1. Download daily BTC-USD OHLCV history.
2. Build return, moving-average, momentum, RSI, ATR, volatility, volume and lag features.
3. Train models to predict the **next-day log return**.
4. Convert the predicted return into a **next daily closing-price forecast**, anchored to the latest completed daily close.
5. Compare Naive, HistGradientBoosting, Ridge, ML Ensemble, Momentum and ARIMA using walk-forward backtesting.
6. Keep Naive visible as the persistence benchmark rather than pretending the ML model automatically wins.
7. Show the live BTC quote separately as an observed market reference.
""")

st.caption("Research/educational forecasting system. Historical performance does not guarantee future results and this is not financial advice or an automated trading system.")
