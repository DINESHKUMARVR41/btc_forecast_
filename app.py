import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BTC Forecast Engine", page_icon="₿", layout="wide")
st.title("₿ BTC Forecast Engine")
st.caption("Round 1 — time-series Bitcoin forecasting with walk-forward backtesting")

metrics_path = Path("results/metrics.csv")
forecast_path = Path("results/forecasts.csv")
latest_path = Path("results/latest_forecast.json")

if not latest_path.exists():
    st.warning("No trained model results found yet.")
    st.code("python -m src.pipeline --start 2023-01-01 --end 2026-09-19")
    st.stop()

latest = json.loads(latest_path.read_text())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current BTC", f"${latest['current_close']:,.0f}")
c2.metric("Next-day forecast", f"${latest['next_day_forecast']:,.0f}")
c3.metric("Expected move", f"{latest['predicted_return_pct']:+.2f}%")
c4.metric("30d annualized vol", f"{latest['latest_annualized_volatility_pct']:.1f}%")

st.divider()

if metrics_path.exists():
    st.subheader("Walk-forward backtest")
    metrics = pd.read_csv(metrics_path)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

if forecast_path.exists():
    fc = pd.read_csv(forecast_path, parse_dates=["date"])
    st.subheader("Forecast vs actual")
    st.line_chart(fc.set_index("date")[["actual_price", "predicted_price"]])

st.info("This is a research forecasting system, not financial advice or an automated trading system.")
