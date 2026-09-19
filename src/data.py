from pathlib import Path
import pandas as pd
import yfinance as yf

def latest_btc_price():
    """Return the latest available BTC-USD intraday close and timestamp."""
    df = yf.download(
        "BTC-USD",
        period="2d",
        interval="1h",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return None, None
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        return None, None
    return float(close.iloc[-1]), close.index[-1]

def download_btc(start="2017-01-01", end=None, cache_path="data/btc_usd.csv"):
    """Download daily BTC-USD OHLCV data and cache it locally."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    df = yf.download(
        "BTC-USD",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No BTC data was downloaded. Check your internet connection or date range.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {missing}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df

def load_btc(path="data/btc_usd.csv"):
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
