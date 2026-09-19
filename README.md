# ₿ Bitcoin Forecast Engine

**Round 1 — Next-Daily-Close Bitcoin Price Forecasting**

A Python-based Bitcoin forecasting system that predicts the **next daily BTC-USD closing price** using historical market data, engineered time-series features, machine-learning models, and walk-forward backtesting.

The project is designed for the **Glimpse Round 1 Bitcoin Forecasting Challenge** and focuses strictly on the Round 1 requirements.

---

## 🎯 What Does This Project Predict?

The system predicts:

> **The next daily BTC-USD closing price after the latest completed daily candle.**

For example:

```text
Latest completed daily close
        ↓
Feature engineering
        ↓
Ridge + HistGradientBoosting
        ↓
ML Ensemble
        ↓
Predicted next-day return
        ↓
Predicted next-day BTC closing price
```

The **live Bitcoin price shown in the dashboard is not the prediction target**.

It is displayed separately as an observed current-market reference.

---

## 🚀 Key Features

* Historical BTC-USD data collection
* Automated feature engineering
* Multiple forecasting approaches
* Ridge Regression
* HistGradientBoosting Regression
* ML ensemble
* Naive persistence benchmark
* Momentum baseline
* ARIMA(5,1,2)
* Walk-forward time-series validation
* Price and return error metrics
* Directional accuracy
* Volatility estimation
* Market-regime indicator
* Interactive Streamlit dashboard
* Historical forecast-vs-actual visualization
* Reproducible Python implementation
* MIT licensed

---

# 🧠 How It Works

The project follows a simple forecasting pipeline.

### 1. Download historical Bitcoin data

The system retrieves historical **BTC-USD daily OHLCV data**.

The main fields are:

```text
Open
High
Low
Close
Volume
```

The latest completed daily candle is used as the forecasting anchor.

A separate live BTC quote can also be displayed by the frontend, but that quote is **not used as the prediction target**.

---

### 2. Create useful time-series features

Raw Bitcoin prices alone are not directly given to the models.

The system creates additional features that describe recent market behavior.

Examples include:

```text
1-day return
3-day return
7-day return
14-day return
30-day return

Moving averages
EMA ratios
Momentum

RSI
ATR
Rolling volatility

Volume changes
Lagged returns
```

These features allow the models to see patterns such as:

* recent price movement
* momentum
* volatility
* short-term trends
* changes in trading volume
* relationship between current price and moving averages

---

### 3. Predict the next-day return

Instead of directly asking the machine-learning models to predict the raw Bitcoin price, the system predicts the **next-day percentage return**.

Conceptually:

```text
Today's Close
      +
Predicted Return
      ↓
Tomorrow's Predicted Close
```

The conversion is approximately:

```text
Predicted Close =
Latest Daily Close × (1 + Predicted Return)
```

This makes the target more suitable for a time-series forecasting model.

---

# 🤖 Models Used

The project evaluates several forecasting approaches.

## 1. Ridge Regression

Ridge is a relatively simple linear machine-learning model.

It attempts to learn a relationship such as:

```text
Market Features
      ↓
Linear relationships
      ↓
Predicted return
```

It is useful as a stable baseline for the machine-learning component.

---

## 2. HistGradientBoosting

HistGradientBoosting is a tree-based machine-learning model.

Unlike Ridge, it can model more complex nonlinear relationships between the engineered features and the target.

Conceptually:

```text
Market Features
      ↓
Decision trees
      ↓
Multiple boosting stages
      ↓
Predicted return
```

It is also relatively practical for CPU-based execution.

---

## 3. ML Ensemble

The final deployed forecasting signal combines the predictions from:

```text
Ridge
   +
HistGradientBoosting
   ↓
ML Ensemble
```

The ensemble produces a single predicted return.

That return is then converted into the predicted next-day Bitcoin closing price.

---

## 4. Naive Persistence Baseline

The Naive model is intentionally simple:

```text
Tomorrow's prediction = Latest completed daily close
```

This is an important benchmark because financial forecasting models should be compared against a simple strategy rather than evaluated in isolation.

The Naive model is therefore treated as a **benchmark**, not the deployed ML forecast.

---

## 5. Momentum Baseline

The Momentum baseline attempts to extend the recent price movement.

It provides another simple reference point for evaluating the machine-learning models.

---

## 6. ARIMA(5,1,2)

ARIMA is included as a traditional statistical time-series model.

The configuration:

```text
ARIMA(5,1,2)
```

was selected to connect the implementation with the time-series methodology discussed in the supplied Bitcoin forecasting research.

ARIMA is evaluated during backtesting but is not forced to become the final model.

---

# 🔄 Walk-Forward Backtesting

A major part of the project is **walk-forward validation**.

Normal random train/test splitting is inappropriate for time-series forecasting because it can allow future information to influence training.

Instead, the project follows the chronological order of the data.

Conceptually:

```text
TRAIN
████████████████

TEST
                 █
```

Then the training window moves forward:

```text
TRAIN
██████████████████

TEST
                   █
```

Then:

```text
TRAIN
████████████████████

TEST
                     █
```

This process continues through the historical dataset.

The model therefore attempts to reproduce a more realistic forecasting situation:

> "Given everything that was known at that point in time, what would the model have predicted for the next day?"

---

# 📊 Evaluation Metrics

The system evaluates the models using several metrics.

## MAE — Mean Absolute Error

MAE measures the average absolute difference between predicted and actual Bitcoin prices.

```text
MAE = average(|actual - predicted|)
```

Lower is better.

---

## RMSE — Root Mean Squared Error

RMSE gives greater importance to larger prediction errors.

```text
RMSE = √average((actual - predicted)²)
```

Lower is better.

---

## MAPE — Mean Absolute Percentage Error

MAPE expresses the forecasting error as a percentage.

```text
MAPE =
average(
    |actual - predicted| / actual
) × 100
```

Lower is better.

---

## Return MAE

The system also measures the error in the predicted daily return.

This helps evaluate the underlying return forecast separately from the final dollar price.

---

## Directional Accuracy

Directional accuracy measures how often the model correctly identifies whether the next-day return is:

```text
Positive ↑
or
Negative ↓
```

It is calculated only when a meaningful non-zero prediction is available.

---

# 📈 Current Walk-Forward Results

The latest local backtest produced the following results:

| Model                |     MAE |    RMSE |   MAPE | Return MAE | Directional Accuracy |
| -------------------- | ------: | ------: | -----: | ---------: | -------------------: |
| Naive                | 1268.18 | 1807.54 | 1.656% |     1.654% |                  N/A |
| MomentumBaseline     | 1284.82 | 1817.71 | 1.676% |     1.676% |               50.46% |
| Ridge                | 1306.13 | 1843.67 | 1.707% |     1.701% |               49.54% |
| ML_Ensemble          | 1341.50 | 1904.55 | 1.753% |     1.745% |               45.87% |
| HistGradientBoosting | 1423.42 | 2022.51 | 1.859% |     1.849% |               46.18% |
| ARIMA(5,1,2)         | 2755.69 | 3934.09 | 3.597% |     3.568% |               51.68% |

### Important

These results are included transparently.

The current ML Ensemble does **not** outperform the Naive persistence benchmark on the reported walk-forward price MAE.

The project does not hide or artificially modify this result.

The purpose of the benchmark is precisely to reveal whether the machine-learning approach provides additional predictive value over a very simple forecasting rule.

---

# 🖥️ Dashboard

The project includes an interactive Streamlit frontend.

Run:

```powershell
streamlit run app.py
```

The dashboard displays:

### Live Bitcoin Price

The latest observed BTC price.

This is **not the forecast**.

---

### Next Daily Close Forecast

The ML Ensemble's predicted next daily closing price.

---

### Predicted Move

The predicted percentage movement from the latest completed daily close.

Example:

```text
Predicted move: -0.15%
```

This means the model predicts a small decrease relative to the latest completed daily close.

---

### Naive Benchmark

The dashboard also displays the prediction produced by the persistence baseline.

Example:

```text
Naive benchmark:
$80,901.46
```

This allows the ML forecast to be compared against the simple baseline.

---

### Forecast vs Daily-Close Baseline

Shows the difference between:

```text
ML forecast
      -
Naive forecast
```

This makes the model's deviation from the simplest possible forecast visible.

---

### Forecast vs Live Price

Shows the difference between the predicted next daily close and the currently observed BTC quote.

This should not be confused with the model's prediction error because the live quote and forecast target represent different points in time.

---

### Volatility

The dashboard displays an annualized volatility estimate derived from historical Bitcoin returns.

Example:

```text
Annualized volatility: 47.2%
```

---

### Market Regime

The system provides a simple market-regime indicator based on recent volatility and drawdown behavior.

Possible regimes include:

```text
NORMAL
ELEVATED_RISK
CRASH_RISK
```

This is a descriptive indicator rather than a guaranteed crash prediction.

---

### Volatility-Based Range

The dashboard also displays a range derived from historical volatility.

For example:

```text
$78,261 – $83,380
```

This is an **estimated volatility-based range**.

It is **not a calibrated statistical confidence interval**.

---

# 🏗️ Project Architecture

```text
btc_forecast_
│
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── ROUND1_SCOPE.md
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── backtest.py
│   ├── pipeline.py
│   └── predict.py
│
└── tests/
```

---

# 📁 File Responsibilities

## `app.py`

Streamlit frontend.

Responsible for displaying:

* current BTC price
* forecast
* model information
* volatility
* market regime
* backtest results
* charts

---

## `src/data.py`

Responsible for obtaining Bitcoin market data.

It handles:

* historical BTC-USD data
* live market reference data
* date handling

---

## `src/features.py`

Responsible for feature engineering.

It creates:

* returns
* moving averages
* EMA-based features
* momentum
* RSI
* ATR
* volatility
* volume-related features
* lagged values
* forecasting target

---

## `src/models.py`

Contains the forecasting models and model utilities.

Main models include:

```text
Ridge
HistGradientBoosting
ARIMA
```

---

## `src/backtest.py`

Contains the walk-forward evaluation logic.

It calculates:

* MAE
* RMSE
* MAPE
* Return MAE
* Directional Accuracy

---

## `src/pipeline.py`

Runs the complete forecasting pipeline.

It handles:

```text
Data
 ↓
Features
 ↓
Training
 ↓
Backtesting
 ↓
Model selection
 ↓
Final forecast
 ↓
Saved artifacts
```

---

## `src/predict.py`

Loads the trained model and generates the latest forecast.

---

## `ROUND1_SCOPE.md`

Documents the project scope and keeps the implementation aligned with the Round 1 requirements.

---

# 🔬 Research Connection

The implementation was influenced by the supplied Bitcoin forecasting research.

The project incorporates ideas from the research such as:

### Classical time-series forecasting

ARIMA is included as a traditional statistical forecasting method.

### Volatility modeling

The project calculates return-based volatility and uses it for market-risk information and the estimated volatility range.

### Machine learning

Tree-based and linear machine-learning approaches are used to model nonlinear and linear relationships in engineered market features.

### Bitcoin crash/volatility research

Recent drawdown and volatility behavior are used to provide a simple market-regime indicator.

The project intentionally avoids unnecessary complexity for Round 1.

---

# ⚙️ Installation

## Requirements

Recommended:

```text
Python 3.10+
```

The project is designed to run on a normal CPU and does not require a GPU.

---

## 1. Clone the repository

```powershell
git clone https://github.com/DINESHKUMARVR41/btc_forecast_.git
cd btc_forecast_
```

---

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run Python commands directly through the environment or adjust your local PowerShell execution policy.

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# ▶️ Running the Forecast Pipeline

Run the complete pipeline with:

```powershell
python -m src.pipeline --start 2023-01-01
```

The pipeline:

1. Downloads historical BTC-USD data.
2. Creates engineered features.
3. Runs walk-forward backtesting.
4. Evaluates multiple models.
5. Generates the latest forecast.
6. Saves the required model artifacts and results.

---

# 🖥️ Running the Dashboard

After the pipeline has completed:

```powershell
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

Typically:

```text
http://localhost:8501
```

---

# 🧪 Running Tests

Run:

```powershell
pytest
```

The tests cover important parts of the forecasting pipeline and ensure the core project components behave as expected.

---

# 📦 Dependencies

The project primarily uses Python data-science and machine-learning libraries.

Major dependencies include:

```text
pandas
numpy
scikit-learn
statsmodels
yfinance
joblib
streamlit
matplotlib
```

See `requirements.txt` for the complete dependency list.

---

# 🔁 Reproducibility

The project is designed so that another user can clone the repository and reproduce the forecasting workflow.

The basic process is:

```text
Clone repository
       ↓
Install requirements
       ↓
Run pipeline
       ↓
Download BTC data
       ↓
Generate features
       ↓
Run walk-forward backtest
       ↓
Train forecasting models
       ↓
Generate latest forecast
       ↓
Launch dashboard
```

Because Bitcoin market data changes over time, the latest forecast and live market values will naturally change when the pipeline is run again.

Therefore, the exact current forecast shown in the README should not be interpreted as a permanent value.

---

# ⚠️ Limitations

Bitcoin is highly volatile and difficult to forecast reliably.

This project has several important limitations.

### 1. Historical data does not guarantee future performance

A model that performs well historically may perform poorly in future market conditions.

---

### 2. Bitcoin is affected by external events

The current system primarily uses market data.

It does not fully model:

* breaking news
* regulatory events
* macroeconomic announcements
* geopolitical events
* social-media sentiment
* exchange-specific order books
* institutional flows
* unexpected market shocks

---

### 3. Machine learning does not automatically outperform simple baselines

The current walk-forward results demonstrate this clearly.

The Naive persistence benchmark currently has lower price MAE than the deployed ML Ensemble.

This is an important finding rather than something the project attempts to hide.

---

### 4. The volatility range is not a confidence interval

The dashboard's displayed range is based on historical volatility.

It should not be interpreted as:

```text
"There is an 80% probability Bitcoin will stay inside this range."
```

The system does not make that calibrated probabilistic claim.

---

### 5. Live price and daily forecast are different quantities

The live BTC price is an observed market reference.

The forecasting target is:

```text
Next daily BTC-USD closing price
```

Therefore:

```text
Live price ≠ Forecast target
```

A difference between the live price and forecast does not itself represent model error.

---

# 🔐 No Automated Trading

This project is a **research and educational forecasting system**.

It does not:

* place trades
* execute orders
* manage cryptocurrency accounts
* provide financial advice
* guarantee profits

The system should not be treated as an automated trading strategy.

---

# 🧩 Round 1 Scope

This project intentionally focuses on the Round 1 forecasting problem.

The implementation does **not** depend on:

* Glimpse crowd data
* Round 2 features
* proprietary trading systems
* paid APIs
* automated trading execution

The primary objective is:

> Build, evaluate, and openly document a Bitcoin next-daily-close forecasting model using historical market data and appropriate time-series/machine-learning techniques.

---

# 📜 License

This project is released under the **MIT License**.

See:

```text
LICENSE
```

for the complete license text.

---

# 👨‍💻 Author

**Dinesh Kumar V R**
**KAJOL K **

GitHub:

`https://github.com/DINESHKUMARVR41`

LinkedIn:

`https://linkedin.com/in/dinesh-kumar-v-r-cse-020b4b381`

---

# ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.

It is not financial advice.

Cryptocurrency markets are highly volatile, and historical model performance does not guarantee future results.

Do not make financial decisions solely based on predictions produced by this software.

---

## ⭐ Project Summary

In one sentence:

> **Bitcoin Forecast Engine is a CPU-friendly Python forecasting system that predicts the next daily BTC-USD closing price using engineered market features, a Ridge + HistGradientBoosting ML ensemble, walk-forward validation, and transparent baseline comparisons.**
