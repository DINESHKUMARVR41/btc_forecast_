# Round 1 Scope

The uploaded Round 1 statement requires a time-series forecasting model that predicts future Bitcoin price, using historical Bitcoin data, the provided research papers and suitable data-science techniques. The submission must include the working model, README, backtesting results and MIT License.

This implementation deliberately does **not** depend on Glimpse crowd data or a Round 2 API. Round 2 details are not specified in the supplied statement, so they are outside this project's scope.

The core experimental question is:

> Do engineered historical market features improve next-day Bitcoin forecasting over a naive baseline under walk-forward evaluation?

The project includes ARIMA as a research-aligned baseline dependency for future/extended experiments, while the main scalable CPU model is scikit-learn HistGradientBoosting.
