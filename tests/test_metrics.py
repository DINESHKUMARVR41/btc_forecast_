import numpy as np
from src.backtest import metrics


def test_metrics_directional_coverage_handles_zero_return_baseline():
    result = metrics(
        actual_price=np.array([100.0, 101.0, 99.0]),
        predicted_price=np.array([100.0, 101.0, 100.0]),
        actual_return=np.array([0.01, -0.02, 0.00]),
        predicted_return=np.array([0.0, 0.0, 0.0]),
    )
    assert np.isnan(result["Directional_Accuracy"])
    assert result["Directional_Coverage"] == 0.0
