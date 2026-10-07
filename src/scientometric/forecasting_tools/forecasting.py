from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .forecast_models import forecast_model
from .metrics import calculate_all

STATISTICAL_MODELS = {"naive", "exponential_smoothing", "arima"}


@dataclass
class SeriesForecastResult:
    target: str
    metrics: pd.DataFrame
    forecasts: pd.DataFrame
    test_predictions: pd.DataFrame
    best_model: str


def evaluate_and_forecast(
    annual: pd.DataFrame,
    target: str,
    models: Iterable[str],
    test_size: int,
    horizon: int,
    lags: int,
    random_state: int,
) -> SeriesForecastResult:
    values = annual[target].to_numpy(dtype=float)
    if len(values) <= test_size + lags:
        raise ValueError(
            f"{target} requires more annual observations. Found {len(values)}, "
            f"but at least {test_size + lags + 1} are needed."
        )
    train, test = values[:-test_size], values[-test_size:]
    future_years = np.arange(int(annual.year.max()) + 1, int(annual.year.max()) + horizon + 1)

    metric_rows = []
    forecast_frame = pd.DataFrame({"year": future_years})
    test_frame = pd.DataFrame({"year": annual.year.iloc[-test_size:].to_numpy(), "actual": test})

    for model_name in models:
        try:
            output = forecast_model(model_name, train, values, test_size, horizon, lags, random_state)
            row = {"model": model_name, "family": "Statistical" if model_name in STATISTICAL_MODELS else "Machine Learning"}
            row.update(calculate_all(test, output.fitted_test))
            metric_rows.append(row)
            forecast_frame[model_name] = np.rint(output.future).clip(min=0)
            test_frame[model_name] = output.fitted_test
        except Exception as exc:
            metric_rows.append({
                "model": model_name,
                "family": "Statistical" if model_name in STATISTICAL_MODELS else "Machine Learning",
                "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "R2": np.nan,
                "error": str(exc),
            })

    metrics = pd.DataFrame(metric_rows).sort_values(["RMSE", "MAE"], na_position="last").reset_index(drop=True)
    valid = metrics.dropna(subset=["RMSE"])
    if valid.empty:
        raise RuntimeError(f"All forecasting models failed for {target}")
    best_model = str(valid.iloc[0]["model"])
    forecast_frame["best_forecast"] = forecast_frame[best_model]
    return SeriesForecastResult(target, metrics, forecast_frame, test_frame, best_model)
