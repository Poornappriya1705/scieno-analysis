from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


@dataclass
class ForecastOutput:
    model: str
    fitted_test: np.ndarray
    future: np.ndarray


def _supervised(values: np.ndarray, lags: int):
    X, y = [], []
    for index in range(lags, len(values)):
        X.append(values[index - lags:index])
        y.append(values[index])
    return np.asarray(X), np.asarray(y)


def _recursive_ml(model, train_values: np.ndarray, test_size: int, horizon: int, lags: int):
    X, y = _supervised(train_values, lags)
    if len(X) < 2:
        raise ValueError("Insufficient observations for lag-based machine-learning model")
    model.fit(X, y)
    history = list(train_values.astype(float))
    test_predictions = []
    for _ in range(test_size):
        pred = max(0.0, float(model.predict(np.asarray(history[-lags:]).reshape(1, -1))[0]))
        test_predictions.append(pred)
        history.append(pred)
    full_history = history.copy()
    future = []
    for _ in range(horizon):
        pred = max(0.0, float(model.predict(np.asarray(full_history[-lags:]).reshape(1, -1))[0]))
        future.append(pred)
        full_history.append(pred)
    return np.asarray(test_predictions), np.asarray(future)


def forecast_model(
    model_name: str,
    train: np.ndarray,
    full: np.ndarray,
    test_size: int,
    horizon: int,
    lags: int,
    random_state: int = 42,
) -> ForecastOutput:
    train = np.asarray(train, dtype=float)
    full = np.asarray(full, dtype=float)

    if model_name == "naive":
        fitted_test = np.repeat(train[-1], test_size)
        future = np.repeat(full[-1], horizon)
    elif model_name == "linear_regression":
        fitted_test, _ = _recursive_ml(LinearRegression(), train, test_size, 0, lags)
        _, future = _recursive_ml(LinearRegression(), full, 0, horizon, lags)
    elif model_name == "random_forest":
        factory: Callable[[], object] = lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=1, random_state=random_state
        )
        fitted_test, _ = _recursive_ml(factory(), train, test_size, 0, lags)
        _, future = _recursive_ml(factory(), full, 0, horizon, lags)
    elif model_name == "gradient_boosting":
        factory = lambda: GradientBoostingRegressor(random_state=random_state, loss="huber")
        fitted_test, _ = _recursive_ml(factory(), train, test_size, 0, lags)
        _, future = _recursive_ml(factory(), full, 0, horizon, lags)
    elif model_name == "svr":
        factory = lambda: SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale")
        fitted_test, _ = _recursive_ml(factory(), train, test_size, 0, lags)
        _, future = _recursive_ml(factory(), full, 0, horizon, lags)
    elif model_name == "exponential_smoothing":
        test_fit = ExponentialSmoothing(train, trend="add", damped_trend=True, initialization_method="estimated").fit()
        fitted_test = np.maximum(0, np.asarray(test_fit.forecast(test_size)))
        full_fit = ExponentialSmoothing(full, trend="add", damped_trend=True, initialization_method="estimated").fit()
        future = np.maximum(0, np.asarray(full_fit.forecast(horizon)))
    elif model_name == "arima":
        test_fit = ARIMA(train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
        fitted_test = np.maximum(0, np.asarray(test_fit.forecast(test_size)))
        full_fit = ARIMA(full, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
        future = np.maximum(0, np.asarray(full_fit.forecast(horizon)))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return ForecastOutput(model=model_name, fitted_test=fitted_test, future=future)
