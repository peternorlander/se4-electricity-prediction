import pandas as pd
from xgboost import XGBRegressor
from features import FEATURE_COLUMNS


def _make_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )


def _fit_models(data: pd.DataFrame) -> tuple:
    """Fit three XGBoost regressors (min/avg/max) on the given data."""
    X = data[FEATURE_COLUMNS].values
    model_min = _make_regressor()
    model_avg = _make_regressor()
    model_max = _make_regressor()
    model_min.fit(X, data["price_min"].values)
    model_avg.fit(X, data["price_avg"].values)
    model_max.fit(X, data["price_max"].values)
    return model_min, model_avg, model_max


def train(training_data: pd.DataFrame) -> tuple:
    """
    Train three XGBoost regressors for daily price min, avg and max.

    Args:
        training_data: Daily DataFrame with feature columns and price targets.

    Returns:
        Tuple of (model_min, model_avg, model_max).
    """
    return _fit_models(training_data)


def predict(models: tuple, forecast_features: pd.DataFrame) -> dict:
    """
    Run inference on forecast features.

    Args:
        models: Tuple of (model_min, model_avg, model_max).
        forecast_features: Daily DataFrame with feature columns.

    Returns:
        Dict keyed by date string (YYYY-MM-DD) with min/avg/max in EUR/MWh.
    """
    model_min, model_avg, model_max = models
    X = forecast_features[FEATURE_COLUMNS].values

    pred_min = model_min.predict(X)
    pred_avg = model_avg.predict(X)
    pred_max = model_max.predict(X)

    dates = pd.to_datetime(forecast_features["date"]).dt.strftime("%Y-%m-%d").values
    predictions = {}

    for i in range(len(dates)):
        predictions[dates[i]] = {
            "min": round(float(pred_min[i]), 4),
            "avg": round(float(pred_avg[i]), 4),
            "max": round(float(pred_max[i]), 4)
        }

    return predictions
