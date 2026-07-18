import pandas as pd
from xgboost import XGBRegressor
from features import FEATURE_COLUMNS, CHEAP2H_FEATURE_COLUMNS


# Target name → (training column, feature columns).
# cheap2h predicts the mean of the day's two cheapest hours — the price a
# ~2-hour charging session can achieve by picking the cheapest points of the
# day. It gets one extra feature (its own lag-1); min/avg/max keep the
# original feature set so their MAE baseline is not perturbed.
TARGETS = {
    "min":     ("price_min",     FEATURE_COLUMNS),
    "avg":     ("price_avg",     FEATURE_COLUMNS),
    "max":     ("price_max",     FEATURE_COLUMNS),
    "cheap2h": ("price_cheap2h", CHEAP2H_FEATURE_COLUMNS),
}


def _make_regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42
    )


def _fit_models(data: pd.DataFrame) -> dict:
    """Fit one XGBoost regressor per target (min/avg/max/cheap2h) on the given data."""
    models = {}
    for name, (target_col, feature_cols) in TARGETS.items():
        model = _make_regressor()
        model.fit(data[feature_cols].values, data[target_col].values)
        models[name] = model
    return models


def train(training_data: pd.DataFrame) -> dict:
    """
    Train one XGBoost regressor per daily price target.

    Args:
        training_data: Daily DataFrame with feature columns and price targets.

    Returns:
        Dict keyed by target name (min/avg/max/cheap2h) with fitted models.
    """
    return _fit_models(training_data)


def predict(models: dict, forecast_features: pd.DataFrame) -> dict:
    """
    Run inference on forecast features.

    Args:
        models: Dict of fitted models keyed by target name.
        forecast_features: Daily DataFrame with feature columns.

    Returns:
        Dict keyed by date string (YYYY-MM-DD) with min/avg/max/cheap2h in EUR/MWh.
    """
    preds = {
        name: models[name].predict(forecast_features[TARGETS[name][1]].values)
        for name in TARGETS
    }

    dates = pd.to_datetime(forecast_features["date"]).dt.strftime("%Y-%m-%d").values
    predictions = {}

    for i in range(len(dates)):
        predictions[dates[i]] = {
            name: round(float(preds[name][i]), 4) for name in TARGETS
        }

    return predictions
