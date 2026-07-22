import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from features import FEATURE_COLUMNS, CHEAP2H_FEATURE_COLUMNS


# Time-decay sample weighting (step 5 — regime handling). Per target: each
# training row is weighted 0.5 ** (age_days / half_life), age measured from the
# most recent training day, so recent data anchors the price level while the full
# ~3-year window still sets the structure. None = uniform weights (no decay).
#
# Deliberately heterogeneous: only `avg` uses decay. A drift-free same-slice A/B
# swept over three runs (2026-07-22) showed decay reliably helps `avg` (~-0.35
# EUR/MWh at hl=500, negative at every half-life in all three runs) but its effect
# on the priority targets `min`/`cheap2h` was NOISE — the delta flipped +0.2 / -0.2
# / ~0 across the three runs, i.e. smaller than the between-run variability — so
# they (and non-priority `max`, also noisy) stay on uniform weights. The `avg`
# model is independent (own regressor, own fit), so weighting it does not touch the
# other targets' predictions. Do NOT "tidy" this into a single scalar. See
# IMPROVEMENT_PLAN.md step 5 and the README rejected table.
HALF_LIFE_DAYS = {"min": None, "avg": 500, "max": None, "cheap2h": None}


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


def _sample_weights(data: pd.DataFrame, half_life_days) -> np.ndarray | None:
    """
    Exponential time-decay weights for the training rows.

    The newest training day gets weight 1.0 and weights halve every
    `half_life_days` of age (age measured from the most recent day in `data`,
    so it is correct for both the production full-window fit and each
    walk-forward slice). Returns None when `half_life_days` is None, which
    leaves XGBoost on uniform weights (the pre-step-5 baseline).
    """
    if half_life_days is None:
        return None
    dates = pd.to_datetime(data["date"])
    age_days = (dates.max() - dates).dt.days.to_numpy()
    return 0.5 ** (age_days / float(half_life_days))


def _fit_models(data: pd.DataFrame, half_life_days=HALF_LIFE_DAYS) -> dict:
    """Fit one XGBoost regressor per target (min/avg/max/cheap2h) on the given data.

    `half_life_days` may be a per-target dict (production; None entries = uniform
    weights), or a single value/None applied to every target (used by A/B sweeps
    that vary one half-life across all targets). Rows are weighted by exponential
    time decay where a half-life is set (see HALF_LIFE_DAYS / _sample_weights).
    """
    def _hl_for(name):
        if isinstance(half_life_days, dict):
            return half_life_days.get(name)
        return half_life_days

    models = {}
    for name, (target_col, feature_cols) in TARGETS.items():
        weights = _sample_weights(data, _hl_for(name))
        model = _make_regressor()
        model.fit(data[feature_cols].values, data[target_col].values,
                  sample_weight=weights)
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
