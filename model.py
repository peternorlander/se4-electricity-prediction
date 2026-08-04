import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBClassifier, XGBRegressor
from features import FEATURE_COLUMNS, CHEAP2H_FEATURE_COLUMNS, MIN_FEATURE_COLUMNS


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
#
# Deliberately heterogeneous: each target uses the feature set its own ablation
# testing validated, since a feature can be load-bearing for one target and dead
# weight for another. `min` is pruned to 15 columns; avg/max keep all 51; cheap2h
# adds its own lag-1 (plus neg_price_proba, appended at fit/serve time by the
# hurdle — see HURDLE_TARGETS). README "Per-Target Feature Sets" has the evidence.
#
# Note for A/B work: ab/variants.py's BASELINE mirrors this dict, so it reflects
# min's PRUNED list. Comparing against the old 51-feature min needs an explicit
# `targets` override rather than BASELINE.
TARGETS = {
    "min":     ("price_min",     MIN_FEATURE_COLUMNS),
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


# Negative-price hurdle (step 5 — regime handling). Negative-price days are a
# distinct physical regime (renewable oversupply + low/weekend demand) that a
# plain regressor has to infer implicitly from the same weather/calendar
# features; a classifier for "will tomorrow's price_min be negative" feeds its
# probability into the regressor as an extra feature to make that regime
# signal explicit.
#
# Validated via a drift-free A/B (ab_test.py, 2026-07-24 snapshot, shifts 0-5):
# cheap2h REAL (mean -0.44 EUR/MWh, all 6 shifts improved) -- the largest,
# cleanest single-run win recorded for the top-priority target. min was
# NOISE by the strict sign-consistency rule (5/6 shifts improved, one
# near-zero flip) -- promising but not confirmed, so HURDLE_TARGETS is
# deliberately cheap2h-only for now. See IMPROVEMENT_PLAN.md step 5 and
# MIN_HURDLE_FOLLOWUP.md (re-test min once more training data accumulates).
NEG_PRICE_THRESHOLD = 0.0  # EUR/MWh
HURDLE_PROBA_FEATURE_NAME = "neg_price_proba"
HURDLE_TARGETS = {"cheap2h"}


def _make_hurdle_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        eval_metric="logloss",
    )


class HurdleAugmentedModel:
    """
    Wraps a negative-price classifier + a price regressor: predict(X) appends
    the classifier's P(price_min < NEG_PRICE_THRESHOLD) as an extra feature
    before calling the regressor. Drop-in replacement for a plain XGBRegressor
    everywhere a fitted model is used (predict.py, evaluate.py) -- both only
    call .predict(X) with the target's normal feature_cols, and
    feature_importances_ is forwarded so get_feature_importance keeps working.
    """
    def __init__(self, classifier: XGBClassifier, regressor: XGBRegressor):
        self.classifier = classifier
        self.regressor = regressor

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.classifier.predict_proba(X)[:, 1]
        return self.regressor.predict(np.column_stack([X, proba]))

    @property
    def feature_importances_(self):
        return self.regressor.feature_importances_


def _fit_hurdle_model(data: pd.DataFrame, feature_cols: list, target_col: str,
                       half_life_days=None) -> HurdleAugmentedModel:
    """
    Fit target_col with an out-of-fold negative-price-probability feature
    appended.

    The regressor trains on OUT-OF-FOLD probabilities (5-fold), not the
    in-sample predictions of a classifier fit on the whole `data` -- an
    in-sample classifier has seen each row's own label, so its probabilities
    would be near-perfect and leak an optimistic signal production can never
    get (the classifier never knows tomorrow's actual label at serving time).
    The classifier stored in the returned HurdleAugmentedModel (fit on the
    full `data`) has genuinely not seen the row it predicts for at serving
    time either, so no leakage there.
    """
    X = data[feature_cols].values
    y = data[target_col].values
    neg_label = (data["price_min"] < NEG_PRICE_THRESHOLD).astype(int).values

    oof_proba = cross_val_predict(
        _make_hurdle_classifier(), X, neg_label,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]

    classifier = _make_hurdle_classifier()
    classifier.fit(X, neg_label)

    weights = _sample_weights(data, half_life_days)
    regressor = _make_regressor()
    regressor.fit(np.column_stack([X, oof_proba]), y, sample_weight=weights)

    return HurdleAugmentedModel(classifier, regressor)


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


def _fit_models(data: pd.DataFrame, half_life_days=HALF_LIFE_DAYS, targets=None) -> dict:
    """Fit one XGBoost regressor per target (min/avg/max/cheap2h) on the given data.

    `half_life_days` may be a per-target dict (production; None entries = uniform
    weights), or a single value/None applied to every target (used by A/B sweeps
    that vary one half-life across all targets). Rows are weighted by exponential
    time decay where a half-life is set (see HALF_LIFE_DAYS / _sample_weights).

    `targets` selects the {name: (target_col, feature_cols)} map to fit over;
    default (None) uses the production TARGETS. The A/B harness passes a variant's
    own targets so the SAME feature lists drive fitting AND prediction -- without
    this, a variant that changes feature_cols (e.g. an ablation) would train on the
    production set but predict on the variant set and mismatch. Production callers
    (model.train, evaluate.walk_forward_validate) pass nothing, so behaviour is
    unchanged.
    """
    if targets is None:
        targets = TARGETS

    def _hl_for(name):
        if isinstance(half_life_days, dict):
            return half_life_days.get(name)
        return half_life_days

    models = {}
    for name, (target_col, feature_cols) in targets.items():
        if name in HURDLE_TARGETS:
            models[name] = _fit_hurdle_model(data, feature_cols, target_col,
                                              half_life_days=_hl_for(name))
            continue
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
