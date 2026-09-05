import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBClassifier, XGBRegressor
from features import FEATURE_COLUMNS, TROUGH_FEATURE_COLUMNS, MIN_FEATURE_COLUMNS, AVG_FEATURE_COLUMNS


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
# other targets' predictions. Do NOT "tidy" this into a single scalar. See the
# README "Features Tested and Rejected" table (time-decay sample weights row).
HALF_LIFE_DAYS = {"min": None, "avg": 500, "max": None, "cheap2h": None}


# Target name → (training column, feature columns).
#
# Deliberately heterogeneous: each target uses the feature set its own ablation
# testing validated, since a feature can be load-bearing for one target and dead
# weight for another. `max` keeps all 51 (not a priority target, never audited).
# `avg` drops its 9-column price/market family (42 of 51, see
# features.AVG_FEATURE_COLUMNS). The two trough targets run pruned lists that
# differ from each other by exactly one column. cheap2h additionally gets
# neg_price_proba, appended at fit/serve time by the hurdle — see HURDLE_TARGETS.
# README "Per-Target Feature Sets" has the evidence.
#
# min (14) and cheap2h (15) differ by `price_se4_max_lag1`, and that difference
# is measured, not stylistic: on one 16-point / four-period grid the same
# removal is REMOVE_HARMFUL for min (-0.552, all four clusters) and
# KEEP_SCENARIO for cheap2h (-0.439 but one cluster genuinely positive). See
# features.MIN_FEATURE_COLUMNS for the numbers. They shared a list from
# 2026-08-05 to 2026-08-06 only because cheap2h had borrowed min's.
#
# Note for A/B work: ab/variants.py's BASELINE mirrors this dict, so it reflects
# the PRUNED lists. Comparing against the old 51/52-column variants needs an
# explicit `targets` override rather than BASELINE.
TARGETS = {
    "min":     ("price_min",     MIN_FEATURE_COLUMNS),
    "avg":     ("price_avg",     AVG_FEATURE_COLUMNS),
    "max":     ("price_max",     FEATURE_COLUMNS),
    "cheap2h": ("price_cheap2h", TROUGH_FEATURE_COLUMNS),
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
# near-zero flip). Re-tested 2026-08-06 on the confound-free four-period
# grid and NOISE again -- clmean -0.104, but +0.079 in the period closest
# to production (see README "Features Tested and Rejected"). HURDLE_TARGETS
# is cheap2h-only, and that is now a closed question, not a pending one.
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


# ---------------------------------------------------------------------------
# Prediction interval for cheap2h (round 19c, adopted 2026-09-05)
# ---------------------------------------------------------------------------
# A point estimate cannot say "I don't know". On 2026-08-18 the model predicted
# 42.7 against a realised 127.9 with exactly the outward confidence it predicts
# 6.0 with on a calm day, and the Home Assistant automation had no way to tell
# those apart. Two quantile regressors on the SAME feature list give it one.
#
# Measured on the round-15b sliding grid, four period clusters (see the README
# "Prediction interval for cheap2h" section):
#   * band width tracks accuracy -- point-estimate MAE runs 7.26 / 13.13 /
#     20.83 / 21.50 across width quartiles, so a wide band really does mean a
#     bad day rather than a shy model;
#   * the RAW band is over-confident: an 80% interval covers 54%;
#   * a rolling conformal correction fixes that to 0.788 with no refit.
#
# This is an ADDITION. The headline cheap2h number is unchanged -- q50 is not
# fitted and the point model still produces `cheap2h`. Nothing here touches
# TARGETS, _fit_models, or anything evaluate.py / the A/B harness calls.
INTERVAL_TARGET = "cheap2h"
INTERVAL_QUANTILES = (0.10, 0.90)

# Split-conformal calibration. The last CONFORMAL_HOLDOUT_DAYS rows are held out
# of the calibration fit, scored, and the band widened by the (1-alpha)
# empirical quantile of the conformity scores. 180 days is what the round-19c
# rolling correction used; shorter reacts faster but estimates the quantile off
# fewer points, and below ~30 the correction is switched off entirely rather
# than trusted.
CONFORMAL_HOLDOUT_DAYS = 180
CONFORMAL_ALPHA = 0.20          # -> a nominal 80% band, matching q10..q90
CONFORMAL_MIN_HOLDOUT = 30


def _make_quantile_regressor(alpha: float) -> XGBRegressor:
    """Production hyperparameters with the objective swapped to pinball loss.

    Deliberately NOT retuned: the question this answers is what the existing
    model can say about its own uncertainty, not whether a differently-tuned
    model scores better. Retuning would confound the two.
    """
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        objective="reg:quantileerror",
        quantile_alpha=alpha,
    )


class IntervalModel:
    """Calibrated [low, high] band for one target.

    predict(X, point) returns (low, high) with three corrections applied in
    order, each for a measured reason:

    1. SORT. The two quantile regressors are independent fits, so nothing makes
       q10 <= q90. Full crossing is rare (0.04% of rows) but real, and an
       unsorted band would occasionally be inverted.
    2. WIDEN by the conformal correction, which is what turns a nominal 80%
       band into an actual one (0.542 -> 0.788 measured).
    3. INCLUDE THE POINT ESTIMATE. The point model minimises squared error and
       so predicts the conditional MEAN; q10/q90 are quantiles of the same
       distribution. Electricity prices are right-skewed, so the mean sits
       above the median (+2.36 EUR/MWh on average) and lands outside the raw
       band on 9.5% of rows -- 1.9% after the conformal widening. Clamping the
       band open around the point costs nothing and means the number the
       automation reads is always inside the range shown next to it.
    """

    def __init__(self, low_model, high_model, correction: float, diagnostics: dict):
        self.low_model = low_model
        self.high_model = high_model
        self.correction = correction
        self.diagnostics = diagnostics

    def predict(self, X: np.ndarray, point: np.ndarray):
        lo_raw = self.low_model.predict(X)
        hi_raw = self.high_model.predict(X)
        low = np.minimum(lo_raw, hi_raw) - self.correction
        high = np.maximum(lo_raw, hi_raw) + self.correction
        return np.minimum(low, point), np.maximum(high, point)


def _conformity_scores(y, low, high) -> np.ndarray:
    """How far outside [low, high] the truth fell. Negative when inside."""
    return np.maximum(low - y, y - high)


def _conformal_correction(scores: np.ndarray, alpha: float = CONFORMAL_ALPHA) -> float:
    """
    The (1-alpha) empirical quantile of the conformity scores, with the standard
    finite-sample (n+1)/n adjustment, floored at 0 so calibration can only ever
    widen the band and never narrow it below what the quantile models produced.
    """
    n = len(scores)
    if n < CONFORMAL_MIN_HOLDOUT:
        return 0.0
    k = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return max(0.0, float(np.quantile(scores, k)))


def train_interval(training_data: pd.DataFrame,
                   target: str = INTERVAL_TARGET) -> IntervalModel:
    """
    Fit the calibrated prediction interval for `target`.

    Split conformal, in three steps:
      1. fit both quantile regressors on everything except the last
         CONFORMAL_HOLDOUT_DAYS rows;
      2. score that holdout to get the conformal correction -- these rows were
         not seen by the calibration fit, which is what makes the resulting
         coverage honest rather than in-sample;
      3. REFIT both regressors on all the data and carry the correction over,
         so the served band uses the full training window like every other
         model here.

    Step 3 assumes the correction transfers between a fit on n-180 rows and one
    on n. It does, in the direction that matters: the full-data models are
    marginally sharper, so carrying the correction over is mildly conservative
    (a slightly wider band than strictly needed), never optimistic.

    Args:
        training_data: The same daily frame train() is given.
        target:        Target name; must be a key of TARGETS.

    Returns:
        IntervalModel, with .diagnostics carrying the holdout coverage before
        and after calibration -- the numbers to watch for drift.
    """
    target_col, feature_cols = TARGETS[target]
    X = training_data[feature_cols].values
    y = training_data[target_col].values

    n_holdout = min(CONFORMAL_HOLDOUT_DAYS, max(0, len(training_data) // 4))
    split = len(training_data) - n_holdout

    if n_holdout >= CONFORMAL_MIN_HOLDOUT:
        lo_cal = _make_quantile_regressor(INTERVAL_QUANTILES[0]).fit(X[:split], y[:split])
        hi_cal = _make_quantile_regressor(INTERVAL_QUANTILES[1]).fit(X[:split], y[:split])
        lo_h = lo_cal.predict(X[split:])
        hi_h = hi_cal.predict(X[split:])
        band_lo, band_hi = np.minimum(lo_h, hi_h), np.maximum(lo_h, hi_h)
        scores = _conformity_scores(y[split:], band_lo, band_hi)
        correction = _conformal_correction(scores)
        diagnostics = {
            "n_holdout": int(n_holdout),
            "coverage_raw": float(np.mean((y[split:] >= band_lo) & (y[split:] <= band_hi))),
            "coverage_calibrated": float(np.mean(
                (y[split:] >= band_lo - correction) & (y[split:] <= band_hi + correction))),
            "correction_eur_mwh": round(correction, 4),
            "nominal_coverage": 1 - CONFORMAL_ALPHA,
        }
    else:
        correction = 0.0
        diagnostics = {"n_holdout": int(n_holdout), "correction_eur_mwh": 0.0,
                       "nominal_coverage": 1 - CONFORMAL_ALPHA,
                       "note": "holdout too small to calibrate; band is uncalibrated"}

    low_model = _make_quantile_regressor(INTERVAL_QUANTILES[0]).fit(X, y)
    high_model = _make_quantile_regressor(INTERVAL_QUANTILES[1]).fit(X, y)
    return IntervalModel(low_model, high_model, correction, diagnostics)


def train(training_data: pd.DataFrame) -> dict:
    """
    Train one XGBoost regressor per daily price target.

    Args:
        training_data: Daily DataFrame with feature columns and price targets.

    Returns:
        Dict keyed by target name (min/avg/max/cheap2h) with fitted models.
    """
    return _fit_models(training_data)


def predict(models: dict, forecast_features: pd.DataFrame,
            interval: IntervalModel = None) -> dict:
    """
    Run inference on forecast features.

    Args:
        models: Dict of fitted models keyed by target name.
        forecast_features: Daily DataFrame with feature columns.
        interval: Optional IntervalModel from train_interval(). When given, each
                  day also gets `<target>_low` / `<target>_high` -- a calibrated
                  band around that day's own prediction. Left as None the output
                  is byte-for-byte what it was before intervals existed, which
                  is what evaluate.py and the A/B harness rely on.

    Returns:
        Dict keyed by date string (YYYY-MM-DD) with min/avg/max/cheap2h in
        EUR/MWh, plus cheap2h_low / cheap2h_high when `interval` is given.
    """
    preds = {
        name: models[name].predict(forecast_features[TARGETS[name][1]].values)
        for name in TARGETS
    }

    band = {}
    if interval is not None:
        target = INTERVAL_TARGET
        low, high = interval.predict(
            forecast_features[TARGETS[target][1]].values, preds[target])
        band = {f"{target}_low": low, f"{target}_high": high}

    dates = pd.to_datetime(forecast_features["date"]).dt.strftime("%Y-%m-%d").values
    predictions = {}

    for i in range(len(dates)):
        predictions[dates[i]] = {
            name: round(float(preds[name][i]), 4) for name in TARGETS
        }
        for name, values in band.items():
            predictions[dates[i]][name] = round(float(values[i]), 4)

    return predictions
