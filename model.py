import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
# docs/REJECTED.md (time-decay sample weights row).
HALF_LIFE_DAYS = {"min": None, "avg": 500, "max": None, "cheap2h": None}


# Coupling hurdle (round 24, adopted 2026-10-16). SE4's trough is bimodal on
# workdays: either decoupled (5-35 EUR/MWh) or priced at continental level, about
# 1.0-1.4 x gas_marginal_cost. A LINEAR logistic model of "coupled" -- cheap2h /
# gas_marginal_cost > COUPLING_RATIO_THRESHOLD -- on weather plus the Nordic
# supply state feeds its probability to the min and cheap2h regressors, the same
# pattern as the negative-price hurdle below. Linear on purpose: the reservoir
# deviation sat at the edge of the training window in autumn 2026, and a tree
# cannot extrapolate along it while a logistic can.
#
# Measured against the round-25 baseline on three long caches (2026-10-16,
# 09-16, 08-21): min REAL on all three, every period negative (NOW -0.67 /
# -0.44 / -0.33 EUR/MWh); cheap2h REAL on two (NOW -0.48 / -0.21) and flat on
# the third (+0.02). `nuclear_outage_se3` is deliberately NOT an input: the
# column is contaminated (docs/FINDINGS.md) and the arm without it measured as
# well or better everywhere. docs/DECISIONS.md "Coupling hurdle" has the record.
COUPLING_RATIO_THRESHOLD = 0.6
COUPLING_PROBA_FEATURE_NAME = "p_couple"
COUPLING_TARGETS = {"min", "cheap2h"}
# Order matters only for bit-identity with the measured arm; keep it.
COUPLING_FEATURE_COLUMNS = [
    "mean_wind_stockholm", "max_wind", "mean_wind_de_north", "mean_temp",
    "radiation_variability", "is_workday",
    "reservoir_norway_deviation", "reservoir_sweden_gwh",
]
# Classifier inputs that are NOT regressor features. They ride along in the
# coupling targets' TARGETS lists (so every caller that slices TARGETS[name][1]
# hands the model what it needs) and CouplingAugmentedModel strips them again
# before the regressor sees the row. A variant that wants one of these as a
# regressor feature has to change this list, not just its feature list.
COUPLING_ONLY_COLUMNS = [c for c in COUPLING_FEATURE_COLUMNS if c not in TROUGH_FEATURE_COLUMNS]
assert COUPLING_ONLY_COLUMNS == [c for c in COUPLING_FEATURE_COLUMNS if c not in MIN_FEATURE_COLUMNS]


def _with_coupling_inputs(feature_cols: list) -> list:
    return list(feature_cols) + [c for c in COUPLING_FEATURE_COLUMNS if c not in feature_cols]


# Target name → (training column, input columns).
#
# Deliberately heterogeneous: each target uses the feature set its own ablation
# testing validated, since a feature can be load-bearing for one target and dead
# weight for another. `max` keeps all 51 (not a priority target, never audited).
# `avg` drops its 9-column price/market family (42 of 51 + the 2 northern-wind columns, see
# features.AVG_FEATURE_COLUMNS). The two trough targets run pruned lists that
# differ from each other by exactly one column. cheap2h additionally gets
# neg_price_proba, appended at fit/serve time by the hurdle — see HURDLE_TARGETS —
# and both trough targets get p_couple from the coupling hurdle above; their
# lists carry the classifier-only inputs (COUPLING_ONLY_COLUMNS) at the end, and
# regressor_columns() gives the list the regressor itself is fitted on.
# docs/MODEL.md "Per-Target Feature Sets" has the evidence.
#
# min (16) and cheap2h (17) differ by `price_se4_max_lag1`, and that difference
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
    "min":     ("price_min",     _with_coupling_inputs(MIN_FEATURE_COLUMNS)),
    "avg":     ("price_avg",     AVG_FEATURE_COLUMNS),
    "max":     ("price_max",     FEATURE_COLUMNS),
    "cheap2h": ("price_cheap2h", _with_coupling_inputs(TROUGH_FEATURE_COLUMNS)),
}


def regressor_columns(name: str, feature_cols: list = None) -> list:
    """The columns `name`'s regressor is fitted on: its TARGETS list (or the
    given one) without the coupling classifier's extra inputs."""
    cols = TARGETS[name][1] if feature_cols is None else feature_cols
    if name not in COUPLING_TARGETS:
        return list(cols)
    return [c for c in cols if c not in COUPLING_ONLY_COLUMNS]


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
# to production (see docs/REJECTED.md). HURDLE_TARGETS
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


def _make_coupling_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000))


def coupling_label(data: pd.DataFrame) -> np.ndarray:
    """1 where the day's trough priced at continental level (cheap2h above
    COUPLING_RATIO_THRESHOLD x the lagged gas marginal cost), else 0."""
    return (data["price_cheap2h"] / data["gas_marginal_cost"]
            > COUPLING_RATIO_THRESHOLD).astype(int).values


def fit_coupling_classifier(data: pd.DataFrame):
    """The coupling classifier fitted on all of `data` -- what serves, and what the
    cheap2h band reads its warning from."""
    return _make_coupling_classifier().fit(data[COUPLING_FEATURE_COLUMNS].values,
                                           coupling_label(data))


class CouplingAugmentedModel:
    """
    Wraps the coupling classifier around a target's own model (a plain regressor,
    or the HurdleAugmentedModel for cheap2h). predict(X) takes the target's
    TARGETS columns, computes P(coupled) from the classifier's columns, and hands
    the inner model its regressor columns with that probability appended -- so
    it is a drop-in model for every caller that slices TARGETS[name][1].
    """
    def __init__(self, classifier, inner, regressor_idx: list, coupling_idx: list,
                 feature_names: list):
        self.classifier = classifier
        self.inner = inner
        self.regressor_idx = regressor_idx
        self.coupling_idx = coupling_idx
        self.feature_names = feature_names

    def coupling_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X[:, self.coupling_idx])[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.coupling_proba(X)
        return self.inner.predict(np.column_stack([X[:, self.regressor_idx], p]))

    @property
    def feature_importances_(self):
        return self.inner.feature_importances_


def _fit_coupled_model(data: pd.DataFrame, name: str, target_col: str, feature_cols: list,
                       half_life_days=None) -> CouplingAugmentedModel:
    """
    Fit `name` with an out-of-fold P(coupled) appended to its regressor columns.
    Same leak-safety argument as _fit_hurdle_model: the regressor trains on
    out-of-fold probabilities, the served classifier is fitted on all of `data`.
    For cheap2h the inner model is the negative-price hurdle, whose classifier
    then also sees p_couple -- which is how it was measured.
    """
    reg_cols = regressor_columns(name, feature_cols)
    missing = [c for c in COUPLING_FEATURE_COLUMNS if c not in feature_cols]
    if missing:
        raise ValueError(f"{name}: coupling inputs {missing} are not in its feature list")

    oof = cross_val_predict(
        _make_coupling_classifier(), data[COUPLING_FEATURE_COLUMNS].values, coupling_label(data),
        cv=KFold(n_splits=5, shuffle=True, random_state=42), method="predict_proba",
    )[:, 1]
    classifier = fit_coupling_classifier(data)

    augmented = data.copy()
    augmented[COUPLING_PROBA_FEATURE_NAME] = oof
    inner_cols = reg_cols + [COUPLING_PROBA_FEATURE_NAME]
    if name in HURDLE_TARGETS:
        inner = _fit_hurdle_model(augmented, inner_cols, target_col, half_life_days=half_life_days)
        names = inner_cols + [HURDLE_PROBA_FEATURE_NAME]
    else:
        inner = _make_regressor()
        inner.fit(augmented[inner_cols].values, augmented[target_col].values,
                  sample_weight=_sample_weights(augmented, half_life_days))
        names = inner_cols

    return CouplingAugmentedModel(
        classifier, inner,
        regressor_idx=[feature_cols.index(c) for c in reg_cols],
        coupling_idx=[feature_cols.index(c) for c in COUPLING_FEATURE_COLUMNS],
        feature_names=names,
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
        if name in COUPLING_TARGETS:
            models[name] = _fit_coupled_model(data, name, target_col, list(feature_cols),
                                              half_life_days=_hl_for(name))
            continue
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
# Measured on the round-15b sliding grid, four period clusters (see
# docs/DECISIONS.md "Prediction interval for cheap2h"):
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
    target_col = TARGETS[target][0]
    X = training_data[regressor_columns(target)].values
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


# ---------------------------------------------------------------------------
# Coupling-aware high for cheap2h (round 24, adopted 2026-10-16)
# ---------------------------------------------------------------------------
# The symmetric band above cannot express "either 10 or 120". On a workday that
# may couple to the continent its q90 sits in between, so it is too high on calm
# days (81 against an outcome of 13) and too low on coupled ones (half the
# autumn-2026 weekday outcomes landed above it). `cheap2h_high` is therefore a
# WARNING LIGHT with two states, and `cheap2h_low` stays the conformal q10:
#
#   * P(coupled) >= COUPLING_WARNING_PROBA: high = the coupled level,
#     COUPLED_LEVEL_QUANTILE of cheap2h / gas_marginal_cost over the coupled
#     training days, times the (frozen) gas marginal cost;
#   * otherwise: a q90 fitted on the DECOUPLED training days only, widened by a
#     split-conformal correction computed PER RISK BIN (round 24b, below).
#
# Checked on unseen days 2026-10-10..15: every expensive day was warned on every
# horizon (9 of 9), outcomes above high fell 43 % -> 10 %. Over the last year it
# warns on ~46 % of days with ~69 % precision and catches ~80 % of expensive
# days. With the risk-binned calibration below (round 24b) fewer outcomes end
# above high than under the old symmetric band in every period measured, while
# calm days keep a low high. It is a warning, not an 80 % interval: one level
# when warned, a risk-dependent q90 otherwise. The threshold was pre-registered at 0.4 and
# lowered to 0.3 after scoring both (12 more expensive days caught a year for 8
# more false alarms); docs/MODEL.md "The cheap2h band" has the contract,
# docs/AB_TESTING.md "Scoring the band" how it is measured.
COUPLING_WARNING_PROBA = 0.3
COUPLED_LEVEL_QUANTILE = 0.75
CALM_HIGH_QUANTILE = 0.90
CALM_HIGH_ALPHA = 0.10           # its one-sided conformal miss rate (1 - CALM_HIGH_QUANTILE)

# Risk bins for the unwarned high (round 24b, adopted 2026-10-19). Calibrated on
# decoupled days alone, the calm q90 sat BELOW the point estimate on days with a
# real but sub-threshold coupling risk -- the point model prices that risk in --
# so the clamp made high == cheap2h on 27-39 % of days at risk 0.1-0.3, and 36-41 %
# of those days ended above it. The calm q90 now gets one split-conformal
# correction per bin of risk, scored over EVERY holdout day the band would not
# warn on (coupled ones the classifier missed included), so ~90 % of unwarned days
# at a given risk end below high. Lower edges; the last bin runs to the warning
# threshold. A bin with fewer than CONFORMAL_MIN_HOLDOUT holdout days falls back
# to the pooled correction over all unwarned holdout days.
RISK_BIN_EDGES = (0.0, 0.1)
RISK_KEY = f"{INTERVAL_TARGET}_risk"
# Keys predict() emits that are not prices: currency and addon conversion pass
# them through untouched.
UNITLESS_KEYS = frozenset({RISK_KEY})


class CheapBand:
    """`cheap2h_low` / `cheap2h_high` / `cheap2h_risk` for a forecast frame.

    predict(frame, point) returns (low, high, risk). low is the IntervalModel's
    calibrated q10; high is the warning described above (the coupled level when
    warned, else the calm q90 plus its risk bin's correction); both are clamped
    open around the point estimate so the number Home Assistant ranks on is
    always inside the range shown next to it.
    """

    warning_threshold = COUPLING_WARNING_PROBA

    def __init__(self, interval: IntervalModel, coupling_classifier, calm_high_model,
                 bin_corrections, coupled_ratio: float, diagnostics: dict):
        self.interval = interval
        self.coupling_classifier = coupling_classifier
        self.calm_high_model = calm_high_model
        self.bin_corrections = np.asarray(bin_corrections, dtype=float)
        self.coupled_ratio = coupled_ratio
        self.diagnostics = diagnostics

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return self.coupling_classifier.predict_proba(frame[COUPLING_FEATURE_COLUMNS].values)[:, 1]

    def predict(self, frame: pd.DataFrame, point: np.ndarray):
        X = frame[regressor_columns(INTERVAL_TARGET)].values
        low, _ = self.interval.predict(X, point)
        risk = self.risk(frame)
        calm_high = self.calm_high_model.predict(X) + self.bin_corrections[_risk_bin(risk)]
        coupled_high = self.coupled_ratio * frame["gas_marginal_cost"].values
        high = np.where(risk >= self.warning_threshold,
                        np.maximum(coupled_high, calm_high), calm_high)
        return low, np.maximum(high, point), risk


def _risk_bin(risk: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(RISK_BIN_EDGES, risk, side="right") - 1,
                   0, len(RISK_BIN_EDGES) - 1)


def train_band(training_data: pd.DataFrame) -> CheapBand:
    """
    Fit the cheap2h band: the conformal interval (for low), the coupling
    classifier (for the warning and the risk), the calm-day q90 with one
    split-conformal correction per risk bin, and the coupled level.

    The risk-bin corrections are split conformal over the last
    CONFORMAL_HOLDOUT_DAYS of the window: the calm q90 and the classifier are
    fitted on the rows before it, every holdout day the band would not warn on is
    scored (y - q90), and each bin takes the (1 - CALM_HIGH_ALPHA) quantile of its
    own days. The served q90 and classifier are then refitted on all rows.
    """
    interval = train_interval(training_data, INTERVAL_TARGET)
    target_col = TARGETS[INTERVAL_TARGET][0]
    reg_cols = regressor_columns(INTERVAL_TARGET)
    label = coupling_label(training_data)

    calm = training_data[label == 0]
    calm_high_model = _make_quantile_regressor(CALM_HIGH_QUANTILE).fit(
        calm[reg_cols].values, calm[target_col].values)

    split = len(training_data) - min(CONFORMAL_HOLDOUT_DAYS, len(training_data) // 4)
    head, hold = training_data.iloc[:split], training_data.iloc[split:]
    head_calm = head[coupling_label(head) == 0]
    head_q = _make_quantile_regressor(CALM_HIGH_QUANTILE).fit(
        head_calm[reg_cols].values, head_calm[target_col].values)
    hold_risk = fit_coupling_classifier(head).predict_proba(
        hold[COUPLING_FEATURE_COLUMNS].values)[:, 1]
    scores = hold[target_col].values - head_q.predict(hold[reg_cols].values)
    unwarned = hold_risk < COUPLING_WARNING_PROBA
    pooled = _conformal_correction(scores[unwarned], alpha=CALM_HIGH_ALPHA)
    edges = list(RISK_BIN_EDGES) + [COUPLING_WARNING_PROBA]
    bin_corrections, bin_days = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (hold_risk >= lo) & (hold_risk < hi)
        bin_days.append(int(in_bin.sum()))
        bin_corrections.append(
            _conformal_correction(scores[in_bin], alpha=CALM_HIGH_ALPHA)
            if in_bin.sum() >= CONFORMAL_MIN_HOLDOUT else pooled)
    served = np.asarray(bin_corrections)[_risk_bin(hold_risk)]
    holdout_coverage = float(np.mean(scores[unwarned] <= served[unwarned])) if unwarned.any() else float("nan")

    ratio = (training_data[target_col] / training_data["gas_marginal_cost"]).values
    coupled_ratio = float(np.quantile(ratio[label == 1], COUPLED_LEVEL_QUANTILE))

    bin_names = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    diagnostics = {
        **interval.diagnostics,
        "calm_high_correction_eur_mwh": {n: round(float(c), 4) for n, c in zip(bin_names, bin_corrections)},
        "calm_high_holdout_days": dict(zip(bin_names, bin_days)),
        "calm_high_holdout_coverage": holdout_coverage,
        "coupled_level_ratio": round(coupled_ratio, 4),
        "coupled_share_training": round(float(label.mean()), 4),
        "warning_threshold": COUPLING_WARNING_PROBA,
    }
    return CheapBand(interval, fit_coupling_classifier(training_data), calm_high_model,
                     bin_corrections, coupled_ratio, diagnostics)


def train(training_data: pd.DataFrame) -> dict:
    """
    Train one XGBoost regressor per daily price target.

    Args:
        training_data: Daily DataFrame with feature columns and price targets.

    Returns:
        Dict keyed by target name (min/avg/max/cheap2h) with fitted models.
    """
    return _fit_models(training_data)


def enforce_coherence(preds: dict) -> dict:
    """
    Make the predicted pair obey the identity min <= cheap2h.

    The two targets are independent regressors, and the true gap between them is tiny
    (over the last year the mean of the two cheapest hours sits a median 0.4 EUR/MWh above
    the cheapest hour), so their predictions cross on 40-50 % of days. Where predicted `min`
    exceeds predicted `cheap2h`, BOTH are replaced by their mean ("split", round 25,
    docs/DECISIONS.md "min <= cheap2h coherence"): when the models disagree about the order
    both are evidently off by about the same, and averaging halves both errors. Days that
    already obey the identity are untouched, and `avg` / `max` are never touched.

    Applied in predict(), in evaluate.walk_forward_validate and in ab.harness.run_walk_forward
    -- the three places that turn fitted models into numbers -- so the headline eval and the
    A/B describe what Home Assistant receives. A no-op unless both targets are present, so a
    variant that predicts only one of them is unaffected. Mutates and returns `preds`.
    """
    if "min" in preds and "cheap2h" in preds:
        low = np.asarray(preds["min"], dtype=float)
        cheap = np.asarray(preds["cheap2h"], dtype=float)
        crossed = low > cheap
        middle = (low + cheap) / 2.0
        preds["min"] = np.where(crossed, middle, low)
        preds["cheap2h"] = np.where(crossed, middle, cheap)
    return preds


def predict(models: dict, forecast_features: pd.DataFrame,
            band: CheapBand = None) -> dict:
    """
    Run inference on forecast features.

    Args:
        models: Dict of fitted models keyed by target name.
        forecast_features: Daily DataFrame with feature columns.
        band: Optional CheapBand from train_band(). When given, each day also
              gets `cheap2h_low` / `cheap2h_high` (EUR/MWh) and `cheap2h_risk`,
              the probability of a coupled -- expensive -- trough. Left as None
              the output is just the four point targets, which is what
              evaluate.py and the A/B harness rely on.

    Returns:
        Dict keyed by date string (YYYY-MM-DD) with min/avg/max/cheap2h in
        EUR/MWh, plus the band keys when `band` is given. `cheap2h_risk` is a
        probability, listed in UNITLESS_KEYS so the SEK and addon conversions
        leave it alone.
    """
    preds = enforce_coherence({
        name: models[name].predict(forecast_features[TARGETS[name][1]].values)
        for name in TARGETS
    })

    extra = {}
    if band is not None:
        target = INTERVAL_TARGET
        low, high, risk = band.predict(forecast_features, preds[target])
        extra = {f"{target}_low": (low, 4), f"{target}_high": (high, 4), RISK_KEY: (risk, 3)}

    dates = pd.to_datetime(forecast_features["date"]).dt.strftime("%Y-%m-%d").values
    predictions = {}

    for i in range(len(dates)):
        predictions[dates[i]] = {
            name: round(float(preds[name][i]), 4) for name in TARGETS
        }
        for name, (values, digits) in extra.items():
            predictions[dates[i]][name] = round(float(values[i]), digits)

    return predictions
