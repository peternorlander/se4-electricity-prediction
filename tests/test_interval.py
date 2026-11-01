"""The cheap2h band's order of operations: sort -> widen -> clamp around the point."""
import numpy as np
import pandas as pd

import model


class _Const:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return self.values


def test_sort_then_widen_then_clamp():
    # row 0: crossed quantiles (q10 30 > q90 10); row 1: point below the band; row 2: above it
    low = _Const([30.0, 20.0, 20.0])
    high = _Const([10.0, 40.0, 40.0])
    band = model.IntervalModel(low, high, correction=5.0, diagnostics={})
    lo, hi = band.predict(np.zeros((3, 1)), point=np.array([20.0, 0.0, 99.0]))
    np.testing.assert_allclose(lo, [5.0, 0.0, 15.0])      # sorted min - 5, then opened to the point
    np.testing.assert_allclose(hi, [35.0, 45.0, 99.0])    # sorted max + 5, then opened to the point


def test_clamp_comes_after_the_widening():
    # A point inside the widened band but outside the raw one must not move the band:
    # clamping before widening would give low = 18 - 5 = 13 instead of 15.
    band = model.IntervalModel(_Const([20.0]), _Const([40.0]), correction=5.0, diagnostics={})
    lo, hi = band.predict(np.zeros((1, 1)), point=np.array([18.0]))
    assert lo[0] == 15.0 and hi[0] == 45.0


def test_conformal_correction_never_narrows():
    assert model._conformal_correction(np.full(100, -3.0)) == 0.0
    assert model._conformal_correction(np.ones(10)) == 0.0       # holdout too small: switched off


class _Classifier:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


def test_cheap_band_high_contains_the_point_and_warns_at_threshold():
    n = 3
    frame = pd.DataFrame({c: np.zeros(n) for c in set(model.COUPLING_FEATURE_COLUMNS)
                          | set(model.regressor_columns(model.INTERVAL_TARGET))})
    frame["gas_marginal_cost"] = 100.0
    interval = model.IntervalModel(_Const([5.0] * n), _Const([30.0] * n), 0.0, {})
    band = model.CheapBand(interval, _Classifier([0.05, 0.29, 0.30]), _Const([30.0] * n),
                           bin_corrections=[1.0, 2.0], coupled_ratio=1.3, diagnostics={})
    low, high, risk = band.predict(frame, point=np.array([40.0, 10.0, 10.0]))
    # calm day clamped to its point / mid-risk calm high + its bin's correction / warned: coupled level
    np.testing.assert_allclose(high, [40.0, 32.0, 130.0])
    np.testing.assert_allclose(low, [5.0, 5.0, 5.0])
    np.testing.assert_allclose(risk, [0.05, 0.29, 0.30])
