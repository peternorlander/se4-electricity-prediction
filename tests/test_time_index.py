"""The time index `avg` reads (plan 2.19, docs/DECISIONS.md "Time index for avg").

Its value is a rule about the date alone, and its reach is a rule about the lists: `avg` only,
never frozen, with the half-life it was measured with.
"""
import numpy as np
import pandas as pd

from se4 import features as feat
from se4 import model as M


def test_t_index_is_years_since_a_fixed_epoch_for_any_frame():
    # The same calendar day gets the same value whether it is a training row or a forecast row,
    # and the epoch is the one the measurement used (a tree splits on the raw value).
    dates = pd.to_datetime(["2021-10-21", "2021-10-22", "2022-10-21", "2026-11-14"])
    out = feat.add_time_features(pd.DataFrame({"date": dates}))
    assert out["t_index"].tolist() == [0.0, 1 / 365, 1.0, (dates[3] - pd.Timestamp("2021-10-21")).days / 365.0]
    assert feat.TIME_INDEX_EPOCH == pd.Timestamp("2021-10-21")
    # date objects, as the training frame carries them, give the same number
    as_dates = feat.add_time_features(pd.DataFrame({"date": [d.date() for d in dates]}))
    assert np.array_equal(as_dates["t_index"].values, out["t_index"].values)


def test_only_avg_reads_the_index_and_it_is_never_frozen():
    assert feat.AVG_FEATURE_COLUMNS[-1] == "t_index"           # appended last, as measured
    assert feat.AVG_FEATURE_COLUMNS[-3:-1] == feat.NORTH_WIND_COLUMNS
    for name in ("min", "max", "cheap2h"):
        assert "t_index" not in M.TARGETS[name][1], name
    assert "t_index" not in feat.FEATURE_COLUMNS
    assert "t_index" not in feat.FORECAST_FROZEN_FEATURES


def test_avg_half_life_is_the_one_measured_with_the_index():
    assert M.HALF_LIFE_DAYS == {"min": None, "avg": 365, "max": None, "cheap2h": None}
    dates = pd.date_range("2026-01-01", periods=366)            # 365 days apart at the two ends
    w = M._sample_weights(pd.DataFrame({"date": dates}), M.HALF_LIFE_DAYS["avg"])
    assert w[-1] == 1.0 and abs(w[0] - 0.5) < 1e-12
