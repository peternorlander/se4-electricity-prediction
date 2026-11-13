"""
The production data checks (data_checks.py) and the DST days, on every cached long
snapshot. Skipped where ab_cache/long/ is absent, as on the Actions runner.
"""
import functools

import numpy as np
import pandas as pd

from se4.ab.snapshot import load_snapshot, LONG_CACHE_ROOT
from se4.data_checks import check_training_data
from se4.features import build_training_data

from conftest import ROOT


@functools.lru_cache(maxsize=None)
def _load(snapshot):
    inputs, _ = load_snapshot(snapshot, cache_root=ROOT / LONG_CACHE_ROOT)
    return inputs, build_training_data(**inputs)


def test_production_checks_pass(snapshot):
    inputs, data = _load(snapshot)
    check_training_data(inputs, data)


def test_dst_days_are_whole_rows(snapshot):
    inputs, data = _load(snapshot)
    prices = inputs["prices_hourly"].copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    hourly = prices.set_index("timestamp")["price_eur_mwh"].resample("1h").mean().dropna()
    counts = pd.Series(1, index=hourly.index.tz_convert("Europe/Stockholm").date).groupby(level=0).sum()
    first, last = data["date"].iloc[0].date(), data["date"].iloc[-1].date()
    dst = counts[(counts != 24) & (counts.index > first) & (counts.index < last)]
    assert len(dst) >= 8 and set(dst.values) <= {23, 25}
    rows = data[data["date"].dt.date.isin(set(dst.index))]
    assert len(rows) == len(dst)                        # every DST day is in the frame...
    assert np.isfinite(rows.select_dtypes("number").values).all()   # ...with every column finite


def test_frame_has_no_date_gaps(snapshot):
    _, data = _load(snapshot)
    assert (pd.to_datetime(data["date"]).diff().dt.days.iloc[1:] == 1).all()
