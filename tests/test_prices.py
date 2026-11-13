"""Daily price targets: the ordering identity and the DST days."""
import numpy as np
import pandas as pd

from se4.features import aggregate_prices_daily, aggregate_weather_daily, complete_swedish_dates
from se4.data_checks import price_order_problems


def _hourly(start_local: str, end_local: str, freq="1h", seed=0) -> pd.DataFrame:
    stamps = pd.date_range(pd.Timestamp(start_local, tz="Europe/Stockholm"),
                           pd.Timestamp(end_local, tz="Europe/Stockholm"), freq=freq, inclusive="left")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"timestamp": stamps.tz_convert("UTC"),
                         "price_eur_mwh": rng.normal(50, 40, len(stamps))})


def test_min_cheap2h_avg_max_ordering_on_every_row():
    daily = aggregate_prices_daily(_hourly("2025-01-01", "2026-01-01"))
    assert len(daily) == 365
    assert price_order_problems(daily) == []
    assert (daily.price_min <= daily.price_cheap2h).all()
    assert (daily.price_cheap2h <= daily.price_avg).all()
    assert (daily.price_avg <= daily.price_max).all()


def test_ordering_check_names_a_broken_row():
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "price_min": [10.0],
                          "price_cheap2h": [9.0], "price_avg": [20.0], "price_max": [30.0]})
    assert price_order_problems(daily) == ["price_min > price_cheap2h on 1 day(s): 2026-01-01"]


def test_quarter_hours_are_resampled_to_hourly_before_the_targets():
    quarter = _hourly("2025-10-01", "2025-10-03", freq="15min")
    hourly = quarter.set_index("timestamp").resample("1h").mean().reset_index()
    a, b = aggregate_prices_daily(quarter), aggregate_prices_daily(hourly)
    pd.testing.assert_frame_equal(a, b)


def test_dst_days_have_23_and_25_hours_and_no_nan():
    prices = _hourly("2025-03-29", "2025-04-01")
    weather = prices.rename(columns={"price_eur_mwh": "temperature"}).assign(windspeed=20.0, radiation=0.0)
    for frame in (prices, weather):
        counts = pd.to_datetime(frame.timestamp).dt.tz_convert("Europe/Stockholm").dt.date.value_counts()
        assert counts[pd.Timestamp("2025-03-30").date()] == 23
    autumn = _hourly("2025-10-25", "2025-10-28")
    counts = pd.to_datetime(autumn.timestamp).dt.tz_convert("Europe/Stockholm").dt.date.value_counts()
    assert counts[pd.Timestamp("2025-10-26").date()] == 25

    for frame in (prices, autumn):
        daily = aggregate_prices_daily(frame)
        assert len(daily) == 3 and not daily.isna().any().any()
        assert set(daily.date) <= complete_swedish_dates(frame)
    daily_w = aggregate_weather_daily(weather)
    assert len(daily_w) == 3 and not daily_w.isna().any().any()


def test_a_day_missing_an_hour_is_not_complete():
    frame = _hourly("2025-10-25", "2025-10-28")
    frame = frame.drop(index=30)          # one hour of 2025-10-26
    assert pd.Timestamp("2025-10-26").date() not in complete_swedish_dates(frame)
