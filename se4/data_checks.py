"""
Invariants of the fetched training data, checked on every production run before the
models are fitted (predict.py) and on every cached snapshot by the test suite
(tests/test_snapshot_data.py).

Each check is one that would have caught a real defect early, and each threshold was
set against the six long caches of 2026-08-06 .. 2026-10-16 with a wide margin
(round 31, IMPROVEMENT_PLAN.md 2.12):

* Wind arrives in km/h. Open-Meteo's `windspeed_100m` default is km/h and
  features._wind_power_curve converts to m/s itself; the 2026-09 review found the curve
  fed km/h as if it were m/s, saturated on ~88 % of hours (docs/FINDINGS.md 1.1). Over
  three years every point's hourly median is 21-27 km/h and the converted curve reaches
  rated speed on 0.6-7.3 % of hours. A median outside 12-45 km/h, or a curve saturated on
  more than a quarter of the hours, means the unit or the conversion changed.
* min <= cheap2h <= avg <= max holds on every row by construction (aggregate_prices_daily).
  A violation means the daily price aggregation broke: zero rows violate it on all six caches.

A failed check raises DataCheckError: a run on corrupted inputs would push confident
numbers that are wrong, and the run-status sensor (ha_client.push_run_status) tells the
automation that the predictions it holds are the previous run's.
"""
import numpy as np
import pandas as pd

WIND_MEDIAN_KMH = (12.0, 45.0)
MAX_RATED_SHARE = 0.25
RATED_SPEED_MS = 13.0
PRICE_ORDER = ("price_min", "price_cheap2h", "price_avg", "price_max")


class DataCheckError(ValueError):
    pass


def wind_columns(hourly: pd.DataFrame) -> list:
    return [c for c in hourly.columns if c.startswith("windspeed")]


def wind_unit_problems(hourly: pd.DataFrame, label: str) -> list:
    """Problems with the hourly wind columns of one weather frame, as readable lines."""
    problems = []
    for col in wind_columns(hourly):
        values = hourly[col].dropna()
        if values.empty:
            problems.append(f"{label}.{col}: no values")
            continue
        median = float(values.median())
        rated = float(((values / 3.6) >= RATED_SPEED_MS).mean())
        if not WIND_MEDIAN_KMH[0] <= median <= WIND_MEDIAN_KMH[1]:
            problems.append(f"{label}.{col}: median {median:.1f} outside {WIND_MEDIAN_KMH} km/h "
                            f"-- is the unit still km/h?")
        if rated > MAX_RATED_SHARE:
            problems.append(f"{label}.{col}: power curve at rated speed on {rated:.0%} of hours "
                            f"(limit {MAX_RATED_SHARE:.0%}) -- km/h fed in as m/s?")
    return problems


def price_order_problems(daily: pd.DataFrame) -> list:
    """Rows where min <= cheap2h <= avg <= max does not hold."""
    problems = []
    for lower, upper in zip(PRICE_ORDER, PRICE_ORDER[1:]):
        bad = daily[daily[lower] > daily[upper] + 1e-9]
        if len(bad):
            days = ", ".join(pd.to_datetime(bad["date"]).dt.strftime("%Y-%m-%d").head(5))
            problems.append(f"{lower} > {upper} on {len(bad)} day(s): {days}")
    return problems


def check_training_data(inputs: dict, training_data: pd.DataFrame) -> None:
    """Raise DataCheckError listing every violated invariant; silent when all hold."""
    problems = []
    for key in ("weather_hourly", "wind_intl_hourly"):
        if inputs.get(key) is not None:
            problems += wind_unit_problems(inputs[key], key)
    problems += price_order_problems(training_data)
    numeric = training_data.select_dtypes("number")
    if not np.isfinite(numeric.values).all():
        problems.append("non-finite values in the training frame")
    if problems:
        raise DataCheckError("training data failed its checks:\n  " + "\n  ".join(problems))
