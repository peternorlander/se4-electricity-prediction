"""
fetch_training_inputs goes through the source cache for every historical source, and gives the
same frames with the cache as without (source_cache.py; docs/DATA_CACHE.md). A source added to
fetch_training_inputs without being wrapped would silently keep the fragile three-year fetch.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

import fetch_data
import source_cache

TODAY = date(2026, 11, 8)


def _hourly(start, end, columns):
    stamps = pd.date_range(pd.Timestamp(start, tz="UTC"),
                           pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23), freq="h")
    return pd.DataFrame({"timestamp": stamps, **{c: 1.0 for c in columns}})


def _daily(start, end, column, step=1):
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.toordinal() % step == 0]       # a fixed weekly grid, like a publisher's
    return pd.DataFrame({"date": days, column: 1.0})


@pytest.fixture
def stubs(monkeypatch):
    """Every historical fetch replaced by a recorder that returns a frame for its window."""
    calls = {}

    def record(name, build):
        def fetch(start, end):
            calls.setdefault(name, []).append((str(start), str(end)))
            return build(start, end)
        return fetch

    def ymd(text):
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))

    def iso(text):
        return date.fromisoformat(text)

    monkeypatch.setattr(fetch_data, "fetch_prices", record(
        "prices", lambda s, e: _hourly(ymd(s), ymd(e) - timedelta(days=1), ["price_eur_mwh"])))
    monkeypatch.setattr(fetch_data, "fetch_market_prices", record(
        "market", lambda s, e: _hourly(ymd(s), ymd(e) - timedelta(days=1), ["price_de", "price_dk2"])))
    monkeypatch.setattr(fetch_data, "fetch_historical", record(
        "weather", lambda s, e: _hourly(iso(s), iso(e), ["temperature", "windspeed", "radiation"])))
    monkeypatch.setattr(fetch_data, "fetch_international_wind_historical", record(
        "wind", lambda s, e: _hourly(iso(s), iso(e), ["windspeed_dk1"])))
    monkeypatch.setattr(fetch_data, "fetch_ttf_prices", record(
        "ttf", lambda s, e: _daily(iso(s), iso(e) - timedelta(days=1), "ttf_close")))
    monkeypatch.setattr(fetch_data, "fetch_eua_prices", record(
        "eua", lambda s, e: _daily(iso(s), iso(e) - timedelta(days=1), "eua_close")))
    monkeypatch.setattr(fetch_data, "fetch_nuclear_outages_se3", record(
        "nuclear", lambda s, e: _daily(ymd(s), ymd(e) - timedelta(days=1), "nuclear_outage_se3")))
    monkeypatch.setattr(fetch_data, "fetch_reservoir_norway", record(
        "norway", lambda s, e: _daily(iso(s), iso(e), "fill_pct", step=7)))
    monkeypatch.setattr(fetch_data, "fetch_reservoir_sweden", record(
        "sweden", lambda s, e: _daily(ymd(s), ymd(e) - timedelta(days=1), "reservoir_sweden_gwh", step=7)))
    monkeypatch.setattr(fetch_data, "fetch_reservoir_norway_median", lambda: pd.DataFrame(
        {"week_number": [1], "median_fill_pct": [0.5], "min_fill_pct": [0.2], "max_fill_pct": [0.9]}))
    # the top-up from the forecast endpoint is not cached and not under test here
    monkeypatch.setattr(fetch_data, "fetch_recent", lambda days: pd.DataFrame())
    monkeypatch.setattr(fetch_data, "fetch_international_wind_recent", lambda days: pd.DataFrame())
    monkeypatch.setattr(fetch_data, "get_non_workdays", lambda s, e: [])
    monkeypatch.setattr(fetch_data, "TRAINING_DAYS", 200)
    return calls


def test_a_second_run_asks_every_historical_source_only_for_the_tail(stubs, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(source_cache.CACHE_DIR_ENV, str(tmp_path))
    fetch_data.fetch_training_inputs(TODAY)
    first = {name: list(windows) for name, windows in stubs.items()}
    stubs.clear()

    fetch_data.fetch_training_inputs(TODAY + timedelta(days=1))

    assert set(stubs) == set(first) == {"prices", "market", "weather", "wind", "ttf", "eua",
                                        "nuclear", "norway", "sweden"}
    full_start = TODAY - timedelta(days=200)
    for name, windows in stubs.items():
        assert len(windows) == 1, name
        asked_from = date.fromisoformat(windows[0][0]) if "-" in windows[0][0] else \
            date(int(windows[0][0][:4]), int(windows[0][0][4:6]), int(windows[0][0][6:]))
        assert asked_from > full_start + timedelta(days=100), \
            f"{name} re-fetched from {asked_from}, i.e. most of its history"


def test_results_with_and_without_the_cache_are_the_same(stubs, tmp_path, monkeypatch):
    monkeypatch.delenv(source_cache.CACHE_DIR_ENV, raising=False)
    plain = fetch_data.fetch_training_inputs(TODAY + timedelta(days=1))

    monkeypatch.setenv(source_cache.CACHE_DIR_ENV, str(tmp_path))
    fetch_data.fetch_training_inputs(TODAY)
    cached = fetch_data.fetch_training_inputs(TODAY + timedelta(days=1))

    for name, frame in plain.items():
        if isinstance(frame, pd.DataFrame):
            pd.testing.assert_frame_equal(frame.reset_index(drop=True),
                                          cached[name].reset_index(drop=True), obj=name)
