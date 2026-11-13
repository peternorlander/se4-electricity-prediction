"""
`ab_test.py fetch --incremental`: a snapshot seeds the production source cache, so the next
snapshot asks each source only for what is new (se4/ab/seed.py).
"""
import json
import os
from datetime import date, timedelta

import pandas as pd
import pytest

from se4 import source_cache as sc
from se4.ab import seed

TODAY = date(2026, 10, 16)


def _daily(last, days=1900, col="value"):
    d = [last - timedelta(days=i) for i in range(days)][::-1]
    return pd.DataFrame({"date": d, col: [x.toordinal() for x in d]})


def _hourly(last_day, days=40, col="value"):
    ts = pd.date_range(end=pd.Timestamp(last_day, tz="UTC") + pd.Timedelta(hours=23),
                       periods=days * 24, freq="1h")
    return pd.DataFrame({"timestamp": ts, col: range(len(ts))})


@pytest.fixture
def base_snapshot():
    inputs = {
        "prices_hourly": _hourly(TODAY),
        "market_prices_hourly": _hourly(TODAY),
        # archive ends today-5; the spliced top-up runs to yesterday
        "weather_hourly": _hourly(TODAY - timedelta(days=1)),
        "wind_intl_hourly": _hourly(TODAY - timedelta(days=1)),
        "ttf_daily": _daily(TODAY - timedelta(days=1), col="ttf_close"),
        "eua_daily": _daily(TODAY - timedelta(days=1), col="eua_close"),
        "nuclear_daily": _daily(TODAY - timedelta(days=1)),
        "norway_reservoir_weekly": _daily(TODAY - timedelta(days=5)),
        "sweden_reservoir_weekly": _daily(TODAY - timedelta(days=12)),
        "norway_reservoir_median": pd.DataFrame({"week_number": [1, 2], "median_fill_pct": [1., 2.]}),
    }
    meta = {"today": TODAY.isoformat(), "fetched_at": "2026-10-16T16:59:38.072943",
            "source_versions": seed.source_versions()}
    return inputs, meta


def test_every_source_is_seeded_and_read_back_as_a_cache_hit(base_snapshot, tmp_path):
    inputs, meta = base_snapshot
    assert seed.seed_source_cache(inputs, meta, tmp_path, archive_lag_days=5) == {}
    assert len(list(tmp_path.glob("*.pkl"))) == len(seed.SNAPSHOT_SOURCES)
    assert set(seed.SNAPSHOT_SOURCES.values()) == set(sc.SOURCES)    # no source forgotten


def test_the_forecast_top_up_is_not_seeded_as_archive(base_snapshot, tmp_path):
    inputs, meta = base_snapshot
    seed.seed_source_cache(inputs, meta, tmp_path, archive_lag_days=5)
    record = sc._load(sc.SOURCES["weather_archive_se4"], tmp_path)
    assert record["frame"]["timestamp"].max() < pd.Timestamp(TODAY - timedelta(days=5), tz="UTC")


def test_a_second_snapshot_asks_only_for_the_tail(base_snapshot):
    inputs, meta = base_snapshot
    calls = []

    def fake(start, end):
        calls.append((start, end))
        return _daily(end, days=(end - start).days + 1, col="ttf_close")

    new_today = TODAY + timedelta(days=20)
    with seed.seeded_cache(inputs, meta, archive_lag_days=5) as (skipped, _):
        out = sc.cached_fetch(sc.SOURCES["yahoo_ttf"], new_today - timedelta(days=1825),
                              new_today, fake)
    source = sc.SOURCES["yahoo_ttf"]
    assert skipped == {}
    assert calls == [(TODAY - timedelta(days=1) - timedelta(days=source.overlap_days), new_today)]
    assert out["date"].max() == new_today and out["date"].is_unique


def test_a_version_mismatch_is_fetched_in_full(base_snapshot):
    inputs, meta = base_snapshot
    meta["source_versions"]["yahoo_ttf"] = 999
    with seed.seeded_cache(inputs, meta, archive_lag_days=5) as (skipped, tmp):
        assert "yahoo_ttf" in skipped and "version" in skipped["yahoo_ttf"]
        assert not (tmp / "yahoo_ttf.pkl").exists()


def test_a_dead_source_fails_unless_staleness_is_allowed_explicitly(base_snapshot):
    inputs, meta = base_snapshot
    new_today = TODAY + timedelta(days=28)      # the real case: Yahoo down, base 4 weeks old

    def dead(start, end):
        return pd.DataFrame(columns=["date", "ttf_close"])

    with seed.seeded_cache(inputs, meta, archive_lag_days=5):
        with pytest.raises(RuntimeError, match="Nothing is served from a cache that old"):
            sc.cached_fetch(sc.SOURCES["yahoo_ttf"], new_today - timedelta(days=1825),
                            new_today, dead)

    with seed.seeded_cache(inputs, meta, archive_lag_days=5, max_stale_days=40):
        out = sc.cached_fetch(sc.SOURCES["yahoo_ttf"], new_today - timedelta(days=1825),
                              new_today, dead)
        assert out["date"].max() == TODAY - timedelta(days=1)


def test_environment_and_stale_limits_are_restored(base_snapshot, monkeypatch):
    inputs, meta = base_snapshot
    monkeypatch.setenv(sc.CACHE_DIR_ENV, "somewhere")
    before = dict(sc.SOURCES)
    with seed.seeded_cache(inputs, meta, archive_lag_days=5, max_stale_days=99):
        assert os.environ[sc.CACHE_DIR_ENV] != "somewhere"
    assert os.environ[sc.CACHE_DIR_ENV] == "somewhere" and dict(sc.SOURCES) == before


def test_save_snapshot_records_source_versions(base_snapshot, tmp_path):
    from se4.ab.snapshot import save_snapshot
    inputs, _ = base_snapshot
    folder = save_snapshot(inputs, TODAY, cache_root=tmp_path, extra_meta={"incremental_from": "x"})
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["source_versions"] == seed.source_versions() and meta["incremental_from"] == "x"


def _production_cache(tmp_path, first, last, version=None):
    source = sc.SOURCES["yahoo_ttf"]
    frame = _daily(last, days=(last - first).days + 1, col="ttf_close")
    frame["ttf_close"] += 0.5                                   # production's rows differ
    sc._save(dataclasses_replace(source, version or source.version), tmp_path, frame,
             covers_from=first)
    return tmp_path


def dataclasses_replace(source, version):
    import dataclasses
    return dataclasses.replace(source, version=version)


def test_a_production_cache_extends_the_snapshot_so_a_dead_source_passes(base_snapshot, tmp_path):
    inputs, meta = base_snapshot
    new_today = TODAY + timedelta(days=28)
    prod = _production_cache(tmp_path, TODAY - timedelta(days=700), new_today - timedelta(days=2))

    def dead(start, end):
        return pd.DataFrame(columns=["date", "ttf_close"])

    with seed.seeded_cache(inputs, meta, archive_lag_days=5, overlay=prod):
        out = sc.cached_fetch(sc.SOURCES["yahoo_ttf"], new_today - timedelta(days=1825),
                              new_today, dead)
    assert out["date"].max() == new_today - timedelta(days=2)    # within the 5-day limit
    assert out["date"].is_unique and out["date"].min() <= TODAY - timedelta(days=1700)
    overlap = out.set_index("date")["ttf_close"]
    assert overlap[TODAY - timedelta(days=10)] == (TODAY - timedelta(days=10)).toordinal() + 0.5


def test_an_overlay_with_a_hole_or_another_version_is_ignored(base_snapshot, tmp_path):
    inputs, meta = base_snapshot
    hole = _production_cache(tmp_path / "a", TODAY + timedelta(days=20), TODAY + timedelta(days=30))
    other = _production_cache(tmp_path / "b", TODAY - timedelta(days=700), TODAY + timedelta(days=5),
                              version=999)
    for overlay in (hole, other):
        with seed.seeded_cache(inputs, meta, archive_lag_days=5, overlay=overlay) as (_, tmp):
            record = sc._load(sc.SOURCES["yahoo_ttf"], tmp)
        assert record["frame"]["date"].max() == TODAY - timedelta(days=1)   # the snapshot's own
