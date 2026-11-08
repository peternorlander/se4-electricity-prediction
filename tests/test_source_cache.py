"""
The production source cache: ask a source only for what is new, never mix versions, and say
so when it is serving old data (source_cache.py, docs/DATA_CACHE.md; born of Yahoo answering
a 3-year request with 1-3 rows on 2026-11-06..08).
"""
from datetime import date, timedelta

import pandas as pd
import pytest

import source_cache as sc

DAILY = sc.Source("t_daily", version=1, key="date", overlap_days=10, max_stale_days=4,
                  max_join_gap_days=5)
STRICT = sc.Source("t_strict", version=1, key="date", overlap_days=10)   # no stale allowance
HOURLY = sc.Source("t_hourly", version=1, key="timestamp", overlap_days=2, max_join_gap_days=2)

START = date(2026, 1, 1)


class Fake:
    """A source whose value on day d is d.toordinal() (+ `shift`), and a call log."""

    def __init__(self, key="date", last_available=None):
        self.key, self.calls, self.shift, self.fail = key, [], 0.0, None
        self.last_available = last_available       # newest day the source can deliver

    def __call__(self, start, end):
        self.calls.append((start, end))
        if self.fail:
            raise self.fail
        top = end if self.last_available is None else min(end, self.last_available)
        days = [start + timedelta(days=i) for i in range((top - start).days + 1)]
        if self.key == "timestamp":
            return pd.DataFrame({
                "timestamp": [pd.Timestamp(d, tz="UTC") for d in days],
                "value": [d.toordinal() + self.shift for d in days]})
        return pd.DataFrame({"date": days, "value": [d.toordinal() + self.shift for d in days]})


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(sc.CACHE_DIR_ENV, str(tmp_path))
    return tmp_path


def test_off_without_a_directory_is_a_plain_call(monkeypatch):
    monkeypatch.delenv(sc.CACHE_DIR_ENV, raising=False)
    fake = Fake()
    out = sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    assert fake.calls == [(START, date(2026, 3, 1))] and len(out) == 60


def test_second_run_fetches_only_the_tail_and_equals_a_full_fetch(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.calls.clear()

    out = sc.cached_fetch(DAILY, START, date(2026, 3, 5), fake)

    assert fake.calls == [(date(2026, 3, 1) - timedelta(days=10), date(2026, 3, 5))]
    full = Fake()(START, date(2026, 3, 5))
    pd.testing.assert_frame_equal(out, full)


def test_fetched_rows_win_over_cached_ones_in_the_overlap(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.shift = 0.5                                  # the source revised its recent rows
    out = sc.cached_fetch(DAILY, START, date(2026, 3, 5), fake).set_index("date")["value"]

    fetch_from = date(2026, 3, 1) - timedelta(days=10)
    assert out[fetch_from] == fetch_from.toordinal() + 0.5       # revised row replaced
    assert out[START] == START.toordinal()                       # older rows untouched


def test_window_start_moves_with_the_cache_trimmed(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    out = sc.cached_fetch(DAILY, date(2026, 1, 11), date(2026, 3, 2), fake)
    assert out["date"].min() == date(2026, 1, 11) and out["date"].max() == date(2026, 3, 2)


def test_timestamp_keys_work_the_same_way(cache):
    fake = Fake(key="timestamp")
    sc.cached_fetch(HOURLY, START, date(2026, 2, 1), fake)
    out = sc.cached_fetch(HOURLY, START, date(2026, 2, 3), fake)
    pd.testing.assert_frame_equal(out, Fake(key="timestamp")(START, date(2026, 2, 3)))
    assert fake.calls[-1][0] == date(2026, 2, 1) - timedelta(days=2)


def test_a_version_bump_forces_a_full_fetch(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.calls.clear()
    bumped = sc.Source("t_daily", version=2, key="date", overlap_days=10)
    sc.cached_fetch(bumped, START, date(2026, 3, 5), fake)
    assert fake.calls == [(START, date(2026, 3, 5))]


def test_a_window_reaching_further_back_than_the_cache_is_fetched_in_full(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, date(2026, 2, 1), date(2026, 3, 1), fake)
    fake.calls.clear()
    sc.cached_fetch(DAILY, START, date(2026, 3, 2), fake)
    assert fake.calls == [(START, date(2026, 3, 2))]


def test_unreadable_cache_is_a_miss_not_a_crash(cache):
    (cache / "t_daily.pkl").write_bytes(b"not a pickle")
    fake = Fake()
    out = sc.cached_fetch(DAILY, START, date(2026, 2, 1), fake)
    assert len(out) == 32 and fake.calls == [(START, date(2026, 2, 1))]


def test_changed_columns_without_a_bump_rebuild_instead_of_mixing(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)

    def renamed(start, end):
        return Fake()(start, end).rename(columns={"value": "other"})
    out = sc.cached_fetch(DAILY, START, date(2026, 3, 5), renamed)
    assert list(out.columns) == ["date", "other"] and len(out) == 64


def test_empty_answer_is_not_cached(cache):
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), lambda s, e: pd.DataFrame(columns=["date", "value"]))
    assert not (cache / "t_daily.pkl").exists()


# --- when the tail fetch fails --------------------------------------------------------------

def test_failed_tail_serves_a_cache_within_the_stale_limit_and_says_so(cache, caplog):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.fail = ConnectionError("yahoo is down")
    with caplog.at_level("WARNING"):
        out = sc.cached_fetch(DAILY, START, date(2026, 3, 4), fake)     # 3 days behind <= 4
    assert out["date"].max() == date(2026, 3, 1)
    assert "serving the cache" in caplog.text and "yahoo is down" in caplog.text


def test_failed_tail_beyond_the_stale_limit_raises(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.fail = ConnectionError("yahoo is down")
    with pytest.raises(RuntimeError, match="yahoo is down"):
        sc.cached_fetch(DAILY, START, date(2026, 3, 9), fake)          # 8 days behind > 4


def test_source_without_a_stale_allowance_never_serves_old_data(cache):
    fake = Fake()
    sc.cached_fetch(STRICT, START, date(2026, 3, 1), fake)
    fake.fail = ConnectionError("entsoe down")
    with pytest.raises(RuntimeError, match="entsoe down"):
        sc.cached_fetch(STRICT, START, date(2026, 3, 2), fake)


def test_thin_answer_within_the_limit_is_merged(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.last_available = date(2026, 3, 2)                   # one new row, nothing after
    out = sc.cached_fetch(DAILY, START, date(2026, 3, 4), fake)
    assert out["date"].max() == date(2026, 3, 2)


def test_thin_answer_far_from_the_end_is_not_trusted(cache):
    """Yahoo 2026-11: it answers, but with rows that stop far short of `end`."""
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    fake.last_available = date(2026, 3, 2)
    with pytest.raises(RuntimeError, match="days before"):
        sc.cached_fetch(DAILY, START, date(2026, 3, 12), fake)


def test_tail_that_leaves_a_hole_is_refused(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)

    def late(start, end):                                    # starts 20 days after the cache ends
        return Fake()(date(2026, 3, 21), end)
    with pytest.raises(RuntimeError, match="missing"):
        sc.cached_fetch(DAILY, START, date(2026, 3, 25), late)


def test_empty_tail_is_a_failure(cache):
    fake = Fake()
    sc.cached_fetch(DAILY, START, date(2026, 3, 1), fake)
    out = sc.cached_fetch(DAILY, START, date(2026, 3, 3),
                          lambda s, e: pd.DataFrame(columns=["date", "value"]))
    assert out["date"].max() == date(2026, 3, 1)             # served from the cache


# --- whole-frame snapshot -------------------------------------------------------------------

SNAP = sc.Source("t_snap", version=1, key=None, max_stale_days=400)


def test_snapshot_falls_back_to_the_last_good_copy(cache):
    good = pd.DataFrame({"week": [1, 2], "x": [0.1, 0.2]})
    assert sc.cached_snapshot(SNAP, lambda: good).equals(good)
    empty = pd.DataFrame(columns=["week", "x"])
    assert sc.cached_snapshot(SNAP, lambda: empty).equals(good)

    def boom():
        raise ConnectionError("nve down")
    assert sc.cached_snapshot(SNAP, boom).equals(good)


def test_snapshot_without_a_copy_behaves_as_before(cache):
    empty = pd.DataFrame(columns=["week", "x"])
    assert sc.cached_snapshot(SNAP, lambda: empty).empty
    with pytest.raises(RuntimeError):
        sc.cached_snapshot(SNAP, lambda: (_ for _ in ()).throw(ConnectionError("x")))
