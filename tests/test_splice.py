"""fetch_data._splice_recent's two rules (docs/DECISIONS.md "Closing the weather-archive lag")."""
from datetime import date

import pandas as pd

from se4.fetch_data import _splice_recent, _drop_partial_last_day


def _frame(start_utc: str, end_utc: str, value: float) -> pd.DataFrame:
    stamps = pd.date_range(start_utc, end_utc, freq="1h", inclusive="left", tz="UTC")
    return pd.DataFrame({"timestamp": stamps, "temperature": value, "windspeed": value, "radiation": value})


TODAY = date(2026, 1, 10)


def test_archive_wins_where_it_has_published():
    archive = _frame("2026-01-01", "2026-01-05 01:00", 1.0)
    recent = _frame("2026-01-03", "2026-01-10 12:00", 2.0)
    out = _splice_recent(archive, recent, TODAY)
    overlap = out[out.timestamp <= archive.timestamp.max()]
    assert (overlap.temperature == 1.0).all() and len(overlap) == len(archive)
    assert (out[out.timestamp > archive.timestamp.max()].temperature == 2.0).all()
    assert out.timestamp.is_unique and out.timestamp.is_monotonic_increasing


def test_cut_at_swedish_midnight_of_today():
    archive = _frame("2026-01-01", "2026-01-05 01:00", 1.0)
    recent = _frame("2026-01-03", "2026-01-10 12:00", 2.0)
    out = _splice_recent(archive, recent, TODAY)
    cutoff = pd.Timestamp(TODAY, tz="Europe/Stockholm").tz_convert("UTC")
    assert out.timestamp.max() == cutoff - pd.Timedelta(hours=1)    # 2026-01-09 23:00 UTC under CET


def test_cut_follows_summer_time():
    archive = _frame("2026-07-01", "2026-07-05 02:00", 1.0)
    recent = _frame("2026-07-03", "2026-07-10 12:00", 2.0)
    out = _splice_recent(archive, recent, date(2026, 7, 10))
    assert out.timestamp.max() == pd.Timestamp("2026-07-09 21:00", tz="UTC")   # 23:00 CEST


def test_without_recent_the_archive_stub_is_dropped():
    archive = _frame("2026-01-01", "2026-01-05 01:00", 1.0)   # spills one hour into 2026-01-05 (Swedish)
    for recent in (None, archive.iloc[:0]):
        out = _splice_recent(archive, recent, TODAY)
        swedish = out.timestamp.dt.tz_convert("Europe/Stockholm").dt.date
        assert swedish.max() == date(2026, 1, 4)
    assert len(_drop_partial_last_day(archive)) == len(archive) - 2    # the two hours of 2026-01-05


def test_nothing_new_in_recent_also_drops_the_stub():
    archive = _frame("2026-01-01", "2026-01-05 01:00", 1.0)
    recent = _frame("2026-01-02", "2026-01-04", 2.0)
    out = _splice_recent(archive, recent, TODAY)
    assert (out.temperature == 1.0).all()
    assert out.timestamp.dt.tz_convert("Europe/Stockholm").dt.date.max() == date(2026, 1, 4)
