"""A thin Yahoo answer is a failed fetch, not a series (2026-11-01: 1 TTF row emptied the training frame)."""
import pandas as pd
import pytest

import sources.yahoo_finance as yf_src

START, END = "2023-11-02", "2026-11-01"


def _frame(dates):
    return pd.DataFrame({"Close": [50.0] * len(dates)},
                        index=pd.DatetimeIndex(dates, name="Date"))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(yf_src.time, "sleep", lambda s: None)


def _patch(monkeypatch, answers):
    calls = iter(answers)
    monkeypatch.setattr(yf_src.yf, "download", lambda *a, **k: next(calls))


def test_thin_answer_is_retried_then_accepted(monkeypatch):
    full = _frame(pd.bdate_range("2023-11-02", "2026-10-30"))
    _patch(monkeypatch, [_frame(["2026-10-30"]), full])
    out = yf_src.download_daily_close("TTF=F", START, END, "ttf_close", "TTF gas")
    assert len(out) == len(full)


def test_never_covering_the_window_raises(monkeypatch):
    _patch(monkeypatch, [_frame(["2026-10-30"])] * yf_src.DOWNLOAD_ATTEMPTS)
    with pytest.raises(RuntimeError, match="never covered"):
        yf_src.download_daily_close("TTF=F", START, END, "ttf_close", "TTF gas")


def test_exception_and_empty_are_retried(monkeypatch):
    full = _frame(pd.bdate_range("2023-11-02", "2026-10-30"))

    def flaky(*a, **k):
        flaky.n += 1
        if flaky.n == 1:
            raise ConnectionError("boom")
        return pd.DataFrame() if flaky.n == 2 else full
    flaky.n = 0
    monkeypatch.setattr(yf_src.yf, "download", flaky)
    assert len(yf_src.download_daily_close("TTF=F", START, END, "ttf_close", "TTF gas")) == len(full)


def test_failed_attempt_log_says_what_yahoo_answered(monkeypatch, caplog):
    """2026-11-06..08: the log named the missing rows but not what Yahoo had sent."""
    monkeypatch.setitem(yf_src.yf_shared._ERRORS, "TTF=F", "YFRateLimitError('Too Many Requests')")
    _patch(monkeypatch, [_frame(["2026-11-06"])] * yf_src.DOWNLOAD_ATTEMPTS)
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
        yf_src.download_daily_close("TTF=F", START, END, "ttf_close", "TTF gas")
    assert "1 row(s)" in caplog.text and "2026-11-06" in caplog.text
    assert "Too Many Requests" in caplog.text
