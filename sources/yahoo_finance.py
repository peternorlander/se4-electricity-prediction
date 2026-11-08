import logging
import time
import warnings

import pandas as pd
import yfinance as yf
from yfinance import shared as yf_shared

logger = logging.getLogger(__name__)

# yfinance 1.7.0 calls pandas methods with `inplace=`, which pandas 3.1 deprecates: four
# warnings per download, ~20 lines per call in the Actions log, burying the one line that
# says why a fetch failed. They are yfinance's to fix; silence them there and nowhere else.
warnings.filterwarnings("ignore", message=r"The inplace keyword", module=r"yfinance(\..*)?")

TTF_TICKER = "TTF=F"

# Yahoo sometimes answers a long daily request with a single row (or none)
# instead of an error -- seen from a GitHub runner on 2026-11-01: 1 TTF record
# for a 3-year window. Downstream, add_ttf_features forward-fills only, so one
# row leaves every earlier day NaN and the final dropna() empties the whole
# training frame. A thin response is therefore treated as a failed fetch.
DOWNLOAD_ATTEMPTS = 4
RETRY_WAIT_SECONDS = (5, 15, 45)
MIN_ROWS_PER_CALENDAR_DAY = 0.4   # real coverage is ~0.69 (weekdays minus holidays)
MAX_EDGE_GAP_DAYS = 10            # first/last row may sit this far inside the window


def _coverage_problem(df: pd.DataFrame, start_date: str, end_date: str):
    """Return why a daily series does not cover [start_date, end_date), or None."""
    if df.empty:
        return "no rows returned"
    span_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    if span_days <= 60:
        return None
    first, last = pd.Timestamp(df["date"].min()), pd.Timestamp(df["date"].max())
    if (first - pd.Timestamp(start_date)).days > MAX_EDGE_GAP_DAYS:
        return f"first row is {first.date()}, window starts {start_date}"
    if (pd.Timestamp(end_date) - last).days > MAX_EDGE_GAP_DAYS:
        return f"last row is {last.date()}, window ends {end_date}"
    if len(df) < MIN_ROWS_PER_CALENDAR_DAY * span_days:
        return f"only {len(df)} rows for a {span_days}-day window"
    return None


def _to_frame(raw: pd.DataFrame, column: str) -> pd.DataFrame:
    """yfinance's answer as a (date, <column>) frame of daily closes."""
    df = raw[["Close"]].copy()
    df.columns = [column]
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df[column] = df[column].astype(float)
    return df


def _describe_answer(raw: pd.DataFrame, ticker: str) -> str:
    """
    What Yahoo actually answered, for the log line of a failed attempt.

    yfinance does not raise on a rate limit or a truncated reply -- it returns what it got and
    keeps the reason in `yfinance.shared._ERRORS`. Without this the log says only that rows
    are missing, never what Yahoo said, which is what 2026-11-06..08 needed.
    """
    parts = [f"{len(raw)} row(s)"]
    if len(raw):
        parts.append(f"{pd.Timestamp(raw.index.min()).date()} -> {pd.Timestamp(raw.index.max()).date()}")
    reason = getattr(yf_shared, "_ERRORS", {}) or {}
    if reason.get(ticker):
        parts.append(f"yfinance reported: {str(reason[ticker])[:300]}")
    return ", ".join(parts)


def download_daily_close(ticker: str, start_date: str, end_date: str,
                         column: str, label: str) -> pd.DataFrame:
    """
    Download daily closes from Yahoo as a (date, <column>) frame, retrying when
    the response is empty or does not cover the requested window.

    Raises RuntimeError if every attempt comes back thin -- training on a
    partly-covered price series is worse than failing the run (the previous
    prediction then stays in place).
    """
    problem = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            raw = yf.download(
                ticker, start=start_date, end=end_date, interval="1d",
                auto_adjust=True, progress=False,
            )
        except Exception as exc:  # yfinance raises a zoo of exception types
            raw, problem = pd.DataFrame(), f"download raised {type(exc).__name__}: {exc}"
        else:
            problem = None

        if not raw.empty:
            df = _to_frame(raw, column)
            problem = _coverage_problem(df, start_date, end_date)
            if problem is None:
                return df
            problem = f"{problem} (answer: {_describe_answer(raw, ticker)})"
        elif problem is None:
            problem = f"no rows returned ({_describe_answer(raw, ticker)})"

        logger.warning(f"{label} ({ticker}) fetch attempt {attempt}/{DOWNLOAD_ATTEMPTS} "
                       f"unusable: {problem}")
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(RETRY_WAIT_SECONDS[attempt - 1])

    raise RuntimeError(
        f"{label} ({ticker}) from Yahoo Finance never covered {start_date} -> {end_date} "
        f"after {DOWNLOAD_ATTEMPTS} attempts (last problem: {problem}). Refusing to train "
        f"on a partly-covered series."
    )


def fetch_ttf_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Dutch TTF Natural Gas futures daily close prices from Yahoo Finance.

    TTF (Title Transfer Facility) is the European benchmark for natural gas prices.
    Gas-fired power plants set the marginal cost of electricity in Germany, which
    propagates to SE4 via imports. TTF futures react faster to geopolitical events
    (e.g. supply disruptions) than electricity prices do, making them a leading
    indicator for SE4 price level shifts.

    Args:
        start_date: Start date string in "YYYY-MM-DD" format (inclusive).
        end_date:   End date string in "YYYY-MM-DD" format (inclusive).

    Returns:
        DataFrame with columns:
            date      - date object (calendar day)
            ttf_close - settlement/close price in EUR/MWh equivalent
    """
    logger.info(f"Fetching TTF gas prices {start_date} → {end_date}")
    df = download_daily_close(TTF_TICKER, start_date, end_date, "ttf_close", "TTF gas")
    logger.info(f"  → {len(df)} TTF price records")
    return df
