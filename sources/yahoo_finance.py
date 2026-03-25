import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TTF_TICKER = "TTF=F"


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

    raw = yf.download(
        TTF_TICKER,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        logger.warning(f"No TTF data returned for {start_date} → {end_date}")
        return pd.DataFrame(columns=["date", "ttf_close"])

    df = raw[["Close"]].copy()
    df.columns = ["ttf_close"]
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ttf_close"] = df["ttf_close"].astype(float)

    logger.info(f"  → {len(df)} TTF price records")
    return df
