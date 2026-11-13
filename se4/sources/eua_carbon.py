import logging

import pandas as pd

from se4.sources.yahoo_finance import download_daily_close

logger = logging.getLogger(__name__)

# SparkChange Physical Carbon EUA ETC — LSE-listed ETC that is physically
# backed by EU Allowances, tracking the spot EUA price ~1:1 (minus fees).
# Listed on London Stock Exchange but denominated in EUR, with multi-year
# daily history on Yahoo Finance.
#
# Alternatives considered:
#   - ^ICEEUA (ICE EUA index): unavailable via yfinance (returns empty).
#   - ECF=F (NYMEX EUA futures): not reliably hosted on Yahoo Finance.
#   - KEUA / KRBN (US ETFs): USD-denominated, fee drag, or global (not EU-only).
EUA_TICKER = "CO2.L"


def fetch_eua_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch EU ETS carbon allowance (EUA) daily close prices from Yahoo Finance.

    The EU Emissions Trading System carbon price is a direct cost driver for
    fossil-fuel power generation (coal, gas). Since gas- and coal-fired plants
    typically set the marginal cost of electricity in Germany, CO₂ price
    movements propagate to SE4 via market coupling. CO₂ prices move more
    slowly than TTF gas (monthly-to-quarterly trends), making them a stable
    regime indicator rather than a noisy daily signal.

    Unit: EUR per tonne CO₂ equivalent.

    Args:
        start_date: Start date string in "YYYY-MM-DD" format (inclusive).
        end_date:   End date string in "YYYY-MM-DD" format (inclusive).

    Returns:
        DataFrame with columns:
            date      - date object (calendar day)
            eua_close - settlement/close price in EUR/tonne
    """
    logger.info(f"Fetching EUA carbon prices {start_date} → {end_date}")
    df = download_daily_close(EUA_TICKER, start_date, end_date, "eua_close", "EU ETS carbon")
    logger.info(f"  → {len(df)} EUA price records")
    return df
