import logging
import pandas as pd

from http_client import get_with_retry

logger = logging.getLogger(__name__)

NVE_API_BASE = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk"


def fetch_reservoir_norway(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weekly Norwegian hydropower reservoir data from NVE.

    NVE publishes reservoir fill levels every Wednesday for the preceding
    Saturday. Data is available from 1995 to present, no authentication needed.

    Args:
        start_date: Start date string in "YYYY-MM-DD" format.
        end_date:   End date string in "YYYY-MM-DD" format.

    Returns:
        DataFrame with columns:
            date        - date object (Saturday of the measurement week)
            fill_pct    - fill percentage (0–1 scale)
            twh_stored  - energy stored in TWh
            capacity_twh - total reservoir capacity in TWh
            weekly_change - week-over-week change in fill percentage
            week_number - ISO week number
    """
    _EMPTY = pd.DataFrame(columns=[
        "date", "fill_pct", "twh_stored", "capacity_twh", "weekly_change", "week_number"
    ])

    logger.info("Fetching NVE reservoir data %s → %s", start_date, end_date)

    try:
        response = get_with_retry(f"{NVE_API_BASE}/HentOffentligData", params={})
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning("NVE reservoir fetch failed: %s", e)
        return _EMPTY

    if not data:
        logger.warning("NVE returned empty response")
        return _EMPTY

    # Filter for Norway total (omrType "NO") — not individual Elspot zones
    records = [r for r in data if r.get("omrType") == "NO"]

    if not records:
        logger.warning("No Norway-total records found in NVE response")
        return _EMPTY

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["dato_Id"]).dt.date
    df["fill_pct"] = df["fyllingsgrad"].astype(float)
    df["twh_stored"] = df["fylling_TWh"].astype(float)
    df["capacity_twh"] = df["kapasitet_TWh"].astype(float)
    df["weekly_change"] = df["endring_fyllingsgrad"].astype(float)
    df["week_number"] = df["iso_uke"].astype(int)

    # Filter to requested date range
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    df = df[["date", "fill_pct", "twh_stored", "capacity_twh", "weekly_change", "week_number"]]
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("  → %d weekly reservoir records", len(df))
    return df


def fetch_reservoir_norway_median() -> pd.DataFrame:
    """
    Fetch 20-year min/max/median reservoir fill by ISO week number from NVE.

    This provides the seasonal norm baseline: comparing current fill against
    the 20-year median reveals whether reservoirs are unusually low or high
    for the time of year — a stronger price signal than the raw level.

    Returns:
        DataFrame with columns:
            week_number    - ISO week number (1–53)
            median_fill_pct - 20-year median fill percentage (0–1 scale)
            min_fill_pct   - 20-year minimum fill percentage
            max_fill_pct   - 20-year maximum fill percentage
    """
    _EMPTY = pd.DataFrame(columns=["week_number", "median_fill_pct", "min_fill_pct", "max_fill_pct"])

    logger.info("Fetching NVE 20-year reservoir median")

    try:
        response = get_with_retry(f"{NVE_API_BASE}/HentOffentligDataMinMaxMedian", params={})
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning("NVE reservoir median fetch failed: %s", e)
        return _EMPTY

    if not data:
        logger.warning("NVE median endpoint returned empty response")
        return _EMPTY

    records = [r for r in data if r.get("omrType") == "NO"]

    if not records:
        logger.warning("No Norway-total records found in NVE median response")
        return _EMPTY

    df = pd.DataFrame(records)
    df["week_number"] = df["iso_uke"].astype(int)
    df["median_fill_pct"] = df["medianFyllingsGrad"].astype(float)
    df["min_fill_pct"] = df["minFyllingsgrad"].astype(float)
    df["max_fill_pct"] = df["maxFyllingsgrad"].astype(float)

    df = df[["week_number", "median_fill_pct", "min_fill_pct", "max_fill_pct"]]
    df = df.drop_duplicates("week_number").sort_values("week_number").reset_index(drop=True)

    logger.info("  → %d week entries", len(df))
    return df
