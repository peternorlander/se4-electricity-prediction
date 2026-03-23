import io
import os
import logging
import zipfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from http_client import get_with_retry

logger = logging.getLogger(__name__)


SE4_AREA_CODE   = "10Y1001A1001A47J"
SE3_AREA_CODE   = "10Y1001A1001A46L"  # Sweden SE3 — source zone for nuclear generation affecting SE4
DE_LU_AREA_CODE = "10Y1001A1001A82H"  # Germany/Luxembourg — price leader for northern Europe
DK2_AREA_CODE   = "10YDK-2--------M"  # Denmark DK2 — directly coupled to SE4
ENTSO_E_API_URL = "https://web-api.tp.entsoe.eu/api"


def _get_token() -> str:
    return os.environ["ENTSO_E_TOKEN"]


def _find_all(root: ET.Element, local_name: str) -> list:
    """Find all elements by local tag name, ignoring XML namespace."""
    return [el for el in root.iter() if el.tag.split("}")[-1] == local_name]


def _find_first(element: ET.Element, local_name: str):
    """Find first matching element by local tag name, ignoring XML namespace."""
    return next(
        (el for el in element.iter() if el.tag.split("}")[-1] == local_name),
        None
    )


def _extract_xml_files(content: bytes) -> list[bytes]:
    """
    Extract all XML files from a ZIP archive, or return the content as a single-item list.

    ENTSO-E unavailability endpoints return a ZIP where each XML file is one
    outage event — reading only the first file would silently discard the rest.
    Price endpoints return raw XML directly (not zipped).
    ZIP archives start with the magic bytes PK (0x50 0x4B).
    """
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            return [zf.read(name) for name in zf.namelist() if name.endswith(".xml")]
    return [content]


def _get_offset_for_position(position: int, resolution: str) -> timedelta:
    if resolution == "PT15M":
        return timedelta(minutes=(position - 1) * 15)
    return timedelta(hours=position - 1)


def _parse_point(point: ET.Element, start_dt: datetime, resolution: str) -> dict | None:
    position_el = _find_first(point, "position")
    price_el = _find_first(point, "price.amount")

    if position_el is None or price_el is None:
        return None

    position = int(position_el.text)
    price = float(price_el.text)
    offset = _get_offset_for_position(position, resolution)

    return {
        "timestamp": start_dt + offset,
        "price_eur_mwh": price
    }


def _parse_period(period: ET.Element) -> list:
    start_el = _find_first(period, "start")
    resolution_el = _find_first(period, "resolution")

    if start_el is None or resolution_el is None:
        return []

    start_dt = datetime.fromisoformat(start_el.text.replace("Z", "+00:00"))
    resolution = resolution_el.text
    records = []

    for point in _find_all(period, "Point"):
        record = _parse_point(point, start_dt, resolution)

        if record is not None:
            records.append(record)

    return records



# ENTSO-E API maximum allowed date range per request
_MAX_RANGE_DAYS = 365


def _fetch_prices_area_chunk(area_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a single chunk (max 365 days) of day-ahead prices from ENTSO-E."""
    params = {
        "documentType": "A44",
        "in_Domain": area_code,
        "out_Domain": area_code,
        "periodStart": f"{start_date}0000",
        "periodEnd": f"{end_date}0000",
        "securityToken": _get_token()
    }

    response = get_with_retry(ENTSO_E_API_URL, params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    records = []

    for period in _find_all(root, "Period"):
        records.extend(_parse_period(period))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def _fetch_prices_area(area_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices from ENTSO-E for a given bidding zone.

    Automatically splits requests longer than 365 days into yearly chunks,
    since the ENTSO-E API enforces a maximum range per request.

    Args:
        area_code:  ENTSO-E bidding zone EIC code.
        start_date: Start date in YYYYMMDD format.
        end_date:   End date in YYYYMMDD format (exclusive).

    Returns:
        DataFrame with columns: timestamp (UTC), price_eur_mwh.
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    chunks = []

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_RANGE_DAYS), end)
        chunks.append(_fetch_prices_area_chunk(
            area_code,
            chunk_start.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        ))
        chunk_start = chunk_end

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    return df


def fetch_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch day-ahead electricity prices from ENTSO-E for SE4."""
    return _fetch_prices_area(SE4_AREA_CODE, start_date, end_date)


def _parse_outage_timeseries(ts: ET.Element) -> dict | None:
    """
    Parse a single TimeSeries element from an A77 unavailability document.

    ENTSO-E A77 documents use start_DateAndOrTime.date / end_DateAndOrTime.date
    for the outage window — NOT the <start>/<end> tags used in price documents.
    """
    mrid_el  = _find_first(ts, "mRID")
    start_el = _find_first(ts, "start_DateAndOrTime.date")
    end_el   = _find_first(ts, "end_DateAndOrTime.date")

    if mrid_el is None or start_el is None or end_el is None:
        return None

    try:
        start_date = datetime.strptime(start_el.text.strip(), "%Y-%m-%d").date()
        end_date   = datetime.strptime(end_el.text.strip(),   "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

    return {
        "mrid":       mrid_el.text,
        "start_date": start_date,
        "end_date":   end_date,
    }


def _fetch_outages_chunk(
    area_code: str, start_date: str, end_date: str, business_type: str
) -> pd.DataFrame:
    """
    Fetch one chunk of nuclear generation unavailability events from ENTSO-E (A77).

    Args:
        area_code:     ENTSO-E bidding zone EIC code.
        start_date:    Start date in YYYYMMDD format.
        end_date:      End date in YYYYMMDD format (exclusive).
        business_type: A53 for planned maintenance, A54 for forced/unplanned outages.

    Returns:
        DataFrame with columns: mrid, start_date, end_date.
    """
    _EMPTY = pd.DataFrame(columns=["mrid", "start_date", "end_date"])

    params = {
        "documentType":       "A77",
        "businessType":       business_type,
        "biddingZone_Domain": area_code,
        "psrType":            "B14",  # Nuclear
        "periodStart":        f"{start_date}0000",
        "periodEnd":          f"{end_date}0000",
        "securityToken":      _get_token(),
    }

    response = get_with_retry(ENTSO_E_API_URL, params)
    response.raise_for_status()

    # Each XML file in the ZIP is one outage event — iterate all of them.
    try:
        xml_files = _extract_xml_files(response.content)
    except Exception as e:
        logger.warning(
            "ENTSO-E outages response could not be extracted (businessType=%s): %s — raw: %s",
            business_type, e, response.content[:200],
        )
        return _EMPTY

    records = []
    for xml_content in xml_files:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.warning("ENTSO-E outages XML parse error (businessType=%s): %s", business_type, e)
            continue

        # Error responses use Acknowledgement_MarketDocument as root element
        root_tag = root.tag.split("}")[-1]
        if root_tag == "Acknowledgement_MarketDocument":
            reason_el = _find_first(root, "text")
            reason_text = reason_el.text if reason_el is not None else "unknown"
            logger.info("ENTSO-E returned no data (businessType=%s %s → %s): %s", business_type, start_date, end_date, reason_text)
            return _EMPTY

        for ts in _find_all(root, "TimeSeries"):
            r = _parse_outage_timeseries(ts)
            if r is not None:
                records.append(r)

    return pd.DataFrame(records) if records else _EMPTY


def fetch_nuclear_outages_se3(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch nuclear generation outages in SE3 from ENTSO-E (planned + forced).

    Planned maintenance (A53) is published months in advance on ENTSO-E,
    making this feature usable for multi-day forecasting. Forced outages (A54)
    are only available for past dates but improve training accuracy.

    Args:
        start_date: Start date in YYYYMMDD format.
        end_date:   End date in YYYYMMDD format (exclusive).

    Returns:
        DataFrame with columns: date (date), nuclear_outage_se3 (int).
        nuclear_outage_se3 is the count of simultaneous active outage events
        per day. Missing dates are filled with 0 (no known outage).
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end   = datetime.strptime(end_date,   "%Y%m%d")

    chunks = []
    for business_type in ["A53", "A54"]:  # planned + forced
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=_MAX_RANGE_DAYS), end)
            chunks.append(_fetch_outages_chunk(
                area_code=SE3_AREA_CODE,
                start_date=chunk_start.strftime("%Y%m%d"),
                end_date=chunk_end.strftime("%Y%m%d"),
                business_type=business_type,
            ))
            chunk_start = chunk_end

    events = pd.concat(chunks, ignore_index=True)
    if not events.empty:
        # mRID is always "1" within each individual document — deduplicate by
        # outage window instead so long events spanning chunk boundaries are not
        # double-counted while distinct events with the same mRID are preserved.
        events = events.drop_duplicates(["start_date", "end_date"])

    logger.info("SE3 nuclear outages: %d unique events (%s → %s)", len(events), start_date, end_date)

    # Expand each outage event to individual dates within its active range
    daily_dates = []
    for _, row in events.iterrows():
        day = row["start_date"]
        while day <= row["end_date"]:
            daily_dates.append(day)
            day += timedelta(days=1)

    # Build full date range and count simultaneous outages per day
    full_range = pd.date_range(start, end - timedelta(days=1), freq="D")
    result = pd.DataFrame({"date": [d.date() for d in full_range]})

    if daily_dates:
        counts = (
            pd.Series(daily_dates)
            .value_counts()
            .reset_index()
            .rename(columns={"index": "date", 0: "nuclear_outage_se3", "count": "nuclear_outage_se3"})
        )
        # value_counts().reset_index() column naming differs by pandas version — normalise
        counts.columns = ["date", "nuclear_outage_se3"]
        result = result.merge(counts, on="date", how="left")
    else:
        result["nuclear_outage_se3"] = 0

    result["nuclear_outage_se3"] = result["nuclear_outage_se3"].fillna(0).astype(int)
    return result


def fetch_market_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices for Germany (DE/LU) and Denmark (DK2).

    These are used as lag-1 features in the model: since the implicit auction
    sets all prices simultaneously, only the *previous* day's prices are valid
    as predictors for SE4.

    Args:
        start_date: Start date in YYYYMMDD format.
        end_date:   End date in YYYYMMDD format (exclusive).

    Returns:
        DataFrame with columns: timestamp (UTC), price_de, price_dk2.
    """
    de = _fetch_prices_area(DE_LU_AREA_CODE, start_date, end_date).rename(
        columns={"price_eur_mwh": "price_de"}
    )
    dk2 = _fetch_prices_area(DK2_AREA_CODE, start_date, end_date).rename(
        columns={"price_eur_mwh": "price_dk2"}
    )
    return pd.merge(de, dk2, on="timestamp", how="inner")
