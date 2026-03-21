import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


SE4_AREA_CODE   = "10Y1001A1001A47J"
DE_LU_AREA_CODE = "10Y1001A1001A82H"  # Germany/Luxembourg — prisledare för norra Europa
DK2_AREA_CODE   = "10YDK-2--------M"  # Danmark DK2 — direkt kopplat till SE4
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


def _fetch_prices_area(area_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch day-ahead prices from ENTSO-E for a given bidding zone.

    Args:
        area_code:  ENTSO-E bidding zone EIC code.
        start_date: Start date in YYYYMMDD format.
        end_date:   End date in YYYYMMDD format (exclusive).

    Returns:
        DataFrame with columns: timestamp (UTC), price_eur_mwh.
    """
    params = {
        "documentType": "A44",
        "in_Domain": area_code,
        "out_Domain": area_code,
        "periodStart": f"{start_date}0000",
        "periodEnd": f"{end_date}0000",
        "securityToken": _get_token()
    }

    response = requests.get(ENTSO_E_API_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    records = []

    for period in _find_all(root, "Period"):
        records.extend(_parse_period(period))

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    return df


def fetch_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch day-ahead electricity prices from ENTSO-E for SE4."""
    return _fetch_prices_area(SE4_AREA_CODE, start_date, end_date)


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
