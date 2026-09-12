import io
import os
import time
import logging
import zipfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from http_client import get_with_retry

logger = logging.getLogger(__name__)


SE4_AREA_CODE     = "10Y1001A1001A47J"
SE3_AREA_CODE     = "10Y1001A1001A46L"  # Sweden SE3 — source zone for nuclear generation affecting SE4
DE_LU_AREA_CODE   = "10Y1001A1001A82H"  # Germany/Luxembourg — price leader for northern Europe
DK2_AREA_CODE     = "10YDK-2--------M"  # Denmark DK2 — directly coupled to SE4
SWEDEN_AREA_CODE  = "10YSE-1--------K"  # Sweden country-level — for hydro reservoir data
ENTSO_E_API_URL = "https://web-api.tp.entsoe.eu/api"

# ENTSO-E API maximum allowed date range per request (default; see
# DOCUMENT_MAX_RANGE_DAYS for the document types that are stricter)
_MAX_RANGE_DAYS = 365


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


# --------------------------------------------------------------------------
# Generic request helpers
#
# The fetchers below this section each build their own params and call
# get_with_retry directly, which is fine for documents whose limits we know and
# whose failures are fatal anyway. These three exist for the other case: an
# exploratory fetch over years of history, where one refused border must not
# abort the run and where the API's own explanation is the only way to find out
# what it wants. experiments/fetch_entsoe_crossborder.py is the caller today;
# docs/FINDINGS.md "If you touch ab_cache/crossborder/" is the war story.
# --------------------------------------------------------------------------

# Per-document-type maximum request range. ENTSO-E does not publish these
# consistently and DOES change them: A11 took 365-day requests on 2026-08-22 and
# rejected them on 2026-09-12 with "larger than maximum allowed period 'P1M' for
# 'NET_CROSS_BORDER_PHYSICAL_FLOWS_R3:XML'", while A78 was unaffected in the same
# run. 28 rather than 30 because P1M is a calendar month. Measured with
# experiments/probe_a11_range_limit.py; fetch_range() recovers if a limit moves
# again, this table only saves the refusals.
DOCUMENT_MAX_RANGE_DAYS = {
    "A11": 28,     # cross-border physical flows
}


def max_range_days(document_type: str) -> int:
    """Largest range worth asking for, for this document type."""
    return DOCUMENT_MAX_RANGE_DAYS.get(document_type, _MAX_RANGE_DAYS)


def date_chunks(start: datetime, end: datetime, days: int = _MAX_RANGE_DAYS):
    """Yield (from, to) pairs of at most `days`, covering [start, end)."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def reason_text(content: bytes) -> str:
    """The Reason code/text ENTSO-E puts in a 4xx body.

    requests' HTTPError message does not carry it, and it is the only place the
    API says WHY a query was refused -- losing it turned a one-line limit change
    into an undiagnosable 27-minute fetch that returned nothing (2026-09-12).
    """
    try:
        root = ET.fromstring(content)
    except Exception:
        return content[:200].decode("utf-8", "replace")
    parts = [el.text for el in root.iter()
             if el.tag.split("}")[-1] in ("code", "text") and el.text]
    return " | ".join(parts)[:300] if parts else content[:200].decode("utf-8", "replace")


def raise_for_status_with_reason(response, label: str) -> None:
    """raise_for_status(), but log WHY first.

    The production fetchers below raise on 4xx, and a bare HTTPError says only
    "400 Client Error" -- which in a GitHub Actions log is indistinguishable
    between a bad token, a bad EIC and a platform limit change. ENTSO-E puts the
    actual reason in the body, and on 2026-09-12 it was a new per-document
    period cap (docs/FINDINGS.md). Same failure, one line of diagnosis.
    """
    if response.status_code >= 400:
        logger.error("ENTSO-E %s failed: HTTP %d: %s",
                     label, response.status_code, reason_text(response.content))
    response.raise_for_status()


def request_documents(params: dict, label: str = "", *, raw_dir=None,
                      raw_name: str = None, log=None) -> tuple[list, bool]:
    """One ENTSO-E call, returning (parsed XML roots, ok).

    `ok` is False only when the REQUEST failed, so a caller can react (see
    fetch_range, which splits the interval and retries). An empty list with
    ok=True means "no data in this window", which is a normal answer. Never
    raises: one unavailable border must not abort an hour-long fetch.

    log: callable taking one string, for callers that keep their own fetch log
    (default: this module's logger).
    """
    emit = log or logger.info
    try:
        resp = get_with_retry(ENTSO_E_API_URL, {**params, "securityToken": _get_token()})
    except Exception as e:
        emit(f"  ERROR  {label}: {type(e).__name__}: {str(e)[:200]}")
        return [], False

    if resp.status_code >= 400:
        emit(f"  ERROR  {label}: HTTP {resp.status_code}: {reason_text(resp.content)}")
        return [], False

    if raw_dir is not None and raw_name is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        ext = ".zip" if resp.content[:2] == b"PK" else ".xml"
        (raw_dir / (raw_name + ext)).write_bytes(resp.content)

    try:
        xmls = _extract_xml_files(resp.content)
    except Exception as e:
        emit(f"  ERROR  {label}: could not extract: {e}")
        return [], False

    roots = []
    for xml_content in xmls:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            emit(f"  ERROR  {label}: XML parse: {e}")
            continue
        if root.tag.split("}")[-1] == "Acknowledgement_MarketDocument":
            reason = _find_first(root, "text")
            emit(f"  EMPTY  {label}: {reason.text if reason is not None else 'no data'}")
            continue
        roots.append(root)
    return roots, True


def fetch_range(params: dict, start: datetime, end: datetime, parse,
                label: str = "", *, min_days: int = 7, log=None,
                raw_dir=None, raw_name: str = None, sleep_s: float = 0.3) -> list:
    """Fetch [start, end), halving the interval and retrying on refusal.

    `parse` maps one XML root to a list of records. Ask for the whole range and
    let the refusals narrow it: the per-document limits are neither documented
    nor stable (see DOCUMENT_MAX_RANGE_DAYS), so a fetch that adapts survives
    the next change with a slow run instead of an empty one.

    min_days floors the recursion. If a week-long window is refused the cause is
    not the range, and splitting further only multiplies a failing request.
    """
    emit = log or logger.info
    roots, ok = request_documents(
        {**params,
         "periodStart": start.strftime("%Y%m%d%H%M"),
         "periodEnd": end.strftime("%Y%m%d%H%M")},
        f"{label} {start.date()}..{end.date()}",
        raw_dir=raw_dir, raw_name=raw_name, log=log,
    )
    if ok:
        records = []
        for root in roots:
            records += parse(root)
        if records:
            emit(f"  OK     {label} {start.date()}..{end.date()}: {len(records)} points")
        return records

    days = (end - start).days
    if days <= min_days:
        emit(f"  GIVEUP {label} {start.date()}..{end.date()} ({days}d)")
        return []

    mid = start + timedelta(days=days // 2)
    emit(f"  SPLIT  {label} {start.date()}..{end.date()} ({days}d) -> 2 x {days // 2}d")
    time.sleep(sleep_s)
    first = fetch_range(params, start, mid, parse, label, min_days=min_days, log=log,
                        raw_dir=raw_dir, raw_name=raw_name and f"{raw_name}_a",
                        sleep_s=sleep_s)
    time.sleep(sleep_s)
    return first + fetch_range(params, mid, end, parse, label, min_days=min_days,
                               log=log, raw_dir=raw_dir,
                               raw_name=raw_name and f"{raw_name}_b", sleep_s=sleep_s)


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
    raise_for_status_with_reason(
        response, f"day-ahead prices (A44) {area_code} {start_date}..{end_date}")

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
    raise_for_status_with_reason(
        response,
        f"nuclear outages (A77/{business_type}) {area_code} {start_date}..{end_date}")

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


def _parse_reservoir_point(point: ET.Element) -> float | None:
    """Extract stored energy quantity from a reservoir data Point element."""
    qty_el = _find_first(point, "quantity")
    return float(qty_el.text) if qty_el is not None else None


def _fetch_reservoir_chunk(area_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch one chunk (max 365 days) of reservoir filling data from ENTSO-E (A72).

    Returns DataFrame with columns: date (date), stored_mwh (float).
    """
    _EMPTY = pd.DataFrame(columns=["date", "stored_mwh"])

    params = {
        "documentType": "A72",
        "processType": "A16",
        "in_Domain": area_code,
        "periodStart": f"{start_date}0000",
        "periodEnd": f"{end_date}0000",
        "securityToken": _get_token(),
    }

    response = get_with_retry(ENTSO_E_API_URL, params)
    raise_for_status_with_reason(
        response, f"reservoir (A72) {area_code} {start_date}..{end_date}")

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.warning("ENTSO-E reservoir XML parse error: %s", e)
        return _EMPTY

    root_tag = root.tag.split("}")[-1]
    if root_tag == "Acknowledgement_MarketDocument":
        reason_el = _find_first(root, "text")
        reason_text = reason_el.text if reason_el is not None else "unknown"
        logger.info("ENTSO-E returned no reservoir data (%s → %s): %s", start_date, end_date, reason_text)
        return _EMPTY

    records = []
    for ts in _find_all(root, "TimeSeries"):
        for period in _find_all(ts, "Period"):
            start_el = _find_first(period, "start")
            resolution_el = _find_first(period, "resolution")
            if start_el is None:
                continue

            period_start = datetime.fromisoformat(start_el.text.replace("Z", "+00:00"))
            resolution = resolution_el.text if resolution_el is not None else "P7D"

            for point in [el for el in period if el.tag.split("}")[-1] == "Point"]:
                pos_el = _find_first(point, "position")
                qty = _parse_reservoir_point(point)
                if pos_el is None or qty is None:
                    continue

                position = int(pos_el.text)
                if resolution == "P7D":
                    point_date = (period_start + timedelta(weeks=position - 1)).date()
                else:
                    point_date = (period_start + timedelta(days=position - 1)).date()

                records.append({"date": point_date, "stored_mwh": qty})

    return pd.DataFrame(records) if records else _EMPTY


def fetch_reservoir_sweden(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Swedish hydro reservoir stored energy from ENTSO-E (A72 document).

    Data is published weekly. Automatically chunks requests longer than 365 days.

    Args:
        start_date: Start date in YYYYMMDD format.
        end_date:   End date in YYYYMMDD format (exclusive).

    Returns:
        DataFrame with columns:
            date              - date object (weekly measurement date)
            reservoir_sweden_gwh - stored energy in GWh
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    chunks = []

    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_RANGE_DAYS), end)
        chunks.append(_fetch_reservoir_chunk(
            SWEDEN_AREA_CODE,
            chunk_start.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        ))
        chunk_start = chunk_end

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        logger.warning("No Sweden reservoir data returned from ENTSO-E")
        return pd.DataFrame(columns=["date", "reservoir_sweden_gwh"])

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["reservoir_sweden_gwh"] = df["stored_mwh"] / 1000.0
    df = df[["date", "reservoir_sweden_gwh"]]

    logger.info("Sweden reservoir: %d weekly records (%s → %s)", len(df), start_date, end_date)
    return df
