from datetime import date

from http_client import get_with_retry


NORDPOOL_API_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
SE4_DELIVERY_AREA = "SE4"


def _extract_se4_prices(data: dict) -> list:
    """Pull the SE4 hourly price entries out of a Nordpool API response."""
    return [
        entry["entryPerArea"][SE4_DELIVERY_AREA]
        for entry in data.get("multiAreaEntries", [])
        if entry.get("entryPerArea", {}).get(SE4_DELIVERY_AREA) is not None
    ]


def fetch_mean_sek_for_date(target_date: date) -> float | None:
    """
    Fetch the SE4 day-ahead daily mean price from Nordpool in SEK/MWh for a
    specific delivery date.

    Args:
        target_date: The delivery date to fetch.

    Returns:
        Daily mean price in SEK/MWh, or None if Nordpool has published no
        prices for that date.
    """
    params = {
        "currency": "SEK",
        "deliveryArea": SE4_DELIVERY_AREA,
        "date": target_date.strftime("%Y-%m-%d")
    }

    response = get_with_retry(NORDPOOL_API_URL, params)

    if response.status_code != 200:
        return None

    prices = _extract_se4_prices(response.json())

    if not prices:
        return None

    return sum(prices) / len(prices)


def _has_prices_for_date(target_date: date) -> bool:
    """Check if Nordpool has published official prices for the given date."""
    params = {
        "currency": "EUR",
        "deliveryArea": SE4_DELIVERY_AREA,
        "date": target_date.strftime("%Y-%m-%d")
    }

    response = get_with_retry(NORDPOOL_API_URL, params)

    if response.status_code != 200:
        return False

    return len(_extract_se4_prices(response.json())) > 0


def get_dates_with_known_prices() -> set:
    """
    Return the set of dates that already have official Nordpool prices published.
    These dates should be excluded from model predictions.

    Returns:
        Set of date objects with known prices.
    """
    known = set()
    today = date.today()

    if _has_prices_for_date(today):
        known.add(today)

    tomorrow = date.fromordinal(today.toordinal() + 1)

    if _has_prices_for_date(tomorrow):
        known.add(tomorrow)
        print(f"  → Tomorrow ({tomorrow}) has official prices, excluding from predictions")
    else:
        print(f"  → Tomorrow ({tomorrow}) has no official prices yet, will be predicted")

    return known
