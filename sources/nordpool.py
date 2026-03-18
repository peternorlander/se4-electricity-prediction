import requests
from datetime import date


NORDPOOL_API_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
SE4_DELIVERY_AREA = "SE4"


def fetch_today_mean_sek() -> float:
    """
    Fetch today's average day-ahead price from Nordpool in SEK/MWh.

    Returns:
        Daily mean price in SEK/MWh.
    """
    params = {
        "currency": "SEK",
        "deliveryArea": SE4_DELIVERY_AREA,
        "date": date.today().strftime("%Y-%m-%d")
    }

    response = requests.get(NORDPOOL_API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    prices = [
        entry["entryPerArea"][SE4_DELIVERY_AREA]
        for entry in data["multiAreaEntries"]
        if entry.get("entryPerArea", {}).get(SE4_DELIVERY_AREA) is not None
    ]

    return sum(prices) / len(prices)


def _has_prices_for_date(target_date: date) -> bool:
    """Check if Nordpool has published official prices for the given date."""
    params = {
        "currency": "EUR",
        "deliveryArea": SE4_DELIVERY_AREA,
        "date": target_date.strftime("%Y-%m-%d")
    }

    response = requests.get(NORDPOOL_API_URL, params=params, timeout=30)

    if response.status_code != 200:
        return False

    data = response.json()
    prices = [
        entry["entryPerArea"][SE4_DELIVERY_AREA]
        for entry in data.get("multiAreaEntries", [])
        if entry.get("entryPerArea", {}).get(SE4_DELIVERY_AREA) is not None
    ]

    return len(prices) > 0


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
