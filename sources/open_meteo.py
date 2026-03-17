import requests
import pandas as pd


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Representative coordinates for SE4 (Malmö)
LOCATION_LAT = 55.6
LOCATION_LON = 13.0
TIMEZONE = "Europe/Stockholm"
HOURLY_VARIABLES = "temperature_2m,windspeed_10m"


def _parse_response(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "windspeed": data["hourly"]["windspeed_10m"]
        }
    )


def fetch_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly weather from Open-Meteo archive.

    Note: Archive has a ~5-day lag before data becomes available.

    Args:
        start_date: Start date as YYYY-MM-DD string.
        end_date: End date as YYYY-MM-DD string.

    Returns:
        DataFrame with columns: timestamp, temperature, windspeed.
    """
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARIABLES,
        "timezone": TIMEZONE
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()

    return _parse_response(response.json())


def fetch_forecast() -> pd.DataFrame:
    """
    Fetch 8-day hourly weather forecast from Open-Meteo.

    Returns:
        DataFrame with columns: timestamp, temperature, windspeed.
    """
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "hourly": HOURLY_VARIABLES,
        "forecast_days": 8,
        "timezone": TIMEZONE
    }

    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()

    return _parse_response(response.json())
