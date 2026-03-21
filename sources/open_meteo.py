import requests
import pandas as pd


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Representative coordinates for SE4 (Malmö)
LOCATION_LAT = 55.6
LOCATION_LON = 13.0
TIMEZONE = "Europe/Stockholm"
HOURLY_VARIABLES = "temperature_2m,windspeed_10m"

# Named extra wind locations for grid correlation features.
# Keys become column names: windspeed_{key} → mean_wind_{key}.
WIND_LOCATIONS = {
    "de_north":   (53.5,  9.9),   # Northern Germany (Schleswig-Holstein) – largest driver of negative SE4 prices
    "dk1":        (56.5,  8.5),   # Western Denmark / Jutland (DK1 bidding zone)
    "dk2":        (55.7, 12.5),   # Eastern Denmark / Zealand (DK2 bidding zone, directly coupled to SE4)
    "utklippan":  (55.96, 15.72), # Utklippan, Blekinge – open-sea station, high Baltic wind correlation
    "karlskrona": (56.16, 15.59), # Karlskrona – captures offshore wind patterns in southern Baltic
    "hano":       (56.01, 14.84), # Hanö – key location for southern Baltic offshore wind parks
}


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


def _fetch_wind_series(lat: float, lon: float, url: str, extra_params: dict) -> pd.Series:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m",
        "timezone": TIMEZONE,
        **extra_params,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return pd.Series(
        data["hourly"]["windspeed_10m"],
        index=pd.to_datetime(data["hourly"]["time"])
    )


def _fetch_all_wind_locations(url: str, extra_params: dict) -> pd.DataFrame:
    series = {
        name: _fetch_wind_series(lat, lon, url, extra_params)
        for name, (lat, lon) in WIND_LOCATIONS.items()
    }
    first = next(iter(series.values()))
    return pd.DataFrame(
        {"timestamp": first.index, **{f"windspeed_{name}": s.values for name, s in series.items()}}
    )


def fetch_international_wind_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly wind speed for all WIND_LOCATIONS from Open-Meteo archive.

    Args:
        start_date: Start date as YYYY-MM-DD string.
        end_date: End date as YYYY-MM-DD string.

    Returns:
        DataFrame with columns: timestamp, windspeed_{location_key}, ...
    """
    return _fetch_all_wind_locations(
        OPEN_METEO_ARCHIVE_URL,
        {"start_date": start_date, "end_date": end_date},
    )


def fetch_international_wind_forecast() -> pd.DataFrame:
    """
    Fetch 8-day hourly wind speed forecast for all WIND_LOCATIONS from Open-Meteo.

    Returns:
        DataFrame with columns: timestamp, windspeed_{location_key}, ...
    """
    return _fetch_all_wind_locations(
        OPEN_METEO_FORECAST_URL,
        {"forecast_days": 8},
    )
