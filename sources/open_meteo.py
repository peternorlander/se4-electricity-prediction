import pandas as pd

from http_client import get_with_retry


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Representative coordinates for SE4 (Malmö)
LOCATION_LAT = 55.6
LOCATION_LON = 13.0

# All requests use UTC to avoid DST transition artifacts when merging with
# ENTSO-E market data (which is always UTC). Daily aggregation in features.py
# then converts to Europe/Stockholm to align with market day boundaries.
TIMEZONE = "UTC"
HOURLY_VARIABLES = "temperature_2m,windspeed_100m,shortwave_radiation"

# Variables fetched per international location (wind + solar).
# windspeed_100m (hub height) is significantly more relevant for large-scale
# wind power than the standard 10m measurement.
INTL_VARIABLES = "windspeed_100m,shortwave_radiation"

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
    hourly = data["hourly"]
    return pd.DataFrame(
        {
            # UTC-aware timestamps — consistent with ENTSO-E data
            "timestamp": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m"],
            "windspeed": hourly["windspeed_100m"],
            "radiation": hourly["shortwave_radiation"],
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
        DataFrame with columns: timestamp, temperature, windspeed, radiation (W/m²).
    """
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARIABLES,
        "timezone": TIMEZONE
    }

    response = get_with_retry(OPEN_METEO_ARCHIVE_URL, params)
    return _parse_response(response.json())


def fetch_forecast() -> pd.DataFrame:
    """
    Fetch 8-day hourly weather forecast from Open-Meteo.

    Returns:
        DataFrame with columns: timestamp, temperature, windspeed, radiation (W/m²).
    """
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "hourly": HOURLY_VARIABLES,
        "forecast_days": 8,
        "timezone": TIMEZONE
    }

    response = get_with_retry(OPEN_METEO_FORECAST_URL, params)
    return _parse_response(response.json())


def _fetch_location_series(lat: float, lon: float, url: str, extra_params: dict) -> pd.DataFrame:
    """Fetch wind speed and solar radiation for a single location."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": INTL_VARIABLES,
        "timezone": TIMEZONE,
        **extra_params,
    }
    response = get_with_retry(url, params)
    hourly = response.json()["hourly"]
    return pd.DataFrame({
        # UTC-aware timestamps — consistent with main SE4 weather fetch
        "time": pd.to_datetime(hourly["time"], utc=True),
        "windspeed": hourly["windspeed_100m"],
        "radiation": hourly["shortwave_radiation"],
    })


def _fetch_all_locations(url: str, extra_params: dict) -> pd.DataFrame:
    location_data = {
        name: _fetch_location_series(lat, lon, url, extra_params)
        for name, (lat, lon) in WIND_LOCATIONS.items()
    }
    first = next(iter(location_data.values()))
    df = pd.DataFrame({"timestamp": first["time"]})
    for name, loc_df in location_data.items():
        df[f"windspeed_{name}"] = loc_df["windspeed"].values
        df[f"radiation_{name}"] = loc_df["radiation"].values
    return df


def fetch_international_wind_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly wind speed for all WIND_LOCATIONS from Open-Meteo archive.

    Args:
        start_date: Start date as YYYY-MM-DD string.
        end_date: End date as YYYY-MM-DD string.

    Returns:
        DataFrame with columns: timestamp, windspeed_{key}, radiation_{key}, ...
    """
    return _fetch_all_locations(
        OPEN_METEO_ARCHIVE_URL,
        {"start_date": start_date, "end_date": end_date},
    )


def fetch_international_wind_forecast() -> pd.DataFrame:
    """
    Fetch 8-day hourly wind speed and solar radiation forecast for all WIND_LOCATIONS.

    Returns:
        DataFrame with columns: timestamp, windspeed_{key}, radiation_{key}, ...
    """
    return _fetch_all_locations(
        OPEN_METEO_FORECAST_URL,
        {"forecast_days": 8},
    )
