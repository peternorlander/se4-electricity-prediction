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
INTL_VARIABLES = "windspeed_100m,shortwave_radiation,temperature_2m"

# Named extra locations for grid correlation features.
# Keys become column names: windspeed_{key} → mean_wind_{key}, temperature_{key} → mean_temp_{key}.
WIND_LOCATIONS = {
    "de_north":    (53.5,  9.9),   # Northern Germany (Schleswig-Holstein) – largest driver of negative SE4 prices
    "dk1":         (56.5,  8.5),   # Western Denmark / Jutland (DK1 bidding zone)
    "dk2":         (55.7, 12.5),   # Eastern Denmark / Zealand (DK2 bidding zone, directly coupled to SE4)
    "karlskrona":  (56.16, 15.59), # Karlskrona – captures offshore wind patterns in southern Baltic
    "stockholm":   (59.33, 18.07), # Stockholm – SE3 load centre, used for SE3↔SE4 temperature gradient
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


def fetch_recent(past_days: int) -> pd.DataFrame:
    """
    Fetch the last `past_days` days of hourly weather from the FORECAST endpoint.

    Fills the archive's ~5-day publication lag. `fetch_historical` can only
    reach today−5, which used to leave the training frame ending ~5 days behind
    the first forecast day — measured at avg +1.70 / min +0.59 / cheap2h +0.37
    EUR/MWh MAE, and +2.2 / +1.0 / +0.8 in the current regime. See the README
    "Closing the weather-archive lag" section for the evidence and for why the
    two products are interchangeable here (windspeed matched the archive to
    0.000 MAE at four of five locations over a 10-day overlap).

    This is the operational NWP model's own recent analysis, not ERA5. That is
    a different product from `fetch_historical`, which is the reason the caller
    splices rather than replacing: the archive stays authoritative wherever it
    has published, and this only covers days it has not reached.

    Args:
        past_days: Days of history to request (Open-Meteo allows up to 92).

    Returns:
        DataFrame with columns: timestamp, temperature, windspeed, radiation (W/m²).
        Runs up to the current hour, so the caller must trim to whole days.
    """
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "hourly": HOURLY_VARIABLES,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TIMEZONE
    }

    response = get_with_retry(OPEN_METEO_FORECAST_URL, params)
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
    """Fetch wind speed, solar radiation and temperature for a single location."""
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
        "temperature": hourly["temperature_2m"],
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
        df[f"temperature_{name}"] = loc_df["temperature"].values
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


def fetch_international_wind_recent(past_days: int) -> pd.DataFrame:
    """
    Fetch the last `past_days` days of hourly wind/radiation/temperature for all
    WIND_LOCATIONS from the FORECAST endpoint — the international counterpart to
    `fetch_recent`, filling the same archive lag.

    Args:
        past_days: Days of history to request (Open-Meteo allows up to 92).

    Returns:
        DataFrame with columns: timestamp, windspeed_{key}, radiation_{key}, ...
        Runs up to the current hour, so the caller must trim to whole days.
    """
    return _fetch_all_locations(
        OPEN_METEO_FORECAST_URL,
        {"past_days": past_days, "forecast_days": 1},
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
