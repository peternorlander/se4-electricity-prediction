import pandas as pd

from se4.http_client import get_with_retry, describe_body


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
    # Northern wind belt (round 25, docs/FINDINGS.md "The northern wind belt"): the fleets behind
    # the SE3 import price. features.NORTH_WIND_SITES averages their power curves into
    # north_wind_power / north_wind_night; the keys must stay in step with that list.
    "norrbotten":   (65.3, 20.9),  # SE1/SE2 north – Markbygden
    "jamtland":     (63.2, 14.6),  # SE2 inland
    "ostrobothnia": (63.1, 21.6),  # Finnish west-coast wind belt
}


# The three global models whose hourly mean is the plan-2.3b blend (IMPROVEMENT_PLAN.md).
# Production serves `best_match` (no `models=` parameter); the blend is only predicted
# next to it as a shadow, so the prediction log can score both on the same days. These
# are the models the backtest blended (docs/FINDINGS.md "Archive weather flatters the eval").
BLEND_MODELS = ("ecmwf_ifs025", "icon_global", "gfs_global")


def _json_with_hourly(response, label: str) -> dict:
    """
    The response's JSON, or an error that says what Open-Meteo answered.

    Open-Meteo reports a refusal (rate limit, bad range, outage) as {"error": true, "reason":
    "..."} and sometimes as a 200 without "hourly". Reading `["hourly"]` blind turned both
    into a bare KeyError that told the Actions log nothing.
    """
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code != 200 or not isinstance(data, dict) or "hourly" not in data:
        raise RuntimeError(f"Open-Meteo {label}: HTTP {response.status_code}, "
                           f"no hourly data: {describe_body(response)}")
    return data


def _hourly_values(hourly: dict, variable: str, models=None) -> list:
    """One hourly variable from an Open-Meteo response: as is, or -- when `models` were
    requested -- the per-hour mean of the `{variable}_{model}` columns, skipping a model
    that has no value for that hour (NaN only where every model is missing)."""
    if not models:
        return hourly[variable]
    per_model = pd.DataFrame({m: pd.to_numeric(pd.Series(hourly[f"{variable}_{m}"]), errors="coerce")
                              for m in models})
    return per_model.mean(axis=1, skipna=True).tolist()


def _model_params(models) -> dict:
    return {"models": ",".join(models)} if models else {}


def _parse_response(data: dict, models=None) -> pd.DataFrame:
    hourly = data["hourly"]
    return pd.DataFrame(
        {
            # UTC-aware timestamps — consistent with ENTSO-E data
            "timestamp": pd.to_datetime(hourly["time"], utc=True),
            "temperature": _hourly_values(hourly, "temperature_2m", models),
            "windspeed": _hourly_values(hourly, "windspeed_100m", models),
            "radiation": _hourly_values(hourly, "shortwave_radiation", models),
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
    return _parse_response(_json_with_hourly(response, f"archive {start_date}..{end_date}"))


def fetch_recent(past_days: int) -> pd.DataFrame:
    """
    Fetch the last `past_days` days of hourly weather from the FORECAST endpoint.

    Fills the archive's ~5-day publication lag. `fetch_historical` can only
    reach today−5, which used to leave the training frame ending ~5 days behind
    the first forecast day — measured at avg +1.70 / min +0.59 / cheap2h +0.37
    EUR/MWh MAE, and +2.2 / +1.0 / +0.8 in the current regime. See docs/DECISIONS.md
    "Closing the weather-archive lag" for the evidence and for why the
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
    return _parse_response(_json_with_hourly(response, f"recent {past_days} days"))


def fetch_forecast(models=None) -> pd.DataFrame:
    """
    Fetch 8-day hourly weather forecast from Open-Meteo.

    Args:
        models: None for Open-Meteo's `best_match` (what production serves), or a
                sequence of model names whose hourly mean is returned (BLEND_MODELS
                for the plan-2.3b shadow).

    Returns:
        DataFrame with columns: timestamp, temperature, windspeed, radiation (W/m²).
    """
    params = {
        "latitude": LOCATION_LAT,
        "longitude": LOCATION_LON,
        "hourly": HOURLY_VARIABLES,
        "forecast_days": 8,
        "timezone": TIMEZONE,
        **_model_params(models),
    }

    response = get_with_retry(OPEN_METEO_FORECAST_URL, params)
    return _parse_response(_json_with_hourly(response, "forecast"), models)


def _fetch_location_series(lat: float, lon: float, url: str, extra_params: dict, models=None) -> pd.DataFrame:
    """Fetch wind speed, solar radiation and temperature for a single location."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": INTL_VARIABLES,
        "timezone": TIMEZONE,
        **extra_params,
        **_model_params(models),
    }
    response = get_with_retry(url, params)
    hourly = _json_with_hourly(response, f"{url.rsplit('/', 1)[-1]} ({lat}, {lon})")["hourly"]
    return pd.DataFrame({
        # UTC-aware timestamps — consistent with main SE4 weather fetch
        "time": pd.to_datetime(hourly["time"], utc=True),
        "windspeed": _hourly_values(hourly, "windspeed_100m", models),
        "radiation": _hourly_values(hourly, "shortwave_radiation", models),
        "temperature": _hourly_values(hourly, "temperature_2m", models),
    })


def _fetch_all_locations(url: str, extra_params: dict, models=None) -> pd.DataFrame:
    location_data = {
        name: _fetch_location_series(lat, lon, url, extra_params, models)
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


def fetch_international_wind_forecast(models=None) -> pd.DataFrame:
    """
    Fetch 8-day hourly wind speed and solar radiation forecast for all WIND_LOCATIONS.

    Args:
        models: as fetch_forecast -- None for `best_match`, or the models to average.

    Returns:
        DataFrame with columns: timestamp, windspeed_{key}, radiation_{key}, ...
    """
    return _fetch_all_locations(
        OPEN_METEO_FORECAST_URL,
        {"forecast_days": 8},
        models,
    )
