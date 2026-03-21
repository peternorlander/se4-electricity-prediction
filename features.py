import pandas as pd
from datetime import timedelta


FEATURE_COLUMNS = [
    "mean_temp", "min_temp", "max_temp",
    "mean_wind", "max_wind",
    # International wind locations (see WIND_LOCATIONS in sources/open_meteo.py)
    "mean_wind_de_north",
    "mean_wind_dk1",
    "mean_wind_dk2",
    "mean_wind_utklippan",
    "mean_wind_karlskrona",
    "mean_wind_hano",
    # Solar radiation (GHI, W/m²) — global horizontal irradiance drives solar output
    "mean_radiation", "max_radiation",          # SE4 local (Malmö / Open-Meteo ERA5)
    "mean_radiation_de_north",                  # Northern Germany — largest solar market affecting SE4
    "mean_radiation_dk1", "mean_radiation_dk2", # Denmark — directly coupled to SE4
    # Market coupling — lag-1 Day-Ahead prices (EUR/MWh) from neighbouring zones
    "price_de_lag1",   # German (DE/LU) price from the previous day
    "price_dk2_lag1",  # Danish (DK2) price from the previous day
    "day_of_week", "month", "day_of_year"
]


def aggregate_weather_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly weather data to daily features.

    Args:
        df: DataFrame with columns: timestamp, temperature, windspeed, radiation.

    Returns:
        DataFrame with one row per day and aggregated weather columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    return df.groupby("date").agg(
        mean_temp=("temperature", "mean"),
        min_temp=("temperature", "min"),
        max_temp=("temperature", "max"),
        mean_wind=("windspeed", "mean"),
        max_wind=("windspeed", "max"),
        mean_radiation=("radiation", "mean"),
        max_radiation=("radiation", "max"),
    ).reset_index()


def aggregate_prices_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly ENTSO-E prices to daily min/avg/max.

    Args:
        df: DataFrame with columns: timestamp (UTC), price_eur_mwh.

    Returns:
        DataFrame with one row per day and price_min/avg/max columns.
    """
    df = df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert("Europe/Stockholm")
    df["date"] = df["timestamp_local"].dt.date

    return df.groupby("date").agg(
        price_min=("price_eur_mwh", "min"),
        price_avg=("price_eur_mwh", "mean"),
        price_max=("price_eur_mwh", "max")
    ).reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features to a daily dataframe.

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        Same DataFrame with day_of_week, month and day_of_year columns added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


def aggregate_international_weather_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly international wind and solar data to daily means.

    Detects all windspeed_{key} and radiation_{key} columns and produces
    mean_wind_{key} and mean_radiation_{key} respectively.
    Driven by WIND_LOCATIONS in sources/open_meteo.py — no changes needed
    here when locations are added or removed.

    Args:
        df: DataFrame with columns: timestamp, windspeed_{key}, radiation_{key}, ...

    Returns:
        DataFrame with one row per day and mean_wind_{key} / mean_radiation_{key} columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    wind_cols = [c for c in df.columns if c.startswith("windspeed_")]
    rad_cols  = [c for c in df.columns if c.startswith("radiation_")]

    agg = {}
    agg.update({f"mean_wind_{c[len('windspeed_'):]}":      (c, "mean") for c in wind_cols})
    agg.update({f"mean_radiation_{c[len('radiation_'):]}" : (c, "mean") for c in rad_cols})

    return df.groupby("date").agg(**agg).reset_index()


def aggregate_market_prices_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly DE/DK2 market prices to daily means (Stockholm-tid).

    Args:
        df: DataFrame with columns: timestamp (UTC tz-aware), price_de, price_dk2.

    Returns:
        DataFrame with one row per day and columns: date, price_de, price_dk2.
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.tz_convert("Europe/Stockholm").dt.date

    return df.groupby("date").agg(
        price_de=("price_de", "mean"),
        price_dk2=("price_dk2", "mean"),
    ).reset_index()


def add_market_lag_features(df: pd.DataFrame, market_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag-1 market prices (DE and DK2) to a daily DataFrame.

    Prices from day D-1 are joined to day D. Since the day-ahead auction runs
    simultaneously across all exchanges, only historical prices can be used as features.

    Args:
        df:           Daily DataFrame with a 'date' column (date objects).
        market_daily: Daily DataFrame from aggregate_market_prices_daily().

    Returns:
        df with added columns price_de_lag1 and price_dk2_lag1.
    """
    lagged = market_daily.copy()
    lagged["date"] = (pd.to_datetime(lagged["date"]) + timedelta(days=1)).dt.date
    lagged = lagged.rename(columns={"price_de": "price_de_lag1", "price_dk2": "price_dk2_lag1"})

    return pd.merge(df, lagged[["date", "price_de_lag1", "price_dk2_lag1"]], on="date", how="left")


def build_training_data(
    prices_hourly: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    wind_intl_hourly: pd.DataFrame,
    market_prices_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge and prepare the full training dataset.

    Args:
        prices_hourly:        Hourly SE4 prices from ENTSO-E.
        weather_hourly:       Hourly weather from Open-Meteo archive (SE4/Malmö).
        wind_intl_hourly:     Hourly wind data for Germany and Denmark.
        market_prices_hourly: Hourly DE/LU + DK2 prices from ENTSO-E.

    Returns:
        Daily DataFrame ready for model training.
    """
    prices_daily = aggregate_prices_daily(prices_hourly)
    weather_daily = aggregate_weather_daily(weather_hourly)
    wind_intl_daily = aggregate_international_weather_daily(wind_intl_hourly)
    market_daily = aggregate_market_prices_daily(market_prices_hourly)

    merged = pd.merge(prices_daily, weather_daily, on="date")
    merged = pd.merge(merged, wind_intl_daily, on="date")
    merged = add_market_lag_features(merged, market_daily)
    merged = add_time_features(merged)
    merged = merged.dropna().reset_index(drop=True)

    return merged


def build_forecast_features(
    forecast_hourly: pd.DataFrame,
    wind_intl_forecast_hourly: pd.DataFrame,
    market_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare forecast weather data as model input features.

    Args:
        forecast_hourly:           Hourly weather forecast from Open-Meteo (SE4/Malmö).
        wind_intl_forecast_hourly: Hourly wind forecast for Germany and Denmark.
        market_daily:              Daily DE/DK2 prices from aggregate_market_prices_daily().

    Returns:
        Daily DataFrame with feature columns ready for inference.
    """
    forecast_daily = aggregate_weather_daily(forecast_hourly)
    wind_intl_daily = aggregate_international_weather_daily(wind_intl_forecast_hourly)

    forecast_daily = pd.merge(forecast_daily, wind_intl_daily, on="date")

    # Future DE/DK2 prices are unknown — we use the most recent known price
    # as a "market regime indicator" for the entire forecast horizon.
    last_known = market_daily.iloc[-1]
    forecast_daily["price_de_lag1"] = last_known["price_de"]
    forecast_daily["price_dk2_lag1"] = last_known["price_dk2"]

    forecast_daily = add_time_features(forecast_daily)

    return forecast_daily
