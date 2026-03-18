import pandas as pd


FEATURE_COLUMNS = [
    "mean_temp", "min_temp", "max_temp",
    "mean_wind", "max_wind",
    "day_of_week", "month", "day_of_year"
]


def aggregate_weather_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly weather data to daily features.

    Args:
        df: DataFrame with columns: timestamp, temperature, windspeed.

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
        max_wind=("windspeed", "max")
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


def build_training_data(prices_hourly: pd.DataFrame, weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Merge and prepare the full training dataset.

    Args:
        prices_hourly: Hourly prices from ENTSO-E.
        weather_hourly: Hourly weather from Open-Meteo archive.

    Returns:
        Daily DataFrame ready for model training.
    """
    prices_daily = aggregate_prices_daily(prices_hourly)
    weather_daily = aggregate_weather_daily(weather_hourly)

    merged = pd.merge(prices_daily, weather_daily, on="date")
    merged = add_time_features(merged)
    merged = merged.dropna().reset_index(drop=True)

    return merged


def build_forecast_features(forecast_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare forecast weather data as model input features.

    Args:
        forecast_hourly: Hourly weather forecast from Open-Meteo.

    Returns:
        Daily DataFrame with feature columns ready for inference.
    """
    forecast_daily = aggregate_weather_daily(forecast_hourly)
    forecast_daily = add_time_features(forecast_daily)

    return forecast_daily
