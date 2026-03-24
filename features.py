import numpy as np
import pandas as pd
from datetime import timedelta


FEATURE_COLUMNS = [
    "mean_temp", "min_temp", "max_temp",
    "mean_wind", "max_wind",
    # International wind locations — 100m hub-height wind speed
    "mean_wind_de_north",
    "mean_wind_dk1",
    "mean_wind_dk2",
    "mean_wind_karlskrona",
    "mean_wind_stockholm",
    # Solar radiation (GHI, W/m²) — global horizontal irradiance drives solar output
    "mean_radiation", "max_radiation",          # SE4 local (Malmö / Open-Meteo ERA5)
    "mean_radiation_de_north",                  # Northern Germany — largest solar market affecting SE4
    "mean_radiation_dk1", "mean_radiation_dk2", # Denmark — directly coupled to SE4
    # Market coupling — lag-1 Day-Ahead prices (EUR/MWh) from neighbouring zones
    "price_de_lag1",   # German (DE/LU) price from the previous day
    "price_dk2_lag1",  # Danish (DK2) price from the previous day
    # SE4 own price lags — autoregressive signal from recent price history
    "price_se4_avg_lag1",  # Previous day's average price
    "price_se4_avg_lag2",  # Two days ago average price
    "price_se4_avg_lag7",  # Same weekday last week (weekly seasonality)
    # Target-specific price lags — direct signal for min/max models
    "price_se4_min_lag1",  # Previous day's min price
    "price_se4_max_lag1",  # Previous day's max price
    # Price momentum and volatility — market regime indicators
    "price_momentum",      # lag1 - lag2: trend direction (positive = rising prices)
    "price_volatility_7d", # Rolling 7-day std of avg price: market stability indicator
    # Residual load lag — captures momentum in supply/demand balance
    "residual_load_lag1",
    # Residual load: demand proxy minus renewable supply — indicates conventional generation need
    "residual_load",
    # Non-linear heating degree days — captures accelerating demand below 0°C
    "hdd_linear",          # max(0, 17 - temp): standard HDD
    "hdd_cold_boost",      # extra quadratic term below 0°C: captures non-linear demand surge
    # SE3↔SE4 temperature gradient — predicts transmission stress and price divergence
    "temp_gradient_se3_se4",
    # Weather forecast uncertainty proxies: rolling variability predicts risk of price spikes
    "wind_variability", "radiation_variability",
    # Nuclear generation availability in SE3 — unplanned outages force SE4 to import expensive power
    "nuclear_outage_se3",
    # Cyclic time encoding via sin/cos — captures periodicity without ordinal discontinuities
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "dow_sin", "dow_cos",
]


def _to_swedish_date(timestamps: pd.Series) -> pd.Series:
    """
    Convert a Series of timestamps to Swedish (Europe/Stockholm) date objects.

    Handles both tz-aware (UTC from Open-Meteo) and tz-naive inputs.
    Using Swedish midnight boundaries ensures weather and price daily
    aggregations are aligned (ENTSO-E prices also converted to Swedish time).
    """
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("UTC")
    return timestamps.dt.tz_convert("Europe/Stockholm").dt.date


def aggregate_weather_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly weather data to daily features.

    Args:
        df: DataFrame with columns: timestamp (UTC-aware), temperature, windspeed, radiation.

    Returns:
        DataFrame with one row per day and aggregated weather columns.
    """
    df = df.copy()
    df["date"] = _to_swedish_date(pd.to_datetime(df["timestamp"]))

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
    df["date"] = _to_swedish_date(df["timestamp"])

    return df.groupby("date").agg(
        price_min=("price_eur_mwh", "min"),
        price_avg=("price_eur_mwh", "mean"),
        price_max=("price_eur_mwh", "max")
    ).reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trigonometric cyclic time features to a daily dataframe.

    Month, day-of-year and day-of-week are encoded as sin/cos pairs so the
    model sees the periodic wrap-around (e.g. December → January, Sunday → Monday)
    as a smooth transition rather than a large numeric jump.

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        Same DataFrame with cyclic time feature columns added.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    month = df["date"].dt.month
    day_of_year = df["date"].dt.dayofyear
    dow = df["date"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def _wind_power_curve(wind_speed: pd.Series, rated_speed: float = 13.0) -> pd.Series:
    """
    Approximate wind turbine power output using a cubic power curve.

    Real wind turbines produce power proportional to v³ below rated speed,
    then cap at rated output. This is far more realistic than linear
    normalization: doubling wind speed from 4→8 m/s increases power ~8×,
    while 12→15 m/s adds very little.

    Args:
        wind_speed:  Wind speed in m/s.
        rated_speed: Wind speed at which turbine reaches rated power (default 13 m/s).

    Returns:
        Normalized power output in [0, 1].
    """
    capped = wind_speed.clip(upper=rated_speed)
    return (capped / rated_speed) ** 3


def add_residual_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an enhanced Residual Load feature.

    Residual Load = Demand Proxy - (Weighted Regional Wind + Weighted Regional Solar)

    Improvements over the basic version:
    1. Cubic wind power curve instead of linear — matches real turbine physics.
    2. Regional wind supply from SE4, DK1, DK2, DE weighted by interconnection
       capacity to SE4 (SE4 prices are heavily coupled to neighbouring zones).
    3. Regional solar supply from SE4, DK1, DK2, DE similarly weighted.
    4. Decomposed components (demand_proxy, wind_supply_total, solar_supply_total)
       exposed as separate features so the model can learn non-linear interactions.

    Interconnection weights (approximate, based on NTC capacity towards SE4):
    - SE4 local:  1.0  (direct)
    - DK2:        0.4  (Øresund link ~1700 MW, significant for SE4)
    - DK1:        0.2  (indirect via DK2 or DE)
    - DE (north): 0.3  (Baltic Cable ~600 MW + indirect via DK)

    Args:
        df: Daily DataFrame with weather and international weather columns.

    Returns:
        Same DataFrame with residual_load and decomposed component columns added.
    """
    df = df.copy()

    # --- Demand proxy: heating degree days (higher when cold) ---
    demand_proxy = 15 - df["mean_temp"]

    # --- Wind supply: cubic power curve, regionally weighted ---
    wind_se4 = _wind_power_curve(df["mean_wind"])
    wind_dk1 = _wind_power_curve(df["mean_wind_dk1"]) if "mean_wind_dk1" in df.columns else 0
    wind_dk2 = _wind_power_curve(df["mean_wind_dk2"]) if "mean_wind_dk2" in df.columns else 0
    wind_de  = _wind_power_curve(df["mean_wind_de_north"]) if "mean_wind_de_north" in df.columns else 0

    wind_supply_total = (1.0 * wind_se4 + 0.4 * wind_dk2 + 0.2 * wind_dk1 + 0.3 * wind_de) / 1.9

    # --- Solar supply: regionally weighted ---
    solar_se4 = df["mean_radiation"] / 500
    solar_dk1 = df["mean_radiation_dk1"] / 500 if "mean_radiation_dk1" in df.columns else 0
    solar_dk2 = df["mean_radiation_dk2"] / 500 if "mean_radiation_dk2" in df.columns else 0
    solar_de  = df["mean_radiation_de_north"] / 500 if "mean_radiation_de_north" in df.columns else 0

    solar_supply_total = (1.0 * solar_se4 + 0.4 * solar_dk2 + 0.2 * solar_dk1 + 0.3 * solar_de) / 1.9

    # --- Composite residual load ---
    df["residual_load"] = demand_proxy - wind_supply_total - solar_supply_total

    return df


def add_hdd_and_temp_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add non-linear Heating Degree Days (HDD) and SE3↔SE4 temperature gradient.

    In the Nordic system, electricity demand is strongly correlated with outdoor
    temperature due to widespread use of electric heating and heat pumps. This
    relationship is non-linear: below 0°C demand increases rapidly per degree,
    while above ~15°C the relationship is weak.

    The SE3↔SE4 temperature gradient captures when SE3 (Stockholm) is significantly
    colder than SE4 (Malmö), which stresses north→south transmission and causes
    SE4 prices to diverge upward from the system price.

    Features added:
    - hdd_linear:           max(0, 17 - temp) — standard Heating Degree Days
    - hdd_cold_boost:       extra quadratic term for sub-zero temps, capturing the
                            accelerating demand response when it's very cold
    - temp_gradient_se3_se4: Stockholm temp minus Malmö temp — negative values mean
                            SE3 is colder → higher transmission load → higher SE4 prices

    Args:
        df: Daily DataFrame with mean_temp and mean_temp_stockholm columns.

    Returns:
        Same DataFrame with HDD and temperature gradient columns added.
    """
    df = df.copy()

    # Standard HDD with 17°C base (common Nordic base temperature)
    df["hdd_linear"] = (17 - df["mean_temp"]).clip(lower=0)

    # Non-linear cold boost: quadratic term only activates below 0°C
    # At -10°C this adds 100, at -20°C it adds 400 — capturing the
    # exponential increase in heating demand during extreme cold
    below_zero = df["mean_temp"].clip(upper=0)
    df["hdd_cold_boost"] = below_zero ** 2

    # Temperature gradient: SE3 (Stockholm) minus SE4 (Malmö)
    # Negative = SE3 is colder → more demand in SE3 → transmission stress
    if "mean_temp_stockholm" in df.columns:
        df["temp_gradient_se3_se4"] = df["mean_temp_stockholm"] - df["mean_temp"]
    else:
        df["temp_gradient_se3_se4"] = 0.0

    return df


def add_weather_variability(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Add rolling weather variability as a proxy for Weather Forecast Uncertainty (WFU).

    Research shows that forecast uncertainty is a strong driver of extreme price
    events on the intraday market. High meteorological variability periods coincide
    with less reliable forecasts and a higher risk of price spikes.

    Uses a rolling standard deviation over `window` days on wind speed and
    radiation. Requires at least 3 observations — earlier rows will be NaN and
    are removed by the downstream dropna() call in build_training_data.

    Args:
        df:     Daily DataFrame sorted by date with mean_wind, mean_radiation.
        window: Rolling window in days (default 7).

    Returns:
        Same DataFrame with wind_variability and radiation_variability columns added.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["wind_variability"] = df["mean_wind"].rolling(window, min_periods=3).std()
    df["radiation_variability"] = df["mean_radiation"].rolling(window, min_periods=3).std()
    return df


def aggregate_international_weather_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly international wind and solar data to daily means.

    Detects all windspeed_{key} and radiation_{key} columns and produces
    mean_wind_{key} and mean_radiation_{key} respectively.
    Driven by WIND_LOCATIONS in sources/open_meteo.py — no changes needed
    here when locations are added or removed.

    Args:
        df: DataFrame with columns: timestamp (UTC-aware), windspeed_{key}, radiation_{key}, ...

    Returns:
        DataFrame with one row per day and mean_wind_{key} / mean_radiation_{key} columns.
    """
    df = df.copy()
    df["date"] = _to_swedish_date(pd.to_datetime(df["timestamp"]))

    wind_cols = [c for c in df.columns if c.startswith("windspeed_")]
    rad_cols  = [c for c in df.columns if c.startswith("radiation_")]
    temp_cols = [c for c in df.columns if c.startswith("temperature_")]

    agg = {}
    agg.update({f"mean_wind_{c[len('windspeed_'):]}":      (c, "mean") for c in wind_cols})
    agg.update({f"mean_radiation_{c[len('radiation_'):]}" : (c, "mean") for c in rad_cols})
    agg.update({f"mean_temp_{c[len('temperature_'):]}" :    (c, "mean") for c in temp_cols})

    return df.groupby("date").agg(**agg).reset_index()


def aggregate_market_prices_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly DE/DK2 market prices to daily means (Stockholm time).

    Args:
        df: DataFrame with columns: timestamp (UTC tz-aware), price_de, price_dk2.

    Returns:
        DataFrame with one row per day and columns: date, price_de, price_dk2.
    """
    df = df.copy()
    df["date"] = _to_swedish_date(df["timestamp"])

    return df.groupby("date").agg(
        price_de=("price_de", "mean"),
        price_dk2=("price_dk2", "mean"),
    ).reset_index()


def add_nuclear_outage(df: pd.DataFrame, nuclear_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily SE3 nuclear outage counts into a daily DataFrame.

    Days without a known outage event default to 0. Planned maintenance is
    published on ENTSO-E months in advance, so this feature is available for
    both training and multi-day forecasting.

    Args:
        df:            Daily DataFrame with a 'date' column (date objects).
        nuclear_daily: DataFrame with columns: date, nuclear_outage_se3.

    Returns:
        df with added column nuclear_outage_se3 (int, 0 = no active outage).
    """
    nuclear = nuclear_daily[["date", "nuclear_outage_se3"]].copy()
    merged = pd.merge(df, nuclear, on="date", how="left")
    merged["nuclear_outage_se3"] = merged["nuclear_outage_se3"].fillna(0).astype(int)
    return merged


def add_se4_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged SE4 prices as autoregressive features.

    Includes average, min and max price lags, plus derived momentum and
    volatility features that help the model understand the current market regime.

    - Lag-1/lag-2 avg: short-term momentum and mean reversion.
    - Lag-7 avg: weekly seasonality (same weekday last week).
    - Lag-1 min/max: target-specific signals — yesterday's min predicts today's
      min far better than yesterday's average does, and vice versa for max.
    - Momentum (lag1 - lag2): captures whether prices are rising or falling.
    - Volatility (7-day rolling std): high-volatility regimes have wider
      min-max spreads and less predictable averages.

    Args:
        df: Daily DataFrame with price_avg, price_min, price_max columns, sorted by date.

    Returns:
        Same DataFrame with price lag, momentum and volatility columns added.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Average price lags
    df["price_se4_avg_lag1"] = df["price_avg"].shift(1)
    df["price_se4_avg_lag2"] = df["price_avg"].shift(2)
    df["price_se4_avg_lag7"] = df["price_avg"].shift(7)

    # Target-specific lags
    df["price_se4_min_lag1"] = df["price_min"].shift(1)
    df["price_se4_max_lag1"] = df["price_max"].shift(1)

    # Momentum: positive = prices rising, negative = falling
    df["price_momentum"] = df["price_se4_avg_lag1"] - df["price_se4_avg_lag2"]

    # Volatility: rolling 7-day standard deviation of average price
    df["price_volatility_7d"] = df["price_avg"].rolling(7, min_periods=3).std()

    return df


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
    nuclear_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge and prepare the full training dataset.

    Args:
        prices_hourly:        Hourly SE4 prices from ENTSO-E.
        weather_hourly:       Hourly weather from Open-Meteo archive (SE4/Malmö).
        wind_intl_hourly:     Hourly wind data for Germany and Denmark.
        market_prices_hourly: Hourly DE/LU + DK2 prices from ENTSO-E.
        nuclear_daily:        Daily SE3 nuclear outage counts from ENTSO-E.

    Returns:
        Daily DataFrame ready for model training.
    """
    prices_daily = aggregate_prices_daily(prices_hourly)
    weather_daily = aggregate_weather_daily(weather_hourly)
    wind_intl_daily = aggregate_international_weather_daily(wind_intl_hourly)
    market_daily = aggregate_market_prices_daily(market_prices_hourly)

    merged = pd.merge(prices_daily, weather_daily, on="date")
    merged = pd.merge(merged, wind_intl_daily, on="date")
    merged = add_se4_price_lags(merged)
    merged = add_market_lag_features(merged, market_daily)
    merged = add_nuclear_outage(merged, nuclear_daily)
    merged = add_residual_load(merged)
    merged = add_hdd_and_temp_gradient(merged)
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["residual_load_lag1"] = merged["residual_load"].shift(1)
    merged = add_weather_variability(merged)
    merged = add_time_features(merged)
    merged = merged.dropna().reset_index(drop=True)

    return merged


def build_forecast_features(
    forecast_hourly: pd.DataFrame,
    wind_intl_forecast_hourly: pd.DataFrame,
    market_daily: pd.DataFrame,
    nuclear_forecast_daily: pd.DataFrame,
    training_daily: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Prepare forecast weather data as model input features.

    Args:
        forecast_hourly:           Hourly weather forecast from Open-Meteo (SE4/Malmö).
        wind_intl_forecast_hourly: Hourly wind forecast for Germany and Denmark.
        market_daily:              Daily DE/DK2 prices from aggregate_market_prices_daily().
        nuclear_forecast_daily:    Daily SE3 planned nuclear outages for the forecast window.
                                   Planned maintenance is published on ENTSO-E in advance,
                                   so this is available for multi-day forecasting.
        training_daily:            Completed training DataFrame; used to seed the rolling
                                   weather variability so that forecast days inherit the
                                   current volatility regime.

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

    # SE4 own price lags — use the last known prices from training data.
    # For multi-day forecasts, the most recent known price is the best available proxy.
    if training_daily is not None and "price_avg" in training_daily.columns:
        sorted_train = training_daily.sort_values("date")
        tail_avg = sorted_train.tail(7)["price_avg"]
        forecast_daily["price_se4_avg_lag1"] = tail_avg.iloc[-1]
        forecast_daily["price_se4_avg_lag2"] = tail_avg.iloc[-2] if len(tail_avg) >= 2 else tail_avg.iloc[-1]
        forecast_daily["price_se4_avg_lag7"] = tail_avg.iloc[0] if len(tail_avg) >= 7 else tail_avg.iloc[-1]

        # Target-specific lags
        forecast_daily["price_se4_min_lag1"] = sorted_train["price_min"].iloc[-1]
        forecast_daily["price_se4_max_lag1"] = sorted_train["price_max"].iloc[-1]

        # Momentum: last known trend direction
        forecast_daily["price_momentum"] = tail_avg.iloc[-1] - tail_avg.iloc[-2] if len(tail_avg) >= 2 else 0.0

        # Volatility: use last known rolling volatility from training data
        forecast_daily["price_volatility_7d"] = sorted_train["price_volatility_7d"].iloc[-1] if "price_volatility_7d" in sorted_train.columns else tail_avg.std()
    else:
        forecast_daily["price_se4_avg_lag1"] = 0.0
        forecast_daily["price_se4_avg_lag2"] = 0.0
        forecast_daily["price_se4_avg_lag7"] = 0.0
        forecast_daily["price_se4_min_lag1"] = 0.0
        forecast_daily["price_se4_max_lag1"] = 0.0
        forecast_daily["price_momentum"] = 0.0
        forecast_daily["price_volatility_7d"] = 0.0

    forecast_daily = add_nuclear_outage(forecast_daily, nuclear_forecast_daily)
    forecast_daily = add_residual_load(forecast_daily)
    forecast_daily = add_hdd_and_temp_gradient(forecast_daily)

    # Residual load lag — use last known value from training data
    if training_daily is not None and "residual_load" in training_daily.columns:
        forecast_daily["residual_load_lag1"] = training_daily["residual_load"].iloc[-1]
    else:
        forecast_daily["residual_load_lag1"] = 0.0

    # Seed rolling variability with the last known values from training data.
    # The current volatility regime is the best proxy for near-term WFU.
    if training_daily is not None:
        forecast_daily["wind_variability"] = training_daily["wind_variability"].iloc[-1]
        forecast_daily["radiation_variability"] = training_daily["radiation_variability"].iloc[-1]
    else:
        forecast_daily["wind_variability"] = 0.0
        forecast_daily["radiation_variability"] = 0.0

    forecast_daily = add_time_features(forecast_daily)

    return forecast_daily
