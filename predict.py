import json
import logging
import sys
from datetime import datetime, timedelta, UTC

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)

from sources.entso_e import fetch_prices, fetch_market_prices, fetch_nuclear_outages_se3
from sources.open_meteo import (
    fetch_historical,
    fetch_forecast,
    fetch_international_wind_historical,
    fetch_international_wind_forecast,
)
from sources.nordpool import get_dates_with_known_prices
from features import (
    build_training_data,
    build_forecast_features,
    aggregate_market_prices_daily,
)
from model import train, predict
from evaluate import walk_forward_validate, get_feature_importance
from currency import calculate_eur_to_sek_rate, convert_predictions_to_sek
from ha_client import fetch_addon_value, apply_addon, push_predictions


# Open-Meteo archive has a ~5-day lag before data becomes available
WEATHER_ARCHIVE_LAG_DAYS = 5

# How many days of historical data to train on
TRAINING_DAYS = 1095


def main():
    today = datetime.now(UTC).date()
    historical_start = today - timedelta(days=TRAINING_DAYS)
    weather_hist_end = today - timedelta(days=WEATHER_ARCHIVE_LAG_DAYS)

    print(f"Fetching ENTSO-E prices {historical_start} → {today}...")
    prices_hourly = fetch_prices(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  → {len(prices_hourly)} records")

    print(f"Fetching historical weather {historical_start} → {weather_hist_end}...")
    weather_hourly = fetch_historical(
        str(historical_start),
        str(weather_hist_end)
    )
    print(f"  → {len(weather_hourly)} records")

    print("Fetching 8-day weather forecast...")
    forecast_hourly = fetch_forecast()
    print(f"  → {len(forecast_hourly)} records")

    print(f"Fetching historical wind data DE/DK {historical_start} → {weather_hist_end}...")
    wind_intl_hourly = fetch_international_wind_historical(
        str(historical_start),
        str(weather_hist_end)
    )
    print(f"  → {len(wind_intl_hourly)} records")

    print("Fetching 8-day wind forecast DE/DK...")
    wind_intl_forecast = fetch_international_wind_forecast()
    print(f"  → {len(wind_intl_forecast)} records")

    print(f"Fetching DE/DK2 market prices {historical_start} → {today}...")
    market_prices_hourly = fetch_market_prices(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  → {len(market_prices_hourly)} records")
    market_daily = aggregate_market_prices_daily(market_prices_hourly)

    print(f"Fetching SE3 nuclear outages {historical_start} → {today}...")
    nuclear_outages = fetch_nuclear_outages_se3(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  → {nuclear_outages['nuclear_outage_se3'].sum()} outage-days found")

    forecast_end = today + timedelta(days=10)
    print(f"Fetching planned SE3 nuclear outages {today} → {forecast_end}...")
    nuclear_outages_forecast = fetch_nuclear_outages_se3(
        today.strftime("%Y%m%d"),
        forecast_end.strftime("%Y%m%d")
    )
    print(f"  → {nuclear_outages_forecast['nuclear_outage_se3'].sum()} outage-days planned")

    print("Deriving EUR/SEK exchange rate...")
    eur_to_sek_rate = calculate_eur_to_sek_rate(prices_hourly)

    print("Checking which dates already have official Nordpool prices...")
    known_price_dates = get_dates_with_known_prices()

    print("Building training data...")
    training_data = build_training_data(prices_hourly, weather_hourly, wind_intl_hourly, market_prices_hourly, nuclear_outages)
    print(f"  → {len(training_data)} days of merged data")

    print("Running walk-forward validation...")
    model_metrics = walk_forward_validate(training_data)

    print("Training models...")
    models = train(training_data)
    feature_importance = get_feature_importance(models)
    print("  → Done")

    print("\n=== Feature importance ===")
    for feature, importance in feature_importance.items():
        print(f"  {feature:<30} {importance:.4f}")

    forecast_features = build_forecast_features(forecast_hourly, wind_intl_forecast, market_daily, nuclear_outages_forecast, training_data)
    forecast_features = forecast_features[
        ~forecast_features["date"].dt.date.isin(known_price_dates)
    ].reset_index(drop=True)

    print(f"Running inference on {len(forecast_features)} days...")
    predictions_eur = predict(models, forecast_features)
    predictions_raw = convert_predictions_to_sek(predictions_eur, eur_to_sek_rate)

    print("Fetching electricity price addon from HA...")
    addon_value = fetch_addon_value()
    predictions_with_addon = apply_addon(predictions_raw, addon_value)

    print("\n=== Price predictions (SEK/kWh) ===")
    print(json.dumps(predictions_raw, indent=2))

    print("\nPushing predictions to Home Assistant...")
    push_predictions(predictions_raw, predictions_with_addon, model_metrics, feature_importance)


if __name__ == "__main__":
    main()
