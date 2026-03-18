import json
from datetime import datetime, timedelta, timezone

import pytz

from sources.entso_e import fetch_prices
from sources.open_meteo import fetch_historical, fetch_forecast
from sources.nordpool import get_dates_with_known_prices
from features import build_training_data, build_forecast_features
from model import train, predict
from currency import calculate_eur_to_sek_rate, convert_predictions_to_sek
from ha_client import fetch_electricity_addon, push_predictions


# Open-Meteo archive has a ~5-day lag before data becomes available
WEATHER_ARCHIVE_LAG_DAYS = 5

# How many days of historical data to train on
TRAINING_DAYS = 365


def apply_addon(predictions_sek: dict, addon: float) -> dict:
    """
    Apply 5% markup and fixed addon to raw SEK/kWh predictions.

    Args:
        predictions_sek: Dict keyed by date string with min/mean/max in SEK/kWh.
        addon: Fixed addon in SEK/kWh from input_number.electricity_price_addon.

    Returns:
        Same structure with addon applied.
    """
    result = {}

    for day, values in predictions_sek.items():
        result[day] = {
            "min": round(values["min"] * 1.05 + addon, 4),
            "mean": round(values["mean"] * 1.05 + addon, 4),
            "max": round(values["max"] * 1.05 + addon, 4)
        }

    return result


def main():
    today = datetime.utcnow().date()
    historical_start = today - timedelta(days=TRAINING_DAYS)
    weather_hist_end = today - timedelta(days=WEATHER_ARCHIVE_LAG_DAYS)

    print(f"Fetching ENTSO-E prices {historical_start} -> {today}...")
    prices_hourly = fetch_prices(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  -> {len(prices_hourly)} records")

    print(f"Fetching historical weather {historical_start} -> {weather_hist_end}...")
    weather_hourly = fetch_historical(
        str(historical_start),
        str(weather_hist_end)
    )
    print(f"  -> {len(weather_hourly)} records")

    print("Fetching 8-day weather forecast...")
    forecast_hourly = fetch_forecast()
    print(f"  -> {len(forecast_hourly)} records")

    print("Deriving EUR/SEK exchange rate...")
    eur_to_sek_rate = calculate_eur_to_sek_rate(prices_hourly)

    print("Checking which upcoming dates already have known Nordpool prices...")
    known_dates = get_dates_with_known_prices(days_ahead=8)

    print("Building training data...")
    training_data = build_training_data(prices_hourly, weather_hourly)
    print(f"  -> {len(training_data)} days of merged data")

    print("Training models...")
    models = train(training_data)
    print("  -> Done")

    forecast_features = build_forecast_features(forecast_hourly)
    forecast_features = forecast_features[
        forecast_features["date"].dt.date.apply(
            lambda d: d > today and d not in known_dates
        )
    ].reset_index(drop=True)

    print(f"Running inference on {len(forecast_features)} days...")
    predictions_eur = predict(models, forecast_features)
    predictions_raw = convert_predictions_to_sek(predictions_eur, eur_to_sek_rate)

    print("Fetching electricity addon from HA...")
    addon = fetch_electricity_addon()
    print(f"  -> addon: {addon} SEK/kWh")

    predictions_with_addon = apply_addon(predictions_raw, addon)

    stockholm_tz = pytz.timezone("Europe/Stockholm")
    timestamp = datetime.now(stockholm_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]

    print("\n=== Price predictions (SEK/kWh) ===")
    print(json.dumps(predictions_raw, indent=2))

    print("\nPushing to Home Assistant...")
    push_predictions(predictions_raw, predictions_with_addon, timestamp)


if __name__ == "__main__":
    main()
