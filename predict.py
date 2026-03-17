import json
from datetime import datetime, timedelta

from sources.entso_e import fetch_prices
from sources.open_meteo import fetch_historical, fetch_forecast
from features import build_training_data, build_forecast_features
from model import train, predict
from currency import calculate_eur_to_sek_rate, convert_predictions_to_sek


# Open-Meteo archive has a ~5-day lag before data becomes available
WEATHER_ARCHIVE_LAG_DAYS = 5

# How many days of historical data to train on
TRAINING_DAYS = 365


def main():
    today = datetime.utcnow().date()
    historical_start = today - timedelta(days=TRAINING_DAYS)
    weather_hist_end = today - timedelta(days=WEATHER_ARCHIVE_LAG_DAYS)
    tomorrow = today + timedelta(days=1)

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

    print("Deriving EUR/SEK exchange rate...")
    eur_to_sek_rate = calculate_eur_to_sek_rate(prices_hourly)

    print("Building training data...")
    training_data = build_training_data(prices_hourly, weather_hourly)
    print(f"  → {len(training_data)} days of merged data")

    print("Training models...")
    models = train(training_data)
    print("  → Done")

    forecast_features = build_forecast_features(forecast_hourly)
    forecast_features = forecast_features[
        forecast_features["date"].dt.date >= tomorrow
    ].reset_index(drop=True)

    print("Running inference...")
    predictions_eur = predict(models, forecast_features)
    predictions_sek = convert_predictions_to_sek(predictions_eur, eur_to_sek_rate)

    print("\n=== Price predictions (SEK/kWh) ===")
    print(json.dumps(predictions_sek, indent=2))


if __name__ == "__main__":
    main()
