import json
import logging
import sys
from datetime import datetime, timedelta, UTC

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)

from sources.open_meteo import fetch_forecast, fetch_international_wind_forecast
from sources.entso_e import fetch_nuclear_outages_se3
from sources.swedish_calendar import get_non_workdays
from sources.nordpool import get_dates_with_known_prices
from fetch_data import fetch_training_inputs, TRAINING_DAYS
from features import (
    build_training_data,
    build_forecast_features,
    aggregate_market_prices_daily,
    aggregate_prices_daily,
)
from model import train, train_interval, predict
from evaluate import walk_forward_validate, get_feature_importance
from currency import calculate_eur_to_sek_rate, convert_predictions_to_sek
from ha_client import fetch_addon_value, apply_addon, push_predictions


def main():
    today = datetime.now(UTC).date()
    historical_start = today - timedelta(days=TRAINING_DAYS)

    training_inputs = fetch_training_inputs(today)

    print("Fetching 8-day weather forecast...")
    forecast_hourly = fetch_forecast()
    print(f"  → {len(forecast_hourly)} records")

    print("Fetching 8-day wind forecast DE/DK...")
    wind_intl_forecast = fetch_international_wind_forecast()
    print(f"  → {len(wind_intl_forecast)} records")

    market_daily = aggregate_market_prices_daily(training_inputs["market_prices_hourly"])

    forecast_end = today + timedelta(days=10)
    print(f"Fetching planned SE3 nuclear outages {today} → {forecast_end}...")
    nuclear_outages_forecast = fetch_nuclear_outages_se3(
        today.strftime("%Y%m%d"),
        forecast_end.strftime("%Y%m%d")
    )
    print(f"  → {nuclear_outages_forecast['nuclear_outage_se3'].sum()} outage-days planned")

    # Calendar needs to cover the forecast horizon too (fetch_training_inputs
    # only covers the historical range), so it's rebuilt here at the wider
    # range and used for both training and forecast features.
    print("Building Swedish workday calendar...")
    forecast_end_cal = today + timedelta(days=10)
    non_workdays = get_non_workdays(str(historical_start), str(forecast_end_cal))
    print(f"  → {len(non_workdays)} non-workdays in range")

    print("Deriving EUR/SEK exchange rate...")
    eur_to_sek_rate = calculate_eur_to_sek_rate(training_inputs["prices_hourly"], today)

    print("Checking which dates already have official Nordpool prices...")
    known_price_dates = get_dates_with_known_prices()

    print("Building training data...")
    training_data = build_training_data(
        **{**training_inputs, "non_workdays": non_workdays}
    )
    print(f"  → {len(training_data)} days of merged data")

    print("Running walk-forward validation...")
    model_metrics = walk_forward_validate(training_data)

    print("Training models...")
    models = train(training_data)
    feature_importance = get_feature_importance(models)
    print("  → Done")

    print("Calibrating the cheap2h prediction interval...")
    interval = train_interval(training_data)
    d = interval.diagnostics
    print(f"  → holdout {d['n_holdout']} days, coverage "
          f"{d.get('coverage_raw', float('nan')):.3f} raw → "
          f"{d.get('coverage_calibrated', float('nan')):.3f} calibrated "
          f"(nominal {d['nominal_coverage']:.2f}), "
          f"widening ±{d['correction_eur_mwh']:.2f} EUR/MWh")

    for target, importances in feature_importance.items():
        print(f"\n=== Feature importance ({target} model) ===")
        for feature, importance in importances.items():
            print(f"  {feature:<30} {importance:.4f}")

    # Freshest known SE4 daily prices (extend past the weather-limited training
    # frame) — used to anchor the SE4 price lags at the most recent price.
    se4_prices_daily = aggregate_prices_daily(training_inputs["prices_hourly"])

    forecast_features = build_forecast_features(
        forecast_hourly, wind_intl_forecast, market_daily, nuclear_outages_forecast,
        training_data, training_inputs["ttf_daily"], training_inputs["norway_reservoir_weekly"],
        training_inputs["norway_reservoir_median"], training_inputs["sweden_reservoir_weekly"],
        non_workdays, training_inputs["eua_daily"], se4_prices_daily=se4_prices_daily,
    )
    forecast_features = forecast_features[
        ~forecast_features["date"].dt.date.isin(known_price_dates)
    ].reset_index(drop=True)

    print(f"Running inference on {len(forecast_features)} days...")
    predictions_eur = predict(models, forecast_features, interval=interval)
    predictions_raw = convert_predictions_to_sek(predictions_eur, eur_to_sek_rate)

    print("Fetching electricity price addon from HA...")
    addon_value = fetch_addon_value()
    predictions_with_addon = apply_addon(predictions_raw, addon_value)

    print("\n=== Price predictions (SEK/kWh) ===")
    print(json.dumps(predictions_raw, indent=2))

    # The interval's own calibration travels with the metrics so drift is
    # visible in Home Assistant without re-running anything: if
    # coverage_calibrated stops sitting near nominal_coverage, the band has
    # stopped meaning what it says.
    model_metrics = {**model_metrics, "cheap2h_interval": interval.diagnostics}

    print("\nPushing predictions to Home Assistant...")
    push_predictions(predictions_raw, predictions_with_addon, model_metrics, feature_importance)


if __name__ == "__main__":
    main()
