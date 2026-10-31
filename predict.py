import json
import logging
import sys
from datetime import datetime, timedelta, UTC

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)

from sources.open_meteo import fetch_forecast, fetch_international_wind_forecast, BLEND_MODELS
from sources.entso_e import fetch_nuclear_outages_se3
from sources.swedish_calendar import get_non_workdays
from sources.nordpool import get_dates_with_known_prices
from fetch_data import fetch_training_inputs, fetch_tomorrow_prices, TRAINING_DAYS
from features import (
    build_training_data,
    build_forecast_features,
    complete_swedish_dates,
    aggregate_market_prices_daily,
    aggregate_prices_daily,
)
from model import train, train_band, predict
from evaluate import get_feature_importance
from currency import calculate_eur_to_sek_rate, convert_predictions_to_sek
from ha_client import fetch_addon_value, apply_addon, push_predictions


def _extend_prices(history: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """History with the fresh fetch appended; both carry today, so overlaps keep history."""
    if fresh.empty:
        return history
    combined = pd.concat([history, fresh], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return (combined.drop_duplicates("timestamp", keep="first")
                    .sort_values("timestamp").reset_index(drop=True))


def _sendable_days(forecast_features, forecast_hourly, wind_intl_forecast, known_price_dates):
    """Whole Swedish days only, and none whose prices are already published.

    The forecast is 8 UTC days, so its last Swedish date holds just the 1-2 hours between
    Swedish and UTC midnight (and today misses its first 1-2); aggregated, such a stub is
    not a forecast for that day. See docs/DECISIONS.md "Whole Swedish days only".
    """
    complete_dates = (complete_swedish_dates(forecast_hourly)
                      & complete_swedish_dates(wind_intl_forecast))
    forecast_dates = forecast_features["date"].dt.date
    incomplete = sorted(set(forecast_dates) - complete_dates)
    if incomplete:
        print(f"  → Dropping incomplete Swedish days: {', '.join(map(str, incomplete))}")
    return forecast_features[
        forecast_dates.isin(complete_dates) & ~forecast_dates.isin(known_price_dates)
    ].reset_index(drop=True)


def predict_blend_shadow(build_features, served_dates, known_price_dates, models, band, eur_to_sek_rate):
    """
    The plan-2.3b shadow: the same fitted models and band, fed the hourly mean of the
    three global NWP models (BLEND_MODELS) instead of the served `best_match` forecast.

    Only the weather differs from the served prediction; every frozen lag, price and fuel
    input is the served run's own. Home Assistant receives it as a separate attribute that
    the automation does not read, and the prediction log stores it with variant = 'blend',
    so both can be scored against realised prices on the same days (IMPROVEMENT_PLAN.md 2.3b).

    Returns SEK/kWh predictions for exactly the served dates, or None. A shadow must never
    cost the served push, so any failure is printed and swallowed.
    """
    try:
        print(f"Blend shadow: fetching the 8-day forecast as the mean of {', '.join(BLEND_MODELS)}...")
        forecast_hourly = fetch_forecast(models=BLEND_MODELS)
        wind_intl_forecast = fetch_international_wind_forecast(models=BLEND_MODELS)
        weather_nan = int(forecast_hourly.isna().sum().sum() + wind_intl_forecast.isna().sum().sum())
        if weather_nan:
            print(f"  → Blend shadow skipped: {weather_nan} hourly values missing in every model")
            return None
        features = _sendable_days(build_features(forecast_hourly, wind_intl_forecast),
                                  forecast_hourly, wind_intl_forecast, known_price_dates)
        if list(features["date"].dt.date) != list(served_dates):
            print(f"  → Blend shadow skipped: its days {list(features['date'].dt.date)} "
                  f"are not the served days {list(served_dates)}")
            return None
        predictions_eur = predict(models, features, band=band)
        return convert_predictions_to_sek(predictions_eur, eur_to_sek_rate)
    except Exception as exc:  # noqa: BLE001 -- see docstring: the shadow must never block the push
        print(f"  → Blend shadow skipped: {type(exc).__name__}: {exc}")
        return None


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

    # The training inputs end on today's Swedish day; the forecast's frozen price lags
    # are anchored on tomorrow instead once the auction has published it (see
    # fetch_data.fetch_tomorrow_prices). Training is unaffected.
    prices_tomorrow, market_tomorrow = fetch_tomorrow_prices(today)
    anchor_prices_hourly = _extend_prices(training_inputs["prices_hourly"], prices_tomorrow)
    anchor_market_hourly = _extend_prices(training_inputs["market_prices_hourly"], market_tomorrow)

    market_daily = aggregate_market_prices_daily(anchor_market_hourly)

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

    print("Training models...")
    models = train(training_data)
    feature_importance = get_feature_importance(models)
    print("  → Done")

    print("Calibrating the cheap2h band (low / high / risk)...")
    band = train_band(training_data)
    d = band.diagnostics
    print(f"  → low: holdout {d['n_holdout']} days, symmetric-band coverage "
          f"{d.get('coverage_calibrated', float('nan')):.3f} (nominal {d['nominal_coverage']:.2f}); "
          f"high: calm-day q90 + per-risk-bin correction {d['calm_high_correction_eur_mwh']} EUR/MWh "
          f"(unwarned holdout coverage {d['calm_high_holdout_coverage']:.3f}), coupled level "
          f"{d['coupled_level_ratio']:.2f} × gas cost when risk ≥ {d['warning_threshold']:.2f} "
          f"({d['coupled_share_training']:.0%} of training days coupled)")

    for target, importances in feature_importance.items():
        print(f"\n=== Feature importance ({target} model) ===")
        for feature, importance in importances.items():
            print(f"  {feature:<30} {importance:.4f}")

    # Freshest known SE4 daily prices (extend past the weather-limited training
    # frame, through tomorrow once published) — used to anchor the SE4 price lags
    # at the most recent price.
    se4_prices_daily = aggregate_prices_daily(anchor_prices_hourly)

    def build_features(weather_hourly, wind_hourly):
        return build_forecast_features(
            weather_hourly, wind_hourly, market_daily, nuclear_outages_forecast,
            training_data, training_inputs["ttf_daily"], training_inputs["norway_reservoir_weekly"],
            training_inputs["norway_reservoir_median"], training_inputs["sweden_reservoir_weekly"],
            non_workdays, training_inputs["eua_daily"], se4_prices_daily=se4_prices_daily,
        )

    forecast_features = _sendable_days(build_features(forecast_hourly, wind_intl_forecast),
                                       forecast_hourly, wind_intl_forecast, known_price_dates)

    print(f"Running inference on {len(forecast_features)} days...")
    predictions_eur = predict(models, forecast_features, band=band)
    predictions_raw = convert_predictions_to_sek(predictions_eur, eur_to_sek_rate)

    predictions_blend_raw = predict_blend_shadow(
        build_features, forecast_features["date"].dt.date, known_price_dates, models, band, eur_to_sek_rate)

    print("Fetching electricity price addon from HA...")
    addon_value = fetch_addon_value()
    predictions_with_addon = apply_addon(predictions_raw, addon_value)

    print("\n=== Price predictions (SEK/kWh) ===")
    print(json.dumps(predictions_raw, indent=2))
    if predictions_blend_raw is not None:
        print("\n=== Blend shadow, not served (SEK/kWh) ===")
        print(json.dumps(predictions_blend_raw, indent=2))

    # The band's own calibration travels with the predictions so drift is visible
    # in Home Assistant without re-running anything (docs/MODEL.md "The cheap2h
    # band" lists which numbers to watch). The run computes no walk-forward MAE:
    # see docs/DECISIONS.md "Walk-forward out of the production run".
    model_metrics = {"cheap2h_interval": band.diagnostics}

    print("\nPushing predictions to Home Assistant...")
    push_predictions(predictions_raw, predictions_with_addon, model_metrics, feature_importance,
                     eur_to_sek_rate=eur_to_sek_rate, predictions_blend_raw=predictions_blend_raw)


if __name__ == "__main__":
    main()
