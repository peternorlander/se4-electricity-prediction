from datetime import date, timedelta

from sources.entso_e import fetch_prices, fetch_market_prices, fetch_nuclear_outages_se3, fetch_reservoir_sweden
from sources.nve import fetch_reservoir_norway, fetch_reservoir_norway_median
from sources.swedish_calendar import get_non_workdays
from sources.open_meteo import fetch_historical, fetch_international_wind_historical
from sources.yahoo_finance import fetch_ttf_prices
from sources.eua_carbon import fetch_eua_prices


# Open-Meteo archive has a ~5-day lag before data becomes available
WEATHER_ARCHIVE_LAG_DAYS = 5

# How many days of historical data to train on
TRAINING_DAYS = 1095


def fetch_training_inputs(today: date) -> dict:
    """
    Fetch every input features.build_training_data() needs.

    Split out of predict.py so the same fetch logic is reusable by the A/B
    backtest flow (ab_test.py / ab.snapshot), which caches the result to disk
    and replays it across many walk-forward shifts without re-fetching. See the
    README "A/B Backtest Flow" section. predict.py's forecast-side fetches (weather/wind
    forecast, planned nuclear outages, the wider-range calendar covering the
    forecast horizon, market_daily aggregation, EUR/SEK rate, known-price
    dates) are not needed for training and stay in predict.py.

    Returned dict keys match build_training_data()'s parameter names exactly,
    so callers can invoke build_training_data(**fetch_training_inputs(today)).

    Args:
        today: The "as of" date; historical fetch windows are derived from it.

    Returns:
        Dict with keys: prices_hourly, weather_hourly, wind_intl_hourly,
        market_prices_hourly, nuclear_daily, ttf_daily, norway_reservoir_weekly,
        norway_reservoir_median, sweden_reservoir_weekly, non_workdays, eua_daily.
    """
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

    print(f"Fetching historical wind data DE/DK {historical_start} -> {weather_hist_end}...")
    wind_intl_hourly = fetch_international_wind_historical(
        str(historical_start),
        str(weather_hist_end)
    )
    print(f"  -> {len(wind_intl_hourly)} records")

    print(f"Fetching DE/DK2 market prices {historical_start} -> {today}...")
    market_prices_hourly = fetch_market_prices(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  -> {len(market_prices_hourly)} records")

    print(f"Fetching TTF gas prices {historical_start} -> {today}...")
    ttf_daily = fetch_ttf_prices(
        str(historical_start),
        str(today)
    )
    print(f"  -> {len(ttf_daily)} TTF price records")

    print(f"Fetching EU ETS carbon prices {historical_start} -> {today}...")
    eua_daily = fetch_eua_prices(
        str(historical_start),
        str(today)
    )
    print(f"  -> {len(eua_daily)} EUA price records")

    print(f"Fetching SE3 nuclear outages {historical_start} -> {today}...")
    nuclear_daily = fetch_nuclear_outages_se3(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  -> {nuclear_daily['nuclear_outage_se3'].sum()} outage-days found")

    print(f"Fetching NVE reservoir data {historical_start} -> {today}...")
    norway_reservoir_weekly = fetch_reservoir_norway(str(historical_start), str(today))
    print(f"  -> {len(norway_reservoir_weekly)} weekly records")

    print("Fetching NVE 20-year reservoir median...")
    norway_reservoir_median = fetch_reservoir_norway_median()
    print(f"  -> {len(norway_reservoir_median)} week entries")

    print(f"Fetching Sweden reservoir data {historical_start} -> {today}...")
    sweden_reservoir_weekly = fetch_reservoir_sweden(
        historical_start.strftime("%Y%m%d"),
        today.strftime("%Y%m%d")
    )
    print(f"  -> {len(sweden_reservoir_weekly)} weekly records")

    print("Building Swedish workday calendar...")
    non_workdays = get_non_workdays(str(historical_start), str(today))
    print(f"  -> {len(non_workdays)} non-workdays in range")

    return {
        "prices_hourly": prices_hourly,
        "weather_hourly": weather_hourly,
        "wind_intl_hourly": wind_intl_hourly,
        "market_prices_hourly": market_prices_hourly,
        "nuclear_daily": nuclear_daily,
        "ttf_daily": ttf_daily,
        "norway_reservoir_weekly": norway_reservoir_weekly,
        "norway_reservoir_median": norway_reservoir_median,
        "sweden_reservoir_weekly": sweden_reservoir_weekly,
        "non_workdays": non_workdays,
        "eua_daily": eua_daily,
    }
