import logging
from datetime import date, timedelta

import pandas as pd

from sources.entso_e import fetch_prices, fetch_market_prices, fetch_nuclear_outages_se3, fetch_reservoir_sweden
from sources.nve import fetch_reservoir_norway, fetch_reservoir_norway_median
from sources.swedish_calendar import get_non_workdays
from sources.open_meteo import (
    fetch_historical, fetch_international_wind_historical,
    fetch_recent, fetch_international_wind_recent,
)
from sources.yahoo_finance import fetch_ttf_prices
from sources.eua_carbon import fetch_eua_prices

logger = logging.getLogger(__name__)


# Open-Meteo archive has a ~5-day lag before data becomes available
WEATHER_ARCHIVE_LAG_DAYS = 5

# Days of history requested from the FORECAST endpoint to top the archive up to
# yesterday. Comfortably more than WEATHER_ARCHIVE_LAG_DAYS so a day where the
# archive lags 6 or 7 instead of 5 still leaves no hole; the splice keeps the
# archive wherever it has published, so over-requesting costs nothing but bytes.
WEATHER_RECENT_TOPUP_DAYS = WEATHER_ARCHIVE_LAG_DAYS + 3

# Daily aggregation in features.py groups by Swedish calendar day, so the
# spliced series must be cut at a Swedish midnight -- otherwise the newest day
# is a partial one (a handful of hours) and would enter training as a full row
# with a badly wrong daily mean.
SWEDISH_TZ = "Europe/Stockholm"

# How many days of historical data to train on
TRAINING_DAYS = 1095


def _splice_recent(archive: pd.DataFrame, recent: pd.DataFrame, today: date) -> pd.DataFrame:
    """
    Extend an archive weather frame with the recent-analysis frame, up to the
    last COMPLETE Swedish day.

    Why this exists: the Open-Meteo archive only reaches today-5, so
    build_training_data() used to produce a frame ending ~5 days before the
    first forecast day and the model was fitted that far behind. Measured cost:
    avg +1.70 / min +0.59 / cheap2h +0.37 EUR/MWh over four evaluation periods,
    and +2.2 / +1.0 / +0.8 in the current regime, positive on 25-28 of 28
    measurements across seven independently fetched caches. See the README
    "Closing the weather-archive lag" section.

    Two rules, both load-bearing:

    1. The archive WINS wherever it has published. Only rows strictly after its
       last timestamp are taken from `recent`, so the three years of archive
       data the model is mostly trained on are untouched, and re-running on a
       later day does not rewrite history as the archive catches up.
    2. The result is cut at Swedish midnight of `today`, which is the end of
       yesterday's Swedish day. `recent` runs to the current hour, and an
       incomplete final day would otherwise be aggregated as if it were whole.

    Args:
        archive: Frame from fetch_historical / fetch_international_wind_historical.
        recent:  Frame from fetch_recent / fetch_international_wind_recent.
        today:   The "as of" date, the same one fetch_training_inputs was called with.

    Returns:
        Concatenation of both, timestamp-sorted, with no duplicate timestamps.
    """
    if recent is None or recent.empty:
        return archive
    if archive is None or archive.empty:
        return recent

    archive = archive.copy()
    recent = recent.copy()
    archive["timestamp"] = pd.to_datetime(archive["timestamp"], utc=True)
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)

    cutoff = pd.Timestamp(today, tz=SWEDISH_TZ).tz_convert("UTC")
    tail = recent[(recent["timestamp"] > archive["timestamp"].max())
                  & (recent["timestamp"] < cutoff)]
    tail = tail.reindex(columns=archive.columns)

    if tail.empty:
        return archive

    return (pd.concat([archive, tail], ignore_index=True)
              .sort_values("timestamp")
              .reset_index(drop=True))


def _fetch_recent_or_none(fetch_fn, label: str):
    """
    Call a recent-analysis fetch, returning None instead of raising.

    The top-up is an accuracy improvement, not a correctness requirement: with
    it the frame reaches yesterday, without it the pipeline is exactly what it
    was before. A scheduled Actions run must not fail outright because
    Open-Meteo's forecast endpoint had a bad minute, so a failure here degrades
    to the archive-only behaviour and says so loudly.
    """
    try:
        return fetch_fn()
    except Exception as e:
        logger.warning(
            "  WARNING: %s top-up failed (%s: %s) - falling back to archive only, "
            "so the training frame will end ~%d days back.",
            label, type(e).__name__, e, WEATHER_ARCHIVE_LAG_DAYS,
        )
        return None


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

    # Top the archive up to yesterday from the forecast endpoint's recent
    # analysis -- see _splice_recent for why this is worth ~0.4-2.2 EUR/MWh.
    print(f"Topping weather up to {today - timedelta(days=1)} "
          f"(archive lags {WEATHER_ARCHIVE_LAG_DAYS} days)...")
    weather_hourly = _splice_recent(
        weather_hourly,
        _fetch_recent_or_none(lambda: fetch_recent(WEATHER_RECENT_TOPUP_DAYS),
                              "SE4 weather"),
        today,
    )
    wind_intl_hourly = _splice_recent(
        wind_intl_hourly,
        _fetch_recent_or_none(
            lambda: fetch_international_wind_recent(WEATHER_RECENT_TOPUP_DAYS),
            "DE/DK wind"),
        today,
    )
    print(f"  -> weather now {len(weather_hourly)} records "
          f"(to {weather_hourly['timestamp'].max()}), "
          f"wind {len(wind_intl_hourly)} records")

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
