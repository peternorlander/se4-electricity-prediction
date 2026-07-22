import pandas as pd
from datetime import date, timedelta

from sources.nordpool import fetch_mean_sek_for_date


# How many days back from `today` to look for a delivery day present in both
# ENTSO-E and Nordpool before giving up.
_RATE_LOOKBACK_DAYS = 7


def calculate_eur_to_sek_rate(entso_e_prices_df: pd.DataFrame, today: date) -> float:
    """
    Derive the EUR/SEK exchange rate that Nordpool applied.

    Nordpool converts the EUR day-ahead price to SEK with a single daily ECB
    fixing, so dividing Nordpool's SEK daily mean by ENTSO-E's EUR daily mean
    for the *same delivery day* recovers that day's exact FX rate. Pairing two
    different days would instead yield FX * (price_dayA / price_dayB) — a wrong
    rate, not merely a stale one — so both sides must sit on one day.

    Today's ENTSO-E rows are frequently not in the fetched frame yet (the fetch
    window ends exclusive of today, and a run straddling midnight can look past
    the last fetched day). Previously that produced a NaN mean → NaN rate →
    every prediction pushed to Home Assistant as NaN. To stay robust we walk
    back from `today` and use the most recent delivery day present in BOTH the
    ENTSO-E frame and Nordpool. FX is near-constant day to day, so a one-day-old
    rate is fine; a NaN or mismatched-day rate is not.

    Args:
        entso_e_prices_df: DataFrame with columns timestamp (UTC) and price_eur_mwh.
        today: The reference date computed once in predict.main(). Passing it in
            (rather than re-deriving a naive date.today() here) removes the
            mid-run rollover race and the UTC-vs-local split.

    Returns:
        Exchange rate as float (SEK per EUR).
    """
    df = entso_e_prices_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert("Europe/Stockholm")
    df["date"] = df["timestamp_local"].dt.date

    for offset in range(_RATE_LOOKBACK_DAYS + 1):
        delivery_day = today - timedelta(days=offset)

        entso_mean_eur = df.loc[df["date"] == delivery_day, "price_eur_mwh"].mean()
        if pd.isna(entso_mean_eur) or entso_mean_eur == 0:
            continue  # ENTSO-E has no usable prices for this day yet

        nordpool_mean_sek = fetch_mean_sek_for_date(delivery_day)
        if nordpool_mean_sek is None:
            continue  # Nordpool has not published this day

        rate = nordpool_mean_sek / entso_mean_eur

        label = "today" if offset == 0 else f"{offset}d ago ({delivery_day})"
        print(f"  Rate delivery day:    {label}")
        print(f"  Nordpool mean:        {nordpool_mean_sek:.2f} SEK/MWh")
        print(f"  ENTSO-E mean:         {entso_mean_eur:.2f} EUR/MWh")
        print(f"  Derived EUR/SEK rate: {rate:.4f}")

        return rate

    raise ValueError(
        f"No delivery day within the last {_RATE_LOOKBACK_DAYS} days is present "
        f"in both the ENTSO-E frame and Nordpool; cannot derive EUR/SEK rate."
    )


def convert_predictions_to_sek(predictions: dict, eur_to_sek_rate: float) -> dict:
    """
    Convert predicted prices from EUR/MWh to SEK/kWh.

    Args:
        predictions: Dict keyed by date string with price targets in EUR/MWh
                     (min/avg/max/cheap2h).
        eur_to_sek_rate: Exchange rate (SEK per EUR).

    Returns:
        Same structure with values in SEK/kWh.
    """
    converted = {}

    for day, values in predictions.items():
        converted[day] = {
            target: round(price * eur_to_sek_rate / 1000, 4)
            for target, price in values.items()
        }

    return converted
