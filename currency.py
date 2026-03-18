import pandas as pd
from datetime import date

from sources.nordpool import fetch_today_mean_sek


def calculate_eur_to_sek_rate(entso_e_prices_df: pd.DataFrame) -> float:
    """
    Derive the EUR/SEK exchange rate used by Nordpool today.

    Compares today's daily average from Nordpool (SEK/MWh)
    with today's daily average from ENTSO-E (EUR/MWh) to back-calculate
    the exact exchange rate Nordpool applied.

    Args:
        entso_e_prices_df: DataFrame with columns timestamp (UTC) and price_eur_mwh.

    Returns:
        Exchange rate as float (SEK per EUR).
    """
    nordpool_mean_sek = fetch_today_mean_sek()

    today = date.today()
    df = entso_e_prices_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert("Europe/Stockholm")
    df["date"] = df["timestamp_local"].dt.date

    today_df = df[df["date"] == today]
    entso_mean_eur = today_df["price_eur_mwh"].mean()

    rate = nordpool_mean_sek / entso_mean_eur

    print(f"  Nordpool today mean: {nordpool_mean_sek:.2f} SEK/MWh")
    print(f"  ENTSO-E today mean:  {entso_mean_eur:.2f} EUR/MWh")
    print(f"  Derived EUR/SEK rate: {rate:.4f}")

    return rate


def convert_predictions_to_sek(predictions: dict, eur_to_sek_rate: float) -> dict:
    """
    Convert predicted prices from EUR/MWh to SEK/kWh.

    Args:
        predictions: Dict keyed by date string with min/avg/max in EUR/MWh.
        eur_to_sek_rate: Exchange rate (SEK per EUR).

    Returns:
        Same structure with values in SEK/kWh.
    """
    converted = {}

    for day, values in predictions.items():
        converted[day] = {
            "min": round(values["min"] * eur_to_sek_rate / 1000, 4),
            "avg": round(values["avg"] * eur_to_sek_rate / 1000, 4),
            "max": round(values["max"] * eur_to_sek_rate / 1000, 4)
        }

    return converted
