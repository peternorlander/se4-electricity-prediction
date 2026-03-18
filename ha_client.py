import os
import requests


def _get_headers() -> dict:
    token = os.environ["HA_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


def _get_base_url() -> str:
    return os.environ["HA_URL"].rstrip("/")


def fetch_electricity_addon() -> float:
    """
    Fetch the electricity price addon value from HA.

    Returns:
        Addon value in SEK/kWh as float.
    """
    url = f"{_get_base_url()}/api/states/input_number.electricity_price_addon"
    response = requests.get(url, headers=_get_headers(), timeout=10)
    response.raise_for_status()

    state = response.json()["state"]

    return float(state)


def push_predictions(predictions_raw: dict, predictions_with_addon: dict, timestamp: str) -> None:
    """
    Push price predictions to a HA sensor via the REST API.

    Creates or updates sensor.electricity_price_predictions with:
      - state: ISO timestamp of when the prediction was made
      - attributes:
          predictions_raw: list of {date, min, mean, max} in SEK/kWh
          predictions_with_addon: same structure with addon applied

    Args:
        predictions_raw: Dict keyed by date string with min/mean/max in SEK/kWh.
        predictions_with_addon: Same structure with addon and VAT applied.
        timestamp: ISO formatted datetime string for the state value.
    """
    def _to_list(predictions: dict) -> list:
        return [
            {
                "date": date_str,
                "min": values["min"],
                "mean": values["mean"],
                "max": values["max"]
            }
            for date_str, values in predictions.items()
        ]

    payload = {
        "state": timestamp,
        "attributes": {
            "predictions_raw": _to_list(predictions_raw),
            "predictions_with_addon": _to_list(predictions_with_addon),
            "friendly_name": "Electricity Price Predictions",
            "unit_of_measurement": "SEK/kWh"
        }
    }

    url = f"{_get_base_url()}/api/states/sensor.electricity_price_predictions"
    response = requests.post(url, headers=_get_headers(), json=payload, timeout=10)
    response.raise_for_status()

    print(f"Pushed predictions to HA: {response.status_code}")
