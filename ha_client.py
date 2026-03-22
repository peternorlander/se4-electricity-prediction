import os
import requests
from datetime import datetime, timezone


SENSOR_ENTITY_ID = "sensor.electricity_price_predictions"


def _get_headers() -> dict:
    token = os.environ["HA_TOKEN"]
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


def _get_base_url() -> str:
    return os.environ["HA_URL"].rstrip("/")


def fetch_addon_value() -> float:
    """
    Fetch the electricity price addon value from HA.

    Returns:
        Addon value in SEK/kWh as float.
    """
    url = f"{_get_base_url()}/api/states/input_number.electricity_price_addon"
    response = requests.get(url, headers=_get_headers(), timeout=10)
    response.raise_for_status()

    value = float(response.json()["state"])
    print(f"  → Fetched electricity_price_addon: {value} SEK/kWh")

    return value


def apply_addon(predictions_raw: dict, addon_value: float) -> dict:
    """
    Apply 5% markup and fixed addon to raw SEK/kWh predictions.

    Formula: adjusted = raw * 1.05 + addon_value

    Args:
        predictions_raw: Dict keyed by date string with min/avg/max in SEK/kWh.
        addon_value: Fixed addon in SEK/kWh from input_number.electricity_price_addon.

    Returns:
        Same structure with addon applied.
    """
    adjusted = {}

    for day, values in predictions_raw.items():
        adjusted[day] = {
            "min": round(values["min"] * 1.05 + addon_value, 4),
            "avg": round(values["avg"] * 1.05 + addon_value, 4),
            "max": round(values["max"] * 1.05 + addon_value, 4)
        }

    return adjusted


def _to_list(predictions: dict) -> list:
    return [
        {
            "date": date_str,
            "min": values["min"],
            "avg": values["avg"],
            "max": values["max"]
        }
        for date_str, values in predictions.items()
    ]


def push_predictions(
    predictions_raw: dict,
    predictions_with_addon: dict,
    model_metrics: dict = None,
    feature_importance: dict = None,
) -> None:
    """
    Push price predictions to a HA sensor via the REST API.

    Creates or updates sensor.electricity_price_predictions with state set to
    the UTC timestamp of when the prediction was made.

    Args:
        predictions_raw:        Dict keyed by date string with min/avg/max in SEK/kWh.
        predictions_with_addon: Same structure with 5% markup and addon applied.
        model_metrics:          Walk-forward MAE stats from walk_forward_validate().
        feature_importance:     Feature importance dict from get_feature_importance().
    """
    predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]

    attributes = {
        "predictions_raw": _to_list(predictions_raw),
        "predictions_with_addon": _to_list(predictions_with_addon),
        "friendly_name": "Electricity Price Predictions",
        "unit_of_measurement": "SEK/kWh"
    }

    if model_metrics is not None:
        attributes["mae_overall"] = model_metrics["mae_overall"]
        attributes["mae_min"] = model_metrics["mae_min"]
        attributes["mae_avg"] = model_metrics["mae_avg"]
        attributes["mae_max"] = model_metrics["mae_max"]

    if feature_importance is not None:
        attributes["feature_importance"] = feature_importance

    payload = {
        "state": predicted_at,
        "attributes": attributes,
    }

    url = f"{_get_base_url()}/api/states/{SENSOR_ENTITY_ID}"
    response = requests.post(url, headers=_get_headers(), json=payload, timeout=10)
    response.raise_for_status()

    print(f"  → Pushed predictions to HA: {SENSOR_ENTITY_ID}")
    print(f"  → State set to: {predicted_at}")
