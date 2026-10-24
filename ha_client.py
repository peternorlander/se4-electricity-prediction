import math
import os
import requests
from datetime import datetime, timezone

from model import UNITLESS_KEYS


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
        predictions_raw: Dict keyed by date string with price targets in SEK/kWh
                         (min/avg/max/cheap2h). Keys in model.UNITLESS_KEYS (the
                         cheap2h_risk probability) are passed through unchanged.
        addon_value: Fixed addon in SEK/kWh from input_number.electricity_price_addon.

    Returns:
        Same structure with addon applied.
    """
    adjusted = {}

    for day, values in predictions_raw.items():
        adjusted[day] = {
            target: price if target in UNITLESS_KEYS else round(price * 1.05 + addon_value, 4)
            for target, price in values.items()
        }

    return adjusted


def _to_list(predictions: dict) -> list:
    return [
        {"date": date_str, **values}
        for date_str, values in predictions.items()
    ]


def _run_provenance(eur_to_sek_rate: float = None) -> dict:
    """
    Run-level facts that go next to the predictions in the HA sensor, so the
    prediction log (Node-RED -> electricity_price_predictions_log) can say which
    code and which exchange rate produced a row.

    eur_to_sek_rate lets a logged SEK/kWh value be turned back into the model's
    EUR/MWh exactly; git_sha (GITHUB_SHA, set on Actions runners) lets a later
    before/after comparison group rows by code instead of guessing from dates.
    A fact that is not available is left out, not sent as null: a local run
    has no GITHUB_SHA.
    """
    provenance = {}

    if eur_to_sek_rate is not None and math.isfinite(eur_to_sek_rate):
        provenance["eur_to_sek_rate"] = round(float(eur_to_sek_rate), 5)

    git_sha = os.environ.get("GITHUB_SHA")
    if git_sha:
        provenance["git_sha"] = git_sha

    return provenance


def push_predictions(
    predictions_raw: dict,
    predictions_with_addon: dict,
    model_metrics: dict = None,
    feature_importance: dict = None,
    eur_to_sek_rate: float = None,
) -> None:
    """
    Push price predictions to a HA sensor via the REST API.

    Creates or updates sensor.electricity_price_predictions with state set to
    the UTC timestamp of when the prediction was made.

    Args:
        predictions_raw:        Dict keyed by date string with min/avg/max/cheap2h in SEK/kWh.
        predictions_with_addon: Same structure with 5% markup and addon applied.
        model_metrics:          Extra attributes merged in as-is; predict.py sends the
                                band's calibration as `cheap2h_interval`.
        feature_importance:     Dict keyed by target name from get_feature_importance().
        eur_to_sek_rate:        SEK per EUR the run converted with; sent as the
                                `eur_to_sek_rate` attribute next to `git_sha`.
    """
    predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4]

    attributes = {
        "predictions_raw": _to_list(predictions_raw),
        "predictions_with_addon": _to_list(predictions_with_addon),
        "friendly_name": "Electricity Price Predictions",
        **_run_provenance(eur_to_sek_rate),
    }

    if model_metrics is not None:
        attributes.update(model_metrics)

    if feature_importance is not None:
        for target, importances in feature_importance.items():
            attributes[f"feature_importance_{target}"] = importances

    payload = {
        "state": predicted_at,
        "attributes": attributes,
    }

    url = f"{_get_base_url()}/api/states/{SENSOR_ENTITY_ID}"
    response = requests.post(url, headers=_get_headers(), json=payload, timeout=10)
    response.raise_for_status()

    print(f"  → Pushed predictions to HA: {SENSOR_ENTITY_ID}")
    print(f"  → State set to: {predicted_at}")
