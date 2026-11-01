"""
Run status to Home Assistant: `sensor.electricity_price_predictions_run`
(IMPROVEMENT_PLAN.md 2.12).

The predictions sensor is only written by a run that succeeds, so a failed run leaves
it as it was and nothing says the numbers are old. This sensor is written by every
run that gets far enough to start Python -- `ok` or `failed`, with when, why and
which commit -- and by the workflow itself when a run fails before predict.py starts
(dependency install, tests). The automation's fallback reads it: `failed`, or a
`last_success_at` more than a day old, means "charge now".

Standard library only, so the workflow's failure step can call it after a failed
`pip install`:

    python run_status.py failed --stage workflow
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

ENTITY_ID = "sensor.electricity_price_predictions_run"
MAX_ERROR_CHARS = 300


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request(method: str, path: str, payload: dict = None):
    request = urllib.request.Request(
        os.environ["HA_URL"].rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        method=method,
        headers={"Authorization": f"Bearer {os.environ['HA_TOKEN']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
    return json.loads(body) if body else None


def _previous_success():
    """last_success_at of the sensor as it stands, so a failure carries it forward."""
    try:
        return (_request("GET", f"/api/states/{ENTITY_ID}") or {}).get("attributes", {}).get("last_success_at")
    except Exception:  # noqa: BLE001 -- a missing sensor (first run) or HA hiccup is not an error here
        return None


def status_payload(status: str, started_at: str = None, error: str = None, stage: str = None,
                   finished_at: str = None, previous_success: str = None) -> dict:
    """The sensor's state and attributes. A fact that is not available is left out."""
    finished_at = finished_at or _now()
    attributes = {"friendly_name": "Electricity Price Predictions run", "finished_at": finished_at}
    if started_at:
        attributes["started_at"] = started_at
    last_success = finished_at if status == "ok" else previous_success
    if last_success:
        attributes["last_success_at"] = last_success
    if error:
        attributes["error"] = error[:MAX_ERROR_CHARS]
    if stage:
        attributes["stage"] = stage
    if os.environ.get("GITHUB_SHA"):
        attributes["git_sha"] = os.environ["GITHUB_SHA"]
    if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        attributes["run_url"] = f"{server}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    return {"state": status, "attributes": attributes}


def push_run_status(status: str, started_at: str = None, error: str = None, stage: str = None) -> bool:
    """
    Write the run sensor. Never raises: reporting a failure must not replace the
    failure itself, and a missing HA_URL / HA_TOKEN (a local run) just skips it.
    Returns whether the push went through.
    """
    if not os.environ.get("HA_URL") or not os.environ.get("HA_TOKEN"):
        return False
    try:
        previous = None if status == "ok" else _previous_success()
        _request("POST", f"/api/states/{ENTITY_ID}",
                 status_payload(status, started_at, error, stage, previous_success=previous))
        print(f"  → Run status pushed to HA: {ENTITY_ID} = {status}")
        return True
    except Exception as exc:  # noqa: BLE001 -- see docstring
        print(f"  → WARNING: could not push run status '{status}': {type(exc).__name__}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Push the run status sensor to Home Assistant.")
    parser.add_argument("status", choices=("ok", "failed"))
    parser.add_argument("--stage", default=None, help="Where the run stopped, e.g. 'workflow'.")
    parser.add_argument("--error", default=None)
    args = parser.parse_args()
    push_run_status(args.status, error=args.error, stage=args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
