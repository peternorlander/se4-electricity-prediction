import time
import logging
import requests


_TIMEOUT = 120
_RETRY_WAITS = (4, 16, 32, 64)  # seconds between attempts 1→2, 2→3, 3→4, 4→5

logger = logging.getLogger(__name__)


def get_with_retry(url: str, params: dict) -> requests.Response:
    """
    Perform a GET request with exponential backoff retry on timeout, connection
    errors, or 5xx server errors (e.g. 503 Service Unavailable).

    Retries up to 4 times, waiting 4s, 16s, 32s, 64s between attempts.
    Does NOT call raise_for_status for non-5xx errors — callers handle those.
    Raises the last exception (or HTTPError for 5xx) if all retries are exhausted.
    """
    last_exc = None
    max_attempts = len(_RETRY_WAITS) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT)
            if response.status_code >= 500:
                last_exc = requests.exceptions.HTTPError(
                    f"{response.status_code} Server Error", response=response
                )
                if attempt < max_attempts:
                    wait = _RETRY_WAITS[attempt - 1]
                    logger.warning(
                        "Request to %s returned %d (attempt %d/%d) — retrying in %ds",
                        url, response.status_code, attempt, max_attempts, wait,
                    )
                    time.sleep(wait)
                continue
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < max_attempts:
                wait = _RETRY_WAITS[attempt - 1]
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %ds",
                    url, attempt, max_attempts, e, wait,
                )
                time.sleep(wait)
    raise last_exc
