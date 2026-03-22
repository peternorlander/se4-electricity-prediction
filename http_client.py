import time
import logging
import requests


_TIMEOUT = 120
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds — doubles each attempt: 2s, 4s, 8s

logger = logging.getLogger(__name__)


def get_with_retry(url: str, params: dict) -> requests.Response:
    """
    Perform a GET request with exponential backoff retry on timeout or connection errors.

    Retries up to _MAX_RETRIES times, waiting 2s, 4s, 8s between attempts.
    Does NOT call raise_for_status — callers are responsible for handling HTTP errors.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT)
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning(
                "Request to %s failed (attempt %d/%d): %s — retrying in %ds",
                url, attempt, _MAX_RETRIES, e, wait,
            )
            time.sleep(wait)
    raise last_exc
