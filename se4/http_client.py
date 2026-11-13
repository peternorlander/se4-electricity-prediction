import re
import time
import logging
import requests


_TIMEOUT = 120
_RETRY_WAITS = (4, 16, 32, 64)  # seconds between attempts 1→2, 2→3, 3→4, 4→5

# 5xx is the server failing; 429 (Open-Meteo "too many requests", shared Actions IPs) and 408
# are the server asking us to come back. All of them are worth another try.
_RETRY_STATUSES = (408, 429)

logger = logging.getLogger(__name__)

_SECRET = re.compile(r"(securityToken|api_key|apikey|token)=[^&\s)'\"]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Hide credentials that a requests error message can carry in the URL it quotes."""
    return _SECRET.sub(r"\1=***", str(text))


def describe_body(response: requests.Response, limit: int = 300) -> str:
    """
    What the API said, as one short line: its own explanation of a refusal.

    A bare "400 Client Error" or a KeyError further down hides the one thing that tells a bad
    token from a changed limit from a rate limit (ENTSO-E <Reason>, Open-Meteo {"reason": ...},
    NVE and Nord Pool plain text). XML and JSON are flattened to their text; whitespace is
    collapsed; credentials are redacted.
    """
    try:
        text = response.text or ""
    except Exception:  # noqa: BLE001
        text = ""
    text = re.sub(r"<[^>]+>", " ", text)          # XML -> its text nodes
    text = re.sub(r"\s+", " ", text).strip()
    return redact(text[:limit]) or "(empty body)"


def get_with_retry(url: str, params: dict) -> requests.Response:
    """
    Perform a GET request with exponential backoff retry on timeout, connection
    errors, 5xx server errors (e.g. 503 Service Unavailable) and 408/429.

    Retries up to 4 times, waiting 4s, 16s, 32s, 64s between attempts.
    Does NOT call raise_for_status for other non-2xx statuses -- callers handle those --
    but logs the status and the API's own explanation of any >= 400 answer, so the log of a
    failed run says what the API said, not only that it failed.
    Raises the last exception (or HTTPError for 5xx/408/429) if all retries are exhausted.
    """
    last_exc = None
    max_attempts = len(_RETRY_WAITS) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=_TIMEOUT)
            if response.status_code >= 500 or response.status_code in _RETRY_STATUSES:
                body = describe_body(response)
                last_exc = requests.exceptions.HTTPError(
                    f"{response.status_code} error: {body}", response=response
                )
                if attempt < max_attempts:
                    wait = _RETRY_WAITS[attempt - 1]
                    logger.warning(
                        "Request to %s returned %d (attempt %d/%d): %s — retrying in %ds",
                        url, response.status_code, attempt, max_attempts, body, wait,
                    )
                    time.sleep(wait)
                continue
            if response.status_code >= 400:
                logger.warning("Request to %s returned %d: %s",
                               url, response.status_code, describe_body(response))
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < max_attempts:
                wait = _RETRY_WAITS[attempt - 1]
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %ds",
                    url, attempt, max_attempts, redact(e), wait,
                )
                time.sleep(wait)
    logger.error("Request to %s gave up after %d attempts: %s: %s",
                 url, max_attempts, type(last_exc).__name__, redact(last_exc))
    raise last_exc
