"""
An external API that refuses must say why in the log (born of 2026-11-06..08, where a failed run
named the missing rows but never what Yahoo / Open-Meteo / ENTSO-E had answered), and a rate
limit is retried like an outage.
"""
import pytest
import requests

from se4 import http_client
from se4.sources import open_meteo


class FakeResponse:
    def __init__(self, status_code, text="", json_data=None):
        self.status_code, self.text, self._json = status_code, text, json_data

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_client.time, "sleep", lambda s: None)


def _answers(monkeypatch, *answers):
    queue = list(answers)

    def fake_get(url, params=None, timeout=None):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(http_client.requests, "get", fake_get)


def test_a_4xx_is_returned_but_its_explanation_is_logged(monkeypatch, caplog):
    xml = "<Acknowledgement><Reason><code>999</code><text>Invalid token</text></Reason></Acknowledgement>"
    _answers(monkeypatch, FakeResponse(401, xml))
    with caplog.at_level("WARNING"):
        response = http_client.get_with_retry("https://api.example/x", {})
    assert response.status_code == 401
    assert "401" in caplog.text and "Invalid token" in caplog.text


def test_rate_limit_is_retried_and_the_reason_is_logged(monkeypatch, caplog):
    limited = FakeResponse(429, '{"error":true,"reason":"Too many concurrent requests"}')
    _answers(monkeypatch, limited, FakeResponse(200, "ok"))
    with caplog.at_level("WARNING"):
        response = http_client.get_with_retry("https://api.example/x", {})
    assert response.status_code == 200
    assert "429" in caplog.text and "Too many concurrent requests" in caplog.text


def test_giving_up_names_the_last_error(monkeypatch, caplog):
    _answers(monkeypatch, *[requests.exceptions.ReadTimeout("read timed out")] * 5)
    with caplog.at_level("WARNING"), pytest.raises(requests.exceptions.ReadTimeout):
        http_client.get_with_retry("https://api.example/x", {})
    assert "gave up after 5 attempts" in caplog.text and "read timed out" in caplog.text


def test_tokens_in_error_text_are_redacted(monkeypatch, caplog):
    boom = requests.exceptions.ConnectionError("Max retries with url: /api?securityToken=SECRET123&x=1")
    _answers(monkeypatch, *[boom] * 5)
    with caplog.at_level("WARNING"), pytest.raises(requests.exceptions.ConnectionError):
        http_client.get_with_retry("https://api.example/x", {})
    assert "SECRET123" not in caplog.text and "securityToken=***" in caplog.text


def test_open_meteo_refusal_is_an_error_that_quotes_the_reason():
    refusal = FakeResponse(429, '{"error":true,"reason":"Daily API request limit exceeded"}',
                           {"error": True, "reason": "Daily API request limit exceeded"})
    with pytest.raises(RuntimeError, match="Daily API request limit exceeded"):
        open_meteo._json_with_hourly(refusal, "archive")


def test_open_meteo_200_without_hourly_is_an_error_not_a_keyerror():
    with pytest.raises(RuntimeError, match="no hourly data"):
        open_meteo._json_with_hourly(FakeResponse(200, "{}", {}), "archive")
