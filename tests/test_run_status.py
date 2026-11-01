"""The run-status sensor (run_status.py): written on success and on failure, never raising."""
import pytest

import predict
import run_status


@pytest.fixture
def ha(monkeypatch):
    monkeypatch.setenv("HA_URL", "http://ha.test:8123")
    monkeypatch.setenv("HA_TOKEN", "test-token")
    for var in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(var, raising=False)
    calls = []
    state = {"attributes": {"last_success_at": "2026-10-30T05:00:00Z"}}

    def fake_request(method, path, payload=None):
        calls.append((method, path, payload))
        return state if method == "GET" else None

    monkeypatch.setattr(run_status, "_request", fake_request)
    return calls


def test_success_sets_last_success(ha):
    assert run_status.push_run_status("ok", started_at="2026-10-31T05:00:00Z")
    method, path, payload = ha[-1]
    assert (method, path) == ("POST", "/api/states/sensor.electricity_price_predictions_run")
    assert payload["state"] == "ok"
    assert payload["attributes"]["last_success_at"] == payload["attributes"]["finished_at"]
    assert not any(m == "GET" for m, _, _ in ha)


def test_failure_carries_the_previous_success_forward(ha):
    run_status.push_run_status("failed", error="ValueError: " + "x" * 1000, stage="workflow")
    payload = ha[-1][2]
    assert payload["state"] == "failed"
    assert payload["attributes"]["last_success_at"] == "2026-10-30T05:00:00Z"
    assert len(payload["attributes"]["error"]) == run_status.MAX_ERROR_CHARS
    assert payload["attributes"]["stage"] == "workflow"


def test_github_context_is_attached(ha, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    run_status.push_run_status("ok")
    attrs = ha[-1][2]["attributes"]
    assert attrs["git_sha"] == "abc123"
    assert attrs["run_url"] == "https://github.com/owner/repo/actions/runs/42"


def test_never_raises(monkeypatch):
    monkeypatch.setenv("HA_URL", "http://ha.test:8123")
    monkeypatch.setenv("HA_TOKEN", "test-token")

    def broken(*args, **kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(run_status, "_request", broken)
    assert run_status.push_run_status("failed") is False
    monkeypatch.delenv("HA_URL")
    assert run_status.push_run_status("ok") is False


def test_predict_run_reports_failure_and_reraises(monkeypatch):
    pushed = []
    monkeypatch.setattr(predict, "push_run_status", lambda status, **kw: pushed.append((status, kw)))

    def boom():
        raise RuntimeError("ENTSO-E down")

    monkeypatch.setattr(predict, "main", boom)
    with pytest.raises(RuntimeError):
        predict.run()
    assert pushed[0][0] == "failed" and pushed[0][1]["error"] == "RuntimeError: ENTSO-E down"

    pushed.clear()
    monkeypatch.setattr(predict, "main", lambda: None)
    predict.run()
    assert pushed[0][0] == "ok"
