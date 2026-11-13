"""Every key production can put in the payload survives the SEK and addon conversions."""
import json

import numpy as np
import pandas as pd

from se4 import model
from se4 import ha_client
from se4.currency import convert_predictions_to_sek


class _Const:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class _Band:
    warning_threshold = model.COUPLING_WARNING_PROBA

    def predict(self, frame, point):
        n = len(frame)
        return np.full(n, -12.5), np.full(n, 140.0), np.full(n, 0.4567)


def _predictions_eur():
    """model.predict on stub models: the exact keys a production run emits."""
    columns = sorted({c for _, cols in model.TARGETS.values() for c in cols})
    frame = pd.DataFrame({c: np.zeros(2) for c in columns})
    frame["date"] = pd.to_datetime(["2026-11-02", "2026-11-03"])
    models = {"min": _Const(10.0), "avg": _Const(50.0), "max": _Const(90.0), "cheap2h": _Const(12.0)}
    return model.predict(models, frame, band=_Band())


def test_predict_emits_the_four_targets_and_the_band():
    preds = _predictions_eur()
    keys = set(next(iter(preds.values())))
    assert keys == set(model.TARGETS) | {"cheap2h_low", "cheap2h_high", model.RISK_KEY}
    assert model.UNITLESS_KEYS == {model.RISK_KEY}


def test_sek_and_addon_convert_prices_and_leave_the_risk_alone():
    preds = _predictions_eur()
    rate = 11.5
    sek = convert_predictions_to_sek(preds, rate)
    with_addon = ha_client.apply_addon(sek, 0.8)
    for day, values in preds.items():
        for key, eur in values.items():
            if key in model.UNITLESS_KEYS:
                assert sek[day][key] == eur and with_addon[day][key] == eur
                assert 0.0 <= eur <= 1.0
            else:
                assert sek[day][key] == round(eur * rate / 1000, 4)
                assert with_addon[day][key] == round(sek[day][key] * 1.05 + 0.8, 4)
    assert sek["2026-11-02"]["cheap2h_low"] < 0     # negative prices stay negative


def test_push_payload_is_json_and_carries_everything(monkeypatch):
    sent = {}

    class _Response:
        def raise_for_status(self):
            pass

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, payload=json)
        return _Response()

    monkeypatch.setenv("HA_URL", "http://ha.test:8123/")
    monkeypatch.setenv("HA_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_SHA", "abc123")
    monkeypatch.setattr(ha_client.requests, "post", fake_post)
    sek = convert_predictions_to_sek(_predictions_eur(), 11.5)
    ha_client.push_predictions(sek, ha_client.apply_addon(sek, 0.8), {"cheap2h_interval": {"n_holdout": 180}},
                               {"cheap2h": {"f": 0.5}}, eur_to_sek_rate=11.5,
                               predictions_shadow={"blend": sek, "Bad-Name": sek, "served": sek})
    assert sent["url"] == "http://ha.test:8123/api/states/sensor.electricity_price_predictions"
    attrs = sent["payload"]["attributes"]
    json.dumps(sent["payload"])
    assert attrs["predictions_raw"][0]["date"] == "2026-11-02"
    assert set(attrs["predictions_shadow"]) == {"blend"}
    assert attrs["eur_to_sek_rate"] == 11.5 and attrs["git_sha"] == "abc123"
    assert attrs["cheap2h_interval"] == {"n_holdout": 180}
    assert attrs["feature_importance_cheap2h"] == {"f": 0.5}


def test_shadow_names_fit_the_log_column():
    ok = ["blend", "a", "a" * 16, "x_1"]
    bad = ["", "Blend", "1abc", "a" * 17, "with-dash"]
    assert all(ha_client.SHADOW_VARIANT_NAME.match(n) for n in ok)
    assert not any(ha_client.SHADOW_VARIANT_NAME.match(n) for n in bad)
