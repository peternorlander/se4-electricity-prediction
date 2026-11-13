"""Wind is fetched in km/h and converted once, inside the power curve (docs/FINDINGS.md 1.1)."""
import pandas as pd
import pytest

import se4.sources.open_meteo as om
from se4.features import _wind_power_curve
from se4.data_checks import wind_unit_problems


def test_power_curve_reads_km_per_hour():
    kmh = pd.Series([0.0, 25.0, 13.0 * 3.6, 80.0])
    power = _wind_power_curve(kmh)
    assert power.iloc[0] == 0.0
    assert 0.0 < power.iloc[1] < 0.2           # an ordinary breeze (7 m/s) is far from rated power
    assert power.iloc[2] == pytest.approx(1.0)
    assert power.iloc[3] == 1.0


class _Response:
    status_code = 200

    def __init__(self, params):
        self.params = params

    def json(self):
        n = 3
        hourly = {"time": [f"2026-01-01T0{i}:00" for i in range(n)]}
        for var in ("temperature_2m", "windspeed_100m", "shortwave_radiation"):
            hourly[var] = [1.0] * n
            for m in om.BLEND_MODELS:
                hourly[f"{var}_{m}"] = [1.0] * n
        return {"hourly": hourly}


def test_no_fetch_overrides_the_wind_unit(monkeypatch):
    calls = []

    def fake_get(url, params):
        calls.append(params)
        return _Response(params)

    monkeypatch.setattr(om, "get_with_retry", fake_get)
    om.fetch_historical("2026-01-01", "2026-01-02")
    om.fetch_recent(8)
    om.fetch_forecast()
    om.fetch_forecast(models=om.BLEND_MODELS)
    om.fetch_international_wind_historical("2026-01-01", "2026-01-02")
    om.fetch_international_wind_recent(8)
    om.fetch_international_wind_forecast()
    om.fetch_international_wind_forecast(models=om.BLEND_MODELS)
    assert len(calls) == 4 + 4 * len(om.WIND_LOCATIONS)
    for params in calls:
        assert not any("unit" in key for key in params), params
        assert "windspeed_100m" in params["hourly"]


def test_wind_check_flags_metres_per_second_and_raw_kmh_curve():
    ok = pd.DataFrame({"windspeed": [5.0, 20.0, 25.0, 30.0, 60.0]})
    assert wind_unit_problems(ok, "w") == []
    in_ms = ok.assign(windspeed=ok.windspeed / 3.6)
    assert any("median" in p for p in wind_unit_problems(in_ms, "w"))
    stormy = pd.DataFrame({"windspeed": [60.0] * 4 + [20.0]})
    assert any("rated speed" in p for p in wind_unit_problems(stormy, "w"))
