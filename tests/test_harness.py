"""The A/B harness hooks of round 31: the target filter, build_fn, and the band's transform."""
import numpy as np
import pandas as pd
import pytest

import ab.verdict as verdict
from ab.variants import BASELINE, Variant


def test_target_filter_keeps_the_coherence_pair():
    available = ("min", "avg", "max", "cheap2h")
    assert verdict.fitted_target_names(("cheap2h",), available) == ("min", "cheap2h")
    assert verdict.fitted_target_names(("min",), available) == ("min", "cheap2h")
    assert verdict.fitted_target_names(("avg",), available) == ("avg",)
    assert verdict.fitted_target_names(("max", "cheap2h"), available) == ("min", "max", "cheap2h")
    assert verdict.fitted_target_names(available, available) == available
    with pytest.raises(ValueError):
        verdict.fitted_target_names(("cheap4h",), available)


def test_restrict_targets_leaves_a_full_variant_untouched():
    assert verdict.restrict_targets(BASELINE, verdict.DEFAULT_TARGET_NAMES) is BASELINE
    pair = verdict.restrict_targets(BASELINE, ("cheap2h",))
    assert tuple(pair.targets) == ("min", "cheap2h")
    assert pair.targets["cheap2h"] == BASELINE.targets["cheap2h"]
    assert BASELINE.targets.keys() == {"min", "avg", "max", "cheap2h"}   # the original is not mutated


def _frame(n=5):
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n), "x": np.arange(n, dtype=float)})


def _drop_a_day(inputs):
    return _frame().iloc[1:].reset_index(drop=True)


def _add_column(inputs):
    return _frame().assign(y=1.0)


def test_build_fn_may_change_columns_never_rows(monkeypatch):
    monkeypatch.setattr(verdict, "build_training_data", lambda **inputs: _frame())
    frames = verdict.build_arm_frames({}, {"base": BASELINE, "cand": Variant("c", build_fn=_add_column)})
    assert "y" in frames["cand"] and "y" not in frames["base"]
    with pytest.raises(ValueError, match="never rows"):
        verdict.build_arm_frames({}, {"base": BASELINE, "cand": Variant("c", build_fn=_drop_a_day)})


def test_arms_without_build_fn_share_one_frame(monkeypatch):
    built = []
    monkeypatch.setattr(verdict, "build_training_data", lambda **inputs: built.append(1) or _frame())
    frames = verdict.build_arm_frames({}, {"base": BASELINE, "cand": Variant("c")})
    assert frames["base"] is frames["cand"] and len(built) == 1


def _double_x(data):
    return data.assign(x=data["x"] * 2)


def test_band_job_applies_the_variants_transform(monkeypatch):
    import ab.band as band
    seen = {}

    def fake_walk_forward(data, variant, train_window):
        seen[variant.name] = data["x"].tolist()
        return pd.DataFrame({"point": [0.0]})

    monkeypatch.setattr(band, "band_walk_forward", fake_walk_forward)
    variants = {"base": BASELINE, "cand": Variant("cand", transform=_double_x)}
    for which in ("base", "cand"):
        band._band_job((_frame(), ("NOW", 0, which), variants, 3))
    assert seen["base"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert seen["cand"] == [0.0, 2.0, 4.0, 6.0, 8.0]
