"""Self-check for forecast-vintage scoring. Run: python ab/check_vintage.py [--snapshot D ...]

There is no pytest in this project; verification is by runnable scripts with asserts
(see ab/check_verdict.py). Two parts:

  1. Offline, synthetic -- always runs, seconds. Pins the mechanics a plausible-looking
     wrong answer would hide in: which test row reads which lead, that frozen columns are
     never swapped, that row 6 is left out of the production-mapped total, that the
     points and their column names come from sources.open_meteo, that a missing vintage
     is an error rather than a silent fallback to archive weather.

  2. On a real snapshot -- with --snapshot (and a vintage cache, see `ab_test.py
     vintages`). The one that matters for the standing numbers:
       * ARCHIVE weather is BIT-IDENTICAL to the harness as it was before vintages
         existed (an embedded copy of the old loop), and scoring vintages in the same
         call does not move it;
       * a "vintage" that is the archive itself reproduces the archive exactly at every
         lead (mapping, alignment and swap are an identity);
       * the real vintages look like the archive they stand in for (units and variable
         mapping: correlation and mean ratio per point and variable at lead 1);
       * only weather-derived columns are swapped, never a frozen one.
     Same snapshot, same thread count: pin OMP_NUM_THREADS (4 for the ledger).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import ab.harness as H
import ab.vintage as V

checks = 0


def ok(cond, label):
    global checks
    assert cond, label
    checks += 1


def raises(exc, fn, label):
    global checks
    try:
        fn()
    except exc:
        checks += 1
        return
    raise AssertionError(f"{label}: expected {exc.__name__}, none raised")


# =====================================================================================
# 1. offline
# =====================================================================================
class _Echo:
    """A 'model' whose prediction is the sum of its feature columns."""

    def predict(self, X):
        return np.asarray(X, dtype=float).sum(axis=1)


def _echo_fit(train, targets):
    return {name: _Echo() for name in targets}


def _synthetic(n=120):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    y = np.sin(np.arange(n) / 3.0) * 10 + 50
    # `w` is the weather feature (equal to the target on archive weather, so the archive
    # error is exactly 0), `lag` a frozen column that is 0 here and 99 in every vintage frame.
    return pd.DataFrame({"date": dates, "y": y, "w": y, "lag": 0.0})


def check_synthetic():
    data = _synthetic()
    targets = {"t": ("y", ["w", "lag"])}
    vintage = {}
    for lead in H.LEADS:
        vf = data.copy()
        vf["w"] = data["y"] + lead          # a forecast that is off by exactly `lead`
        vf["lag"] = 99.0                    # frozen: must never reach a test row
        vintage[lead] = vf
    err = H.run_walk_forward_errors(data, _echo_fit, targets, iterations=6, frozen_features=["lag"],
                                    vintage_frames=vintage)
    ok(set(err) == {H.ARCHIVE, *H.LEADS}, "conditions are archive + the seven leads")
    ok(err[H.ARCHIVE]["t"].shape == (6, 7), "error arrays are (windows, rows)")
    ok(np.array_equal(err[H.ARCHIVE]["t"], np.zeros((6, 7))), "archive error of the echo model is exactly 0")
    for lead in H.LEADS:
        ok(np.allclose(err[lead]["t"], lead), f"lead {lead}: error == {lead} everywhere (frozen lag=99 not swapped)")
    s = H.summarize_errors(err, "t")
    # production-mapped: row j at lead j+2, rows 0..5 -> leads 2..7; row 6 (lead 8) not sent
    ok(np.allclose(s["forecast_by_row"], [2, 3, 4, 5, 6, 7]), "row j reads lead j+2")
    ok(np.isclose(s["forecast"], 4.5), "production-mapped total is the mean of leads 2..7")
    ok(np.isclose(s["archive"], 0) and np.isclose(s["archive_sent"], 0) and np.isclose(s["penalty"], 4.5),
       "penalty = forecast - archive over the same six rows")
    ok(s["by_lead"] == {L: float(L) for L in H.LEADS}, "by-lead view scores every row at lead L")
    ok(len(s["archive_by_row"]) == 7 and len(s["forecast_by_row"]) == 6, "row 6 is archive-only")

    # the archive condition is untouched by scoring vintages in the same call
    plain = H.run_walk_forward(data, _echo_fit, targets, iterations=6, frozen_features=["lag"])
    ok(np.array_equal(plain["t"], H.window_means(err[H.ARCHIVE]["t"])), "vintages do not move the archive numbers")

    # a vintage frame identical to the archive swaps nothing and reproduces it
    same = {lead: data.copy() for lead in H.LEADS}
    e2 = H.run_walk_forward_errors(data, _echo_fit, targets, iterations=6, frozen_features=["lag"],
                                   vintage_frames=same)
    ok(all(np.array_equal(e2[lead]["t"], e2[H.ARCHIVE]["t"]) for lead in H.LEADS), "identical vintage == archive")

    # swap rule: differing columns, frozen excluded
    vf = data.copy()
    vf["w"] += 1
    vf["lag"] = 5.0
    ok(H.swap_columns(data, vf, ["lag"]) == ["w"], "swap = differing columns minus frozen")
    ok(H.swap_columns(data, vf, []) == ["w", "lag"], "...and a column not frozen is swapped")

    # a CANDIDATE's transform runs on every lead frame, and shifts truncate them in step:
    # a column derived from the weather column carries the FORECAST's value on a forecast row
    from ab.variants import Variant
    from ab.verdict import _point_errors

    def derive(frame):
        out = frame.copy()
        out["w2"] = out["w"] * 2.0
        return out
    long_data = _synthetic(420)
    long_vint = {}
    for lead in H.LEADS:
        vf = long_data.copy()
        vf["w"] = long_data["y"] + lead
        vf["lag"] = 99.0
        long_vint[lead] = vf
    variant = Variant(name="derived", transform=derive, fit_fn=_echo_fit,
                      targets={"t": ("y", ["w2", "lag"])}, frozen_features=["lag"])
    pe = _point_errors(long_data, variant, shift=5, train_window=None, vintage_frames=long_vint)
    y = long_data["y"].values[:-5][-7:]             # the last window after a 5-row shift
    ok(pe[H.ARCHIVE]["t"].shape == (52, 7), "52 windows of 7 rows")
    ok(np.allclose(pe[H.ARCHIVE]["t"][-1], np.abs(2.0 * y - y)), "archive: transform on the shifted archive frame")
    ok(np.allclose(pe[3]["t"][-1], np.abs(2.0 * (y + 3) - y)),
       "forecast: the derived column is computed from the lead frame, shifted in step with the archive")

    # alignment is enforced, not assumed
    bad = {lead: v.iloc[:-1] for lead, v in vintage.items()}
    raises(ValueError, lambda: H.run_walk_forward_errors(data, _echo_fit, targets, iterations=6,
                                                         vintage_frames=bad), "misaligned vintage frame")
    raises(ValueError, lambda: H.run_walk_forward_errors(data, _echo_fit, targets, iterations=6, step=5,
                                                         vintage_frames=vintage), "step != 7 with vintages")


def check_points_and_columns():
    from sources import open_meteo as om
    pts = V.weather_points()
    ok(list(pts) == [V.SE4_POINT, *om.WIND_LOCATIONS], "points = the SE4 point + every WIND_LOCATIONS entry, in order")
    ok(pts[V.SE4_POINT] == (om.LOCATION_LAT, om.LOCATION_LON), "SE4 coordinates come from the source")
    for name, xy in om.WIND_LOCATIONS.items():
        ok(pts[name] == xy, f"{name} coordinates come from the source")

    # The column names must be what production's fetch produces, not what this module assumes.
    real = om._parse_response({"hourly": {"time": ["2026-01-01T00:00"], "temperature_2m": [1.0],
                                          "windspeed_100m": [2.0], "shortwave_radiation": [3.0]}})
    key, cols = V.frame_columns(V.SE4_POINT)
    ok(key == "weather_hourly" and set(cols.values()) == set(real.columns) - {"timestamp"},
       "SE4 columns match sources.open_meteo._parse_response")
    orig = om._fetch_location_series
    om._fetch_location_series = lambda lat, lon, url, extra: pd.DataFrame({
        "time": pd.to_datetime(["2026-01-01T00:00"], utc=True), "windspeed": [1.0], "radiation": [2.0], "temperature": [3.0]})
    try:
        intl = om._fetch_all_locations("x", {})
    finally:
        om._fetch_location_series = orig
    produced = set(intl.columns) - {"timestamp"}
    mine = set()
    for name in om.WIND_LOCATIONS:
        key, cols = V.frame_columns(name)
        ok(key == "wind_intl_hourly", f"{name} lives in wind_intl_hourly")
        mine |= set(cols.values())
    ok(mine == produced, "WIND_LOCATIONS columns match sources.open_meteo._fetch_all_locations exactly")


def check_hours_and_coverage():
    # Europe/Stockholm DST days have 23 and 25 hours; the hour set must follow local dates
    short, long_ = pd.Timestamp("2026-03-29").date(), pd.Timestamp("2026-10-25").date()
    for day, n in ((short, 23), (long_, 25), (pd.Timestamp("2026-06-01").date(), 24)):
        ok(len(V._hours_of([day])) == n, f"{day}: {n} hours")
    days = [pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-06-02").date()]
    hours = V._hours_of(days)
    cols = [c for api, _ in V._API_VARIABLES for c in V.vintage_columns(api)]
    frame = pd.DataFrame(1.0, index=hours, columns=cols)
    V.check_coverage({"malmo": frame}, days)
    ok(True, "full coverage passes")
    holey = frame.copy()
    holey.iloc[5, 3] = np.nan
    raises(V.VintageCoverageError, lambda: V.check_coverage({"malmo": holey}, days), "a NaN hour is an error")
    raises(V.VintageCoverageError, lambda: V.check_coverage({"malmo": frame.iloc[:-3]}, days),
           "a missing hour is an error")
    # nothing is fetched when it must not be, and a window before the archive starts is refused first
    import tempfile
    with tempfile.TemporaryDirectory() as empty:
        raises(V.VintageCoverageError, lambda: V.ensure_vintages(pd.Timestamp("2026-01-05").date(), root=empty,
                                                                 fetch=False), "empty cache with fetching off")
        before = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=400, freq="D")})
        raises(V.VintageCoverageError, lambda: V.VintageSet.for_snapshot({}, before, [0], root=empty, fetch=False),
               "a window before the Previous Runs archive starts")
    dates = pd.Series(pd.date_range("2026-01-01", periods=400, freq="D"))
    ev = V.evaluated_dates(dates, [0, 10], iterations=2, step=7)
    ok(len(ev) == 14 + 10 and ev[-1] == dates.iloc[-1].date(), "evaluated days = last iterations*step rows per shift")
    ok(V.default_root(Path("ab_cache/long")) == Path("ab_cache/vintages"), "vintage root of ab_cache/long")
    ok(V.default_root(Path("ab_cache")) == Path("ab_cache/vintages"), "vintage root of ab_cache")


def check_hybrid():
    stamps = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    inputs = {
        "weather_hourly": pd.DataFrame({"timestamp": stamps, "temperature": 1.0, "windspeed": 2.0, "radiation": 3.0}),
        "wind_intl_hourly": pd.DataFrame({"timestamp": stamps, "windspeed_dk1": 5.0, "radiation_dk1": 6.0,
                                          "temperature_dk1": 7.0}),
        "prices_hourly": object(),
    }
    mk = lambda v: pd.DataFrame({f"{a}_previous_day{L}": v for a, _ in V._API_VARIABLES for L in V.LEADS},
                                index=stamps)
    vint = {"malmo": mk(10.0), "dk1": mk(20.0), "ghost": mk(30.0)}
    vint["dk1"].iloc[4, :] = np.nan                       # one hour without a vintage
    out = V.hybrid_inputs(inputs, vint, 3, log=lambda *_: None)
    ok((out["weather_hourly"][["temperature", "windspeed", "radiation"]] == 10.0).all().all(), "SE4 point swapped")
    ok((out["wind_intl_hourly"].loc[stamps != stamps[4], ["windspeed_dk1", "radiation_dk1", "temperature_dk1"]] == 20.0)
       .all().all(), "WIND_LOCATIONS point swapped")
    ok(out["wind_intl_hourly"].loc[4, "windspeed_dk1"] == 5.0, "an hour without a vintage keeps archive")
    ok((inputs["weather_hourly"]["temperature"] == 1.0).all(), "the snapshot's own frame is not mutated")
    ok(out["prices_hourly"] is inputs["prices_hourly"], "non-weather inputs are the snapshot's own objects")
    ok(set(out["wind_intl_hourly"].columns) == set(inputs["wind_intl_hourly"].columns), "no columns added")


def run_offline():
    check_synthetic()
    check_points_and_columns()
    check_hours_and_coverage()
    check_hybrid()
    print(f"[ok] offline: {checks} checks (lead mapping, frozen columns, row 6, points from the source, "
          f"coverage errors, hybrid inputs)")


# =====================================================================================
# 2. on a real snapshot
# =====================================================================================
def _reference_run_walk_forward(data, fit_fn, targets, step, iterations, frozen_features, frozen_rolling,
                                train_window):
    """The loop exactly as ab.harness.run_walk_forward had it before vintages existed."""
    from features import apply_forecast_freeze
    from model import enforce_coherence
    data = data.reset_index(drop=True)
    n = len(data)
    min_train = n - iterations * step
    mae_lists = {name: [] for name in targets}
    for iteration in range(iterations):
        train_end = min_train + iteration * step
        test_end = train_end + step
        train_start = 0 if train_window is None else train_end - train_window
        train_slice = data.iloc[train_start:train_end]
        raw_test = data.iloc[train_end:test_end]
        models = fit_fn(train_slice, targets=targets)
        frozen = apply_forecast_freeze(raw_test, train_slice, frozen_features=frozen_features,
                                       frozen_rolling=frozen_rolling)
        preds = enforce_coherence({name: models[name].predict(frozen[feature_cols].values)
                                   for name, (_, feature_cols) in targets.items()})
        for name, (target_col, feature_cols) in targets.items():
            actual = raw_test[target_col].values
            errors = np.abs(preds[name] - actual)
            mae_lists[name].append(float(errors.mean()))
    return {name: np.array(values) for name, values in mae_lists.items()}


def run_snapshot(args):
    import features as feat
    from ab.snapshot import load_snapshot
    from ab.variants import BASELINE

    root = Path(args.root)
    inputs, meta = load_snapshot(args.snapshot, cache_root=root)
    data = feat.build_training_data(**inputs)
    print(f"snapshot {meta['today']}: {len(data)} rows, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    shift, it, tw = args.shift, args.iterations, H.PRODUCTION_TRAIN_ROWS
    kw = dict(frozen_features=BASELINE.frozen_features, frozen_rolling=BASELINE.frozen_rolling, train_window=tw)
    shifted = H.apply_shift(data, shift)

    # --- archive bit-identity against the pre-vintage loop ---------------------------
    ref = _reference_run_walk_forward(shifted, BASELINE.fit_fn, BASELINE.targets, 7, it, **kw)
    new = H.run_walk_forward(shifted, BASELINE.fit_fn, BASELINE.targets, iterations=it, **kw)
    for name in ref:
        ok(np.array_equal(ref[name], new[name]), f"run_walk_forward != the pre-vintage loop on {name}")
    print(f"[ok] run_walk_forward is bit-identical to the pre-vintage loop "
          f"({it} windows x {len(ref)} targets, shift {shift}): "
          + " ".join(f"{k}={float(np.mean(v)):.6f}" for k, v in new.items()))

    # --- a vintage that IS the archive: identity at every lead -----------------------
    stamps_w = pd.to_datetime(inputs["weather_hourly"]["timestamp"], utc=True)
    pseudo = {}
    for point in V.weather_points():
        key, cols = V.frame_columns(point)
        frame = inputs[key]
        if not all(c in frame.columns for c in cols.values()):
            continue
        idx = pd.to_datetime(frame["timestamp"], utc=True)
        pseudo[point] = pd.DataFrame({f"{api}_previous_day{L}": frame[col].values
                                      for api, col in cols.items() for L in V.LEADS}, index=idx)
    pframes = V.build_vintage_frames(inputs, pseudo, data, log=lambda *_: None)
    ok(all(H.swap_columns(data, f, feat.FORECAST_FROZEN_FEATURES) == [] for f in pframes.values()),
       "archive-as-vintage swaps no column")
    shifted_p = {L: H.apply_shift(f, shift) for L, f in pframes.items()}
    err = H.run_walk_forward_errors(shifted, BASELINE.fit_fn, BASELINE.targets, iterations=it,
                                    vintage_frames=shifted_p, **kw)
    for name in ref:
        ok(np.array_equal(H.window_means(err[H.ARCHIVE][name]), ref[name]),
           f"scoring vintages moved the archive numbers on {name}")
        for L in H.LEADS:
            ok(np.array_equal(err[L][name], err[H.ARCHIVE][name]), f"archive-as-vintage != archive at lead {L} ({name})")
    print(f"[ok] archive weather unchanged by scoring vintages in the same call; archive-as-vintage "
          f"reproduces it exactly at all 7 leads")

    # --- the real vintages ------------------------------------------------------------
    vroot = Path(args.vintage_root) if args.vintage_root else V.default_root(root)
    days = V.evaluated_dates(data["date"], [shift], iterations=it)
    vintages = V.ensure_vintages(max(days), root=vroot, fetch=False)
    V.check_coverage(vintages, days)
    print(f"[ok] vintage cache {vroot}: {len(vintages)} points cover every evaluated hour "
          f"({days[0]} -> {days[-1]}) at leads 1-7")
    # Variable / unit / timestamp mapping: the lead-1 forecast must look like the archive it
    # stands in for -- over two years, because a few weeks of autumn radiation is a biased
    # sample (lead-1 radiation runs 0.87-0.92x ERA5 over the full span, 0.78x in autumn).
    # Measured 2026-10-25 on every point: lag-0 correlation 0.90-0.99, above the +-1 h lags for
    # wind and radiation; wind mean ratio 0.97-1.05.
    wide = V.evaluated_dates(data["date"], [shift], iterations=104)
    hours = V._hours_of(wide)
    V.check_coverage(vintages, wide)
    corrs, wind_ratios = [], []
    for point, v in vintages.items():
        key, cols = V.frame_columns(point)
        if not all(c in inputs[key].columns for c in cols.values()):
            continue
        arch = inputs[key].assign(timestamp=pd.to_datetime(inputs[key]["timestamp"], utc=True)).set_index("timestamp")
        for api, col in cols.items():
            a = arch[col].reindex(hours).astype(float)
            f = v[f"{api}_previous_day1"].reindex(hours).astype(float)
            c0, cm, cp = a.corr(f), a.corr(f.shift(1)), a.corr(f.shift(-1))
            corrs.append((c0, point, api))
            ok(c0 > 0.85, f"{point}/{api}: lead-1 vintage correlates {c0:.2f} with archive -- wrong variable or unit?")
            if api != "temperature_2m":     # smooth and diurnal: 2 m temperature peaks a touch off lag 0
                ok(c0 > cm and c0 > cp, f"{point}/{api}: lead-1 vintage is best aligned off lag 0 -- shifted timestamps?")
            if api == "wind_speed_100m":    # a km/h vs m/s slip is a factor 3.6
                ratio = float(f.mean() / a.mean())
                wind_ratios.append(ratio)
                ok(0.7 < ratio < 1.4, f"{point}/{api}: lead-1 vintage mean is {ratio:.2f}x the archive's")
    lo = min(corrs)
    print(f"[ok] lead-1 vintages look like the archive over {len(wide)} days: min correlation {lo[0]:.3f} "
          f"({lo[1]}/{lo[2]}), wind and radiation best aligned at lag 0 at every point, wind mean ratio "
          f"{min(wind_ratios):.2f}..{max(wind_ratios):.2f}")

    frames = V.build_vintage_frames(inputs, vintages, data, log=lambda *_: None)
    frozen = set(feat.FORECAST_FROZEN_FEATURES)
    swapped = H.swap_columns(data, frames[3], frozen)
    differing_frozen = [c for c in frozen if c in data.columns and not H.column_equal(data[c], frames[3][c])]
    ok(len(swapped) > 0 and not set(swapped) & frozen, "real vintages swap weather columns and no frozen one")
    print(f"[ok] {len(swapped)} columns are swapped into test rows; {len(differing_frozen)} frozen columns differ "
          f"between the frames and are deliberately left at archive: {sorted(differing_frozen)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None, help="also run the real-snapshot checks on this snapshot date")
    ap.add_argument("--root", default="ab_cache/long", help="snapshot cache root")
    ap.add_argument("--vintage-root", default=None)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=8, help="windows for the bit-identity check")
    args = ap.parse_args()
    run_offline()
    if args.snapshot:
        run_snapshot(args)
    print(f"all vintage checks passed ({checks})")


if __name__ == "__main__":
    main()
