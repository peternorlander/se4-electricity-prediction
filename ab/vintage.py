"""
Forecast-vintage weather for the A/B harness. See docs/AB_TESTING.md, "Archive
weather is a perfect forecast".

Every weather input in the eval is ERA5 archive -- a forecast with no error.
Production pays the error. A paired A/B cancels it for a feature-set change but
not for a weather-derived feature, whose value is the forecast skill it carries.
This module supplies the other half: for every Open-Meteo point production reads,
the forecast that was issued L days before each hour (Open-Meteo "Previous Runs",
`<var>_previous_dayL`, L = 1..7, `best_match` -- what production's forecast fetch
resolves to, because it passes no `models=`). Public, no token, from 2024-03.

How the harness uses it (ab.harness.run_walk_forward_errors):

  * the model is trained on archive weather exactly as in production;
  * a second copy of the merged frame is built per lead by the UNMODIFIED
    features.build_training_data on inputs whose weather columns carry the
    vintage, so every derived column (residual load, HDD, wind_night, the
    northern power curves ...) is the production recipe on forecast inputs;
  * only the per-day columns that differ are swapped into the test rows. Frozen
    lags and rolling regime signals are never swapped: production holds them at
    values from before the run, and apply_forecast_freeze does the same here.

The points are DERIVED from sources.open_meteo (the SE4 point plus every
WIND_LOCATIONS entry), so a point added to production is fetched and swapped
here without touching this file. A cached point whose coordinates no longer match
the source is refetched.

A vintage that is missing for an evaluated hour is an ERROR, not a fallback: a
silent fallback to archive would make the newest test days look perfectly
forecast, which is exactly the optimism this module exists to remove.

Cache layout, outside every snapshot (snapshots stay read-only):

    ab_cache/vintages/best_match/<point>.pkl    hourly UTC frame, 21 columns
    ab_cache/vintages/best_match/meta.json      lat/lon + requested range per point

A vintage is by definition a forecast issued earlier, so history never changes
and one cache serves every snapshot; a newer snapshot only extends the tail.
"""
import json
import os
import time
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pandas as pd

from ab.harness import LEADS, column_equal
from ab.snapshot import DEFAULT_CACHE_ROOT

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "best_match"
# The Previous Runs archive starts here (ECMWF IFS from 2024-04, which best_match
# resolves to where it is the better model). Test rows before it cannot be scored.
START = date(2024, 3, 1)
# On an extension the last days already cached are fetched again, so a tail that
# was incomplete when it was first saved is replaced by the settled values.
TAIL_OVERLAP_DAYS = 3
# Open-Meteo weights a call by (variables / 10) x (days / 14) against 600 per
# minute; the pause after a request keeps a run of large fetches under that.
_WEIGHT_PER_MINUTE = 600.0

VINTAGE_CACHE_ROOT = DEFAULT_CACHE_ROOT / "vintages"
SE4_POINT = "malmo"

# API variable -> the unsuffixed column production's weather frames use.
_API_VARIABLES = (
    ("temperature_2m", "temperature"),
    ("wind_speed_100m", "windspeed"),
    ("shortwave_radiation", "radiation"),
)

# Column families a weather vintage cannot touch. The swap rule needs no list of
# weather columns (whatever differs between the archive and a vintage frame IS
# weather-derived, by construction), so this is only a tripwire that the frames
# differ where they must not: a prefix here that starts differing means the
# hybrid inputs changed more than weather.
NON_WEATHER_PREFIXES = ("price_", "ttf_", "co2_", "gas_", "nuclear_", "reservoir_")


class VintageCoverageError(RuntimeError):
    """A vintage is missing for an hour the run evaluates."""


# =====================================================================================
# the points production reads
# =====================================================================================
def weather_points() -> dict:
    """{point: (lat, lon)} -- the SE4 point and every WIND_LOCATIONS entry."""
    from sources.open_meteo import LOCATION_LAT, LOCATION_LON, WIND_LOCATIONS
    assert SE4_POINT not in WIND_LOCATIONS, f"WIND_LOCATIONS now has a '{SE4_POINT}' key"
    return {SE4_POINT: (LOCATION_LAT, LOCATION_LON), **WIND_LOCATIONS}


def frame_columns(point: str) -> tuple:
    """
    (snapshot key, {API variable: column}) for a point -- the naming
    sources.open_meteo gives it: unsuffixed in weather_hourly for the SE4 point,
    `<name>_<key>` in wind_intl_hourly for every WIND_LOCATIONS entry.
    """
    if point == SE4_POINT:
        return "weather_hourly", {api: col for api, col in _API_VARIABLES}
    return "wind_intl_hourly", {api: f"{col}_{point}" for api, col in _API_VARIABLES}


def vintage_columns(api_variable: str) -> list:
    return [f"{api_variable}_previous_day{lead}" for lead in LEADS]


def default_root(cache_root) -> Path:
    """
    Where the vintage cache for snapshots under `cache_root` lives:
    ab_cache/long -> ab_cache/vintages, ab_cache -> ab_cache/vintages. One
    directory for every snapshot, next to them and never inside one.
    """
    cache_root = Path(cache_root)
    base = cache_root.parent if cache_root.name == "long" else cache_root
    return base / "vintages"


# =====================================================================================
# fetch + cache
# =====================================================================================
def _fetch_range(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """One point, all variables x all leads, hourly UTC, `start`..`end` inclusive."""
    from http_client import get_with_retry
    hourly = ",".join(col for api, _ in _API_VARIABLES for col in vintage_columns(api))
    params = {"latitude": lat, "longitude": lon, "hourly": hourly, "models": MODEL,
              "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "UTC"}
    response = get_with_retry(PREVIOUS_RUNS_URL, params)
    if response.status_code == 429:           # per-minute cap: wait it out once
        time.sleep(65)
        response = get_with_retry(PREVIOUS_RUNS_URL, params)
    response.raise_for_status()
    body = response.json()["hourly"]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(body.pop("time"), utc=True)})
    for key, values in body.items():
        # single-model requests name columns without the model suffix; strip it if present
        frame[key[: -len(MODEL) - 1] if key.endswith("_" + MODEL) else key] = values
    expected = [c for api, _ in _API_VARIABLES for c in vintage_columns(api)]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Previous Runs response lacks {missing}")
    return frame[["timestamp"] + expected].astype({c: float for c in expected})


def _pause(start: date, end: date):
    weight = (len(_API_VARIABLES) * len(LEADS) / 10.0) * (((end - start).days + 1) / 14.0)
    time.sleep(weight / _WEIGHT_PER_MINUTE * 60.0 + 1.0)


def _meta_path(root) -> Path:
    return Path(root) / MODEL / "meta.json"


def _read_meta(root) -> dict:
    try:
        with open(_meta_path(root), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"model": MODEL, "points": {}}


def _write_meta(root, meta: dict):
    path = _meta_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _point_path(root, point: str) -> Path:
    return Path(root) / MODEL / f"{point}.pkl"


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([old, new], ignore_index=True)
    return (both.drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp").reset_index(drop=True))


def newest_available() -> date:
    """The latest date requested: yesterday (UTC); today's hours are still being issued."""
    return datetime.now(UTC).date() - timedelta(days=1)


def ensure_vintages(end: date, root=VINTAGE_CACHE_ROOT, start: date = START,
                    fetch: bool = True, log=print) -> dict:
    """
    Load the cached vintage frames for every production weather point, fetching
    what is missing first: a point with no cache (or whose coordinates changed, or
    whose cache starts after `start`) is fetched in full, one whose cache stops
    before `end` is extended from TAIL_OVERLAP_DAYS before its old end.

    `end` is capped at newest_available(); a snapshot that reaches past it fails
    the coverage check later with the dates, rather than here.

    Returns {point: DataFrame indexed by UTC timestamp, 21 `<api var>_previous_dayL`
    columns}. Nothing is written for a point that is already covered, so a run on
    a warm cache is offline.
    """
    root = Path(root)
    end = min(end, newest_available())
    meta = _read_meta(root)
    out = {}
    pause_for = None
    for point, (lat, lon) in weather_points().items():
        entry = meta["points"].get(point)
        path = _point_path(root, point)
        usable = (entry is not None and path.exists() and entry["lat"] == lat and entry["lon"] == lon
                  and date.fromisoformat(entry["requested_start"]) <= start)
        if usable and date.fromisoformat(entry["requested_end"]) >= end:
            out[point] = pd.read_pickle(path)
            continue
        if not fetch:
            raise VintageCoverageError(
                f"vintage cache {root}/{MODEL} does not cover {point} through {end} "
                f"(have {'nothing' if entry is None else entry['requested_end']}) and fetching is off")
        if usable:
            old = pd.read_pickle(path)
            lo = max(start, date.fromisoformat(entry["requested_end"]) - timedelta(days=TAIL_OVERLAP_DAYS))
        else:
            old, lo = None, start
        if pause_for:
            _pause(*pause_for)
        pause_for = (lo, end)
        t0 = time.time()
        new = _fetch_range(lat, lon, lo, end)
        frame = new if old is None else _merge(old, new)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".pkl.partial")      # an interrupted write must not leave a torn cache
        frame.to_pickle(partial)
        os.replace(partial, path)
        meta["points"][point] = {"lat": lat, "lon": lon, "requested_start": start.isoformat(),
                                 "requested_end": end.isoformat(),
                                 "fetched_at": datetime.now(UTC).isoformat()}
        _write_meta(root, meta)
        log(f"  vintages {point:<13} {lo} -> {end}  {len(new)} h "
            f"({'extended' if old is not None else 'fetched'}) [{time.time() - t0:.0f}s]")
        out[point] = frame
    return {p: f.set_index("timestamp") for p, f in out.items()}


# =====================================================================================
# which hours does a run evaluate, and are they covered
# =====================================================================================
def evaluated_dates(frame_dates, shifts, iterations: int = 52, step: int = 7) -> list:
    """
    The Swedish dates that are test rows at any of `shifts` -- the last
    iterations*step rows of the frame after each shift's tail truncation.
    """
    dates = pd.to_datetime(pd.Series(list(frame_dates)))
    n = len(dates)
    days = set()
    for shift in shifts:
        days |= set(dates.iloc[n - shift - iterations * step: n - shift].dt.date)
    return sorted(days)


def _hours_of(days) -> pd.DatetimeIndex:
    """Every UTC hour whose Europe/Stockholm date is in `days` (23, 24 or 25 per day)."""
    days = set(days)
    lo, hi = pd.Timestamp(min(days)), pd.Timestamp(max(days))
    hours = pd.date_range(lo - pd.Timedelta(days=1), hi + pd.Timedelta(days=2), freq="h", tz="UTC")
    local = pd.Series(hours.tz_convert("Europe/Stockholm").date, index=hours)
    return hours[local.isin(days).values]


def check_coverage(vintages: dict, days, points=None):
    """
    Raise VintageCoverageError unless every point has a finite vintage, at every
    variable and lead, for every hour of `days`. The message names the worst
    offenders and the first/last missing hour.
    """
    hours = _hours_of(days)
    problems = []
    for point in (points or list(vintages)):
        frame = vintages[point].reindex(hours)
        for api, _ in _API_VARIABLES:
            for col in vintage_columns(api):
                gaps = frame.index[frame[col].isna().values]
                if len(gaps):
                    problems.append((point, col, len(gaps), gaps.min(), gaps.max()))
    if problems:
        shown = "\n".join(f"    {p:<13} {c:<38} {n:>5} h  {a:%Y-%m-%d %H:%M} -> {b:%Y-%m-%d %H:%M}"
                          for p, c, n, a, b in problems[:12])
        more = f"\n    ... and {len(problems) - 12} more series" if len(problems) > 12 else ""
        raise VintageCoverageError(
            f"{len(problems)} vintage series have gaps over the {len(hours)} evaluated hours "
            f"({min(days)} -> {max(days)}):\n{shown}{more}\n"
            f"Vintages exist from {START}. A snapshot's newest days need a day or two to be "
            f"published (re-run later); anything earlier than {START} cannot be scored.")


# =====================================================================================
# hybrid inputs and frames
# =====================================================================================
def hybrid_inputs(inputs: dict, vintages: dict, lead: int, log=print) -> dict:
    """
    A copy of the snapshot's inputs whose weather columns carry the forecast
    issued `lead` days before each hour, for every production point the snapshot
    has columns for. Hours without a vintage keep the archive value (outside the
    evaluated days this is harmless; check_coverage guards the evaluated ones).
    Only weather_hourly / wind_intl_hourly are copied -- every other input is the
    snapshot's own object.
    """
    out = dict(inputs)
    for key in {frame_columns(p)[0] for p in vintages}:
        out[key] = inputs[key].copy()
    skipped = []
    for point, vintage in vintages.items():
        key, columns = frame_columns(point)
        frame = out[key]
        if not all(col in frame.columns for col in columns.values()):
            skipped.append(point)
            continue
        stamps = pd.to_datetime(frame["timestamp"], utc=True)
        for api, col in columns.items():
            new = vintage[f"{api}_previous_day{lead}"].reindex(stamps).values
            frame[col] = np.where(np.isnan(new), frame[col].values, new)
    if skipped and lead == LEADS[0]:
        log(f"  note: the snapshot carries no columns for {skipped}; those points are not swapped "
            f"(the model trained on it cannot read them either)")
    return out


def build_vintage_frames(inputs: dict, vintages: dict, archive: pd.DataFrame,
                         build=None, leads=LEADS, log=print) -> dict:
    """
    {lead: merged frame} built by `build` (default features.build_training_data,
    unmodified) on hybrid inputs, row-for-row aligned with `archive`, the frame
    built on the same snapshot's archive weather.

    Asserted on every frame: same dates in the same order, and no price / fuel /
    carbon / reservoir / nuclear column moved (see NON_WEATHER_PREFIXES).
    """
    if build is None:
        from features import build_training_data as build
    frames = {}
    for lead in leads:
        frame = build(**hybrid_inputs(inputs, vintages, lead, log=log))
        if list(frame["date"]) != list(archive["date"]):
            raise AssertionError(f"lead {lead}: the vintage frame has different rows than the archive frame "
                                 f"({len(frame)} vs {len(archive)})")
        if set(frame.columns) != set(archive.columns):
            raise AssertionError(f"lead {lead}: the vintage frame has different columns than the archive frame")
        moved = [c for c in archive.columns
                 if c.startswith(NON_WEATHER_PREFIXES) and not column_equal(archive[c], frame[c])]
        if moved:
            raise AssertionError(f"lead {lead}: vintage weather moved non-weather columns {moved}")
        frames[lead] = frame[list(archive.columns)]
    return frames


class VintageSet:
    """
    The vintage frames of one snapshot, one per lead, plus how to describe them.
    Build with VintageSet.for_snapshot; the harness only needs `.frames`.
    """

    def __init__(self, frames: dict, root, points: list):
        self.frames = frames
        self.root = root
        self.points = points

    @property
    def leads(self) -> tuple:
        return tuple(sorted(self.frames))

    @classmethod
    def for_snapshot(cls, inputs: dict, archive: pd.DataFrame, shifts, root, fetch: bool = True,
                     iterations: int = 52, step: int = 7, build=None, log=print) -> "VintageSet":
        """
        Fetch/extend the cache to cover the snapshot, check that every hour the
        shifts evaluate has a vintage at every lead, and build the lead frames.
        """
        days = evaluated_dates(archive["date"], shifts, iterations=iterations, step=step)
        if pd.Timestamp(min(days)).date() < START:
            raise VintageCoverageError(
                f"the shifts evaluate from {min(days)}, before the Previous Runs archive starts "
                f"({START}); lower the shifts' reach or run with --no-vintage")
        # A Swedish day's hours all fall on its own UTC date or the evening before, so the
        # last evaluated day needs the cache through that date and no further.
        vintages = ensure_vintages(max(days), root=root, fetch=fetch, log=log)
        check_coverage(vintages, days)
        t0 = time.time()
        frames = build_vintage_frames(inputs, vintages, archive, build=build, log=log)
        log(f"  vintage frames: {len(frames)} leads x {len(archive)} rows built in "
            f"{time.time() - t0:.0f}s from {len(vintages)} points; evaluated days "
            f"{min(days)} -> {max(days)} ({len(days)}) all covered")
        return cls(frames, root, list(vintages))
