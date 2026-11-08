"""
Versioned on-disk cache for the big historical source fetches.

WHY: every production run used to re-download three years of every source, although a
day-old price, temperature or gas close never changes. That made the run only as reliable
as the least reliable API on the worst day: Yahoo answering a 3-year request with 1-3 rows
(2026-11-01, -06, -07, -08: "first row is 2026-11-06, window starts 2023-11-09"), Open-Meteo
archive read timeouts on the first attempt of almost every run. With this cache a run asks
each source only for what is new -- the last `overlap_days` and everything after -- and
merges that into what it already holds. See docs/DATA_CACHE.md.

HOW: `cached_fetch(SOURCES[name], start, end, fetch_window)` wraps the existing fetch
function. It is OFF unless the environment variable SOURCE_CACHE_DIR is set (the production
workflow sets it), so a local run, the tests and the A/B snapshot fetch behave exactly as
before. Per source:

  * no cache, a cache from another `version`, an unreadable file or a window that reaches
    further back than the cache does  -> one full fetch, exactly as before;
  * otherwise fetch from `last cached day - overlap_days`; where cache and fetch both hold
    a row, THE FETCHED ROW WINS (sources revise, and a parser fix lands in the overlap);
  * if that tail fetch fails or comes back unusable, the cache is served instead -- but only
    for a source with `max_stale_days`, only while the cache is no older than that, and
    loudly. Without it the failure propagates as it always did.

THE VERSION RULE (the part that must not be forgotten): a cache stores what the code made of
a response. If you change how a source is requested, parsed or reduced -- so that the SAME
response would now give DIFFERENT rows or columns -- bump that source's `version` below, in
the same commit. The next run then re-fetches that source in full instead of mixing old and
new meaning. tests/test_source_cache_versions.py fails when a source's code changes without
this being acknowledged; docs/DATA_CACHE.md says what to do about it.
"""
import hashlib
import importlib
import inspect
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR_ENV = "SOURCE_CACHE_DIR"


@dataclass(frozen=True)
class Source:
    """One cached source. Edit `version` when the code listed in `code` changes meaning."""
    name: str
    version: int
    key: str | None                 # "timestamp" / "date"; None for a whole-frame snapshot
    overlap_days: int = 21          # re-fetched every run, and the fetched rows win
    max_stale_days: int | None = None   # serve the cache this far behind `end` when the
                                        # fetch fails; None = a failed fetch fails the run
    max_join_gap_days: int = 5      # a tail starting later than last cached day + this left a hole
    code: tuple = ()                # "module:attr" refs fingerprinted by the version test


# Everything the entso_e price parsers depend on (shared by SE4 and DE/DK2 on purpose: a
# change there changes both caches).
_ENTSOE_PRICE_CODE = tuple(f"sources.entso_e:{n}" for n in (
    "_MAX_RANGE_DAYS", "_VARIABLE_BLOCK_CURVE", "_RESOLUTION_STEPS", "_DAY_AHEAD_CONTRACT",
    "_SDAC_SEQUENCE", "_SEQUENCE_PARAM", "_SEQUENCE_PINNED_AREAS", "_get_offset_for_position",
    "_parse_point", "_period_positions", "_fill_variable_blocks", "_parse_period",
    "_timeseries_meta", "_parse_price_document", "_select_one_price_per_timestamp",
    "_fetch_prices_area_chunk", "_fetch_prices_area"))

SOURCES = {s.name: s for s in (
    Source("entsoe_prices_se4", version=1, key="timestamp", overlap_days=21,
           max_join_gap_days=2,
           code=_ENTSOE_PRICE_CODE + ("sources.entso_e:SE4_AREA_CODE",
                                      "sources.entso_e:fetch_prices")),
    Source("entsoe_prices_de_dk2", version=1, key="timestamp", overlap_days=21,
           max_join_gap_days=2,
           code=_ENTSOE_PRICE_CODE + ("sources.entso_e:DE_LU_AREA_CODE",
                                      "sources.entso_e:DK2_AREA_CODE",
                                      "sources.entso_e:fetch_market_prices")),
    # The archive publishes with a 5-day lag and fetch_data tops it up 8 days from the
    # forecast endpoint, so a cache up to 3 days behind still leaves no hole after the splice.
    Source("weather_archive_se4", version=1, key="timestamp", overlap_days=21,
           max_stale_days=3, max_join_gap_days=2,
           code=tuple(f"sources.open_meteo:{n}" for n in (
               "LOCATION_LAT", "LOCATION_LON", "HOURLY_VARIABLES", "TIMEZONE",
               "OPEN_METEO_ARCHIVE_URL", "_hourly_values", "_model_params",
               "_parse_response", "fetch_historical"))),
    Source("wind_archive_intl", version=1, key="timestamp", overlap_days=21,
           max_stale_days=3, max_join_gap_days=2,
           code=tuple(f"sources.open_meteo:{n}" for n in (
               "WIND_LOCATIONS", "INTL_VARIABLES", "TIMEZONE", "OPEN_METEO_ARCHIVE_URL",
               "_hourly_values", "_model_params", "_fetch_location_series",
               "_fetch_all_locations", "fetch_international_wind_historical"))),
    Source("yahoo_ttf", version=1, key="date", overlap_days=21, max_stale_days=5,
           max_join_gap_days=6,
           code=("sources.yahoo_finance:TTF_TICKER", "sources.yahoo_finance:_to_frame",
                 "sources.yahoo_finance:fetch_ttf_prices")),
    Source("yahoo_eua", version=1, key="date", overlap_days=21, max_stale_days=5,
           max_join_gap_days=6,
           code=("sources.eua_carbon:EUA_TICKER", "sources.yahoo_finance:_to_frame",
                 "sources.eua_carbon:fetch_eua_prices")),
    # Outage announcements keep arriving after the fact, so look back further.
    Source("entsoe_nuclear_se3", version=1, key="date", overlap_days=60,
           max_join_gap_days=2,
           code=tuple(f"sources.entso_e:{n}" for n in (
               "SE3_AREA_CODE", "_parse_outage_timeseries", "_fetch_outages_chunk",
               "fetch_nuclear_outages_se3"))),
    # Weekly series: published with a lag of up to ~11 days, so "stale" is measured in weeks.
    Source("nve_reservoir_norway", version=1, key="date", overlap_days=60,
           max_stale_days=14, max_join_gap_days=10,
           code=("sources.nve:NVE_API_BASE", "sources.nve:fetch_reservoir_norway")),
    Source("entsoe_reservoir_sweden", version=1, key="date", overlap_days=60,
           max_stale_days=21, max_join_gap_days=10,
           code=tuple(f"sources.entso_e:{n}" for n in (
               "SWEDEN_AREA_CODE", "_parse_reservoir_point", "_RESERVOIR_STEPS",
               "_fill_reservoir_positions", "_fetch_reservoir_chunk",
               "fetch_reservoir_sweden"))),
    # Not a time series: one small frame, kept only as a fallback for a failed fetch.
    Source("nve_reservoir_median", version=1, key=None, max_stale_days=400,
           code=("sources.nve:NVE_API_BASE", "sources.nve:fetch_reservoir_norway_median")),
)}


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------

def cache_dir() -> Path | None:
    """The cache directory, or None when caching is off (SOURCE_CACHE_DIR unset or empty)."""
    value = os.environ.get(CACHE_DIR_ENV, "").strip()
    return Path(value) if value else None


def _path(root: Path, source: Source) -> Path:
    return root / f"{source.name}.pkl"


def _load(source: Source, root: Path):
    """The stored record for `source`, or None -- with the reason printed."""
    path = _path(root, source)
    if not path.exists():
        print(f"  -> {source.name}: no cache yet, fetching in full")
        return None
    try:
        with open(path, "rb") as handle:
            record = pickle.load(handle)
        if not isinstance(record, dict) or not isinstance(record.get("frame"), pd.DataFrame):
            raise ValueError("not a cache record")
    except Exception as exc:  # noqa: BLE001 -- any unreadable file is just a miss
        logger.warning("%s: cache file unreadable (%s: %s) -- fetching in full",
                       source.name, type(exc).__name__, exc)
        return None
    if record.get("version") != source.version:
        print(f"  -> {source.name}: cache is version {record.get('version')}, code wants "
              f"version {source.version} -- fetching in full")
        return None
    return record


def _save(source: Source, root: Path, frame: pd.DataFrame, **extra) -> None:
    """Write atomically; a crash mid-write must never leave a half file for the next run."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        record = {"version": source.version, "frame": frame,
                  "saved_at": datetime.now(UTC).isoformat(), **extra}
        with tempfile.NamedTemporaryFile(dir=root, delete=False, suffix=".tmp") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp = Path(handle.name)
        os.replace(tmp, _path(root, source))
    except Exception as exc:  # noqa: BLE001 -- failing to save costs the next run, not this one
        logger.warning("%s: could not write the cache (%s: %s)",
                       source.name, type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# time-series helpers
# ---------------------------------------------------------------------------

def _normalise(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    frame = frame.copy()
    if key == "timestamp":
        frame[key] = pd.to_datetime(frame[key], utc=True)
    else:
        frame[key] = pd.to_datetime(frame[key]).dt.date
    return frame


def _day(value) -> date:
    return value.date() if isinstance(value, pd.Timestamp) else value


def _bound(day: date, key: str):
    """`day` as a value comparable with the key column."""
    return pd.Timestamp(day, tz="UTC") if key == "timestamp" else day


def _revisions(cached: pd.DataFrame, fresh: pd.DataFrame, key: str) -> tuple:
    """(rows both hold, of those how many differ in a numeric column)."""
    both = cached.merge(fresh, on=key, suffixes=("_old", "_new"))
    changed = np.zeros(len(both), dtype=bool)
    for column in fresh.columns:
        if column == key or f"{column}_old" not in both or f"{column}_new" not in both:
            continue
        old, new = both[f"{column}_old"], both[f"{column}_new"]
        if pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new):
            changed |= ~np.isclose(old.astype(float), new.astype(float),
                                   rtol=1e-9, atol=1e-9, equal_nan=True)
    return len(both), int(changed.sum())


def _tail_problem(source: Source, fresh, cached: pd.DataFrame, end: date):
    """Why the fetched tail cannot be merged, or None."""
    if fresh is None or len(fresh) == 0:
        return "no rows returned"
    if source.key not in fresh.columns:
        return f"no '{source.key}' column in the answer"
    fresh_key = _normalise(fresh, source.key)[source.key]
    first, last = _day(fresh_key.min()), _day(fresh_key.max())
    cached_last = _day(cached[source.key].max())
    if first > cached_last + timedelta(days=source.max_join_gap_days):
        return (f"tail starts {first} but the cache ends {cached_last}: "
                f"the days between would be missing")
    if source.max_stale_days is not None and (end - last).days > source.max_stale_days:
        return f"tail ends {last}, {(end - last).days} days before {end}"
    return None


def cached_fetch(source: Source, start: date, end: date, fetch_window) -> pd.DataFrame:
    """
    `fetch_window(start, end)` for the window [start, end], answered from the cache where
    possible. `fetch_window` is the source's existing fetch function wrapped to take two
    `date`s; it must return a frame with a `source.key` column.

    Returns what a full fetch would have (a frame with key >= start, sorted by key), except
    that rows older than the overlap come from the cache.
    """
    root = cache_dir()
    if root is None:
        return fetch_window(start, end)

    record = _load(source, root)
    if record is not None and record["covers_from"] > start:
        print(f"  -> {source.name}: cache starts {record['covers_from']}, window starts "
              f"{start} -- fetching in full")
        record = None

    def full_fetch():
        frame = fetch_window(start, end)
        if frame is not None and len(frame):
            _save(source, root, _normalise(frame, source.key), covers_from=start)
        return frame

    if record is None:
        return full_fetch()

    cached = record["frame"]
    cached_last = _day(cached[source.key].max())
    fetch_from = min(max(start, cached_last - timedelta(days=source.overlap_days)), end)
    print(f"  -> {source.name}: cache holds {len(cached)} rows to {cached_last}; "
          f"fetching {fetch_from} -> {end}")

    try:
        fresh = fetch_window(fetch_from, end)
        problem = _tail_problem(source, fresh, cached, end)
    except Exception as exc:  # noqa: BLE001 -- decided below: serve the cache or re-raise
        fresh, problem = None, f"{type(exc).__name__}: {exc}"

    if problem is not None:
        behind = (end - cached_last).days
        if source.max_stale_days is not None and behind <= source.max_stale_days:
            logger.warning("%s: refresh failed (%s) -- serving the cache, which ends %s "
                           "(%d days before %s, limit %d)", source.name, problem, cached_last,
                           behind, end, source.max_stale_days)
            return cached[cached[source.key] >= _bound(start, source.key)].reset_index(drop=True)
        raise RuntimeError(
            f"{source.name}: refresh failed ({problem}) and the cache ends {cached_last}, "
            f"{behind} days before {end}"
            + ("" if source.max_stale_days is None
               else f" (limit {source.max_stale_days})")
            + ". Nothing is served from a cache that old."
        )

    if set(fresh.columns) != set(cached.columns):
        # The code changed shape without a version bump: the cache cannot be trusted to
        # mean what the new rows mean. Rebuild rather than mix, and say so.
        logger.warning("%s: columns changed (%s) without a version bump in source_cache.py "
                       "-- fetching in full", source.name,
                       sorted(set(fresh.columns) ^ set(cached.columns)))
        return full_fetch()

    fresh = _normalise(fresh, source.key)
    overlapping, revised = _revisions(cached, fresh, source.key)
    merged = (pd.concat([cached, fresh], ignore_index=True)
                .drop_duplicates(source.key, keep="last")
                .sort_values(source.key, kind="mergesort"))
    merged = merged[merged[source.key] >= _bound(start, source.key)].reset_index(drop=True)
    print(f"  -> {source.name}: {len(merged)} rows ({len(fresh)} fetched, {revised} of "
          f"{overlapping} overlapping rows differ from the cache)")
    _save(source, root, merged, covers_from=max(start, record["covers_from"]))
    return merged


def cached_snapshot(source: Source, fetch) -> pd.DataFrame:
    """
    `fetch()` for a frame that is not a time series, with the last good copy as a fallback.

    The copy is served only when the fetch raises or returns an empty frame, and only while
    it is no older than `max_stale_days`.
    """
    root = cache_dir()
    if root is None:
        return fetch()

    problem, frame = None, None
    try:
        frame = fetch()
        if frame is not None and len(frame):
            _save(source, root, frame)
            return frame
        problem = "no rows returned"
    except Exception as exc:  # noqa: BLE001
        problem = f"{type(exc).__name__}: {exc}"

    record = _load(source, root)
    if record is not None:
        age = (datetime.now(UTC) - datetime.fromisoformat(record["saved_at"])).days
        if age <= (source.max_stale_days or 0):
            logger.warning("%s: refresh failed (%s) -- serving the copy from %d days ago",
                           source.name, problem, age)
            return record["frame"]
    if frame is not None:
        return frame          # the fetch's own (empty) answer, as before
    raise RuntimeError(f"{source.name}: fetch failed ({problem}) and there is no usable copy")


# ---------------------------------------------------------------------------
# the version guard
# ---------------------------------------------------------------------------

def _normalised_source(obj) -> str:
    """A function's source without comments, blank lines or trailing spaces (so a reformat
    or a comment edit does not count as a change of meaning; a docstring edit does)."""
    lines = [line.rstrip() for line in inspect.getsource(obj).splitlines()]
    return "\n".join(line for line in lines if line.strip() and not line.strip().startswith("#"))


def fingerprint(source: Source) -> str:
    """
    A short hash of the code that decides what `source` stores: the functions and constants
    listed in `source.code`. tests/test_source_cache_versions.py compares it with the value
    recorded there, which is how a change to that code gets noticed and a decision forced:
    bump `version` (the stored rows would now mean something else) or leave it (they would not).
    """
    digest = hashlib.sha256()
    for ref in source.code:
        module_name, attr = ref.split(":")
        obj = getattr(importlib.import_module(module_name), attr)
        text = _normalised_source(obj) if callable(obj) else repr(obj)
        digest.update(f"{ref}\n{text}\n".encode("utf-8"))
    return digest.hexdigest()[:12]


if __name__ == "__main__":
    # Prints the table tests/test_source_cache_versions.py expects, after you have decided
    # whether the version needs bumping.
    for source in SOURCES.values():
        print(f'    "{source.name}": ({source.version}, "{fingerprint(source)}"),')
