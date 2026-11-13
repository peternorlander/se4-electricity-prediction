"""
Seed the production source cache from an existing A/B snapshot, so a new snapshot fetch only
asks the APIs for what is new.

WHY: `ab_test.py fetch` re-downloads ~5 years of every source. When one of them is down for
long windows (Yahoo's TTF=F answering 0 rows, 2026-11-13) the whole snapshot is blocked,
although 99 % of it already sits in the newest ab_cache/long/ snapshot. Production solved the
same problem with se4.source_cache (docs/DATA_CACHE.md); this module feeds that same machinery
from a snapshot directory instead of from the Actions cache, so the merge rules, the overlap
and the stale-cache limits are the ones production already trusts -- nothing is re-implemented.

HOW: the snapshot's frames are written into a throw-away directory in the cache's own record
format, `SOURCE_CACHE_DIR` is pointed at it for the duration of the fetch, and
`fetch_data.fetch_training_inputs` runs unchanged. The directory is never inside the snapshot
folder (load_snapshot unpickles every *.pkl it finds there).
"""
import contextlib
import dataclasses
import os
import tempfile
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

import pandas as pd

from se4 import source_cache
from se4.source_cache import SOURCES

# snapshot file stem -> source_cache source name
SNAPSHOT_SOURCES = {
    "prices_hourly": "entsoe_prices_se4",
    "market_prices_hourly": "entsoe_prices_de_dk2",
    "weather_hourly": "weather_archive_se4",
    "wind_intl_hourly": "wind_archive_intl",
    "ttf_daily": "yahoo_ttf",
    "eua_daily": "yahoo_eua",
    "nuclear_daily": "entsoe_nuclear_se3",
    "norway_reservoir_weekly": "nve_reservoir_norway",
    "sweden_reservoir_weekly": "entsoe_reservoir_sweden",
    "norway_reservoir_median": "nve_reservoir_median",
}

# Snapshot weather is archive + a forecast-endpoint top-up (fetch_data._splice_recent). The
# production cache holds the archive part only, so the top-up must not be seeded as archive.
_ARCHIVE_SOURCES = ("weather_archive_se4", "wind_archive_intl")


def source_versions() -> dict:
    """{source name: version} of the running code -- recorded in every snapshot's meta.json."""
    return {name: source.version for name, source in SOURCES.items()}


def _first_day(frame: pd.DataFrame, key: str) -> date:
    first = frame[key].min()
    return first.date() if isinstance(first, pd.Timestamp) else first


def seed_source_cache(inputs: dict, meta: dict, root: Path, archive_lag_days: int) -> dict:
    """
    Write `inputs` (a loaded snapshot) into `root` as source-cache records.

    Returns {source name: reason} for every source that was NOT seeded (so it will be fetched
    in full): a version mismatch, or a frame the cache cannot use. An empty dict means all
    ten were seeded.
    """
    root = Path(root)
    today = date.fromisoformat(meta["today"])
    recorded = meta.get("source_versions")
    fetched = datetime.fromisoformat(meta["fetched_at"]) if meta.get("fetched_at") else None
    if fetched is not None and fetched.tzinfo is None:
        fetched = fetched.astimezone()          # meta.json stores local naive time
    saved_at = (fetched or datetime.now(UTC)).astimezone(UTC).isoformat()
    skipped = {}

    for stem, name in SNAPSHOT_SOURCES.items():
        source = SOURCES[name]
        frame = inputs.get(stem)
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            skipped[name] = f"{stem} missing or empty in the snapshot"
            continue
        if recorded is not None and recorded.get(name) != source.version:
            skipped[name] = (f"snapshot was built with version {recorded.get(name)}, "
                             f"code is at {source.version}")
            continue

        extra = {}
        if source.key is not None:
            frame = source_cache._normalise(frame, source.key)
            if name in _ARCHIVE_SOURCES:
                cut = pd.Timestamp(today - timedelta(days=archive_lag_days), tz="UTC")
                frame = frame[frame[source.key] < cut].reset_index(drop=True)
            if frame.empty:
                skipped[name] = "nothing left after cutting the forecast top-up"
                continue
            extra["covers_from"] = _first_day(frame, source.key)
        # cached_snapshot judges the age of its fallback copy from saved_at, so a snapshot
        # copy must carry the time of the fetch that made it, not the time of seeding.
        source_cache._save(source, root, frame, saved_at=saved_at, **extra)
    return skipped


def overlay_production_cache(root: Path, overlay: Path) -> dict:
    """
    Lay the rows of a production source cache (downloaded from the Actions cache, see
    .github/workflows/export_source_cache.yml) over the seeded cache in `root`.

    Production's cache is 3 years long, so it cannot stand in for a 5-year A/B window; it can
    only EXTEND the snapshot it is laid over. Where both hold a row the production row wins
    (it is the newer fetch). A source is left alone -- and so keeps the snapshot's rows -- when
    the production file has another version, is unreadable, or would leave a hole: its first
    row must reach back to the snapshot's last day (+ the source's join gap).

    Returns {source name: one-line outcome} for the sources it looked at.
    """
    overlay = Path(overlay)
    if (overlay / "source_cache").is_dir():
        overlay = overlay / "source_cache"
    report = {}
    for name, source in SOURCES.items():
        if source.key is None:
            continue
        record = source_cache._load(source, overlay)        # prints why a version/file is rejected
        base = source_cache._load(source, root)
        if record is None or base is None:
            report[name] = "not overlaid (no usable production file or no seeded rows)"
            continue
        new, old = record["frame"], base["frame"]
        old_last = source_cache._day(old[source.key].max())
        new_first = source_cache._day(new[source.key].min())
        if new_first > old_last + timedelta(days=source.max_join_gap_days):
            report[name] = f"not overlaid: starts {new_first}, the snapshot ends {old_last}"
            continue
        if set(new.columns) != set(old.columns):
            report[name] = "not overlaid: columns differ"
            continue
        overlapping, revised = source_cache._revisions(old, new, source.key)
        merged = (pd.concat([old, new], ignore_index=True)
                    .drop_duplicates(source.key, keep="last")
                    .sort_values(source.key, kind="mergesort").reset_index(drop=True))
        source_cache._save(source, root, merged, covers_from=base["covers_from"],
                           saved_at=base["saved_at"])
        new_last = source_cache._day(merged[source.key].max())
        report[name] = (f"ends {old_last} -> {new_last} ({revised} of {overlapping} "
                        f"overlapping rows differ from the snapshot)")
    return report


@contextlib.contextmanager
def seeded_cache(inputs: dict, meta: dict, archive_lag_days: int, max_stale_days=None,
                 overlay=None):
    """
    Context in which `fetch_data.fetch_training_inputs` is answered from the seeded cache.

    `max_stale_days`: when given, replaces every source's own stale limit (only those that
    have one). It lets a snapshot be built while a source such as Yahoo is down, at the price
    of a final stretch of forward-filled values -- the caller must say so loudly and record it.
    Without it, production's limits apply and a source that cannot be refreshed fails the fetch.

    `overlay`: a downloaded production source cache laid over the snapshot's rows first
    (overlay_production_cache).

    Yields (skipped, directory) -- `skipped` as returned by seed_source_cache.
    """
    saved_env = os.environ.get(source_cache.CACHE_DIR_ENV)
    saved_sources = dict(SOURCES)
    with tempfile.TemporaryDirectory(prefix="ab_seed_") as tmp:
        try:
            skipped = seed_source_cache(inputs, meta, Path(tmp), archive_lag_days)
            if overlay is not None:
                print(f"Overlaying production cache {overlay}:")
                for name, outcome in overlay_production_cache(Path(tmp), overlay).items():
                    print(f"  {name}: {outcome}")
            if max_stale_days is not None:
                for name, source in saved_sources.items():
                    if source.max_stale_days is not None:
                        SOURCES[name] = dataclasses.replace(source, max_stale_days=max_stale_days)
            os.environ[source_cache.CACHE_DIR_ENV] = tmp
            yield skipped, Path(tmp)
        finally:
            SOURCES.clear()
            SOURCES.update(saved_sources)
            if saved_env is None:
                os.environ.pop(source_cache.CACHE_DIR_ENV, None)
            else:
                os.environ[source_cache.CACHE_DIR_ENV] = saved_env
