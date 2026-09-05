"""
Local snapshot cache for the A/B backtest flow. See the README "A/B Backtest
Flow" section.

A snapshot is a pickled copy of one fetch_training_inputs() result, saved
under ab_cache/<YYYY-MM-DD>/ so ab_test.py can replay the exact same fetched
data across many walk-forward shifts without re-fetching (fast local
iteration; no repeated ENTSO-E/Open-Meteo/NVE calls). Pickle is used instead
of parquet because pyarrow is not a project dependency and these files are
local-only, self-produced, and gitignored (see .gitignore: ab_cache/).

predict.py never reads this cache -- production always fetches fresh data.
"""
import json
import pickle
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_ROOT = Path("ab_cache")

# Long-window snapshots (ab_test.py fetch --days N, N > fetch_data.TRAINING_DAYS)
# live in a separate root, never the default one. Grid scripts assume every
# snapshot in DEFAULT_CACHE_ROOT is a normal-length (currently 1095-day) window
# and load_snapshot(None) picks the newest by mtime -- mixing a 5-year fetch
# into the same directory would let a routine `ab_test.py run` silently pick it
# up and compare BASELINE vs CANDIDATE on the wrong window length. Keeping long
# snapshots physically separate makes that impossible rather than merely
# documented.
LONG_CACHE_ROOT = DEFAULT_CACHE_ROOT / "long"


# Snapshots taken before 2026-08-23 were fetched with weather ending
# ~WEATHER_ARCHIVE_LAG_DAYS back; afterwards fetch_data tops the archive up to
# yesterday (README "Closing the weather-archive lag"). The merged frame is
# therefore ~4 rows longer on a newer snapshot, and the two are NOT
# interchangeable inside one measurement grid. Rather than stamp a version
# number that has to be remembered and bumped, every snapshot records the fact
# it actually depends on -- how far its weather reaches -- and load_snapshot
# derives the same fields for older snapshots that predate this.
def describe_weather_tail(inputs: dict, today: date) -> dict:
    """
    How far behind `today` a snapshot's weather stops.

    `weather_tail_days` is the number the caller usually wants: 1 means the
    frame reaches yesterday (post-top-up), ~5 means archive-only. Comparing it
    across snapshots is how you tell whether two caches can share a grid.

    Returns an empty dict if the snapshot has no weather frame at all, so this
    can never be the reason a save or load fails.
    """
    weather = inputs.get("weather_hourly")
    if weather is None or getattr(weather, "empty", True) or "timestamp" not in weather:
        return {}

    last = pd.to_datetime(weather["timestamp"], utc=True).max()
    return {
        "weather_last_timestamp": last.isoformat(),
        "weather_tail_days": int((today - last.date()).days),
    }


def read_meta(date_str: str, cache_root=DEFAULT_CACHE_ROOT) -> dict:
    """
    Read one snapshot's meta.json, deriving the weather-tail fields when the
    snapshot predates them.

    Cheap by design: it unpickles ONLY weather_hourly.pkl in the fallback path,
    never the whole snapshot, so `ab_test.py list` can report the tail for every
    cache without paying for a full load. `weather_tail_derived` marks the
    fallback, so "recorded at save time" and "worked out afterwards" stay
    distinguishable.

    Args:
        date_str:   Snapshot directory name (YYYY-MM-DD).
        cache_root: Root directory for all snapshots.

    Returns:
        The meta dict. Missing/unreadable meta.json returns {}.
    """
    snapshot_dir = Path(cache_root) / date_str
    try:
        with open(snapshot_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}

    if "weather_tail_days" in meta or "today" not in meta:
        return meta

    weather_path = snapshot_dir / "weather_hourly.pkl"
    if not weather_path.exists():
        return meta
    with open(weather_path, "rb") as f:
        weather = pickle.load(f)
    derived = describe_weather_tail({"weather_hourly": weather},
                                    date.fromisoformat(meta["today"]))
    return {**meta, **derived, "weather_tail_derived": True} if derived else meta


def save_snapshot(inputs: dict, today: date, cache_root=DEFAULT_CACHE_ROOT) -> Path:
    """
    Persist a fetch_training_inputs(today) result to disk.

    Args:
        inputs:     Dict returned by fetch_data.fetch_training_inputs().
        today:      The "as of" date the inputs were fetched for; also the
                    snapshot's directory name (YYYY-MM-DD).
        cache_root: Root directory for all snapshots.

    Returns:
        Path to the snapshot directory.
    """
    cache_root = Path(cache_root)
    snapshot_dir = cache_root / today.isoformat()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for key, value in inputs.items():
        with open(snapshot_dir / f"{key}.pkl", "wb") as f:
            pickle.dump(value, f)

    meta = {
        "today": today.isoformat(),
        "fetched_at": datetime.now().isoformat(),
        **describe_weather_tail(inputs, today),
    }
    with open(snapshot_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return snapshot_dir


def list_snapshots(cache_root=DEFAULT_CACHE_ROOT) -> list:
    """Return available snapshot date strings, oldest first."""
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return []
    return sorted(
        p.name for p in cache_root.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )


def load_snapshot(date_str: str = None, cache_root=DEFAULT_CACHE_ROOT):
    """
    Load a saved snapshot.

    Args:
        date_str:   Snapshot directory name (YYYY-MM-DD). None = newest available.
        cache_root: Root directory for all snapshots.

    Returns:
        (inputs, meta) -- inputs is the fetch_training_inputs()-shaped dict,
        meta is {"today", "fetched_at", "weather_last_timestamp",
        "weather_tail_days"}. The last two are read from meta.json when the
        snapshot was saved with them and derived from the data otherwise (with
        "weather_tail_derived": True), so snapshots taken before 2026-08-23
        report the same fields as new ones.
    """
    cache_root = Path(cache_root)
    available = list_snapshots(cache_root)
    if not available:
        raise FileNotFoundError(
            f"No snapshots found under {cache_root}. Run 'python ab_test.py fetch' first."
        )

    if date_str is None:
        date_str = available[-1]
    elif date_str not in available:
        raise FileNotFoundError(f"Snapshot '{date_str}' not found. Available: {available}")

    snapshot_dir = cache_root / date_str
    meta = read_meta(date_str, cache_root=cache_root)

    inputs = {}
    for pkl_path in sorted(snapshot_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            inputs[pkl_path.stem] = pickle.load(f)

    return inputs, meta
