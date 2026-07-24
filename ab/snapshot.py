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

DEFAULT_CACHE_ROOT = Path("ab_cache")


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

    meta = {"today": today.isoformat(), "fetched_at": datetime.now().isoformat()}
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
        meta is {"today": ..., "fetched_at": ...}.
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
    with open(snapshot_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    inputs = {}
    for pkl_path in sorted(snapshot_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            inputs[pkl_path.stem] = pickle.load(f)

    return inputs, meta
