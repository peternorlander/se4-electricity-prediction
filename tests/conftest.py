"""Shared fixtures. The suite runs from the repository root: `python -m pytest`."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ab.snapshot import LONG_CACHE_ROOT, list_snapshots  # noqa: E402

SNAPSHOTS = list_snapshots(ROOT / LONG_CACHE_ROOT)


def pytest_generate_tests(metafunc):
    # Tests that take `snapshot` run once per cached long snapshot, and are skipped where
    # there is none (the Actions runner: ab_cache/ is gitignored).
    if "snapshot" in metafunc.fixturenames:
        metafunc.parametrize("snapshot", SNAPSHOTS or [pytest.param(None, marks=pytest.mark.skip(
            reason=f"no snapshot under {LONG_CACHE_ROOT}/"))])
