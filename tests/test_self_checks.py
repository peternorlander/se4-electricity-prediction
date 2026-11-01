"""The harness's own self-check scripts, offline parts. ab/check_vintage.py's stub of
sources.open_meteo went stale with the 2.3b `models` argument and nothing ran it; here it runs."""
import subprocess
import sys

import pytest

from conftest import ROOT


@pytest.mark.parametrize("script", ["ab/check_verdict.py", "ab/check_vintage.py"])
def test_self_check_passes(script):
    result = subprocess.run([sys.executable, script], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
