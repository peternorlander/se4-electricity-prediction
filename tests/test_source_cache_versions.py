"""
A source's code may not change meaning without its cache version being looked at.

The production cache (source_cache.py) stores what the code made of a response. If a parser,
a request parameter or a reduction changes so that the same response would now give different
rows or columns, the cached history no longer means what new rows mean, and a merge would mix
the two silently. The cure is cheap -- bump `version` in source_cache.SOURCES, and the next
run re-fetches that source in full -- but only if somebody remembers. This test makes the
change impossible to miss: it fails when the code a source lists in `Source.code` differs from
the recorded fingerprint, and when the recorded version differs from the live one.

WHAT TO DO WHEN IT FAILS (docs/DATA_CACHE.md "When you change a source"):
  1. Decide: would the SAME API response now produce different rows/columns/values?
       yes -> bump that source's `version` in source_cache.SOURCES
       no  -> (a log line, a comment, a refactor) leave it
  2. Run `python -m se4.source_cache`, copy the line(s) for the changed source(s) into RECORDED.
  Both versions and fingerprints are recorded here on purpose: changing only one of them is
  visible in the diff.
"""
import pytest

from se4.source_cache import SOURCES, fingerprint

RECORDED = {
    "entsoe_prices_se4": (1, "a4408c09d59e"),
    "entsoe_prices_de_dk2": (1, "7799f4527557"),
    "weather_archive_se4": (1, "d5f8774fc137"),
    "wind_archive_intl": (1, "314b6718497f"),
    "yahoo_ttf": (1, "b4ce85157f95"),
    "yahoo_eua": (1, "f4a3e74148db"),
    "entsoe_nuclear_se3": (1, "fb9af971e7d3"),
    "nve_reservoir_norway": (1, "ea9f5ba1d195"),
    "entsoe_reservoir_sweden": (1, "439e341bf5ff"),
    "nve_reservoir_median": (1, "203bd478a5d2"),
}


def test_every_source_is_recorded():
    assert set(RECORDED) == set(SOURCES), (
        "source_cache.SOURCES and RECORDED in this file list different sources: "
        f"{sorted(set(RECORDED) ^ set(SOURCES))}")


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_source_code_change_is_acknowledged(name):
    source = SOURCES[name]
    recorded_version, recorded_print = RECORDED[name]
    live = fingerprint(source)
    assert (source.version, live) == (recorded_version, recorded_print), (
        f"The code behind cache source '{name}' (or its version) changed.\n"
        f"  recorded: version {recorded_version}, fingerprint {recorded_print}\n"
        f"  now:      version {source.version}, fingerprint {live}\n"
        "If the same API response would now give different rows, columns or values, bump "
        "`version` in source_cache.SOURCES so the next run fetches this source in full. "
        "Then run `python -m se4.source_cache` and update RECORDED here. See docs/DATA_CACHE.md.")


def test_listed_code_exists():
    """A renamed function would otherwise drop out of the fingerprint instead of failing."""
    for source in SOURCES.values():
        assert source.code, f"{source.name} lists no code to fingerprint"
        fingerprint(source)   # raises if a listed module or attribute is gone
