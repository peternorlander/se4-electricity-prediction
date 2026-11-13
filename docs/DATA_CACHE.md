# The source cache

Production does not re-download three years of every source on each run. It keeps what it
has fetched and asks each source only for what is new. The code is
[`se4/source_cache.py`](../se4/source_cache.py); the wiring is `fetch_data.fetch_training_inputs`.

**Why it exists.** A run used to be as reliable as the least reliable API on its worst day.
On 2026-11-06, -07 and -08 Yahoo answered the 3-year TTF request with 1–3 rows
(`first row is 2026-11-06, window starts 2023-11-09`) in every attempt of every retry, and
the run, correctly, refused to train on it. Waiting a few hours fixed it. Open-Meteo's
archive timed out on the first attempt of nearly every run. Historical closes, prices and
temperatures do not change, so a run needs the last few weeks from the API, not the last
three years.

## How it behaves

| | |
|---|---|
| **Off by default.** | Only a run with `SOURCE_CACHE_DIR` set uses it. The production workflow sets it. Local runs, the tests and the A/B snapshot fetch (`se4/ab/snapshot.py`) are unchanged, so an A/B snapshot is always a fresh fetch. |
| **What is cached** | Everything `fetch_training_inputs` fetches over the 3-year window: SE4 prices, DE/DK2 prices, the Open-Meteo **archive** (SE4 weather and the DE/DK wind points), TTF, EUA, SE3 nuclear outages, the NVE and Swedish reservoirs. The NVE 20-year median is kept only as a fallback copy. |
| **What is not** | Anything that has to be fresh: the 8-day weather and wind forecast, the recent-analysis top-up that fills the archive's lag, the tomorrow-price anchor, the planned-outage forecast, Nord Pool's published-date check. |
| **Each run** | Fetch from `last cached day − overlap_days` to today and merge. **Where cache and fetch both hold a row, the fetched row wins**; the log says how many overlapping rows differ. The result is trimmed to the window and equals a full fetch (`tests/test_fetch_data_cache.py` checks it). The new total is saved. |
| **A full fetch instead** | No cache, a cache of another version, an unreadable file, a window reaching further back than the cache does, or a fetch that returns other columns than the cache holds. |
| **When the tail fetch fails** | Sources with `max_stale_days` (Yahoo 5, weather archive 3, reservoirs 14–21) serve the cache if it is no older than that, with a `WARNING` that says why. The weather figure is 3 because `fetch_data` tops the archive up 8 days from the forecast endpoint, so an archive 3 days behind still leaves no hole. Every other source (prices, nuclear) lets the failure fail the run, as before, so the previous prediction stays in Home Assistant. A tail that starts later than `max_join_gap_days` after the cache ends, or stops more than `max_stale_days` before today, is a failed fetch too. |
| **Where it lives** | `source_cache/<name>.pkl` on the runner, kept between runs by `actions/cache` (restore the newest entry, save a new one each run, also when the run fails after some sources succeeded). Entries unused for 7 days are evicted by GitHub; a lost cache costs one full fetch, exactly today's behaviour. |

## The cold start

An empty cache (the first run after this was added, or after GitHub evicted it) takes one
full fetch, which is today's behaviour and as fragile as it was. Probed from a laptop on
2026-11-08, during the outage that prompted this: `TTF=F` returned **one row (2026-11-06)
for every request shape tried** (`yf.download` with start/end over 1, 2 and 3 years,
`period="5y"`, `period="max"`, `Ticker.history`), while `CO2.L` returned its 758 rows. So the
fault was Yahoo's data for that ticker, not the window length, the runner's IP or the
request, and retrying with another shape would not have helped. A warm cache is not hit by it
(the tail needs the last day or two; one row is enough), but a cold start during such an
outage fails until Yahoo recovers. If that happens in practice, add a seed file to the repo
(`source_cache` would read it when no cache exists) -- not built, because the cache is
expected to be warm every day.

## When you change a source: the version rule

The cache stores **what the code made of a response**. If a change means the same response
would now give different rows, columns or values, the cached history no longer means what new
rows mean, and merging them would mix two definitions silently. **Bump that source's
`version` in `source_cache.SOURCES`, in the same commit.** The next run then fetches that
source in full.

Bump when you change:

- how a source is **requested**: a URL, a parameter, a unit, a location list, a document
  type, a window or chunking that changes the answer;
- how it is **parsed**: a new curve rule, a fill, a dedupe, a different column, a renamed
  column, a changed dtype or unit;
- how it is **reduced** before it is returned: the nuclear day-count, a unit conversion.

Do not bump for a log line, a comment, a refactor with identical output, or a change in how
`se4/features.py` *uses* the frame afterwards (the cache holds the fetch result, not features).

Two things enforce it:

1. `tests/test_source_cache_versions.py` fingerprints the functions and constants each source
   lists in `Source.code`. If one changes, the test fails and prints what to do: decide
   whether the version needs bumping, then run `python -m se4.source_cache` and paste the new
   line into the test's `RECORDED` table. Version and fingerprint are recorded separately so
   a fingerprint updated without a bump is visible in the diff. The test cannot decide for
   you; it only makes sure somebody did.
2. A source whose code changes *shape* without a bump is caught at run time as well: columns
   that differ from the cache trigger a full fetch with a warning.

Adding a source: add a `Source` to `SOURCES` (key `timestamp` or `date`, `overlap_days`
long enough to cover the source's revisions, `max_stale_days` only if a slightly old copy is
better than no run, and a `code` list), wrap its fetch in `fetch_training_inputs` with
`cached_fetch(SOURCES[...], start, end, lambda s, e: ...)`, and add it to the test's table.
`tests/test_fetch_data_cache.py` lists the sources a second run must ask only for the tail.

To throw the whole cache away: bump `v1` to `v2` in both `source-cache-v1-` keys of
`.github/workflows/daily_predict.yml`, or delete the entries (`gh cache delete --all`).

## Limits worth knowing

- **Revisions older than `overlap_days` are not picked up.** Measured on 12 pairs of
  independently fetched snapshots, SE4 prices, weather, nuclear outages and TTF were
  byte-identical across fetches ([FINDINGS.md](FINDINGS.md#cached-history-is-stable-except-for-one-column-2026-09-17)),
  which is what makes caching them safe. Not measured: the final ERA5 values that replace the
  archive's preliminary ones weeks later, and late A77 outage notices (hence 60 days for
  nuclear). The per-run log line `N of M overlapping rows differ from the cache` is how a
  source that does revise shows itself. If one does, raise its `overlap_days` (no version
  bump: it only widens what is re-fetched).
- **A parser fix needs a bump to reach old rows.** The overlap re-fetches only the last
  weeks; older cached rows keep their old meaning until the version changes. That is what the
  version rule is for.
- **The cache is only as good as the run that wrote it.** A tail merged from a bad answer is
  saved. The existing guards still apply on every run (`data_checks.check_training_data`,
  the coverage checks inside the fetchers), and the overlap logging shows when a source starts
  revising rows.
- **A/B snapshots never read the production cache.** `ab_test.py fetch` is uncached by default;
  `fetch --incremental` seeds a *temporary* cache from the newest A/B snapshot with the same
  merge rules (`se4/ab/seed.py`, [AB_TESTING.md](AB_TESTING.md#fetching-a-new-snapshot-when-a-source-is-unreliable---incremental)).
  A cached production frame is never an A/B input.
