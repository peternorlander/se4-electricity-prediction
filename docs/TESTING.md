# Tests

`tests/` guards **contracts and invariants**: rules about the code that a defect broke once,
or would break silently. It does not judge whether the model is good. Model quality is
measured by the A/B harness ([AB_TESTING.md](AB_TESTING.md)). Production data is guarded by
`data_checks.py` inside `predict.py`. The three never stand in for each other:

| Guard | Asks | Runs |
|---|---|---|
| `tests/` | Does the code still do what it promises? | Every push and pull request (`.github/workflows/tests.yml`), and locally before a commit |
| `data_checks.py` | Is today's fetched data sane (wind unit, price order, finite)? | Every production run, before fitting |
| A/B harness | Is a change better? | When a change is proposed |

Run it with `pip install -r requirements-dev.txt`, then `python -m pytest` from the repository
root. The whole suite takes under a minute.

## When a test is added

- **Every fixed defect gets one.** It must fail on the code before the fix. Examples: the
  km/h-into-m/s wind curve in `test_wind.py`, and the one-row Yahoo answer in
  `test_yahoo_download.py`.
- **Every new contract with the outside gets one.** That means a key Home Assistant or the
  prediction log reads, a unit an API is asked for, or a rule another module relies on (the
  `min`/`cheap2h` coherence pair, a `build_fn`'s rows).
- **A harness hook gets one.** An identity case shows the new path is bit-identical to the
  old one, and a non-trivial case shows it actually reaches what it claims to (see
  `test_harness.py`).
- **A change to how a source is requested, parsed or reduced** trips
  `test_source_cache_versions.py`. That is intended: decide whether the production cache
  version must be bumped, then update the test's table ([DATA_CACHE.md](DATA_CACHE.md)).
- A change validated by an A/B does **not** need a test for its effect. Its evidence is the
  ledger entry. It does need a test for any new contract it introduces.

## Rules for a test

1. **Offline.** No network and no tokens. Stub the source with `monkeypatch`: replace the
   download function or `requests` call and hand back a small frame. Patch `time.sleep` when
   the code retries (as `test_yahoo_download.py` does).
2. **Small and synthetic.** Build the few rows the rule needs, with a seeded
   `np.random.default_rng`. A test that fits a model fits a stub (`_Const` in
   `test_interval.py` and `test_payload.py`), not XGBoost. XGBoost differs between platforms
   ([finding](FINDINGS.md#windows-and-linux-fit-different-models-from-the-same-data-round-31)).
3. **No MAE or score thresholds.** A threshold on model error is an A/B question, and it
   breaks with every legitimate model change.
4. **No dependence on today's date.** Fix the dates in the test. A suite that fails on its own
   one morning teaches everyone to ignore it.
5. **Snapshot tests use the `snapshot` argument.** A test that takes `snapshot` runs once per
   cache under `ab_cache/long/` and is skipped where there is none, as on the runner (see
   `conftest.py`). Cache what it loads (`functools.lru_cache`, as in
   `test_snapshot_data.py`): six snapshots × several tests is slow otherwise. These tests
   only run locally, so nothing that must hold in CI may depend on them.
6. **One file per area**, named `test_<area>.py`. The module docstring says which rule the
   file guards and where it is documented (a DECISIONS entry, a finding, or the date of the
   defect). A test name says the rule, not the mechanics: `test_thin_answer_is_retried_then_accepted`.

## Checking that a test catches what it should

Before committing a test for a fix, re-introduce the defect (revert the fix, or flip the
condition), run the test and see it fail, then restore. A test that passes on the broken
code guards nothing. Round 31 did this for all nine defects its tests cover.
