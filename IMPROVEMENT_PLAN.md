# SE4 Prediction — Improvement Plan

The to-do list for model accuracy: only what is **not yet done**. Since 2026-10-24 this is
the one plan file — the 2026-08 list (the round-12 → round-19 follow-ups) and the
September 2026 review (round 21 onwards, written 2026-09-08 as `IMPROVEMENT_PLAN_2026-09.md`)
were merged here. What is open, in the order it should be done, is
[section 4](#4-suggested-order-2026-10-23).

**Item numbers are stable.** They are the September review's, and the docs, ledgers and
commits refer to items by number ("plan 2.19", "2.1-C"), so a closed item leaves a gap
(2.1, 2.4, 2.7–2.9 and 2.15–2.17 are closed) and nothing is renumbered; a new item takes the
next free number. "Old item N" — a single number — means an item of the 2026-08 list, as
it does in older commit messages that say "`IMPROVEMENT_PLAN.md` item N".

Everything settled lives under [docs/](docs/): the current model in
[MODEL.md](docs/MODEL.md), shipped changes in [DECISIONS.md](docs/DECISIONS.md), the
rejected ledger in [REJECTED.md](docs/REJECTED.md), the dated write-ups in
[FINDINGS.md](docs/FINDINGS.md) and the methodology in [AB_TESTING.md](docs/AB_TESTING.md).
Read AB_TESTING.md in full first — the harness, "How to run an experiment", "How changes
are validated" (the adoption bar), "Measurement grids" and "Which verdict function to
call" — and REJECTED.md before trying anything. This file assumes that context and does
not repeat it.

**Priority: `cheap2h` first, then `min`. `avg` improvements are welcome; `max`
is explicitly not a priority.**

**Standing rules for everything below.**

- Re-opening a rejected idea, or trying a new one, needs a stated mechanism for why it
  should work — not "we changed a lot, maybe it's different now." If you can't state
  one, that's the answer.
- Screen on one period cluster, confirm on the others, and pre-register the
  confirmation bar before looking at the screen.
- Measure on a long snapshot (`ab_test.py fetch --days 1825`) on the rolling grid —
  NOW / −6M / FAR today, see
  [Measurement grids](docs/AB_TESTING.md#measurement-grids-training-geometry-and-period-clusters-read-before-designing-a-run)
  — and judge with `classify_clustered` / `classify_ablation_clustered`, not the plain
  `classify` / `classify_ablation`
  ([which one](docs/AB_TESTING.md#which-verdict-function-to-call)). `OMP_NUM_THREADS=4`.

**How this file is maintained.** A measured item is cut back to what is still open — or
deleted outright — as soon as its verdict is in the ledgers:
[DECISIONS.md](docs/DECISIONS.md) if it shipped, [REJECTED.md](docs/REJECTED.md) if it did
not, plus a section in [FINDINGS.md](docs/FINDINGS.md) when the round explained something
([README](README.md#how-changes-are-documented) has the rules). A half-done item keeps only
what it still owes and where it stands. The ledgers are the record; this file is the
to-do list.

**Status 2026-10-25.** Half done and carried with what they still owe: 2.2 (the
prediction log has run since 2026-10-18; the scorer is open), 2.3b/c/d, 2.5-HS, 2.10 (a
re-check) and 2.13 (its production check). **Nothing that reads the prediction log is
started before 2026-11-15** (Peter's call). Round 28 (2026-10-23) closed 2.7 and turned up
the two newest items, 2.19 (a time index for `avg`) and 2.20 (the decomposed nuclear column
as production sees it), which run together on the first long cache from mid-November 2026.
Round 29 (2026-10-25) closed 2.1: the realised physical residual load is a −1.2 `cheap2h`
ceiling, the weather reaches 11–21 % of it, and the all-horizon version of C failed
([FINDINGS.md](docs/FINDINGS.md#the-physical-residual-load-is-a-high-ceiling-that-the-weather-reaches-little-of-round-29)).
One lead survives it, a near-horizon column, and lives in 2.18; the same mid-November cache
serves it.

`SLOT_PREDICTION_PLAN.md` (round 20) is unchanged and still in the queue; 2.6 has its one
sequencing rule.

---

## 1. The findings the items are built on — in docs/FINDINGS.md

The four findings of the September 2026 review that most items below were built on,
and the source probes and reproduction recipes behind them, are in
[docs/FINDINGS.md](docs/FINDINGS.md#1-what-this-review-measured-2026-09-08-snapshot-ab_cache2026-09-05):

| | Finding | Drives |
|---|---|---|
| §1.1 | [`residual_load` is temperature in disguise, and its wind term is saturated](docs/FINDINGS.md#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated) | 2.1 (the primary item; closed in round 29); re-opens the intraday block |
| §1.2 | [The regime the model was tuned on is moving away from it](docs/FINDINGS.md#12-the-regime-the-model-was-tuned-on-is-moving-away-from-it) | 2.5, 2.6 |
| §1.3 | [Sources verified as reachable, with what they carry](docs/FINDINGS.md#13-sources-verified-as-reachable-with-what-they-carry-appendix-a-has-the-probes) | 2.3, 2.7, 2.14 |
| §1.4 | [The weather grid has no point under the northern wind fleet](docs/FINDINGS.md#14-the-weather-grid-has-no-point-under-the-northern-wind-fleet) | 2.13 |

Appendix A (source probes, 2026-09-08) and Appendix B (reproduction recipes)
moved with them.

---

## 2. Items, in priority order

Each item: mechanism → what to build → arms and the pre-registered bar → what a
failure would mean. Effort is a rough guess of agent time excluding compute.

### 2.2 Production prediction log and realised scorecard

**Mechanism.** Every accuracy number in docs/MODEL.md is a backtest with two known
optimisms that no backtest can remove: archive weather in place of the forecast
(measured 2026-10-11 on vintages: +1.63 cheap2h / +1.69 min / +2.81 avg at production's
leads, [FINDINGS.md](docs/FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b);
the log shows what it is in practice) and the gap1 fit tail (+0.155 / +0.258 /
+0.591). The only instrument that measures what Home Assistant actually
received is the record of what was sent, scored against what was realised.
Until 2026-10-18 there was no such record: `predictions_raw` is overwritten on every push.

**Status 2026-10-18: the log exists; the scorer is open.** The log is not in the repo and
not an Actions artefact. It lives in Peter's Home Assistant MariaDB (table
`electricity_price_predictions_log`), written by a Node-RED function node from the
sensor's `predictions_raw` right after each run — raw SEK/kWh, without the 5 % markup
and addon, which are a transform that only makes a logged value harder to score. One row
per (`prediction_date`, `target_date`, `variant`): `horizon_days` (computed from the two
dates, so it stays right if the first predicted day shifts), `min` / `avg` / `max` /
`cheap2h` (`*_sek_kwh`), `cheap2h_low` / `cheap2h_high` / `cheap2h_risk`, and two
run-level columns: `eur_to_sek_rate` (turns a logged SEK value back into the model's
EUR/MWh exactly) and `git_sha` (the commit the run executed, so a before/after read
groups rows by code instead of guessing from dates). The last two reach the sensor from
`ha_client._run_provenance`; rows written before that change carry NULL. A rerun on the
same day replaces that day's rows — what Home Assistant last saw. `variant` defaults to
`'served'`; the slot is for shadow rows (below). The Node-RED flow is not in the repo.

The older wide table `electricity_price_predictions` (d1…d6 min/avg/max, **with** addon,
from at least 2026-09-22) is kept as the baseline from before the round-24/25 changes and
is **to be brought into the scoring** (bullet below): its values have the addon baked in
(`ha_client.apply_addon`: raw × 1.05 + addon), so they cannot be scored until the addon
is known. It has no `cheap2h`, and the new
table starts after the 2026-10-12 and 2026-10-16 changes, so the log has **no before-period
for `cheap2h`**: production is judged against what the backtest predicted for it, not
against an earlier production.

**Not before 2026-11-15.** That is four weeks after the log started, the earliest point
at which it is worth reading, so `score_production.py` and every step that reads the log
(the 2.3b forward replay, 2.3c, 2.3d, the 2.10 and 2.13 production checks) wait until
then; it is not written earlier either. What it buys: at ~4 weeks a first, descriptive
look at MAE per horizon, at the band (it warns on ~46 % of days, so ~13 warned days — thin)
and at the decision regret; at ~13 weeks the optimism re-derivation below. The clock is
four weeks of daily rows from the day the Node-RED node is writing every run; if that is
later than 2026-10-18, the date moves with it.

**Reading the data (decided 2026-10-18).** The database is the record. The scorer reads
a CSV export of the whole table (phpMyAdmin, dated file name) from
`ab_cache/prediction_log/`: gitignored with the rest of `ab_cache/`, ignored by
`list_snapshots` (no `meta.json`), and disposable because it is re-exported on demand.
The old wide table is exported at the same time and not before (decided 2026-10-18), together
with the addon value for its period.

**What remains.**

- `score_production.py`: joins the log to realised daily prices (ENTSO-E, already
  fetched) and reports MAE per target and per horizon; the warning-light band scored with `ab.band.score_band` on what
  was served (its threshold 0.3 and coupled level were chosen post hoc on a historical
  walk-forward with archive weather, so the log is the only honest check); and the
  *decision* regret of the charging rule the automation actually uses (README HA
  section) — the metric that the slot plan and the q90 rule were both scored on in
  backtests but never in production.
- **Bring the old log in.** Export `electricity_price_predictions` together with the new
  table and treat it as extra min/avg/max history from at least 2026-09-22 — about four more
  weeks than the new table has. Three things have to be recovered first:
  - *The addon.* raw = (value − addon) / 1.05, with the addon each run read live from
    `input_number.electricity_price_addon`. Peter has to state the value and the dates it
    changed, if it ever did: the HA recorder keeps states for its purge window only
    (typically 10 days) and an `input_number` has no long-term statistics, so the history
    may not be recoverable from HA.
  - *The target days.* Only `d1_date` is stored; d2…d6 are the following days, which holds
    because the forecast set is consecutive days. Horizon = `d1_date` − `prediction_date`
    + (k − 1).
  - *The rate.* No `eur_to_sek_rate` was stored; re-derive it as the pipeline does
    (`currency.calculate_eur_to_sek_rate`) to get EUR/MWh.
  There is no `git_sha`, so score it in blocks split at the known change dates (2026-10-12,
  2026-10-16) and never pool it blindly with the new table.
- Re-derive the two optimism figures empirically once ~13 weeks are logged:
  production MAE at D+2 minus backtest MAE at D+2 is the sum of the weather and
  gap1 effects, and the per-horizon slope is the weather part.
- A shadow arm for 2.3b: the pipeline would push the blend's prediction as a second
  attribute and the flow would write it with `variant = 'blend'`. Not built.
- Not logged: the run's `cheap2h_interval` diagnostics (`coupled_level_ratio`,
  `calm_high_holdout_coverage`, …), which is where a new regime shows first. Add a JSON
  column if drift tracking needs them.
- **`horizon_days = 8` rows are not forecasts.** Until 2026-10-21 every run sent a
  seventh day, D+8: since the UTC switch of 2026-03-22 the 8-day forecast's last Swedish
  date holds only the 1–2 hours between Swedish and UTC midnight, and nothing dropped it
  ([DECISIONS.md](docs/DECISIONS.md#whole-swedish-days-only)). The 2026-10-17 test row and
  every run before the fix logged one. They carry no information and can be deleted from
  the table; the scorer should reject `horizon_days > 7` regardless, until a longer
  horizon is shipped on purpose (2.11). The old wide table never had the stub — its flow
  kept six days.

**Bar.** None — this is instrumentation. It is the prerequisite for reading
item 2.3 honestly and for the forward replays of 2.10 and 2.13. It cannot say whether the
two changes of 2026-10-12 helped by a before/after on `cheap2h` (no before exists, see
above); the question it can answer is whether served `cheap2h` lands near the backtest's
expectation.

**Effort.** Log: done. Scorer: a few hours. Zero model risk.

### 2.3 The weather axis: price it, then improve it

**2.3a — measured 2026-10-11, no longer an item.** Archive weather flatters the
eval by +1.63 cheap2h / +1.69 min / +2.81 avg EUR/MWh at production's leads,
almost all of it from D+5 (+1.0 … +2.1 at L5, +3.0 … +4.7 at L7 on cheap2h):
[FINDINGS.md](docs/FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b).
The README and MODEL.md numbers are read with that in mind until the baseline is
re-measured on vintages.

**2.3b Multi-model blend at serve time — measured, missed its strict bar.** The
hourly mean of ECMWF, ICON and GFS recovers ~25 % of the penalty (cheap2h −0.37,
min −0.42, avg 0) and is negative in all 15 cache × cluster cells, but the
pre-registered per-lead bar was met on 0 of 5 ([REJECTED.md](docs/REJECTED.md)).
Next step is a *forward* replay, not another backtest: log the blend's prediction next
to the served forecast (a `variant = 'blend'` row in the 2.2 log, which needs the shadow
push from the pipeline), and score both on the
days the log covers — not before 2.2's 2026-11-15, and the shadow push has to run for
those weeks too, so it is built first, with a bar written for production's lead mapping (row j at
lead j + 2): production-mapped cheap2h and min negative in every cluster-equivalent
period and the pooled change ≤ −0.2. Cost at serve time: three model requests for
nine points.

**2.3c Train on what you serve.** The headroom is now a number: +1 … +5 EUR/MWh on
cheap2h at L5–L7. Mechanism unchanged — the model learns price = f(analysis
weather) and is served f(forecast weather) — but vintages exist only from 2024-03,
so far-lead rows would train on ~2.6 years. Per-lead models or a stacked set with
a lead column; start only after the 2.3b replay shows whether input quality or
model mismatch carries the penalty. Bar as 2.3b.

**2.3d Ensemble spread as an interval conditioner.** Mechanism: the conformal band
is one width for the whole forecast set, adjusted by a single correction. Weather
uncertainty varies day to day and is observable as the 51-member spread of
`wind_speed_100m`. Condition the conformity quantile on spread bins (or add spread
to the two quantile regressors). Scored on pinball loss and coverage per cluster,
never MAE. Live only — the ensemble has no vintage archive — so scored
prospectively via 2.2.

**Effort.** 2.3b replay: one day plus waiting for data; 2.3c three days; 2.3d one day.

### 2.5 The hurdle's label under regime drift — only the slot-specific arm is left

H5 (threshold 5 EUR/MWh) and HQ (trailing 10th-percentile label) were measured on
five caches and are NOISE ([REJECTED.md](docs/REJECTED.md)). What remains is
**HS**: `P(cheap2h_day < 0)` and `P(cheap2h_night < 0)` as two probabilities, the
slot-specific hurdle of `SLOT_PREDICTION_PLAN.md` §4.4, which needs the slot targets
from 2.6 to exist first. Bar: `classify_clustered` on cheap2h; do not re-test the
hurdle on `min` (closed twice).

### 2.6 Slot targets (round 20)

Unchanged: `SLOT_PREDICTION_PLAN.md` is the spec. It used to wait for the residual-load
rebuild (old 2.1), which would have given the model a feature that tells a night-wind trough
from a midday-solar one; that item closed in round 29 (the weather reaches 11–21 % of what a
physical residual is worth, [FINDINGS.md](docs/FINDINGS.md#the-physical-residual-load-is-a-high-ceiling-that-the-weather-reaches-little-of-round-29)),
so the slot targets are built on the current features.

Do not wire them in mid-A/B. `ab.variants.BASELINE` mirrors `model.TARGETS`
(docs/AB_TESTING.md), so the slot targets must not be added while another candidate is
being measured, and vice versa.

### 2.10 `min ≤ cheap2h` coherence — shipped 2026-10-12, re-check in ~3 months

Record and arms: [DECISIONS.md](docs/DECISIONS.md#min--cheap2h-coherence-round-25). Every
number there is a full evaluation year (52 weekly windows) and the three clusters are three
such years, so the pooled gain is a year-long effect — but it is modest, and the freshest
year is its weak spot: `min` in the NOW cluster is +0.03 on the newest cache and only
−0.05 … −0.13 on the others, while FAR and −6M carry most of the pooled −0.17 … −0.23.
That is worth confirming rather than assuming. **When a long cache from mid-January 2027
or later exists**, measure the change on and off (patch `ab.harness.enforce_coherence` to the
identity for the "off" arm, as `experiments/integration_check_coherence.py` does) on NOW and
−6M. Bar, written now: `min` NOW ≤ 0 and `cheap2h` NOW not REAL-harmful on at least two of
three later caches. If `min` NOW is positive on most of them, revert to the raw pair (the
"lower" arm is already rejected). The production log (2.2) is the second source for the
same question.

### 2.11 Longer horizon for the automation (product decision, cheap)

Open-Meteo serves `forecast_days=16`; the pipeline requests 8 and sends **six**
predicted days (D+2..D+7): today and tomorrow have published prices, and the last
Swedish date of a UTC request is a 1–2 hour stub that `predict.py` drops (until
2026-10-21 it went out as D+8, [DECISIONS.md](docs/DECISIONS.md#whole-swedish-days-only)).
Extending to D+2..D+14 is a parameter change plus the nuclear-outage and calendar
windows (`forecast_end = today + 10`). Keep the whole-day filter, and request one day
more than the last day sent: Open-Meteo changes the last hours of a series with
`forecast_days` (de_north, 2026-10-18), so a longer request is not bit-identical on the
previous last day. Accuracy
beyond D+7 is unmeasured — the eval uses 7-day windows — so ship it only with
the interval, and report it separately in 2.2. Whether the automation wants a
14-day view is Peter's call; the model cost is nil.

### 2.12 Engineering hygiene

- **Tests.** There are none. A small `pytest` suite with the checks that would
  have caught 1.1: fetched wind speed median in a plausible m/s range,
  `min ≤ cheap2h ≤ avg ≤ max` on every training row, DST days have 23/25
  hours and produce no NaN, `_splice_recent`'s two rules, the interval's
  sort → widen → clamp order, `convert_predictions_to_sek` on a payload with
  every key production can emit. Run in the Actions workflow before
  `predict.py`.
- **Harness.** A `build_fn(inputs) -> frame` hook on `ab.variants.Variant`.
  `Variant.transform` receives the merged daily frame, after the intraday and residual-load
  columns were computed from the hourly inputs, so it cannot rebuild them; rounds 25, 27 and 28
  each worked around it by building their frames (and, for per-window as-of columns, their own
  loop) in the experiment script (`experiments/run_round25.py`, `run_round28.py`; the
  pattern is in [AB_TESTING.md](docs/AB_TESTING.md#score-the-test-rows-as-production-could-know-them-the-af-condition-round-28)).
  2.3 needs the hook too. Also a `--targets` filter on
  `run_walk_forward` / `run_ab` (the slot plan §3.4 asks for it) so a
  cheap2h-only screen does not pay for `max`.
- **Run status to Home Assistant.** The workflow has no failure path; a failed
  run leaves the sensor silently stale. Push a `last_run_status` attribute (or
  a separate binary sensor) from a `finally:` block so the automation can fall
  back to "charge now" when predictions are older than a day.
- **A NaN last close reaches the forecast features (found 2026-10-12).** The cached
  EUA series from 2026-10-10 ends with a NaN close (Yahoo returned no `CO2.L` price for
  2026-10-09). `build_forecast_features` takes the *last row* of `eua_daily` (and
  `ttf_daily`) as `co2_price_lag1` / `ttf_price_lag1` without dropping NaN, so on such a
  day every forecast row gets NaN `co2_price_lag1` and `gas_marginal_cost` — the latter is
  in the trough lists — and XGBoost treats it as missing without a word. The training side
  forward-fills, the forecast side does not. Fix: take the last non-NaN value (and log it);
  add the case to the test suite above.
- ~~**Does ENTSO-E A72 (Swedish reservoir) omit repeated weeks?**~~ **Closed 2026-10-14:**
  no — 0 omitted of 158 weeks over three years. Record:
  [FINDINGS.md](docs/FINDINGS.md#entso-e-documents-four-traps-found-validating-the-round-25-fetch-2026-10-11).
  `_fetch_reservoir_chunk` carries a no-op A03 fill + warning as a guard.
- **Dependencies.** `requirements.txt` is unpinned and the workflow runs
  Python 3.13 while local runs are 3.14; pin at least `xgboost` and `pandas`
  majors so a bit-identity check across machines means something.

### 2.13 Northern wind points — shipped 2026-10-12, replayed 2026-10-22

Record: [DECISIONS.md](docs/DECISIONS.md#northern-wind-points-round-25). Hydro
precipitation and a fourth site (Sundsvall) were measured and rejected
([REJECTED.md](docs/REJECTED.md)). The replay on `long/2026-10-16` and the `min` vintage
re-score are done (round 27): it replicates on all six caches under the coupling hurdle,
forecast-weather gain on the newest window cheap2h −0.45 / `min` −0.52 clmean. One thing
is left:

- **The production log (2.2) is where the real gain shows** — not before 2026-11-15.
  There is no before/after for `cheap2h`: the log starts 2026-10-18, after the change
  (see 2.2), so score the served `cheap2h` against the forecast-weather expectation in
  DECISIONS.md (NOW cluster −0.7 … −1.2, of which −1.1 at D+2…D+4 and −0.7 at D+5…D+7).

Not tested, low prior: other fleets (NO3/NO4, northern Finland beyond Ostrobothnia) —
each single site gave ~half the three-site gain, so the regional average is the
signal and three points seem to saturate it.

### 2.14 Nord Pool UMM as the outage source

**Mechanism.** The open question in old item 0 / 2.4 was whether ENTSO-E A78
is usable as a *forward-looking* feature, because the API returns the current
version of a record and revisions are invisible, so a backtest could know
things production could not. **Round 21 settled that the A78 route cannot
answer it**: every record for an event before 2025-11 was re-published in
November 2025, so the cache holds no vintage at all before that date
(docs/FINDINGS.md "If you touch ab_cache/crossborder/"). With the trough targets
having rejected this data outright and the `avg` export-headroom candidate now
closed on its serve-time encoding, **the one consumer UMM had was the nuclear features of
item 2.7, and round 28 closed it** (the realised nuclear output, as a leaky ceiling, moves
`avg` by −0.03 and `min` by +0.18; [FINDINGS.md](docs/FINDINGS.md#the-nuclear-column-and-the-week-ahead-load-carry-nothing-a-model-can-use-round-28)).
**Parked 2026-10-23: build it only if a new per-day column from a revisioned document appears.**
The interconnector half has no consumer either. What follows is what was verified, kept
for that day. Nord Pool's UMM API is the upstream of those
records and keeps **every version of every message with its
`publicationDate`** — `GET /messages/{id}/{version}` returns the record as it
stood then — for production units (nuclear with installed and available MW per
period), transmission units (in/out area, available MW) and large consumers,
back to at least 2016 for the SE4 borders, without authentication. That is the
vintage information A77/A78 lack, and it answers the leak question by
construction: for any backtest date D, use the latest version of each message
with `publicationDate ≤ D`. Verified on the Baltic Cable trip
(`e911027b…`): v1 2026-06-21 end 06-29 → v2 06-23 end 08-31 → v3 08-17 end
09-18 → v4 09-11 end 09-14. Also in the feed for SE4 (Jun–Dec 2026): SE3→SE4
(40 messages), SE4↔PL (129), SE4↔DK2 (35), SE4↔LT (23), SE4↔DE-LU (13), plus
the SE2→SE3 and FI→SE3 cuts that set the import price.

**What to build.**

- `sources/umm.py`: fetch messages by `eventStartDate`/`eventStopDate` window
  (paginate with `skip`; `messageTypes=1` is production, `2` consumption,
  `3` transmission — not the order the names suggest; `areas=<SE4 EIC>`
  filters all three, 176 SE4 transmission messages for Jun–Dec 2026), keep
  every version, build two daily series
  *as of* a given date: unavailable nuclear MW in SE3 (replaces the event
  count in `nuclear_outage_se3`, old item 4 / 2.7). The same fetch also yields
  unavailable interconnector MW per SE4 border; parse it only if a feature
  needs it, since the one that would have is closed.
- Cache the full message history once under `ab_cache/umm/`; the daily
  production fetch only needs the current window (events overlapping
  today..today+16).
- Encode "as of run time" in the eval: `apply_forecast_freeze` freezes lag
  features; these are per-day features whose *value* depends on the run date,
  so the walk-forward must rebuild them per window from messages published
  before the window start. That is a small extension of the harness (a
  per-window feature builder). Round 25 got its forecast-vintage frames by building one
  frame per variant instead (`experiments/run_round25_vintage.py`), which does not
  generalise to a per-window as-of feature — so this is its own piece of machinery.

**Bar.** Whatever the new column's arms are, rebuilt as-of, against the same arms under the `AF` condition
of [AB_TESTING.md](docs/AB_TESTING.md#score-the-test-rows-as-production-could-know-them-the-af-condition-round-28);
`classify_clustered`.

**Effort.** One day for the source and cache, one day for the per-window
builder shared with 2.3a.

### 2.18 A near-horizon model that reads the price (added 2026-10-22)

**Idea (Peter's).** A frozen price lag is information about tomorrow's price and noise
about next week's. Today one model serves every day, so it either reads the price on
every day or on none. The price/market family was pruned from `min`, `cheap2h` and `avg`
because, frozen across all seven rows, it hurt: cheap2h +1.00, min +0.74, avg +1.08,
harmful in every period ([REJECTED.md](docs/REJECTED.md)). The question was never asked
for the near days alone. A separate *near model*, with the price family, for the first
N sent days, next to today's model for the rest, is a new mechanism. It is not the
round-18 price-level features or the shelved per-horizon models re-opened.

**Why now.** Since 2026-10-22 the anchor is D+1
([DECISIONS.md](docs/DECISIONS.md#price-anchor-on-d1)), so D+2's lag1 is exactly
yesterday's price, as in every training row: no train/serve mismatch on the first sent
day. Measured on that change: a fresher anchor helps cheap2h on D+2/D+3 (up to −0.44)
and nothing from D+4. That is the shape this item would exploit, on a model that barely
reads the price today.

**Design.**
- For each horizon h in the near set (D+2 = staleness 1, D+3 = staleness 2, …), train
  on the price family at the age that day will see. Use lag-h columns (`price_se4_*`
  shifted h days), not lag1 frozen and labelled as fresh. The near model for D+3 is
  trained on prices two days old.
- Score per eval row (1-based row k = D+(k+1), staleness k) against today's model on
  the same rows. Choose N as the last horizon where the near model wins in every
  cluster, and stop at the first horizon where it does not.
- Targets: `cheap2h` and `min` first (`ab_test.py band` as well, since both feed the
  band), `avg` after. `max` already reads the family.
- **Weather columns are near-horizon candidates too (round 27, post hoc).** On forecast
  vintages the MW-scale residual load is worth `avg` −0.1 … −0.4 at D+2…D+4 and nothing
  or worse at D+5…D+7, and the northern wind points on `avg` the same
  ([FINDINGS.md](docs/FINDINGS.md#a-wind-carrying-residual-load-pays-only-near-the-run-day-round-27)).
  A near model is where such a column could live. That makes it a hypothesis for this
  item with its own bar, scored on vintages, not a result to carry over.
  **Evaluate it in the same run as the price family (Peter, 2026-10-22).** Read off the
  stored round-27 errors (no new run), `avg` with `residual_load_mw` +
  `residual_load_min_mw` appended on D+2…D+k and today's model after, on forecast weather,
  six caches, NOW / −6M / FAR: k = D+3 −0.02 / −0.07 / −0.20, D+4 −0.05 / −0.15 / −0.20,
  **D+5 −0.14 / −0.24 / −0.36** (every cache and cluster negative, NOW −0.08 … −0.25),
  against ≈ 0 in NOW with the columns on all six days. The split was picked after seeing
  these numbers, so it is pre-registered here, before the 2.18 run: **near `avg` = AVG
  list + the two `_mw` columns for D+2…D+5, today's `avg` for D+6…D+7**. Bar: production-
  mapped forecast-weather delta negative in every cluster on ≥ 2 long caches with
  genuinely new NOW windows (one fetched mid-November 2026 or later), and the six-day
  total not worse. `min` and cheap2h do not get the columns at any horizon (worse on a
  forecast at every lead, round 27). Build it on the same per-horizon routing as the
  price-family near model; it is not worth a second avg model on its own (−0.14 in NOW
  is near the ~0.10–0.15 `avg` noise floor).

**A calibrated residual load is a third near-horizon candidate (round 29, post hoc).**
`CALRn`: two columns, `cal_resid_mean` and `cal_resid_min`, the prediction of the realised
eight-zone load − wind − solar (A75 + A65, GW) from the day's weather and calendar by an
XGBoost first stage (200 trees, depth 3, no time index, out-of-fold on the training rows),
for D+2…D+4 on `cheap2h` and `min`. Round 29, forecast vintages at production's map:
`cheap2h` D+2…D+4 **−0.36** (NOW −0.37 / −6M −0.36 / FAR −0.34, every cache), `min` −0.25,
and −0.03 / −0.11 at D+5…D+7; on archive weather only −0.13 on `cheap2h` (REAL on the primary,
NOISE on four caches), so it is a gain that exists on a forecast and not on the archive,
chosen after the numbers were seen as the better of two first-stage variants
([FINDINGS.md](docs/FINDINGS.md#the-physical-residual-load-is-a-high-ceiling-that-the-weather-reaches-little-of-round-29)).
Neutral in the newest four weeks. **Bar, written 2026-10-25 before any replication exists:** on a
long cache fetched mid-November 2026 or later (with its own vintage fetch and the A75/A65
top-up, `experiments/fetch_round29_entsoe.py --since 2026-10-01`), the production-mapped
D+2…D+4 delta is negative in every cluster on `cheap2h` and not positive on `min`, the
archive delta is not REAL-harmful in any cluster, the six-day total is not worse, and
`ab_test.py band` shows no breach. Run it with `experiments/run_round29_cal.py --arms
base,CALRn` in the same pass as the price family. Cost if it holds: A75 + A65 for eight zones
fetched on every run (about five minutes with the token production already uses) and a first
stage refit per run; it would be a column of the near model, never of the all-horizon lists.
The `avg` version fails on vintages (NOW +0.62) and is not part of this.

*Scale, so the size is not misread (Peter's question, 2026-10-25):* −0.36 is about 2.8 % of the
`cheap2h` base error over D+2…D+4 (12.7 EUR/MWh), −0.25 about 2 % of `min`'s (12.4); over all six
sent days the lead is −0.19 (about 1.3 % of 14.8), and the all-horizon `CALR` was −0.19 archive /
−0.16 vintage, inside the ~0.13–0.15 noise floor. The 11–21 % in FINDINGS is the share of the
oracle's gain that the weather reaches, not an improvement of the forecast.

**It is part of the horizon-split evaluation, not a separate run.** The near/far split N (the
"Measurement geometry" paragraph below) is chosen per horizon, and `CALRn` is scored in that
same run as one more candidate for the near model, next to the price family and the `avg`
`_mw` columns: per horizon D+2 … D+7 on forecast vintages, with the rolling origins, so that
the horizon where it stops paying decides where it is switched off. Until that run the
production model gets nothing from it.

**Bar** (write it down before the run). Per horizon, `classify_clustered` REAL and
favourable on ≥ 2 long caches with genuinely different NOW windows. The total over
D+2…D+7 must not get worse.

**Measurement geometry: decide before the run (Peter, 2026-10-24).** The step-7 grid is
right for all-horizon verdicts, but this item makes one per horizon, and two properties of
the grid then matter. Within one grid point, eval row k is always the same weekday, so a
per-horizon delta is also a per-weekday delta. And each horizon gets only 52 test days per
point. Consider rolling origins instead: fit once per test week as now, then issue a
forecast from each of the seven start days in that week, each with its own freeze anchor.
Every test day is then scored at every horizon and every horizon sees every weekday, for
about the cost in fits of the current grid. Both arms share the same fits, so the model
being 0–6 days old at an origin (production: ~1) does not bias the comparison. Three
things to settle in writing first: (1) the extra origins sharpen each point's per-horizon
estimate but are not replication, because the same day at D+3 and D+6 is strongly
correlated; the cluster stays the unit of the verdict
([AB_TESTING.md](docs/AB_TESTING.md)), and per-window spreads from overlapping origins are
optimistic. (2) Score D+2…D+7 only; row 7 (D+8) is not sent. (3) Run it on forecast
vintages as well as archive weather. On archive weather a day's D+3 and D+6 differ only by
the age of the frozen lags, while most of the real decay with horizon is weather forecast
error (+1.3 cheap2h / +1.5 min / +2.4 avg on the six sent days). The near/far split N is
the quantity most exposed to that.

**Cost.** One extra model per near horizon, plus routing in `predict.py` and
`model.predict` by horizon. The harness needs a per-row target, the same shape as
`run_round25_vintage.py`'s per-row scoring.

### 2.19 A time index for `avg` (added 2026-10-23, found as a control in round 28)

**What was found.** Adding `t_index` (days since 2021-10-21 / 365) to the `avg` model is REAL on all
six long caches — six-cache mean **−0.28** (−0.18 … −0.43), negative in NOW, −6M and FAR on every cache,
per-window std ×0.96 – 1.00 — on top of the half-life-500 weights `avg` already has. The date is known for every
forecast day and nothing in the column is weather, so neither the forecast-vintage trap nor the AF condition
applies. On `cheap2h` (+0.23) and `min` (+0.16) the same column is REAL-harmful on five caches
([REJECTED.md](docs/REJECTED.md), [FINDINGS.md](docs/FINDINGS.md#the-nuclear-column-and-the-week-ahead-load-carry-nothing-a-model-can-use-round-28)).
It is a control that turned out to be a result, found after the main round was read, and measured only on
the six overlapping caches — a lead, not a change.

**Mechanism to test, not assume.** `avg` is the one target that wants recency. Half-life weights give recent
rows more weight; a time index lets the trees give the newest regime its own leaf, and every forecast row
lands in it. If the second is only a cheaper way to get the first, a shorter half-life reproduces the gain and the
index is redundant; if the index beats every half-life, it carries something weights cannot. The hazard is
the same property: through a regime break the model keeps predicting the newest leaf.

**Arms** (`avg` only; `min` and `cheap2h` are closed): `TI` alone; half-lives 250 and 365 without it
(production is 500); `TI` with each; a step version, whole years since 2021 (is the gain the slope or the
levels?); SEED7 as the null. Against production `avg`.

**Bar** (written now, before any run). On the six existing caches: `classify_clustered` REAL, six-cache mean
≤ −0.15, std ratio ≤ 1.05, favourable on ≥ 4 of 6. **And** on a cache fetched at least four weeks after
`long/2026-10-16` (mid-November 2026 or later), because the six caches share most of their NOW window and the
gain in NOW is partly the recent coupled weeks: NOW negative there, or the item stays a lead. **And** the four
weeks from 2026-08-17 (the regime break) are scored apart and must not be worse than `base`. `max` reads the
full list, so it is reported (a REAL-harmful `max` is Peter's call). `ab_test.py band` is not needed — `avg`
feeds no band.

**Effort.** Half a day; about one hour of compute (one target, 12 points per cache). No new fetch except the
replication cache.

### 2.20 The decomposed nuclear column, replayed as production sees it (added 2026-10-24, run with 2.19)

**What is open.** Round 28 closed the nuclear column on its pre-registered bar (scored with every record
visible, HS). Scored as production actually serves it (AF: planned outages visible, forced ones only once
started — which is what `predict.py` fetches today), two arms read slightly better than today's column on
`avg`: `Rac` (distinct nuclear units + distinct non-nuclear units, replacing `nuclear_outage_se3`) −0.05,
negative on all six caches, and `Rall` (those two + unavailable nuclear MW + cancelled records) −0.07, negative
on five. About half of it is the old column losing more than the new ones when future forced outages are
invisible. It is inside the `avg` noise floor, the six caches share most of their windows, and the two arms
were the best of seven, so it is a lead picked after the fact, not a result
([FINDINGS.md](docs/FINDINGS.md#the-nuclear-column-and-the-week-ahead-load-carry-nothing-a-model-can-use-round-28)).

**When and how.** In the same run as 2.19, on the first long cache fetched at least four weeks after
`long/2026-10-16` (mid-November 2026 or later), plus a top-up of the A77 records to that date
(`experiments/fetch_round28_entsoe.py --only a77 --since <four weeks before the cache>`). Re-use
`experiments/run_round28.py` (`kind avg`, arms `base`, `Rac`, `Rall`; the data layer picks the newest
`ab_cache/entsoe_r28/` directory). Only the new cache's own grid counts; the six old caches are not re-run.

**Bar (written 2026-10-24, before the run).** For `Rac` and `Rall` separately, delta against `base` under AF on
the new cache: negative in each of NOW, −6M and FAR, cluster mean ≤ −0.05, and the HS delta not REAL-harmful.
If one arm passes, it is a parser change in `sources/entso_e.py` (keep unit, `psrType` and document status —
the guard in `experiments/results/nuclear_parser_fix.patch` is the starting point) and needs `max` reported and
the local integration check before it ships. If both pass, `Rac` (two columns, less to maintain) unless `Rall`
is ahead by more than 0.05. If neither passes, 2.20 is closed for good and the column stays.

**Effort.** No extra compute beyond 2.19's run (one target, two arms); the parser change about two hours if it
passes.

## 3. Carried at low priority, and what stays closed

### `ttf_vs_30d` — a regime-abnormality ratio, low expected yield

**Mechanism.** World events do not hit SE4 prices directly; they transmit through fuel
markets (`ttf_price_lag1`, `ttf_rolling_7d`, `co2_price_lag1`, `gas_marginal_cost`),
which are already covered. The gap: trees split on absolute levels, so a level split
learned from 2023 data does not generalise to a 2026 shock (the motivating case was the
2026-07 Iran-war price spike). `ttf_vs_30d = ttf_price_lag1 / ttf_rolling_30d` (a new
30-day rolling mean next to the existing 7-day one) encodes "how far above normal" as a
stationary ratio, and should react before a shock fully lands in the electricity-price
lags.

**Prior.** Low. Its sibling `price_vs_30d` was rejected in round 18 (harmful on `min` and
`cheap2h`, three independent measurements, [REJECTED.md](docs/REJECTED.md)). This one is a
fuel-price ratio, not an own-price ratio, so the "own frozen price lag hurts the trough
targets" mechanism does not apply as such. Round 24's finding that the autumn-2026 misses
are a gas-price-and-coupling regime — shipped as the coupling hurdle
([DECISIONS.md](docs/DECISIONS.md#coupling-hurdle-and-the-warning-light-band-round-24)) —
is the stronger route to the same problem.

### The rest

**Also low priority, carried:** `cpl_gap7` on cheap2h and the q90 decision rule at H=7
(round 19 follow-ups); `B6_fuel_co2` on min, pending post-crisis training windows
(~2026-10, per the rejected table).

**Not re-opened, no new mechanism:** price-level features for the trough targets
(round 18), the SE3 / system price lag among them — its physical route, the northern wind
points (2.13), shipped instead; anchored targets (19b); window length (15); per-target
hyperparameters (13.1); `reg:absoluteerror`; seed ensembling; the avg-anchored clamp. The
solar-capacity index was superseded by 2.1-C, which closed in round 29 ([REJECTED.md](docs/REJECTED.md)); not re-opened.

**News, geopolitical-event and oil features stay out**, also on top of `ttf_vs_30d`. TTF
prices an event in faster and more quantitatively than any hand-built flag, a binary
crisis dummy has only 1–2 occurrences in a three-year window (unlearnable), and daily news
indices at this frequency mostly give XGBoost something to overfit. Oil/Brent is not worth
adding either: gas, not oil, sets the European marginal cost, and gas is covered.

**Not a feature source:** ENTSO-E day-ahead load (A65) and wind/solar generation
forecasts (A69) reach D+1 only, and D+1 is never predicted (the horizon column of
[FINDINGS.md §1.3](docs/FINDINGS.md#13-sources-verified-as-reachable-with-what-they-carry-appendix-a-has-the-probes)),
so they cannot be forecast features. They would be calibration data at best, which A75
actuals already serve (2.1-C, closed in round 29).

**Structural note, not an item.** Nordic flow-based market coupling went live
2024-10-30, inside the training window. If a later analysis finds a step
change in SE4–SE3 coupling around that date, a single dated indicator is a
legitimate feature; do not add it speculatively — it would act as a time
index.

---

## 4. Suggested order (2026-10-25)

Round 27 (2026-10-22) replicated the northern wind points and closed the MW-scale residual load; round 28
(2026-10-23) ran all of 2.7 and closed it: no nuclear feature of any kind, and no week-ahead load; round 29
(2026-10-25) measured the ceiling of 2.1-C, found it high and out of the weather's reach, and closed 2.1.

1. **2.2 prediction log** — the log has existed since 2026-10-18 (MariaDB) and the clock is
   running. **Nothing to do until 2026-11-15**; then `score_production.py` (reads a CSV
   export in `ab_cache/prediction_log/`). The 2.3b shadow push is the one piece that can
   be built earlier, because it has to run for those weeks. It prices everything
   weather-related for real, and the forward replays of 2.3b, 2.10 and 2.13 wait on it.
2. **2.19 a time index for `avg`, 2.20 the decomposed nuclear column under AF, and the 2.18
   calibrated-residual lead** — one run on the first long cache from mid-November 2026 or later (the same
   fetch 2.18 needs), plus an A77 top-up (`fetch_round28_entsoe.py --only a77`), an A75/A65 top-up
   (`fetch_round29_entsoe.py --since 2026-10-01`) and the vintage fetch. All three bars are written down.
   Run `TI` and a residual arm next to any `avg` result (round 28: a slowly moving column can be a clock) and a
   permuted null of the same width next to any block of more than a few columns (round 29: it costs +0.33).
3. **2.12 hygiene** — the NaN last close into the forecast fuel/carbon lags; the `build_fn`
   hook; the tests. (The A72 probe is done: reservoir arrives complete.)
4. **2.6 slot targets** — unchanged; best built once the feature set above has settled,
   so the scoring run measures the final features. 2.5-HS follows it.
5. **2.3b forward replay / 2.3c** — not before 2026-11-15, and only if the 2.3b shadow push
   has been running since before then.
6. **2.10 coherence re-check** — when a long cache from mid-January 2027 exists.
7. 2.11 horizon. (2.14 UMM is parked: nothing consumes it.)
8. **2.18 near-horizon price model**: to be taken up separately (Peter, 2026-10-22).
   The price family needs no new fetch and any long cache serves; the calibrated-residual
   lead in it needs the mid-November cache and the A75/A65 top-up.
