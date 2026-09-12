# SE4 Prediction — Improvement Plan, September 2026 (round 21 onwards)

**Status: PLANNED 2026-09-08. Nothing in this file is implemented.** It is the
successor to `IMPROVEMENT_PLAN.md` (2026-08), which stays in place as the
record of the round-12 → round-19 follow-ups. Where an item carries over it is
said so; where this file changes a priority it says why, with the measurement
that changed it.

Read first, as before: [docs/AB_TESTING.md](docs/AB_TESTING.md) in full (the
harness, "How changes are validated", "Measurement grids", "Which verdict
function to call") and [docs/REJECTED.md](docs/REJECTED.md). The standing rules from `IMPROVEMENT_PLAN.md` are unchanged and not
repeated here: state a mechanism before measuring, screen on one period cluster
and confirm on the others, pre-register the bar, use the round-15b sliding grid
on a long snapshot with `classify_clustered` / `classify_ablation_clustered`,
`OMP_NUM_THREADS=4`, priority cheap2h → min → avg, `max` not a priority.

`SLOT_PREDICTION_PLAN.md` (round 20) is unchanged and still next in the queue;
section 2.6 below says how it interacts with the new item 2.1.

**How this file is maintained (2026-09-12).** A measured item is cut back to
what is still open — or deleted outright — as soon as its verdict is in
the ledgers under [docs/](docs/) — [DECISIONS.md](docs/DECISIONS.md) if it
shipped, [REJECTED.md](docs/REJECTED.md) if it did not, plus a section in
[FINDINGS.md](docs/FINDINGS.md) when the round explained something. The ledgers
are the record; this file is the to-do list. 2.4 is the first item maintained
that way.

---

## 1. What this review measured — moved to docs/FINDINGS.md

The four findings this plan was built on, and the source probes and reproduction
recipes behind them, are now in
[docs/FINDINGS.md](docs/FINDINGS.md#1-what-this-review-measured-2026-09-08-snapshot-ab_cache2026-09-05):

| | Finding | Drives |
|---|---|---|
| §1.1 | [`residual_load` is temperature in disguise, and its wind term is saturated](docs/FINDINGS.md#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated) | 2.1 (the primary item); re-opens the intraday block |
| §1.2 | [The regime the model was tuned on is moving away from it](docs/FINDINGS.md#12-the-regime-the-model-was-tuned-on-is-moving-away-from-it) | 2.5, 2.6, 2.9 |
| §1.3 | [Sources verified as reachable, with what they carry](docs/FINDINGS.md#13-sources-verified-as-reachable-with-what-they-carry-appendix-a-has-the-probes) | 2.3, 2.7, 2.14 |
| §1.4 | [The weather grid has no point under the northern wind fleet](docs/FINDINGS.md#14-the-weather-grid-has-no-point-under-the-northern-wind-fleet) | 2.13 |

Appendix A (source probes, 2026-09-08) and Appendix B (reproduction recipes)
moved with them.

---

## 2. Items, in priority order

Each item: mechanism → what to build → arms and the pre-registered bar → what a
failure would mean. Effort is a rough guess of agent time excluding compute.

### 2.1 Rebuild residual load in physical units — the primary item

**Mechanism.** [FINDINGS.md §1.1](docs/FINDINGS.md#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated). The feature meant to carry "demand minus renewable
supply" carries demand only. A residual load whose three terms are on one scale
(MW, or MW-equivalents) can be low on a windy mild night *and* on a sunny mild
midday, which is what the trough targets are; the current one can only be low
when it is warm.

**What to build.** Three layers, each testable on its own:

1. **Unit fix.** Divide by 3.6 *inside* `_wind_power_curve` (or request
   `wind_speed_unit=ms` and fix the raw columns' scale everywhere). Do it inside
   the curve for the A/B: tree splits are invariant to a linear rescale of a
   single column, so leaving the raw `mean_wind_*` / `max_wind` columns as-is
   keeps every untouched target bit-identical, which is the wiring check the
   docs/AB_TESTING.md asks for. Fix the raw columns' unit in a separate, documented commit
   after adoption (docs/FEATURES.md says "10m" and the power-curve docstring
   says "m/s"; both are wrong).
2. **Scale.** Express each term in MW-equivalents:
   `residual = demand(T) − Σ_zone cap_wind[zone, year] · curve(v_zone) − Σ_zone cap_solar[zone, year] · f(rad_zone)`,
   with the zone weights `WIND_*_W` / `SOLAR_*_W` retained as coupling
   multipliers. The daily feature, `residual_load_min`, `residual_load_range`,
   `wind_night` and `radiation_midday` all derive from the hourly series, so
   this happens in `add_residual_load` / `aggregate_intraday_features`, not in
   a `transform` (see the harness note below).
3. **Calibration instead of constants.** Fit the three mappings on the training
   window from ENTSO-E actuals, refit at every run: demand(T) from A65 actual
   total load for SE4 against `mean_temp` / HDD (gives MW per °C); the
   aggregate wind power curve per zone from A75 actual wind generation
   (onshore B19 + offshore B18) against Open-Meteo wind speed, per year so the
   capacity buildout is absorbed; solar likewise from A75 B16 against
   radiation. This is the principled version of the rejected "solar-capacity
   index" (old item 1): the scaling comes from measured generation, not a
   hand-drawn line, and applies to wind as much as solar. A68 installed
   capacity is the fallback if A75 is patchy for a zone.

**Harness note (the trap from old item 1).** `ab.variants.Variant.transform`
receives the merged daily frame, after these columns were computed from the
hourly inputs. It cannot rebuild them. Add an optional `build_fn(inputs) ->
frame` to `Variant` (default `features.build_training_data`) so a candidate can
rebuild the frame from the same snapshot; the harness must still assert
identical `date` columns between arms. Do this once — items 2.3 and 2.4 need it
too.

**Arms, pre-registered.**

| arm | change | expectation |
|---|---|---|
| U | unit fix only | small; mostly a wiring check, and it moves `wind_night` |
| S | unit fix + MW scaling with fixed per-zone capacities (one constant per zone) | the primary arm for `residual_load` / `residual_load_min` |
| C | S with per-year calibrated curves from A75/A65 | the primary arm overall; expected to matter most in the NOW cluster where solar has grown |
| C+ | C plus `wind_night` and `radiation_midday` re-added to the trough list | the re-opening of the intraday block, on a feature that now varies |

Bar: `classify_clustered` on cheap2h, then min, four clusters; per-window std
not inflated; untouched targets (`avg`, `max`) bit-identical in arm U. Screen S
and C on NOW, confirm on −6M/−12M/−21M. C+ is judged against C, not against
production. Check `corr(delta, min_train)` as always; the grid keeps
`min_train` constant so it should be ~0.

**If it fails.** If S and C are NOISE on both trough targets, the reading is
that the raw wind columns already carry what the trees need and residual load
is a temperature channel by accident that happens to work. Then rename it
honestly (`demand_proxy_min`) and close the intraday block for good. Either
answer is worth having; today the ledger has verdicts on a feature nobody knew
was degenerate.

**Effort.** 1 day for U+S and the harness hook; 2–3 days more for C (three
ENTSO-E fetches, a calibration module, a production source change).

### 2.2 Production prediction log and realised scorecard

**Mechanism.** Every accuracy number in docs/MODEL.md is a backtest with two known
optimisms that no backtest can remove: archive weather in place of the forecast
(unmeasured, grows with horizon) and the gap1 fit tail (+0.155 / +0.258 /
+0.591). The only instrument that measures what Home Assistant actually
received is the record of what was sent, scored against what was realised.
There is no such record today: `predictions_raw` is overwritten on every push.

**What to build.**

- In `predict.main`, after conversion, append one row per (run date, target
  date, horizon) to a CSV with `cheap2h`, `min`, `avg`, `max`, `cheap2h_low`,
  `cheap2h_high` in EUR/MWh, plus the run's `coverage_calibrated`. Persist it
  where the Actions run can write and later runs can read: simplest is a
  `predictions-log` orphan branch the workflow commits to (`actions/checkout`
  with `ref`, `git commit`, `git push`), or an HA long-term statistic if the
  branch is unwanted. Start today — the value is in the months of history.
- `score_production.py`: joins the log to realised daily prices (ENTSO-E,
  already fetched), reports MAE per target and per horizon, interval coverage
  per month, and the *decision* regret of the charging rule the automation
  actually uses (README HA section) — the metric that the slot plan and the
  q90 rule were both scored on in backtests but never in production.
- Re-derive the two optimism figures empirically once ~13 weeks are logged:
  production MAE at D+2 minus backtest MAE at D+2 is the sum of the weather and
  gap1 effects, and the per-horizon slope is the weather part.

**Bar.** None — this is instrumentation. It is the prerequisite for reading
item 2.3 honestly.

**Effort.** Half a day. Zero model risk.

### 2.3 The weather axis: measure it, then improve it

The README's Known Limitations list "weather in evaluation is archive, not
forecast" as a known
unmeasured limitation. It is now measurable back to 2024-03 (FINDINGS.md §1.3), which is
~2.5 years — enough for a four-cluster grid on the last two years.

**2.3a Forecast-vintage evaluation (measure first).** Fetch, for every hour in
the window, the value forecast N days earlier (`*_previous_dayN`, N = 1..7) for
wind, temperature and radiation at the six locations, one model (ECMWF IFS
0.25° is the natural default; ICON has one more month of history). Build a
second frame where the *test* rows' weather comes from the vintage matching the
horizon (`h = 1..7`) while the training rows keep archive weather — that is what
production does. Report MAE by horizon against the archive-weather eval.
Expectation: the far horizons are worse by an amount comparable to the whole
gap5 effect, and this is the number that decides whether 2.3b–d are worth
anything. Note `mae_by_horizon` is weekday-confounded (docs/MODEL.md); compare
like
with like by evaluating both frames on the same grid.

**2.3b Multi-model blend at serve time.** Mechanism: forecast error at D+3..D+7
is the largest un-modelled error source, and averaging independent NWP models
is the cheapest known reduction of it. Fetch ECMWF + ICON + GFS with `models=`
and use the mean (or the median) for every weather input; for the Nordic points
add `metno_seamless` (MET Norway MEPS) and `dmi_seamless`, the regional 2.5 km
models that are the reference for Scandinavian wind at D+1..D+2 and fall back
to a global model beyond that. Measurable only on
the 2.3a frame — Previous Runs serves all three models, so the blend's vintage
history exists too. Bar: lower MAE at every horizon ≥ 3 in every cluster, on
cheap2h.

**2.3c Train on what you serve.** Mechanism: the model learns
price = f(analysis weather) and is served f(forecast weather); at D+5 the
input distribution differs (smoother, biased). Training the far-horizon rows on
the previous-run forecast of matching lead time removes the mismatch. This is
horizon-aware modelling by a different route than the shelved
`forecast_horizon` feature; it needs per-horizon models or a stacked training
set with a horizon column. Only worth it if 2.3a shows the far-horizon
degradation is large. Bar as 2.3b.

**2.3d Ensemble spread as an interval conditioner.** Mechanism: the conformal
band is one width for the whole forecast set, adjusted by a single correction.
Weather uncertainty varies day to day and is directly observable as the
51-member spread of `wind_speed_100m`. Condition the conformity quantile on
spread bins (or add spread as a feature to the two quantile regressors).
Scored on pinball loss and coverage per cluster, never MAE (docs/DECISIONS.md,
the interval entry). Live only — the ensemble has no vintage archive — so this is scored
prospectively via 2.2, or on the weeks of data that accumulate before the
next review.

**Effort.** 2.3a two days (fetch is 6 locations × 3 variables × 7 vintages ×
~900 days, well within Open-Meteo's free tier if cached); 2.3b one day; 2.3c
three days; 2.3d one day.

### 2.4 Cross-border capacity — MEASURED 2026-09-12, closed for the trough targets

Old item 0 was run in full (flows + outages, four period clusters, break-week
replay). Verdicts and the A78 data traps are in docs/FINDINGS.md "Cross-border
capacity and flows (round 21)"; the scripts are `experiments/run_round21_*.py`. Three
results change this plan:

* **Closed for `cheap2h`/`min`.** Every arm NOISE, headroom harmful on cheap2h,
  and a *leaky* capacity schedule NOISE as well — so there is no version of
  this data, however privileged, that pays on the trough targets. Do not
  re-open with another encoding.
* **The break needs the northern supply balance, not the interconnector.** The
  `capacity × calm` interaction fired on exactly 08-18/19/20 and moved nothing,
  because the same state in Sep 2024 came with cheap2h 6.6. This promotes 2.13.
* **What is left is `avg` only:** `export_headroom_lag1` is REAL on both
  5-year snapshots (−0.423 / −0.218, all four clusters, std falling) and is the
  one open candidate from the whole item. To confirm it:
  1. Re-encode for serve time. The A/B used *realised* A11 flows on the run
     day; production knows tomorrow's **scheduled** exchanges (Nord Pool
     `DayAheadFlow`, or ENTSO-E day-ahead scheduled commercial exchanges) plus
     the outages posted for that day. Fetch whichever has history and rebuild
     the column from it.
  2. Run on a **fresh** long snapshot and a fresh cross-border fetch (the
     parser was fixed 2026-09-12), in the same batch as the other items.
  3. Re-check after 2.1 ships — `residual_load` is in avg's list and verdicts
     are scoped to the model.
  Priority is below 2.1/2.13: it is an `avg`-only, −0.2 to −0.4 effect against
  two new production fetches.

**Effort.** One day, mostly the scheduled-flow source.

### 2.5 The hurdle's label under regime drift

**Mechanism.** The classifier predicts `P(price_min < 0)`. Positives were
23–27 % of days in 2023–2025 and are 7 % in 2026 (30 of the last 365). A rarer
class is learned worse, and — more importantly — the physical regime the hurdle
was built to flag (oversupply, low demand) still occurs; it now clears at 0–5
EUR/MWh instead of below zero (84 days under 5 in the last year). The label has
drifted away from the regime.

**Arms.**
- H5: threshold 5 EUR/MWh (`NEG_PRICE_THRESHOLD`).
- HQ: a trailing-quantile label, `price_min` below the training window's 10th
  percentile — stationary across regimes by construction.
- HS: `P(cheap2h_day < 0)` and `P(cheap2h_night < 0)` as two probabilities —
  the slot-specific hurdle from `SLOT_PREDICTION_PLAN.md` §4.4, which needs
  the slot targets from 2.6 to exist first.

Bar: `classify_clustered` on cheap2h with the hurdle's own evidence in mind —
the −6M/−12M clusters are the negative-rich ones and the NOW cluster is the
one that matters; a candidate that wins only where positives are plentiful is
not a fix for drift. Do not re-test the hurdle on `min` (closed twice).

**Effort.** Half a day for H5/HQ; HS after 2.6.

### 2.6 Slot targets (round 20) — sequencing with 2.1

Unchanged: `SLOT_PREDICTION_PLAN.md` is the spec. One addition. The slot
values are a night-wind versus midday-solar split, and 2.1 is the item that
gives the model a feature that can tell those apart. Two acceptable orders:

- 2.1 first, then 2.6 (cleanest — the slot scoring run then runs on the
  rebuilt feature and its numbers stay valid), or
- 2.6 first on the current features, with 2.1 then measured on all seven
  targets (the slots become the guard rail the slot plan already describes).

Do not interleave them mid-A/B. `ab.variants.BASELINE` mirrors `model.TARGETS`
(docs/AB_TESTING.md), so the slot targets must not be wired in while a 2.1
candidate is
being measured, and vice versa.

### 2.7 ENTSO-E documents that publish a week ahead

Because D+1 is never predicted, only documents that describe D+2..D+7 *in
advance* are forecast features. Three qualify and none is used:

- **A65 week-ahead load forecast** (processType A31): daily min/max load for
  SE4 for the coming week. Stated mechanism: a TSO's load forecast embeds
  industrial calendars and known large-consumer schedules that temperature and
  `is_workday` do not. Small expected effect (demand is the stable side of SE4
  pricing), cheap to test.
- **A77 nuclear outages weighted by MW** (old item 4). The documents carry
  `nominal_P`; the feature today counts events. A 1400 MW Oskarshamn-3 outage
  and a small unit both score 1.
- **A80 aggregated generation unavailability** for SE4 and DK2 — planned
  outages of CHP, gas and the Karlshamn reserve; and the same for SE3 hydro.
  Lower prior; group with A77 in one fetch.

Test as one "week-ahead block" first (block tests are more sensitive than
singles, docs/AB_TESTING.md), then split if the block wins.

**Effort.** 1 day for the fetches, half a day per A/B.

### 2.8 Neighbour-zone holiday calendar

**Mechanism.** `is_workday` is a top-5 feature and is Swedish. SE4's trough is
now 0.89-correlated with DK2's and 0.73 with DE's (FINDINGS.md §1.2), and DE/DK
holidays
that Sweden does not share (Whit Monday, German Unity Day, Corpus Christi in
the large southern states, Danish Constitution Day) cut continental demand and
push export prices down. Forecastable to any horizon, one line with the
`holidays` package (`holidays.Germany`, `holidays.Denmark`). Also add the
Swedish bridge days the current calendar misses (only the Friday after
Ascension is handled).

**Bar.** `classify_clustered`, cheap2h then min; expect a small effect and be
prepared for INCONCLUSIVE — the grid's resolution floor is ~0.2 EUR/MWh.

**Effort.** One hour.

### 2.9 Time-decay on `min` / `cheap2h` — re-test with a stated drift mechanism

Old item 2 said "re-test the measurement". This review supplies the mechanism
it lacked: FINDINGS.md §1.2 documents a monotone drift in the trough's *composition* (slot
share, negative-price share, coupling) over exactly the training window. A
uniform three-year fit averages a regime that no longer exists into one that
does. Run on the sliding grid, half-life sweep {365, 500, 730}, **screen and
decide on the NOW cluster** (the far clusters describe a world where the drift
had not yet happened). Same honest caveat as before: 0.2 is at the resolution
floor; INCONCLUSIVE is a legitimate answer.

### 2.10 `min ≤ cheap2h` coherence (old item 3, carried unchanged)

Re-measure the violation rate on the current models first (the 31-of-60 figure
predates the list split), then the three arms (raise cheap2h / lower min /
split). Payoff is coherence for Home Assistant, not MAE. If 2.6 ships first,
the sibling identity `min(slots) ≥ cheap2h` gets the same treatment.

### 2.11 Longer horizon for the automation (product decision, cheap)

Open-Meteo serves `forecast_days=16`; the pipeline requests 8 and, after
removing today and tomorrow, sends **six** predicted days (D+2..D+7), not the
"8–10" the README says. Extending to D+2..D+14 is a parameter change plus the
nuclear-outage and calendar windows (`forecast_end = today + 10`). Accuracy
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
- **Docs.** docs/FEATURES.md "Local Weather" says 10 m wind; the fetch is
  100 m. The
  power-curve docstring says m/s; the data is km/h. Fix both with 2.1's unit
  commit. README "Purpose" says 8–10 days; it is six.
- **Harness.** The `build_fn` hook from 2.1. Also a `--targets` filter on
  `run_walk_forward` / `run_ab` (the slot plan §3.4 asks for it) so a
  cheap2h-only screen does not pay for `max`.
- **Run status to Home Assistant.** The workflow has no failure path; a failed
  run leaves the sensor silently stale. Push a `last_run_status` attribute (or
  a separate binary sensor) from a `finally:` block so the automation can fall
  back to "charge now" when predictions are older than a day.
- **Dependencies.** `requirements.txt` is unpinned and the workflow runs
  Python 3.13 while local runs are 3.14; pin at least `xgboost` and `pandas`
  majors so a bit-identity check across machines means something.

### 2.13 Northern wind points, and hydro precipitation

**Mechanism.** [FINDINGS.md §1.4](docs/FINDINGS.md#14-the-weather-grid-has-no-point-under-the-northern-wind-fleet). The same mechanism `mean_wind_stockholm` already
carries — northern surplus sets the SE3 price SE4 imports at — with the actual
fleet under the measurement instead of a load centre 500 km south of it.

**What to build.**

1. Add to `sources.open_meteo.WIND_LOCATIONS`: `norrbotten` (65.3, 20.9),
   `jamtland` (63.2, 14.6), `ostrobothnia` (63.1, 21.6). Adding a key is
   production-safe on its own — `aggregate_international_weather_daily` emits
   `mean_wind_<key>` for every key and no model reads it until it is in a
   feature list — and it means the next `ab_test.py fetch` carries the columns,
   so this item needs no `build_fn` hook. Existing snapshots do not have them;
   fetch a fresh long snapshot after the edit.
2. Two encodings, both pre-registered:
   - **N-block** (control): the three raw daily means added to the trough list,
     the same encoding the existing wind columns use.
   - **`north_wind_power`** (primary): one column, the mean over the three
     sites of the power curve (in m/s — see 2.1-U; do not feed km/h into it)
     with a night-window sibling `north_wind_night`. A blend is the encoding
     that survived for the south, and it is the natural "SE3 import price"
     term for 2.1's rebuilt residual load if that ships first.
   Sundsvall is a fourth candidate only if the three leave it something to add
   (screen it last; its 0.017 is likely absorbed by Jämtland + Norrbotten).
3. Retire nothing. `mean_wind_stockholm` stays; the question of whether it is
   still load-bearing next to the northern points is a later ablation
   (`classify_ablation_clustered`), not part of this A/B.

**Bar.** `classify_clustered` on cheap2h, then min; `avg` reported. Screen on
NOW, confirm on the three far clusters. Expect the effect to be larger in NOW
than pooled (FINDINGS.md §1.4 shows the 2026 gain above the pooled one), which
is the
opposite of the training-size artefact the far clusters used to manufacture,
but check `corr(delta, min_train)` anyway.

**Hydro precipitation (second-order, `avg` first).** The Nordic hydrological
balance sets the system price level and the model sees it only through weekly
reservoir levels, which lag. A 7-day precipitation *forecast* over the hydro
regions is forecastable at run time and is what hydro producers price against.
Screen: trailing 7-day precipitation at Bergen correlates −0.33 with `avg`,
the *leading* 7-day sum −0.17 with `avg` and −0.18 with `cheap2h` (Jämtland
−0.19 / −0.11 / −0.15). Encode as the run-time 7-day forecast precipitation sum
at three points — Vestland (60.4, 5.3), Jämtland (already fetched above),
Lule älv (66.5, 20.0) — as an anomaly against the same-week archive mean, one
frozen scalar across the horizon (add to `FORECAST_FROZEN_FEATURES`; it is a
lag-type feature in the eval). Spring snowmelt timing (temperature at the same
inland points in April–May) is the same mechanism and the same fetch. Judge on
`avg` first; a trough effect is a bonus. Test after 2.1 and the wind points.

**Effort.** Wind points: an hour to wire, one fresh fetch, one A/B batch.
Precipitation: half a day.

### 2.14 Nord Pool UMM as the outage source

**Mechanism.** The open question in old item 0 / 2.4 was whether ENTSO-E A78
is usable as a *forward-looking* feature, because the API returns the current
version of a record and revisions are invisible, so a backtest could know
things production could not. **Round 21 settled that the A78 route cannot
answer it**: every record for an event before 2025-11 was re-published in
November 2025, so the cache holds no vintage at all before that date
(docs/FINDINGS.md "If you touch ab_cache/crossborder/"). Since the trough targets rejected the
data outright, the remaining value of UMM is (a) the MW-weighted nuclear
feature below and (b) a vintage-honest capacity history for 2.4's `avg`
candidate. Nord Pool's UMM API is the upstream of those
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
  count in `nuclear_outage_se3`, old item 4 / 2.7) and unavailable
  interconnector MW per SE4 border (feeds 2.4's `export_headroom`).
- Cache the full message history once under `ab_cache/umm/`; the daily
  production fetch only needs the current window (events overlapping
  today..today+16).
- Encode "as of run time" in the eval: `apply_forecast_freeze` freezes lag
  features; these are per-day features whose *value* depends on the run date,
  so the walk-forward must rebuild them per window from messages published
  before the window start. That is a small extension of the harness (a
  per-window feature builder), and it is the same machinery 2.3a needs for
  forecast vintages — build it once.

**Bar.** MW-weighted nuclear vs the current count, cheap2h then min then
`avg`, `classify_clustered`. Interconnector availability is scored inside 2.4.

**Effort.** One day for the source and cache, one day for the per-window
builder shared with 2.3a.

---

## 3. Carried at low priority, and what stays closed

**Low priority, carried:** `ttf_vs_30d` (old item 5, unchanged reasoning);
`cpl_gap7` on cheap2h and the q90 decision rule at H=7 (round 19 follow-ups);
`B6_fuel_co2` on min, pending post-crisis training windows (~2026-10, per the
rejected table).

**Not re-opened, no new mechanism:** price-level features for the trough
targets (round 18), anchored targets (19b), window length (15), per-target
hyperparameters (13.1), `reg:absoluteerror`, seed ensembling, the avg-anchored
clamp, news / geopolitical / oil features. The solar-capacity index is
*superseded* by 2.1-C rather than re-opened.

**Structural note, not an item.** Nordic flow-based market coupling went live
2024-10-30, inside the training window. If a later analysis finds a step
change in SE4–SE3 coupling around that date, a single dated indicator is a
legitimate feature; do not add it speculatively — it would act as a time
index.

---

## 4. Suggested order

1. **2.2 prediction log** — start collecting now; every other measurement of
   production accuracy waits on months of history.
2. **2.12 harness hook + unit test** — half a day, unblocks 2.1 and 2.3.
3. **2.1 U and S arms** — the cheapest test with the strongest mechanism in
   the file. Decide on C from S's result.
4. **2.13 northern wind points** — add the three keys to `WIND_LOCATIONS`
   before the fresh fetch below so the same snapshot serves 2.1 and 2.13; run
   the two A/Bs back to back.
5. **2.6 slot targets** (or 2.1-C first; see 2.6).
6. **2.3a forecast-vintage eval** — measure before 2.3b–d; it also re-baselines
   the docs/MODEL.md table on an honest weather axis. Build the per-window feature
   builder here and reuse it for 2.14.
7. **2.5 hurdle label**, **2.8 holidays**, **2.9 time-decay**, **2.13
   precipitation** — short A/Bs that share one long snapshot; run as one batch.
8. **2.14 UMM source** (now justified by nuclear MW, not by interconnectors),
   then **2.7 week-ahead documents** and the `avg` remainder of **2.4** — the
   source work; 2.14 needs no token, the ENTSO-E parts do, so Peter runs those
   fetches.
9. **2.10 coherence**, **2.11 horizon** — when convenient.

Fetch one fresh long snapshot (`python ab_test.py fetch --days 1825`) before
step 3 so the whole batch shares a grid; the current long caches are 5d-tail
and predate the top-up (docs/AB_TESTING.md "Snapshot generations").

