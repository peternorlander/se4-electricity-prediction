# SE4 Prediction — Improvement Plan, September 2026 (round 21 onwards)

**Status 2026-10-12 (after round 25).** Written 2026-09-08 as a plan. Since then
2.4, 2.5 (labels), 2.8, 2.9 and 2.16 were measured and rejected and are gone from
this file (rows in [REJECTED.md](docs/REJECTED.md)); 2.10 (`min ≤ cheap2h`) and 2.13
(northern wind points) were measured and shipped ([DECISIONS.md](docs/DECISIONS.md));
2.3a was measured and written up in [FINDINGS.md](docs/FINDINGS.md); 2.1-S and 2.3b
were measured with results that need a decision. What is open, in the order it should
be done, is section 4. It is the successor to `IMPROVEMENT_PLAN.md` (2026-08), which
stays as the record of the round-12 → round-19 follow-ups.

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
are the record; this file is the to-do list. 2.4 was the first item maintained
that way and is now gone from here entirely: three REAL measurements, closed
anyway, all of it in [REJECTED.md](docs/REJECTED.md).

---

## 1. What this review measured — moved to docs/FINDINGS.md

The four findings this plan was built on, and the source probes and reproduction
recipes behind them, are now in
[docs/FINDINGS.md](docs/FINDINGS.md#1-what-this-review-measured-2026-09-08-snapshot-ab_cache2026-09-05):

| | Finding | Drives |
|---|---|---|
| §1.1 | [`residual_load` is temperature in disguise, and its wind term is saturated](docs/FINDINGS.md#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated) | 2.1 (the primary item); re-opens the intraday block |
| §1.2 | [The regime the model was tuned on is moving away from it](docs/FINDINGS.md#12-the-regime-the-model-was-tuned-on-is-moving-away-from-it) | 2.5, 2.6 |
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

**Unit fix — SHIPPED 2026-09-17 (round 23).** Divide by 3.6 *inside*
`_wind_power_curve`; adopted on a cheap2h REAL verdict. Record and numbers:
[DECISIONS.md](docs/DECISIONS.md#residual-load-wind-curve-unit-fix-round-23).
**Still open, deliberately not done in that commit:** the raw `mean_wind_*` /
`max_wind` columns and their unit labels (docs/FEATURES.md, the old
docstring) are still in km/h — a separate, documentation-only commit. The
unit fix alone does not cure the scale mismatch below (FINDINGS.md §1.1);
expect layers 2–3 to matter more, not less, now that 1 is in.

**What's left to build.** Layer 2 (scale) was measured with fixed constants in
round 25; layer 3 (calibration) is open and its data now exists.

2. **Scale — S, measured 2026-10-11.** MW-equivalent terms with fixed per-zone
   capacities (wind SE4 2.5 / Karlskrona 1.0 / DK2 2.5 / DK1 5.0 / DE-north 30 GW,
   solar SE4 2.0 / DK2 1.5 / DK1 3.5 / DE-north 20 GW × 0.8, demand 0.5 GW/°C, the
   existing `WIND_*_W` / `SOLAR_*_W` kept as coupling multipliers); the correlation
   with temperature falls from −0.999 to −0.78. Rolling grid, five caches:
   **`avg` −0.35 … −0.40 (VALIDATED, 12/12 points on every cache)**, **`min` −0.06 …
   −0.17 (REAL on three caches, BORDERLINE one, NOISE one, none harmful)**,
   **`cheap2h` NOISE on four and REAL-harmful +0.21 on one**
   ([REJECTED.md](docs/REJECTED.md)). Doubling the supply terms makes cheap2h
   harmful (+0.34), halving them helps avg more (−0.54): the scale matters, and the
   constants were guesses — the A68 data now fetched shows several too high (DK2
   wind 1.4 GW not 2.5, DK1 4.2 not 5.0, SE4 2.1 not 2.5). S stacks on the
   northern wind points for `min` (−0.20 … −0.31 further) and `avg` (−0.37 … −0.39
   further). **Not shipped.** Because cheap2h does not want it, shipping means
   *extra* columns (`residual_load_mw`, `residual_load_min_mw`) in the `avg` and,
   after a replay, `min` lists only; the cheap2h list keeps today's. Re-score the
   winner on forecast vintages first ([AB_TESTING.md](docs/AB_TESTING.md#archive-weather-is-a-perfect-forecast-re-score-weather-candidates-on-vintages)).
3. **Calibration instead of constants — C.** Fit the three mappings on the training
   window from ENTSO-E actuals, refit at every run: demand(T) from A65 actual load
   for SE4 against `mean_temp` / HDD (MW per °C); the aggregate wind power curve per
   zone from A75 actual wind generation (B19 + B18) against Open-Meteo wind speed,
   per year so the buildout is absorbed; solar likewise from A75 B16 against
   radiation. This is the principled version of the rejected solar-capacity index:
   the scaling comes from measured generation, not a hand-drawn line
   ([FINDINGS.md](docs/FINDINGS.md#solar-capacity-buildout-is-visible-in-the-data-a-linear-index-still-doesnt-capture-it-round-22)).
   **The data is fetched:** `ab_cache/entsoe_r25/2026-10-11` (A75 wind/solar for
   SE1–SE4, DK1, DK2, DE_LU, FI from 2021-07; A65 load; A68 only patchy — Swedish
   zones have 2026 wind only, so calibration must come from A75). Read
   [FINDINGS.md](docs/FINDINGS.md#entso-e-documents-four-traps-found-validating-the-round-25-fetch-2026-10-11)
   first: filter `gen` on `direction == "in"`, A03 is already filled.
   **Measure the ceiling before building the calibration.** Add the *realised*
   zonal generation (SE1+SE2 wind MW, DK1/DK2 wind, DE_LU wind and solar — leaky, not
   knowable at run time) as columns to the trough models. If that oracle arm does
   not beat the northern-wind points by more than the grid's noise (~0.1 on cheap2h,
   see AB_TESTING), a calibrated weather→generation mapping has no headroom and C is
   not worth its 2–3 days. If it does, the same A75 series trains the per-zone
   mapping, and the honest version is scored on vintages.

**Harness note (the trap from old item 1).** `ab.variants.Variant.transform`
receives the merged daily frame, after these columns were computed from the
hourly inputs. It cannot rebuild them. Round 25 worked around it by building the
frames in the experiment script (`experiments/run_round25.py`) and feeding them to
the unmodified `run_walk_forward`; an optional `build_fn(inputs) -> frame` on
`Variant` is still the clean fix, and 2.3 and 2.7 need it too.

**Arms.**

| arm | change | state |
|---|---|---|
| ~~U~~ | unit fix only | **shipped 2026-09-17** |
| ~~S~~ | MW scaling, fixed constants | measured: avg VALIDATED, min PROMISING, cheap2h rejected (above) |
| C | per-year calibrated curves from A75/A65 | open; gated on the oracle ceiling above |
| C+ | C plus `wind_night` and `radiation_midday` re-added to the trough list | open; judged against C |

Bar for C/C+: `classify_clustered` on cheap2h, then min, NOW / −6M / FAR on the
rolling grid, per-window std not inflated; a vintage re-score for the winner.

**Effort.** U and S done. Oracle ceiling: half a day. C: 2–3 days.

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
Next step is a *forward* replay, not another backtest: log the blend's inputs next
to the served forecast from the first day item 2.2 exists, and score both on the
days the log covers, with a bar written for production's lead mapping (row j at
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
  **A production defect sits under this item, and fixing it was tested and rejected
  (round 26, 2026-10-12).** `fetch_nuclear_outages_se3` sends `psrType=B14` but the API
  ignores it for A77: the SE3 response also holds gas CHP (B04), hydro (B12) and wind (B19)
  units, the parser counts them, and it counts cancelled records (docStatus A09) and every
  partial derate as a full event. Over five years the daily `nuclear_outage_se3` averages
  2.39 against 1.12 for distinct non-cancelled nuclear units and differs on 70 % of days.
  The column is in `FEATURE_COLUMNS`, so `avg` and `max` read it; the trough lists do not.
  **The straightforward fix is REAL-harmful on `avg` (+0.13 on 3 of 5 caches)** — see
  [REJECTED.md](docs/REJECTED.md) — so production keeps the old, misnamed column and the item
  becomes a decomposition: from the saved A77 records build (a) distinct non-cancelled nuclear
  units, (b) the same MW-weighted (`nominal − available`), (c) a count of non-nuclear
  production units out (hydro, gas, wind), (d) a count of cancelled records, and test each
  and the old column's replacement by the pair (a)+(c) on `avg` with `classify_clustered`.
  Keep the old column until one arm beats it. The fixed parser is
  `experiments/results/nuclear_parser_fix.patch`; it also has a guard worth keeping — it
  raises if the response stops carrying `psrType`/unit, instead of zeroing the feature.
  Parsed data: `ab_cache/entsoe_r25/2026-10-11/unavail.pkl`, raw responses beside it
  (`experiments/fetch_round25_entsoe.py`).

  **The data is fetched** (`ab_cache/entsoe_r25/2026-10-11`: week-ahead min/max load for SE4, SE3, DK2, DE_LU from 2021-07 with ~98 % day coverage; A77/A80 with unit, `psr`, nominal and available MW, revision and a `cancelled` flag; the traps are in [FINDINGS.md](docs/FINDINGS.md#entso-e-documents-four-traps-found-validating-the-round-25-fetch-2026-10-11)).

Test as one "week-ahead block" first (block tests are more sensitive than
singles, docs/AB_TESTING.md), then split if the block wins.

**Effort.** 1 day for the fetches, half a day per A/B.

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
- **A NaN last close reaches the forecast features (found 2026-10-12).** The cached
  EUA series from 2026-10-10 ends with a NaN close (Yahoo returned no `CO2.L` price for
  2026-10-09). `build_forecast_features` takes the *last row* of `eua_daily` (and
  `ttf_daily`) as `co2_price_lag1` / `ttf_price_lag1` without dropping NaN, so on such a
  day every forecast row gets NaN `co2_price_lag1` and `gas_marginal_cost` — the latter is
  in the trough lists — and XGBoost treats it as missing without a word. The training side
  forward-fills, the forecast side does not. Fix: take the last non-NaN value (and log it);
  add the case to the test suite above.
- **Dependencies.** `requirements.txt` is unpinned and the workflow runs
  Python 3.13 while local runs are 3.14; pin at least `xgboost` and `pandas`
  majors so a bit-identity check across machines means something.

### 2.13 Northern wind points — shipped 2026-10-12, replay still owed

Record: [DECISIONS.md](docs/DECISIONS.md#northern-wind-points-round-25). Hydro
precipitation and a fourth site (Sundsvall) were measured and rejected
([REJECTED.md](docs/REJECTED.md)). Three things are left, all follow-ups:

1. **Replay on `ab_cache/long/2026-10-16`** (fetch it with the fix in; the caches
   carry the sites already) — the first genuinely later NOW window. The expectation
   to score is "about half the archive gain": cheap2h −0.4 … −0.5 on forecast
   weather, −1.0 … −1.2 on archive.
2. **The same vintage re-score on `min`** at the production lead map, per cluster,
   to put a number next to the cheap2h one (the audit has it for two caches).
3. **The production log (2.2) is where the real gain shows.** Compare the served
   `cheap2h` against the realised price before and after 2026-10-12.

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
closed on its serve-time encoding, **the remaining value of UMM is the
MW-weighted nuclear feature below** — the interconnector half has no consumer
left, so build it only if something asks for it. Nord Pool's UMM API is the upstream of those
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
  per-window feature builder), and it is the same machinery 2.3a needs for
  forecast vintages — build it once.

**Bar.** MW-weighted nuclear vs the current count, cheap2h then min then
`avg`, `classify_clustered`.

**Effort.** One day for the source and cache, one day for the per-window
builder shared with 2.3a.

---

### 2.15 German prices: identify the right curve, then re-test the DE lag

**DONE 2026-10-11.** a: the second curve is sequence 2, the German 15-minute
intraday auction (both curves labelled A01); production now loads SDAC
(sequence 1, request pinned for DE/LU only), 100 % equal to SMARD 2021–2026, and
ambiguity is a hard error. b: nothing favourable on the clean series — best arm
cheap2h + `price_de_lag1` NOISE −0.047; the trough targets' price-level rejection
and 14b are closed on a clean instrument; `max` −0.32 from the fix itself.
All long caches in `ab_cache/long/` are corrected in place to SDAC. Record:
[DECISIONS.md](docs/DECISIONS.md#one-day-ahead-price-per-hour-chosen-deterministically),
[REJECTED.md](docs/REJECTED.md),
[FINDINGS.md](docs/FINDINGS.md#german-prices-two-auctions-in-one-response-round-215).
The text below is the plan as written.

**Mechanism (new, and the reason this is not just a re-test of a closed
question).** Until 2026-10-10 `price_de` was a coin flip between two curves on
95 % of hours — median gap 9.74, max 695 EUR/MWh — re-rolled on every fetch and
on every production run
([FINDINGS.md](docs/FINDINGS.md#cached-history-is-stable-except-for-one-column-2026-09-17),
[DECISIONS.md](docs/DECISIONS.md#one-day-ahead-price-per-hour-chosen-deterministically)).
So every measurement that ever used a German lag used a column that was partly
noise and not reproducible: round 18's `price_de_lag1` arms on min/cheap2h,
round 14b's price-family drop on `avg`, the price-block arms in rounds 11 and
14a, and `max`'s own feature list. The selection is now deterministic. That is a
changed *instrument*, not a changed hypothesis, which is exactly what the
scoping rule says re-opens a verdict.

**a. Decide what we should be loading.** Run `python
experiments/probe_a44_curves.py --probe` (needs `ENTSO_E_TOKEN`) and answer
three things: how many TimeSeries an A44 response for DE/LU carries, what
distinguishes them (`contract_MarketAgreement.type`, `businessType`,
resolution), and whether adding `contract_MarketAgreement.type=A01` to the
request removes the second curve at the source. If the second curve is an
intraday auction, pin the request and turn the parser-side tiebreak into a hard
error — a day-ahead model should never silently train on intraday prices. Check
SE4 and DK2 the same way: they showed no conflicts across 12 cache pairs, but
they run through the same function and a conflict there would reach the priority
targets. Half a day, no model risk. **Before b**, which can only measure
whatever a settles on.

**b. Re-test the DE lag on a deterministic series.** Pre-registered arms:
`price_de_lag1` alone on cheap2h and min (round 18's exact arm), and the
nine-column price family back on `avg` (round 14b's arm, inverted). Rolling
grid, `classify_clustered`, on a long snapshot fetched after a. The honest prior
is that nothing moves: a noisy column is a handicap, so cleaning it can only
help arms that were rejected as *harmful*, and round 18's were harmful by
+0.21…+0.61 — more than noise in one of fifteen columns plausibly explains.
Both outcomes are worth the day. If an arm turns favourable, the trough targets'
blanket rejection of a price level ([REJECTED.md](docs/REJECTED.md)) was partly
an artefact of a broken column and has to be re-read. If nothing moves, that
rejection is finally measured on a clean instrument and can be closed instead of
being re-opened every time someone notices that DE leads SE4. Note that `max` is
the only target that currently reads the column and has never been audited: if a
changes which curve gets loaded, re-baseline `max` in
[MODEL.md](docs/MODEL.md) too — not as a priority, but because the number there
was measured on the coin flip.

**Effort.** Half a day for a; one A/B batch for b, sharing a snapshot with the
next batch.

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

## 4. Suggested order (2026-10-11)

1. **Done 2026-10-12: the northern wind points (2.13) and the `min ≤ cheap2h` coherence**
   ([DECISIONS.md](docs/DECISIONS.md)). The replay of the first is 2.13 above; the headline
   MAE table in MODEL.md predates both.
2. **2.2 prediction log** — start the clock today; it is the instrument that prices
   everything weather-related for real, and the forward replay for 2.3b and for 2.13
   waits on it.
3. **2.7: decompose `nuclear_outage_se3`, then the week-ahead block.** The column counts
   gas, hydro, wind and cancelled records as nuclear outages (2.39 vs 1.12), but the plain fix
   made `avg` worse, so it is a question of what the extra information is; the data is in
   `ab_cache/entsoe_r25/2026-10-11`.
4. **2.1 oracle ceiling, then C** — the A75/A65 data is in hand; half a day tells
   whether a calibrated weather→generation mapping has headroom beyond the northern
   wind points. In the same pass, the `avg` (and replayed `min`) MW-scale columns.
5. **2.6 slot targets** — unchanged; the guard-rail targets are better built after
   the feature set above is settled, so the scoring run measures the final features.
6. **2.3b forward replay / 2.3c** — once the log has a few weeks.
7. 2.12 hygiene (incl. the `build_fn` hook), 2.11 horizon, 2.14 UMM, 2.5-HS after 2.6.

**Done 2026-09-12: `ab_cache/long/2026-09-12` is fetched** — 1825 days, **1787
usable rows** from 2021-10-21, weather tail **1d**. It is the first long cache
taken after the archive top-up, so run the whole batch on it and do **not** mix
it with `long/2026-08-06` or `long/2026-08-21` in one grid (those are 5d-tail;
docs/AB_TESTING.md "Snapshot generations"). Its row count differs from both
older caches for a second, harmless reason as well — see docs/AB_TESTING.md
"How long a snapshot to fetch" before updating any `EXPECTED_ROWS` constant.

