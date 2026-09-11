# SE4 Prediction — Improvement Plan, September 2026 (round 21 onwards)

**Status: PLANNED 2026-09-08. Nothing in this file is implemented.** It is the
successor to `IMPROVEMENT_PLAN.md` (2026-08), which stays in place as the
record of the round-12 → round-19 follow-ups. Where an item carries over it is
said so; where this file changes a priority it says why, with the measurement
that changed it.

Read first, as before: README "A/B Backtest Flow", "How changes are validated",
"Measurement grids", "Which verdict function to call", and "Features Tested and
Rejected". The standing rules from `IMPROVEMENT_PLAN.md` are unchanged and not
repeated here: state a mechanism before measuring, screen on one period cluster
and confirm on the others, pre-register the bar, use the round-15b sliding grid
on a long snapshot with `classify_clustered` / `classify_ablation_clustered`,
`OMP_NUM_THREADS=4`, priority cheap2h → min → avg, `max` not a priority.

`SLOT_PREDICTION_PLAN.md` (round 20) is unchanged and still next in the queue;
section 2.6 below says how it interacts with the new item 2.1.

---

## 1. What this review measured (2026-09-08, snapshot `ab_cache/2026-09-05`)

Four findings, each of which moves a priority. Numbers are reproducible with
the recipes in Appendix B.

### 1.1 `residual_load` is temperature in disguise, and its wind term is saturated

| quantity | value |
|---|---|
| corr(`residual_load`, `mean_temp`) | **−0.9997** |
| corr(`residual_load_min`, `max_temp`) | **−0.9988** |
| daily std of the demand term `15 − mean_temp` | 6.60 |
| daily std of the wind term `(min(v,13)/13)³` | 0.125 |
| daily std of the SE4 solar term `radiation/500` | 0.202 |
| share of *hours* with the SE4 wind term at 1.0 (rated) | **88 %** (DE-north 84 %, DK1 89 %, DK2 86 %, Karlskrona 85 %, Stockholm 83 %) |
| share of *days* with the SE4 daily wind term at 1.0 | **94 %** |

Two independent defects:

1. **Unit mismatch.** `sources/open_meteo.py` never passes `wind_speed_unit`, so
   Open-Meteo returns `windspeed_100m` in its default **km/h** (median 26.6 at
   Malmö, i.e. 7.4 m/s — a normal 100 m wind). `features._wind_power_curve`
   documents its input as m/s with `rated_speed = 13`. 13 km/h is 3.6 m/s, cut-in
   speed, so the cubic curve is at rated output on almost every hour and the
   "wind supply" term only distinguishes dead calm from everything else.
   Dividing by 3.6 inside the curve restores the intended shape: the daily
   term's std goes 0.125 → 0.266 and its share at 1.0 goes 94 % → 4 %.
   `wind_night`'s correlation with `price_cheap2h` roughly doubles
   (−0.116 → −0.218) from that one change.
2. **Scale mismatch, which the unit fix does not cure.** Even corrected, the wind
   term spans [0, 1] and the solar term [0, ~1.5] while the demand proxy is in
   °C and spans ~40. One unit of "full wind" is worth one degree of temperature.
   So `residual_load` and `residual_load_min` remain ≥ 0.998 correlated with
   temperature after the fix; the "physical trough driver" the README describes
   does not exist in the model. The trough targets get their wind information
   only through the raw speed columns `max_wind`, `mean_wind_stockholm`,
   `mean_wind_de_north` — no power curve, no regional weighting, no night
   window.

**Why this matters for the ledger.** Several rejected entries were measured on
this feature: the intraday trough block (`wind_night`, `radiation_midday`,
`residual_load_range`) as a year-round addition, the seasonal interaction round
12, and the solar-capacity scaling. Each was a test of a column whose wind
component was ~constant. That is a stated mechanism for re-opening them — after
the feature is rebuilt, not before. `residual_load_min`'s LOO verdict ("keep,
harmful to remove") is consistent with it being the trough list's only
temperature channel, not evidence that the trough physics is captured.

### 1.2 The regime the model was tuned on is moving away from it

| | 2023 (Sep–Dec) | 2024 | 2025 | 2026 (Jan–Sep) |
|---|---|---|---|---|
| share of days with `price_min < 0` | 26 % | 27 % | 23 % | **7 %** |
| share of days with `cheap2h ≤ 0` | 24 % | 25 % | 21 % | **7 %** |
| slot of the daily min: night 00–07 | 66 % | 45 % | 38 % | **26 %** |
| slot of the daily min: day 08–15 | 4 % | 31 % | 46 % | **63 %** |
| corr(SE4 cheap2h, DK2 cheap2h, same day) | 0.71 | 0.67 | 0.79 | **0.89** |
| corr(SE4 cheap2h, DE cheap2h, same day) | 0.44 | 0.47 | 0.66 | **0.73** |

Last 365 days: 30 negative-price days, 84 days with cheap2h < 5 EUR/MWh.

Three consequences: the hurdle classifier's positive class (`price_min < 0`) is
thinning in the NOW regime (item 2.5); the trough is now a midday-solar event
more often than a night-wind one, which is exactly the split the slot plan
formalises (2.6) and which the current blended features cannot express (2.1);
and SE4's trough is increasingly the DK2 trough, so anything that improves the
DK/DE side of the weather → supply mapping pays more every year (2.1, 2.3).

### 1.3 Sources verified as reachable, with what they carry (Appendix A has the probes)

| source | what it adds | horizon | history |
|---|---|---|---|
| Open-Meteo **Previous Runs API** | the forecast that was *issued* 1–7 days before each hour, per model | — | ICON / GFS from **2024-03-01**, ECMWF IFS from **2024-04-01** |
| Open-Meteo `models=` and **Ensemble API** | ECMWF / ICON / GFS / Météo-France side by side; 51 ECMWF ensemble members | 8 d (ensemble), **16 d** (`forecast_days=16` works) | live only |
| Energinet `Forecasts_Hour` | Energinet's own wind/solar forecasts for DK1/DK2, with the day-ahead vintage stored | D+1 only | from 2019-10-31 |
| Energinet `Transmissionlines` | hourly `ImportCapacity` / `ExportCapacity` / scheduled and physical exchange per DK border, incl. **DK2–SE4** | published with the day-ahead result | depth not verified (rate-limited during the probe) |
| Nordpool `DayAheadFlow` | tomorrow's *scheduled* day-ahead flow per SE4 border (SE3, SE3-SWL, DK2, PL, GER, LT), 15-min | D+1 | per-date query; depth not verified |
| ENTSO-E, not yet used | A65 **week-ahead** load (processType A31); A75 actual generation per type; A68 installed capacity per type; A77 with MW; A80 generation unavailability | see 2.7 | needs `ENTSO_E_TOKEN` |
| Open-Meteo **Nordic regional models** (`metno_seamless`, `dmi_seamless`) | MET Norway MEPS and DMI Harmonie (2.5 km), the reference short-range wind models for Scandinavia; "seamless" falls back to a global model beyond their range | 8 d | Previous Runs verified for 2025-06; earlier not probed |
| Nord Pool **UMM API** (`ummapi.nordpoolgroup.com/messages`) | every production / transmission / consumption unavailability message with **`publicationDate` and `version`**, installed and available MW per period, in/out area EICs, reason text; nuclear units carry installed 1081–1172 MW | published ahead of the event; **every earlier version retrievable** (`/messages/{id}/{version}`) | since 2015 (SE4 borders: 67 messages in March 2016 alone), ~115k messages, no auth |
| Additional Open-Meteo points, northern wind belt (1.4) | SE2 (Norrbotten, Jämtland) and Finnish Ostrobothnia wind — the fleets behind the SE3 import price | 8–16 d | archive to 1940 |

Note the horizon column. Production runs after tomorrow's prices are published,
so D+1 is never predicted and any source that only reaches D+1 is useless as a
*forecast* feature — it can still be valuable for **calibration** (2.1) and for
the **coupling state** (2.4), which is quasi-static across the week.

### 1.4 The weather grid has no point under the northern wind fleet

`mean_wind_stockholm` is the most load-bearing trough feature (LOO: +1.36 on
cheap2h, +1.02 on min if removed), yet Stockholm is a load centre with little
wind capacity. What it proxies is the Nordic surplus that reaches SE4 through
SE3: when the north is windy, SE3 clears low and SE4 imports at that price
(the 2026-08-17 break was the opposite — Stockholm wind at its 2nd percentile).
The fleets that create that surplus sit in SE2 (Markbygden in Norrbotten,
Jämtland, Västernorrland — Sweden's largest wind zone), SE1 and Finnish
Ostrobothnia, and none has a point in `WIND_LOCATIONS`. The southern side is
covered three times over (DK1, DK2, DE-north).

Screen, 2026-09-11: linear incremental R² on `cheap2h` from adding a candidate
point's daily and night-window power-curve terms to the existing six sites plus
radiation, temperature and calendar; 2023-09 → 2026-08 (n = 1091) and 2026
alone. In-sample and linear, so a ranking, not an effect size.

| candidate (lat, lon) | represents | ΔR² all | ΔR² 2026 | corr with nearest existing site |
|---|---|---|---|---|
| **Norrbotten** (65.3, 20.9) | SE2 north, Markbygden | **+0.041** | **+0.060** | 0.33 |
| **Ostrobothnia** (63.1, 21.6) | FI wind belt | **+0.037** | **+0.049** | 0.59 |
| **Jämtland** (63.2, 14.6) | SE2 inland | **+0.025** | **+0.056** | 0.49 |
| Sundsvall (62.4, 17.3) | SE2 coast | +0.017 | +0.039 | 0.50 |
| Bergen (60.4, 5.3) | NO hydro, precipitation | +0.006 | +0.018 | 0.44 |
| Gdańsk (54.4, 18.6) | PL Pomerania | +0.004 | +0.004 | 0.64 |
| Göteborg (57.7, 12.0) | SE3 west coast | +0.003 | +0.021 | 0.81 |
| Rostock (54.1, 12.1) | DE Mecklenburg / Baltic coast | +0.003 | +0.003 | 0.90 |
| Munich (48.4, 11.6), radiation | DE south solar | +0.002 | +0.022 | 0.84 |
| South Baltic offshore (55.0, 13.5) | Kriegers Flak / Arkona | +0.001 | +0.001 | 0.91 |
| German North Sea (54.3, 7.0) | offshore cluster | +0.000 | +0.009 | 0.75 |

For scale, removing an *existing* site from the same regression costs 0.002–0.007
except Stockholm at 0.049. The three northern points are each worth more than any
existing site but Stockholm, carry information the grid does not have
(correlation 0.33–0.59 with the nearest existing point), and matter more in 2026
than pooled. Every southern candidate is redundant with DK/DE-north at
correlation 0.75–0.91. DE-south solar is nil pooled and +0.022 in 2026 — the
rising solar coupling of 1.2 — and belongs with the day-slot work (2.1-C+, 2.6),
not with a wind-point addition.

---

## 2. Items, in priority order

Each item: mechanism → what to build → arms and the pre-registered bar → what a
failure would mean. Effort is a rough guess of agent time excluding compute.

### 2.1 Rebuild residual load in physical units — the primary item

**Mechanism.** Section 1.1. The feature meant to carry "demand minus renewable
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
   README asks for. Fix the raw columns' unit in a separate, documented commit
   after adoption (README says "10m" and "m/s" in places; both are wrong).
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

**Mechanism.** Every accuracy number in the README is a backtest with two known
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

The README lists "weather in evaluation is archive, not forecast" as a known
unmeasured limitation. It is now measurable back to 2024-03 (1.3), which is
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
anything. Note `mae_by_horizon` is weekday-confounded (README); compare like
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
Scored on pinball loss and coverage per cluster, never MAE (README, interval
section). Live only — the ensemble has no vintage archive — so this is scored
prospectively via 2.2, or on the weeks of data that accumulate before the
next review.

**Effort.** 2.3a two days (fetch is 6 locations × 3 variables × 7 vintages ×
~900 days, well within Open-Meteo's free tier if cached); 2.3b one day; 2.3c
three days; 2.3d one day.

### 2.4 Cross-border capacity and coupling state (old item 0, refined)

Carried over as the route to the regime-break class of error (the 2026-08-17
week). What this review adds is two cheaper, fresher observables and a
sharper statement of the leak risk.

**Fresher observables.**
- **Nordpool `DayAheadFlow`** gives tomorrow's *scheduled* flow per border at
  run time. On 2026-09-08 it shows SE3 → SE4 ~1.9 GW, DK2 432 MW export, PL
  100, LT 700, **GER 0/0** — the Baltic Cable outage, visible without A78. As a
  feature it is frozen for D+2..D+7 like every lag, but it is *one day fresher
  than any lag* and it is a physical quantity in MW, which round 18 did not
  reject (round 18 rejected price levels). Encode as `net_export_south_d1`,
  `se3_import_d1`, `de_link_open_d1` (0/1). History: query per date — check
  how far back the endpoint answers before planning on it; A11 physical flows
  from the existing cache are the fallback and are already on disk for 5 years.
- **Energinet `Transmissionlines`** publishes hourly import/export *capacity*
  for DK2–SE4 with the day-ahead result. It is the one SE4 border with a
  capacity time series that does not depend on A61 (empty for Nordic borders)
  or on A78 revision history. Verify depth first (Appendix A).

**The leak question, resolved by 2.14.** ENTSO-E A78 returns only the current
record, but Nord Pool's UMM API keeps every version of every outage message
with its publication time, so a backtest can use exactly the end date that was
visible on each run date. The Baltic Cable record shows why that matters: the
trip was posted 2026-06-21 with end 06-29 (v1), moved to 08-31 on 06-23 (v2),
to 09-18 on 08-17 (v3) and to 09-14 on 09-11 (v4). On 2026-08-16 the knowable
answer was "back on 08-31", not the 09-18 that A78 shows today. Build the
outage features from UMM versions as-of the run date (2.14); keep the
conservative A78 encoding ("active now and already started at run time") only
as the fallback for anything UMM does not carry. Realised and scheduled flows
are by construction what was knowable.

**Primary arm, pre-registered:** the interaction `export_headroom ×
low_wind`, where `export_headroom` = available southbound capacity (A78-derived
or Energinet + A11 nominal) − scheduled export, and `low_wind` is
`mean_wind_stockholm` below its 10th percentile. That is the mechanism the
post-mortem correction identified: crippled export capacity is the background
condition, northern calm is the trigger. Score the 2026-08-17 week separately
(`experiments/run_round19_spike.py`) as well as the four clusters.

**Effort.** 2 days for the observables and freeze-list wiring; the A78
snapshotting is an hour.

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
(README), so the slot targets must not be wired in while a 2.1 candidate is
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
singles, README), then split if the block wins.

**Effort.** 1 day for the fetches, half a day per A/B.

### 2.8 Neighbour-zone holiday calendar

**Mechanism.** `is_workday` is a top-5 feature and is Swedish. SE4's trough is
now 0.89-correlated with DK2's and 0.73 with DE's (1.2), and DE/DK holidays
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
it lacked: 1.2 documents a monotone drift in the trough's *composition* (slot
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
- **Docs.** README "Local Weather" says 10 m wind; the fetch is 100 m. The
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

**Mechanism.** Section 1.4. The same mechanism `mean_wind_stockholm` already
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
than pooled (1.4 shows the 2026 gain above the pooled one), which is the
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
things production could not. Nord Pool's UMM API is the upstream of those
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
   the README table on an honest weather axis. Build the per-window feature
   builder here and reuse it for 2.14.
7. **2.5 hurdle label**, **2.8 holidays**, **2.9 time-decay**, **2.13
   precipitation** — short A/Bs that share one long snapshot; run as one batch.
8. **2.14 UMM source**, then **2.4 cross-border** and **2.7 week-ahead
   documents** — the source work; 2.14 needs no token, the ENTSO-E parts do,
   so Peter runs those fetches.
9. **2.10 coherence**, **2.11 horizon** — when convenient.

Fetch one fresh long snapshot (`python ab_test.py fetch --days 1825`) before
step 3 so the whole batch shares a grid; the current long caches are 5d-tail
and predate the top-up (README "Snapshot generations").

---

## Appendix A — Source probes, 2026-09-08

All from this machine without credentials; ENTSO-E was not probed (token not
in the session).

**Open-Meteo Previous Runs**
`https://previous-runs-api.open-meteo.com/v1/forecast?latitude=55.6&longitude=13.0&hourly=wind_speed_100m_previous_day1,...&start_date=...&models=icon_seamless,ecmwf_ifs025,gfs_seamless`
Variables `wind_speed_100m`, `temperature_2m`, `shortwave_radiation` with
`_previous_day1..7` all return values. First month with data: **2024-03** for
`icon_seamless` and `gfs_seamless`, **2024-04** for `ecmwf_ifs025`; 2024-02 and
earlier return nulls for all three. Units default to km/h — pass
`wind_speed_unit=ms`.

**Open-Meteo forecast / ensemble**
`models=ecmwf_ifs025,icon_seamless,gfs_seamless,meteofrance_seamless` returns
one column per model; `forecast_days=16` returns 384 hours; the Ensemble API
(`ensemble-api.open-meteo.com/v1/ensemble`, `models=ecmwf_ifs025`) returns
`wind_speed_100m` plus `_member01.._member50` for 8 days.

**Energinet**
`https://api.energidataservice.dk/dataset/Forecasts_Hour` — PriceArea DK1/DK2,
ForecastType {Offshore Wind, Onshore Wind, Solar}, columns
`ForecastDayAhead / Forecast5Hour / Forecast1Hour / ForecastCurrent`,
earliest row 2019-10-31, latest HourUTC = tomorrow 23:00 (day-ahead only).
`.../dataset/Transmissionlines` — columns `PriceArea, ConnectedArea,
ImportCapacity, ExportCapacity, ScheduledExchangeDayAhead,
ScheduledExchangeIntraday, PhysicalExchangeNonvalidated,
PhysicalExchangeSettlement, CongestionIncomeEUR`. The DK2–SE4 filter hit the
rate limit (429, "try again in 110 seconds") — verify depth and whether
capacity is present for the *next* day with a single filtered call. Filter
syntax: `filter={"PriceArea":["DK2"],"ConnectedArea":["SE4"]}` (URL-encoded).

**Nordpool**
`https://dataportal-api.nordpoolgroup.com/api/DayAheadFlow?date=2026-09-08&deliveryArea=SE4`
returns 15-min `byConnectionArea` import/export/netPosition for SE3, SE3 (SWL),
DK2, PL, GER, LT, `status: Final`, updated 11:04 UTC the day before.
`DayAheadCapacities` with the same parameters returns 204 (empty) — the
endpoint exists but wants other parameters; not resolved.

**ENTSO-E cross-border cache** (`ab_cache/crossborder/2026-08-23`, 5 years):
302k A11 flow rows, 0 A61 rows (expected — explicit-auction document), 2478
A78 events (DE_LU 405 planned / 80 forced, DK2 304/33, LT 199/41, PL 688/99,
SE3 568/61). Planned-outage median duration 0.4 d (DE_LU, PL) to 3.7 d (SE3);
`available_mw` is populated (`nominal_mw` is not). Daily SE4 southbound net
export in 2026: 1856 MW (Jan) falling to 576 MW (Aug).

**Nord Pool UMM** (`https://ummapi.nordpoolgroup.com/messages`, probed
2026-09-11, no auth, `curl` works while Python `urllib` gets 403 — set a
browser-like `User-Agent`). Parameters that worked: `messageTypes` (**1
production, 2 consumption, 3 transmission** — a first probe misread 2 as
transmission and found nothing; use 3), `areas=<EIC>` (works for all three
types; SE4 transmission Jun–Dec 2026 = 176 messages),
`eventStartDate`/`eventStopDate`, `publicationStartDate`/`publicationStopDate`,
`limit` (≤ 100 returned per page), `skip`. Response
`{"items": [...], "total": N}`; each item has `messageId`, `version`,
`publicationDate`, `unavailabilityType` (1 unplanned / 2 planned),
`unavailabilityReason`, and `productionUnits[]` / `transmissionUnits[]` /
`consumptionUnits[]` with `installedCapacity` and `timePeriods[]` of
`availableCapacity`, `eventStart`, `eventStop`. SE3 nuclear example rows:
Ringhals 4 (1134 MW) available 0 from 2026-09-01, Forsmark 1 (1098 MW) 0 from
2026-09-06, Ringhals 3 (1081 MW) 0 to 2026-10-31 — all with version numbers
2–9. Total messages 114 758. Version history: `GET
/messages/{messageId}/{version}` returns each earlier version (HTTP 200) —
Baltic Cable message `e911027b-e42e-460b-b860-705a07ab5db7` v1–v4 as quoted in
2.14. Depth: SE4 transmission messages total 67 for March 2016, 35 for March
2019, 50 for March 2022 (one-month event windows; full-year windows time out
at 60 s — page by month). SE4-related transmission unit names seen in the
feed: `PL → SE4`, `SE4 → PL`, `SE3 → SE4`, `SE2 → SE3`, `FI → SE3`,
`SE4 → DK2`, `DK2 → SE4`, `LT → SE4`, `DE-LU → SE4`, `SE4 → LT`,
`SE4 → DE-LU`, `SE4 → SE3`, `SE3 → DK1`, `SE3 → NO1`.

**Location screen** (`scratchpad/screen_locations.py`, Open-Meteo archive
with `wind_speed_unit=ms`): base linear R² over the existing six sites +
radiation + temperature + calendar is 0.481 (cheap2h), 0.477 (min), 0.519
(avg). Full table in 1.4; per-candidate daily frames were saved as
`cand_<name>.pkl` next to the script during the session and are
re-fetchable in seconds.

**Naive references, last 364 days, cheap2h MAE** (for context when reading the
15.04 headline): persistence lag1 18.2, lag7 26.7, frozen 7-day anchor 24.5,
monthly climatology 23.9. Weekday means: Mon 40, Tue 49, Wed 47, Thu 38, Fri
34, Sat 23, Sun 20 EUR/MWh.

## Appendix B — Recipes

Reproduce 1.1 (needs a snapshot, no token):

```python
from ab.snapshot import load_snapshot
import features as F
inputs, meta = load_snapshot("2026-09-05")
base = F.build_training_data(**inputs)
print(base["residual_load"].corr(base["mean_temp"]))        # -0.9997
orig = F._wind_power_curve
F._wind_power_curve = lambda v, rated_speed=13.0: orig(v / 3.6, rated_speed)
fixed = F.build_training_data(**inputs)
F._wind_power_curve = orig
print(fixed["wind_night"].corr(fixed["price_cheap2h"]))    # -0.218 vs -0.116
```

Reproduce 1.2 from `ab_cache/2026-09-05/prices_hourly.pkl`: resample to hourly
means, Europe/Stockholm dates, `idxmin` per day for the slot of the minimum,
`(price_min < 0).mean()` per year; DK2/DE from `market_prices_hourly.pkl` with
the same `nsmallest(2).mean()` per day.
