# SE4 Electricity Price Predictor

A machine learning pipeline that predicts day-ahead electricity prices for the Swedish SE4 bidding zone (Malmö/southern Sweden) and pushes the predictions to Home Assistant as a sensor.

## Purpose

The predictions are used in Home Assistant to make smart decisions about **schedulable energy consumption** — most importantly EV charging. If prices are expected to be lower in the coming days, charging can be deferred. If prices are expected to rise, the car charges now. The sensor exposes daily min/avg/cheap2h prices up to 8–10 days ahead, giving automation rules enough lead time to act.

The pipeline runs via GitHub Actions, triggered from Node-RED/Home Assistant, and updates a Home Assistant sensor with predictions in SEK/kWh, including a configurable distribution cost addon.

## Architecture

```
Data sources (API)
    │
    ├── ENTSO-E      → SE4/DE/DK2 day-ahead prices, SE3 nuclear outages, Sweden reservoir (A72)
    ├── Open-Meteo   → Historical weather archive + 8-day forecast (SE4/Malmö + 5 regional locations)
    ├── Yahoo Finance → TTF natural gas futures, EU ETS carbon allowance (EUA)
    ├── NVE          → Norwegian hydro reservoir levels + 20-year seasonal median
    └── Nordpool     → Published prices (to skip days already known)
         │
         ▼
Feature engineering (features.py)
    → 47 shared features (+1 cheap2h-specific lag), daily resolution, 3-year training window
         │
         ▼
Model training (model.py)
    → 4 separate XGBoost regressors: price_min, price_avg, price_max, price_cheap2h
    → price_cheap2h additionally gets a negative-price hurdle: an XGBoost
      classifier's P(price_min < 0) feeds the regressor as an extra feature
    → Walk-forward validation: 52 windows × 7 days (evaluate.py)
         │
         ▼
Inference → EUR/MWh → SEK/kWh → Home Assistant sensor (ha_client.py)
```

## Current MAE Baseline

Evaluation uses **52-window (1-year) walk-forward validation** on 3 years of training data. Each target is reported **individually** — min, avg, max and cheap2h have different physical drivers, so a blended metric would hide the per-target movement that matters when tuning. `cheap2h` and `min` are the numbers to watch for scheduling; `max` is not a priority.

The evaluation is **horizon-honest**: each test window's price/market/fuel/reservoir lags are frozen to their last-known value, exactly as `build_forecast_features` does in production for all 8 forecast days. A validation that instead fed each test day its *true* previous-day price as `lag1` — knowledge the live model only has for day+1 — would make days 2–8 look far more accurate than they are. This baseline is the accuracy Home Assistant actually receives.

Baseline (52-window/1-year eval, volatility leak fix; wiggles ~±0.5 run-to-run as the window rolls). **The 1-year eval is not comparable to the previous 35-window/8-month numbers** — the mean is lower mainly because it now includes the easy low-price summer 2025 weeks (per-window MAE 3–9), not because accuracy improved:

| Target  | MAE (EUR/MWh) | Std |
|---------|--------------|------|
| min     | 16.11        | ±7.15 |
| avg     | 17.83        | ±7.63 |
| max     | 36.15        | ±19.13 |
| cheap2h | 15.93        | ±7.29 |

**Intraday trough features (2026-07):** the daily min/cheap2h is set at a specific intraday trough (overnight wind or midday solar) that the daily-mean `residual_load` diluted. Exposing the trough directly — chiefly `residual_load_min` (daily minimum of hourly residual load) — cut cheap2h ~0.66 EUR/MWh and held min, with `residual_load_min` landing as a top-4 feature in all three priority models. See the [Intraday Trough Features](#intraday-trough-features) section.

**Price-lag anchor freshening (2026-07):** the SE4 price lags — the dominant min/cheap2h features — are now frozen at the freshest *known* ENTSO-E price (`se4_prices_daily`), not the training frame's last row. The training frame ends ~`WEATHER_ARCHIVE_LAG_DAYS` behind because it inner-joins prices with the lagging weather archive, so production was previously anchoring the price lags ~5 days stale (the DE/DK2 lags were already fresh — this closes the same gap for SE4's own lags). The walk-forward reports an **anchor-staleness sensitivity** (fresh d0 vs stale d5 ≈ old pipeline). Measured saving from freshening: **~1.1 EUR/MWh (min/cheap2h), ~2.3 (avg)** — real and free, but modest, because yesterday's min and six-days-ago min are similar.

**Negative-price hurdle (2026-07-24, cheap2h only):** an XGBoost classifier predicts P(tomorrow's `price_min` < 0 EUR/MWh) — a distinct physical regime (renewable oversupply + low/weekend demand) a plain regressor otherwise has to infer implicitly from the same weather/calendar features. Its out-of-fold probability feeds into the `cheap2h` regressor as an extra feature (`neg_price_proba`, now the #1 feature for cheap2h at ~0.25 importance). Validated via `ab_test.py` on a real cached snapshot, shifts 0–5: **REAL, all 6 shifts improved, mean −0.44 EUR/MWh** — the largest, cleanest single-run win recorded for the top-priority target; local reproduction via `evaluate.walk_forward_validate` matched the A/B exactly (14.74 vs the same run's min/avg/max, which were unaffected). The table above predates this change; expect cheap2h to drop roughly this much once confirmed on the next Actions run. `min` showed the same direction on 5 of 6 shifts but didn't clear the strict sign-consistency bar (one near-zero flip) — **not yet productionized for `min`**, see [MIN_HURDLE_FOLLOWUP.md](MIN_HURDLE_FOLLOWUP.md) for the re-test plan once more training data accumulates.

**Per-horizon MAE (`mae_by_horizon`) is weekday-confounded — do not read it as pure horizon decay.** The walk-forward steps by 7 days with 7-day windows, so horizon ≡ weekday (d+1 is always the same weekday as the window start). The curve mixes horizon and day-of-week and is non-monotonic. The clean measure of horizon/lag-staleness cost is the anchor-staleness sensitivity above (~1 EUR), **not** the per-horizon spread. This is why horizon-aware modeling (a `forecast_horizon` feature / per-horizon models) was evaluated and **shelved**: the true stale-lag headroom is ~1 EUR, and the bulk of the ~17 EUR error is regime-driven (winter cold-snap volatility, spring solar/negative-price ramp), not horizon-driven.

Feature importance is reported for **min, avg and cheap2h models only** — including max would dilute the signal for what actually matters for scheduling decisions.

## Target Definition

The SDAC day-ahead auction switched from hourly to 15-minute MTU on **delivery day 2025-10-01**. ENTSO-E returns 24 points/day before that date and 96 points/day after. This matters because a min over 96 quarter-hours is systematically lower than a min over 24 hours — mixing both in one training window teaches the model the wrong level for exactly the period it is evaluated on.

**Current definitions (since 2026-07): all price targets are computed on hourly means.** 15-minute data is resampled to hourly before daily aggregation (`aggregate_prices_daily`), so every target has one consistent definition across the whole 3-year window:

- `price_min` / `price_max` — cheapest / most expensive hour of the day
- `price_avg` — daily mean (numerically unaffected by resampling)
- `price_cheap2h` — mean of the day's **two cheapest hours, not necessarily adjacent**. This is the decision-relevant target for EV charging: a ~2h charging session picks the cheapest points of the day wherever they are. It is also statistically smoother than the pointwise min (average of 2 values instead of an extremum).

**Backtracking plan — when to switch to native 15-minute targets:**

| Date | 15-min share of 3-year window | Action |
|------|------------------------------|--------|
| 2025-10-01 | 0% → growing | 15-min MTU go-live (delivery day) |
| ~2027-10 | ~67% (2 years) | Run the experiment: walk-forward compare (a) hourly-harmonized targets on the full window vs (b) native 15-min targets (`price_cheap2h` = mean of 8 cheapest quarters) on a 15-min-only window (`TRAINING_DAYS ≈ 730`). Switch only if (b) wins on min/cheap2h MAE. |
| 2028-10-01 | 100% | Hourly era has left the window — mixing ends by itself; native 15-min targets become a pure definition choice with no data-quality downside. |

Note: since charging sessions span hours, the *pointwise* 15-min min mostly adds noise and may never be the right target. The natural evolution is `cheap2h` computed on the 15-min curve (8 cheapest quarters), which the charging schedule can exploit since it can use individual non-contiguous 15-min MTUs.

## Data Sources

| Source | What it provides | Auth |
|--------|-----------------|------|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | SE4/DE/DK2 hourly day-ahead prices, SE3 nuclear outages (A77), Sweden reservoir fill (A72) | `ENTSO_E_TOKEN` env var |
| [Open-Meteo](https://open-meteo.com) | Hourly weather archive + 8-day forecast for SE4 (Malmö) and 5 regional locations | None |
| [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | TTF natural gas futures (`TTF=F`), EU ETS carbon allowances (`CO2.L`) | None |
| [NVE](https://biapi.nve.no/magasinstatistikk) | Norwegian hydro reservoir fill levels + 20-year min/max/median by week | None |
| [Nordpool](https://www.nordpoolgroup.com) | Published SE4 prices (used to exclude already-known days from predictions) | None |

## Features (51 shared + 1 cheap2h-specific)

(cheap2h's model also gets a `neg_price_proba` input computed at fit/serve time by
the negative-price hurdle classifier — see [Negative-Price Hurdle](#negative-price-hurdle-cheap2h-only).
It's not a column in `FEATURE_COLUMNS`/`CHEAP2H_FEATURE_COLUMNS` below, since it's
model-internal rather than fetched/engineered by `features.py`.)

### Local Weather (SE4/Malmö)
- `mean_temp`, `min_temp`, `max_temp` — daily temperature aggregates
- `mean_wind`, `max_wind` — daily wind speed (10m)
- `mean_radiation`, `max_radiation` — global horizontal irradiance (GHI W/m²)

### Regional Wind & Solar (5 locations, 100m hub-height)
Captures wind and solar generation in coupled markets that flow into SE4.
- **Karlskrona** — Baltic offshore wind patterns
- **DK2** — directly coupled to SE4 via Øresund (~1700 MW)
- **Stockholm** — SE3 load centre (also used for temperature gradient)
- **DK1** — western Denmark/Jutland
- **DE North** — northern Germany, Baltic Cable (~600 MW)

### Market Coupling
- `price_de_lag1`, `price_dk2_lag1` — previous day's prices in neighbouring zones
- Only lag-1 is valid: day-ahead auction clears all zones simultaneously

### SE4 Own Price Lags (autoregressive)
- `price_se4_avg_lag1` — strongest feature for the **avg** model (~0.24 importance)
- `price_se4_avg_lag2`, `price_se4_avg_lag7` — momentum and weekly seasonality
- `price_se4_min_lag1` — yesterday's min; **strongest feature for both min and cheap2h (~0.23–0.25)** — a better predictor of today's cheap2h than yesterday's cheap2h is
- `price_se4_max_lag1` — yesterday's max
- `price_se4_cheap2h_lag1` — yesterday's cheap2h (**cheap2h model only** via `CHEAP2H_FEATURE_COLUMNS`; kept out of min/avg/max to avoid perturbing their baseline with a correlated lag)
- `price_momentum` — lag1 minus lag2 (rising vs falling trend)
- `price_volatility_7d` — rolling 7-day std (market regime stability)

### Residual Load
Engineered composite feature: demand proxy minus weighted wind/solar supply.
- Demand proxy: `15 - mean_temp` (heating-based)
- Wind: cubic power curve `(v/13)³` applied per location, weighted by interconnection capacity to SE4. Weights: SE4/Malmö 1.0, **Karlskrona 0.5** (also SE4 — southern Baltic offshore wind), DK2 0.4, DK1 0.2, DE-north 0.3 (normaliser 2.4). Karlskrona was previously fetched but missing from the blend.
- Solar: `radiation / 500` per location, same weights minus Karlskrona (normaliser 1.9)
- Also exposed as `residual_load_lag1` for momentum

### Intraday Trough Features
The daily price minimum (and `cheap2h`) is set at a specific intraday trough — the hour of highest renewable supply / lowest net demand — which the daily-*mean* `residual_load` dilutes. These expose that trough directly from the hourly weather forecast, using the same power curve and interconnection weights (`aggregate_intraday_features`). All four are per-day forecastable (not lagged, not frozen), so they help at every horizon.
- `residual_load_min` — daily **minimum** of hourly residual load: the physical driver of the daily price min
- `residual_load_range` — daily max − min of hourly residual load: proxy for the intraday min↔peak spread (drives the min–avg gap)
- `wind_night` — interconnection-weighted wind power, 00–06 local mean (overnight wind-driven trough)
- `radiation_midday` — interconnection-weighted solar supply, 10–16 local mean (midday negative-price window)

### Heating Degree Days
- `hdd_linear` — `max(0, 17 - temp)`: standard Nordic HDD, a key heating-demand signal (`residual_load` and `max_temp` usually rank higher)
- `hdd_cold_boost` — quadratic term below 0°C: captures non-linear demand surge during extreme cold

### Temperature Gradient (SE3↔SE4)
- `temp_gradient_se3_se4` — Stockholm minus Malmö temperature
- Negative = SE3 colder → transmission stress → SE4 prices diverge upward

### Fuel & Carbon Costs
- `ttf_price_lag1`, `ttf_rolling_7d` — Dutch TTF natural gas futures (leading indicator for German marginal cost)
- `co2_price_lag1`, `co2_rolling_7d` — EU ETS carbon allowance price (EUR/tonne, via `CO2.L` on Yahoo Finance)
- `gas_marginal_cost` — synthetic feature: `TTF + 0.35 × CO₂`, approximating CCGT short-run marginal cost

### Hydro Reservoir Levels (weekly, forward-filled)
- `reservoir_norway_deviation` — Norway fill % minus 20-year median for same ISO week. Seasonal anomaly: low reservoirs in autumn = structural scarcity
- `reservoir_sweden_gwh` — Sweden stored energy in GWh (ENTSO-E A72)
- `reservoir_sweden_change` — week-over-week change (filling vs draining trend)

### Nuclear Outages (SE3)
- `nuclear_outage_se3` — count of simultaneous SE3 nuclear outages per day
- Planned maintenance (A53) is published months ahead → usable for multi-day forecasts
- Forced outages (A54) historical only → improves training accuracy

### Calendar
- `is_workday` — 1 for normal workdays, 0 for weekends, Swedish public holidays, and bridge days (Friday after Ascension). Demand drops 20–40% on non-workdays. (**3rd–5th most important feature**)
- `month_sin/cos`, `day_of_year_sin/cos`, `dow_sin/cos` — cyclic encoding to avoid ordinal discontinuities

### Forecast Uncertainty Proxies
- `wind_variability`, `radiation_variability` — rolling 7-day std of weather variables. High variability = less reliable forecasts = higher price spike risk

## Model

Four separate XGBoost regressors (min/avg/max/cheap2h targets):

```python
XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.5,
)
```

Separate models per target because the targets have different physical drivers: min occurs at renewable oversupply moments, avg smooths out noise, max occurs at peak demand/scarcity, and cheap2h tracks the depth of the daily price trough. All models share `FEATURE_COLUMNS`; cheap2h additionally gets `price_se4_cheap2h_lag1`.

### Negative-Price Hurdle (cheap2h only)

`model.HURDLE_TARGETS = {"cheap2h"}` routes the cheap2h target through
`HurdleAugmentedModel` instead of a plain `XGBRegressor`: a shallow XGBoost
classifier (`max_depth=3`, 200 trees) predicts P(`price_min` < 0 EUR/MWh tomorrow —
`model.NEG_PRICE_THRESHOLD`), and that probability is appended as an extra feature
(`neg_price_proba`) before the regressor runs. Negative-price days are a distinct
physical regime — renewable oversupply plus low/weekend demand — that the regressor
otherwise has to infer implicitly from the same weather/calendar inputs; making the
regime signal explicit was the bet, and it paid off (`neg_price_proba` is now the
**#1 feature for cheap2h**, ~0.25 importance).

**Leak-safety is the subtle part.** The regressor trains on **out-of-fold** (5-fold)
classifier probabilities (`model._fit_hurdle_model`), not the in-sample predictions
of a classifier fit on the whole training slice — an in-sample classifier has seen
each row's own label, so its probabilities would be near-perfect and would leak an
optimistic signal that production can never actually get (at serving time nobody
knows tomorrow's true label). The classifier stored inside `HurdleAugmentedModel`
for serving is fit on the *full* training slice and has genuinely never seen the row
it predicts for, so no leakage at inference time either — the same asymmetry
(OOF-for-training, full-fit-for-serving) walk-forward validation and production
inference already rely on for every other feature.

Validated 2026-07-24 via `ab_test.py` — cheap2h REAL (mean −0.44 EUR/MWh across 6
simulated shifts, see "Current MAE Baseline"). `min` showed the same direction but
didn't clear the strict A/B bar, so it stays on a plain regressor for now — see
[MIN_HURDLE_FOLLOWUP.md](MIN_HURDLE_FOLLOWUP.md).

## A/B Backtest Flow

`ab_test.py` is the tool for evaluating any candidate feature or model change before
committing it. It exists because MAE effects on the priority targets are small
(often < 0.3 EUR/MWh) and **period-dependent**: the same change can look like an
improvement on one day's data and a regression on the next. Historically that forced
re-running an experiment across several real calendar days to tell signal from noise
— one change every few days. The A/B flow simulates those different run-days from a
single data snapshot, so a verdict takes one sitting.

### How it works

**Key insight — one shift axis.** `walk_forward_validate` derives its 52-window grid
backwards from the *end* of the data (`min_train = n - iterations*step`). So
truncating the last `s` rows of the merged frame reproduces the eval exactly as it
would have run `s` days earlier — window placement **and** the training tail both
move together, the same way a real earlier run would differ. One knob (`shift`)
captures the whole between-day axis; running shifts `0..5` gives six simulated
run-days from one fetch.

**The pieces:**
- `fetch_data.py::fetch_training_inputs(today)` — the training-side fetch, shared
  with `predict.py`. `ab_test.py fetch` calls it and caches the result.
- `ab/snapshot.py` — saves/loads a fetched-inputs snapshot under `ab_cache/<date>/`
  (pickled, gitignored, local-only; pickle rather than parquet because pyarrow is
  not a dependency). Snapshots accumulate — one directory per `fetch` day, a few MB
  each; nothing prunes them automatically.
- `ab/harness.py` — `run_walk_forward()`, a copy of the `evaluate.py` loop
  parameterized by a variant, plus `apply_shift()` (tail truncation). It deliberately
  does **not** import or modify `evaluate.walk_forward_validate`, so the headline eval
  and every recorded baseline stay untouched; the two are pinned together by an
  acceptance test (shift-0 `BASELINE` must reproduce `walk_forward_validate`'s
  per-window MAE exactly).
- `ab/variants.py` — a `Variant` dataclass and the two instances the harness runs:
  `BASELINE` (mirrors production) and `CANDIDATE` (the only thing you edit per
  experiment).
- `ab/verdict.py` — runs both variants across the shifts and classifies each target.

**The verdict rule** (`ab/verdict.py::classify`), applied per target on the list of
per-shift deltas (`candidate_MAE − baseline_MAE`, negative = candidate better):
- **`NO_CHANGE`** — every delta is exactly 0 (candidate ≡ baseline).
- **`NOISE`** — the delta's sign flips across shifts. The effect is smaller than the
  between-day variability → reject.
- **`REAL`** — sign is consistent across all shifts **and** `|mean delta| ≥` the
  between-shift spread (`max − min`) → adopt.
- **`BORDERLINE`** — sign-consistent but the effect is smaller than the spread →
  replay on a **different real-day snapshot** before deciding.

This never touches the headline eval (`evaluate.walk_forward_validate`, still
step=7 / 52 windows) or production (`predict.py` always fetches fresh, never reads
`ab_cache/`).

### How to run an experiment (agent playbook)

Follow this whenever testing a feature or model change from the improvement plan:

1. **Get a snapshot.** `python ab_test.py fetch` (needs `ENTSO_E_TOKEN`; runnable
   locally from VS Code). Reuse an existing one with `python ab_test.py list` if a
   recent snapshot is already cached — a snapshot a few days old is fine for
   iterating.
2. **Express the change as `CANDIDATE`** in `ab/variants.py`. This is the only file
   you edit, usually a handful of lines. A `Variant` has:
   - `transform(data) -> data` — adds or modifies **columns** on the merged daily
     frame (e.g. scale a feature, add a derived feature). **Must not add, drop, or
     reorder rows** — `BASELINE` and `CANDIDATE` are compared on identical slices, and
     the harness asserts the row count and `date` column are unchanged.
   - `fit_fn(train_slice) -> models` — override to change training (objective, sample
     weights, hyperparameters). Default: `model._fit_models` (production).
   - `targets` — a `{name: (target_col, feature_cols)}` dict; override to add a new
     feature column to the models' feature lists (a `transform` that only *creates* a
     column has no effect until the column is added to `feature_cols` here).
   - `frozen_features` / `frozen_rolling` — override only if the change adds a
     lag-type feature that must be frozen in the horizon-honest eval (default `None` =
     production lists).
3. **Run it.** `python ab_test.py run` (add `--snapshot YYYY-MM-DD` to pick a
   specific one, `--shifts 0-5` to change the grid). Read the per-target verdict
   table. **Priority order is cheap2h first, then min**; `avg` is welcome, `max` is
   not a priority.
4. **Decide by the verdict**, weighing cheap2h/min and checking std isn't inflated:
   - `REAL` → adopt: implement the change properly in the real modules, then revert
     `CANDIDATE` to the no-op `Variant(name="candidate")`.
   - `NOISE` / `NO_CHANGE` → reject: record it in the **Features Tested and Rejected**
     table below with the numbers, and revert `CANDIDATE`.
   - `BORDERLINE` → fetch a snapshot on a later real day (`ab_test.py fetch` again
     after a day or two) and `python ab_test.py run --snapshot <older>` vs the new
     one; adopt only if the sign agrees. The cross-snapshot replay is the *only* case
     that needs multiple caches — it covers real data revisions (weather backfill,
     price corrections) that tail-truncation alone can't simulate.
5. **Confirm on the next real Actions run.** An `ab_test.py` verdict is for fast
   pre-commit iteration; an adopted change is still confirmed against the production
   run's `Walk-forward validation` output before it's considered done.

**Worked example** — testing whether scaling a radiation feature by an installed-solar
index helps (as a `transform`; revert after):

```python
def _scale_radiation(data):
    out = data.copy()
    years = (pd.to_datetime(out["date"]) - pd.Timestamp("2023-01-01")).dt.days / 365.0
    out["radiation_midday"] = out["radiation_midday"] * (1 + years / 3)
    return out

CANDIDATE = Variant(name="solar_scaled", transform=_scale_radiation)
```

Because it scales an existing feature in place, no `targets` override is needed. Run
`python ab_test.py run` and read the cheap2h row.

## Features Tested and Rejected

Keep this list updated — it prevents re-testing things that didn't work.

| Feature | Reason rejected |
|---------|----------------|
| `price_se4_min_lag7` | Importance 0.009, added variance to min/avg MAE. With only 3 years training data, insufficient weekly-min samples. |
| `reservoir_norway_fill_pct` (raw) | Redundant with `reservoir_norway_deviation` which is the more informative signal. Removed to reduce noise. |
| `reservoir_norway_change` | Low importance (0.009), already captured implicitly by `reservoir_sweden_change`. Removed to reduce noise. |
| 5 reservoir features (initial) | Trimmed to 3 after observing that raw fill % and Norway change added noise without improving min/avg MAE. |
| `^ICEEUA` (ICE EUA index) | Returns empty via yfinance. Use `CO2.L` instead. |
| `ECF=F` (NYMEX EUA futures) | Not reliably available on Yahoo Finance. |
| Svenska Kraftnät hydro API | No public REST API for reservoir data. Use ENTSO-E A72 instead. |
| `objective="reg:absoluteerror"` (vs default `reg:squarederror`) | Drift-free same-slice A/B: worse on both priority targets — cheap2h +0.61, min +0.14, avg +0.78 EUR/MWh; only max (non-priority) improved (2026-07). Squared error stays. |
| Seed ensembling (3-seed avg per target) | No improvement, 3× the train compute. The reported MAE std is between-window regime dispersion, which averaging seeds cannot reduce — it slightly *raised* std on min/avg/cheap2h (2026-07). |
| `min ≤ cheap2h ≤ avg ≤ max` coherence clamp (avg-anchored) | Drift-free same-slice A/B (8-month and 1-year): only ever clipped cheap2h (min/avg/max deltas exactly 0.00; fired 99–128×, all on cheap2h) and made cheap2h *worse* (+0.05 to +0.11). When predicted cheap2h > predicted avg the incoherence means avg was too low, not cheap2h too high, so clipping toward avg moves it away from truth; it also erased the volatility-leak-fix's cheap2h gain (2026-07). Zero upside, reverted. |
| Time-decay sample weights on `min`/`cheap2h`/`max` (`weight = 0.5 ** (age_days / half_life)`) | Drift-free same-slice A/B, swept over **three** runs (2026-07-22, stride-3, 121 overlapping windows). On the priority targets the effect is **noise**: min's delta at hl=500 went +0.22 → −0.22 → −0.01 across the three runs and cheap2h's collapsed to ~0 — i.e. the effect (~±0.2) is smaller than the run-to-run variability (~0.4 swing). `max` was likewise noisy/mixed and is not a priority. A single favorable stride-3 run (−0.22/−0.20) did *not* replicate — the lesson being that a small effect needs replication across days, not one A/B. **`avg` is the exception and IS kept** with decay (half-life 500): it improved at *every* half-life in *all three* runs (~−0.35 EUR/MWh at hl=500), so `HALF_LIFE_DAYS` weights only the (independent) avg model and leaves min/cheap2h/max on uniform weights. |
| Solar-capacity scaling of radiation (placeholder linear index: `mean_radiation` / `radiation_midday` × `1 + (1/3)·years_since_2023`, the "SE4 PV roughly doubled over 3 years" prior) | Drift-free same-slice A/B with internal day-alignment phases, run on two ~1-delivery-day-apart data samples (2026-07-22 and 07-23): no replicable gain on the priority targets. cheap2h was consistent *within* each run but **flipped sign between them** — all three phases −0.01…−0.13 (better) in the morning, all three +0.09…+0.15 (worse) in the evening — the classic period-noise signature (effect ~0.1 EUR/MWh < day-to-day swing). min was mixed-sign in both runs; avg looked good in the morning (−0.20, all phases) but evaporated in the evening (−0.03, mixed); max non-priority and inconsistent. Likely redundant with `price_se4_min_lag1` / `residual_load_min`, which already absorb the buildout. Real ENTSO-E A68 per-zone capacity *could* be re-tested, but the first-order interaction shows nothing so the prior is low (2026-07). |
| Spread-decomposition for min/cheap2h (predict `avg` + `spread_down = avg − min` resp. `avg − cheap2h`, reconstruct `min`/`cheap2h = avg_pred − spread_pred`) | `ab_test.py` A/B on the 2026-07-24 snapshot across shifts 0–5: `avg`/`max` deltas were exactly 0.00 at every shift (fit_fn self-check — those two are untouched, confirming the harness wiring), but min/cheap2h were classified **NOISE** (sign flip across shifts) with a **positive (worse) mean delta** — min +0.50 (deltas +0.77/+0.57/+0.02/+0.78/−0.16/+1.00), cheap2h +0.40 (+0.76/+0.49/−0.05/+0.54/−0.37/+1.03). Reconstructing from two independently-fit models (avg error + spread error) compounds error rather than cancelling it; no shift showed a clear win. Rejected — direct min/cheap2h regression stays (2026-07-24). |

## Known Limitations

- **3-year training window**: Unusual market periods (e.g. energy crisis 2021–2022) have outsized weight. Reservoir features will become more valuable as more data accumulates.
- **Daily resolution**: The model predicts daily aggregates, not 24 hourly prices. Hour-level predictions would be more actionable for EV scheduling but require significantly more feature engineering. The `cheap2h` target partially addresses this: it predicts what a ~2h charging session picking the day's cheapest hours would pay, which is the number the "charge today or wait" decision needs.
- **Forecast horizon**: All forecast days (1–8) use the same features and a single model that doesn't distinguish horizon. Horizon-aware modelling was **evaluated and shelved** (2026-07): the anchor-staleness sensitivity showed the true stale-lag cost is only ~1 EUR/MWh for min/cheap2h, so a `forecast_horizon` feature has little headroom. The scary-looking per-horizon curve is a weekday artifact of the step-7 walk-forward (see Current MAE Baseline), not real horizon decay. The price-lag anchor is kept fresh regardless, since that was a free win.
- **Weather in evaluation**: walk-forward uses archive weather as a stand-in for the forecast, so results are optimistic on the weather axis — real day+7 weather forecasts are worse than archive. This optimism grows at far horizons and is not captured by the anchor-staleness or per-horizon metrics.
- **Max prediction accuracy** (~34 EUR/MWh MAE, the highest of the four targets): Intentionally not optimized. Max prices are driven by rare spike events that are hard to predict from daily features.
- **EUR/SEK rate**: Derived daily from Nordpool vs ENTSO-E prices. If data is unavailable, the rate may be stale.

## Setup

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ENTSO_E_TOKEN` | ENTSO-E Transparency Platform API token |
| `HA_URL` | Home Assistant base URL (e.g. `https://homeassistant.local:8123`) |
| `HA_TOKEN` | Home Assistant long-lived access token |

### Installation

```bash
pip install -r requirements.txt
python predict.py
```

### GitHub Actions

The pipeline is defined in `.github/workflows/daily_predict.yml` and runs on `workflow_dispatch` — triggered via the GitHub API from a Node-RED flow in Home Assistant (no fixed schedule). Secrets are configured in the repository settings matching the environment variables above.

## Home Assistant Integration

The pipeline creates/updates `sensor.electricity_price_predictions` with attributes:
- `predictions_raw` — EUR-to-SEK converted prices, indexed by date (min/avg/max/cheap2h per day)
- `predictions_with_addon` — prices adjusted by `input_number.electricity_price_addon` (distribution costs etc.) with a 5% markup
- `mae_min` / `mae_avg` / `mae_max` / `mae_cheap2h` — current per-target MAE values from horizon-honest walk-forward validation
- `mae_by_horizon` — per-target MAE broken down by forecast horizon (day+1 … day+7); shows how accuracy decays with forecast distance
- `feature_importance_min` / `feature_importance_avg` / `feature_importance_cheap2h` — top features per model (for debugging)

For charging decisions ("charge today or wait for cheaper days"), compare `cheap2h` across days — it is the expected price of a ~2h session picking the day's cheapest hours, which is what the schedule can actually achieve.

The addon value is fetched live from Home Assistant each run, so distribution cost changes take effect immediately without redeploying.
