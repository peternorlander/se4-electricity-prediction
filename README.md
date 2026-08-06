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
    → 51 engineered features, daily resolution, 3-year training window
         │
         ▼
Model training (model.py)
    → 4 separate XGBoost regressors: price_min, price_avg, price_max, price_cheap2h
    → per-target feature sets: avg/max use all 51, while the two trough targets
      (min, cheap2h) share a validated 15-feature subset (TROUGH_FEATURE_COLUMNS)
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

Current, measured on the **2026-08-04** data snapshot with the production code as
it stands (negative-price hurdle on cheap2h, `min` and `cheap2h` both on the
pruned 15-feature trough list):

| Target  | MAE (EUR/MWh) | Std |
|---------|--------------|------|
| min     | 15.70        | ±8.16 |
| avg     | 19.11        | ±8.73 |
| max     | 34.39        | ±17.34 |
| cheap2h | **15.94**    | ±7.98 |

The cheap2h row is the only one that moved when the trough list was adopted
(2026-08-05): re-measuring the *same snapshot* before and after gives
**16.85 → 15.94 (−0.91)**, while `min`, `avg` and `max` reproduced their previous
values to the last digit. That exact reproduction of three untouched targets is
what makes this particular before/after meaningful — same snapshot, same windows,
same code path, one thing changed. It is **not** the evidence for the change
(that is the A/B, −0.86 pooled over 16 measurements); it is a consistency check
on the wiring.

**Do not compare this table against an older one to judge a change.** The headline
number moves with the *evaluation period*, not just with the model. Against the
table before it (min 16.11 / avg 17.83 / max 36.15 / cheap2h 15.93), measured on
an earlier window:

| Target | Old → new | Model changed between the two? |
|--------|-----------|-------------------------------|
| `avg` | 17.83 → 19.11 (**+1.28**) | **No** — pure period effect |
| `max` | 36.15 → 34.39 (**−1.76**) | **No** — pure period effect, *opposite direction* |
| `min` | 16.11 → 15.70 (−0.41) | yes, pruned (A/B: −1.01) |
| `cheap2h` | 15.93 → 15.94 (+0.01) | yes, hurdle (A/B: −0.44) **and** prune (A/B: −0.86) |

Two models that did not change at all moved **+1.28** and **−1.76** across those
two windows — larger than either cheap2h change, and in opposite directions. The
cheap2h row is the trap in miniature: it reads as "+0.01, nothing happened" across
periods, while the two changes it actually contains are worth about −1.3 together
on same-slice measurement.

So "min improved by 0.41" understates a −1.01 change that happened to face a
period headwind. Cross-run comparison cannot separate the two.

The honest measure of any change is its **drift-free A/B delta** — same data, same
slices, one thing varied. Those are the numbers to trust:

| Change | A/B delta | Replication |
|--------|-----------|-------------|
| `min` pruned feature list | **−1.01 EUR/MWh** | 14 measurements, 4 snapshots, all negative |
| `cheap2h` pruned feature list | **−0.86 EUR/MWh** | 16 measurements, **3 evaluation periods**, all negative (after 17/17 in an earlier round) |
| Negative-price hurdle (cheap2h) | **−0.44 EUR/MWh** | 6 measurements, 1 snapshot, all negative |

See [How changes are validated](#how-changes-are-validated) for why this project
reports it that way.

**Intraday trough features (2026-07):** the daily min/cheap2h is set at a specific intraday trough (overnight wind or midday solar) that the daily-mean `residual_load` diluted. Exposing the trough directly — chiefly `residual_load_min` (daily minimum of hourly residual load) — cut cheap2h ~0.66 EUR/MWh and held min, with `residual_load_min` landing as a top-4 feature in all three priority models. See the [Intraday Trough Features](#intraday-trough-features) section.

**Price-lag anchor freshening (2026-07):** the SE4 price lags — the dominant min/cheap2h features — are now frozen at the freshest *known* ENTSO-E price (`se4_prices_daily`), not the training frame's last row. The training frame ends ~`WEATHER_ARCHIVE_LAG_DAYS` behind because it inner-joins prices with the lagging weather archive, so production was previously anchoring the price lags ~5 days stale (the DE/DK2 lags were already fresh — this closes the same gap for SE4's own lags). The walk-forward reports an **anchor-staleness sensitivity** (fresh d0 vs stale d5 ≈ old pipeline). Measured saving from freshening: **~1.1 EUR/MWh (min/cheap2h), ~2.3 (avg)** — real and free, but modest, because yesterday's min and six-days-ago min are similar.

**Negative-price hurdle (2026-07-24/25, cheap2h only):** an XGBoost classifier predicts P(tomorrow's `price_min` < 0 EUR/MWh) — a distinct physical regime (renewable oversupply + low/weekend demand) a plain regressor otherwise has to infer implicitly from the same weather/calendar features. Its out-of-fold probability feeds into the `cheap2h` regressor as an extra feature (`neg_price_proba`, now the #1 feature for cheap2h at ~0.25 importance). Validated via `ab_test.py` on a real cached snapshot, shifts 0–5: **REAL, all 6 shifts improved, mean −0.44 EUR/MWh** — the largest, cleanest single-run win recorded for the top-priority target; local reproduction via `evaluate.walk_forward_validate` matched the A/B exactly (14.74 vs the same run's min/avg/max, which were unaffected). **Confirmed on a real Actions run and committed 2026-07-25**; the table above now includes it. `min` showed the same direction on 5 of 6 shifts but didn't clear the strict sign-consistency bar (one near-zero flip) — **not productionized for `min`**, see [MIN_HURDLE_FOLLOWUP.md](MIN_HURDLE_FOLLOWUP.md) for the re-test plan once more training data accumulates.

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

## Features (51 engineered + 1 cheap2h-specific)

The catalogue below is the full engineered set (`FEATURE_COLUMNS`), used in its
entirety by the **avg** and **max** models. **Not every model uses every
feature** — see [Per-Target Feature Sets](#per-target-feature-sets):

| Model | Features used |
|-------|---------------|
| avg, max | all 51 |
| **min, cheap2h** | **a validated 15-feature subset** (`TROUGH_FEATURE_COLUMNS`) — cheap2h additionally gets `neg_price_proba` at fit/serve |

(cheap2h's model also gets a `neg_price_proba` input computed at fit/serve time by
the negative-price hurdle classifier — see [Negative-Price Hurdle](#negative-price-hurdle-cheap2h-only).
It's not a column in `FEATURE_COLUMNS` below, since it's model-internal rather
than fetched/engineered by `features.py`.)

Note that **36 of the 51 columns below are not used by the two priority targets**.
They are not dead code — avg and max use all of them — but if you are reading this
to understand what drives the cheap-price forecast, the 15 in
[Per-Target Feature Sets](#per-target-feature-sets) are the list that matters.

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
- `price_se4_min_lag1` — yesterday's min; historically the highest-importance feature for both min and cheap2h (~0.23–0.25). **No longer used by either trough model** — ablation testing showed removing it (with the rest of the prune) *improves* both, since importance reflects in-sample usage rather than marginal value, and the lag is frozen stale across each forecast window anyway. Still used by avg/max. See [Per-Target Feature Sets](#per-target-feature-sets).
- `price_se4_max_lag1` — yesterday's max; the **only** SE4 price lag the pruned trough list retains
- `price_se4_cheap2h_lag1` — yesterday's cheap2h. **No longer used by any model** (2026-08-05). It was built for the cheap2h model and was that model's highest-importance feature, but a direct A/B found it *actively harmful* once the rest of the list was pruned — worse on 13 of 13 directly-comparable measurements. The same lesson as `price_se4_min_lag1`: a target's own lag looks essential by importance and is frozen stale across the forecast window in practice. Still computed by `add_se4_price_lags` because the A/B experiment scripts use it as their reference arm.
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

Separate models per target because the targets have different physical drivers: min occurs at renewable oversupply moments, avg smooths out noise, max occurs at peak demand/scarcity, and cheap2h tracks the depth of the daily price trough.

### Per-Target Feature Sets

The same reasoning that justifies separate models also applies to their inputs: a
feature can be load-bearing for one target and dead weight for another. `model.TARGETS`
is therefore deliberately heterogeneous.

| Model | Feature list | Count |
|-------|--------------|-------|
| `avg`, `max` | `FEATURE_COLUMNS` | 51 |
| `min`, `cheap2h` | `TROUGH_FEATURE_COLUMNS` — cheap2h also gets `neg_price_proba` appended by the hurdle at fit/serve time | **15** (+1 for cheap2h) |

The 15 kept features:

`max_wind`, `mean_wind_de_north`, `mean_wind_stockholm`, `price_se4_max_lag1`,
`residual_load`, `residual_load_min`, `temp_gradient_se3_se4`, `radiation_variability`,
`ttf_price_lag1`, `gas_marginal_cost`, `reservoir_norway_deviation`,
`reservoir_sweden_gwh`, `reservoir_sweden_change`, `is_workday`, `dow_sin`

**How this list was arrived at (2026-07 → 2026-08).** A systematic per-target
ablation program tested every one of the 51 columns against all four targets, then
tested combined drop sets per target. `min` adopted the result first: **−1.01
EUR/MWh pooled across 14 measurements spanning four independently-fetched
snapshots**, every measurement negative, per-window std *decreasing* (−0.19).

`cheap2h` came to the same list by a different route. Its own per-feature evidence
was unusable — re-running identical windows on an independently fetched snapshot
reproduced the sign of its per-feature deltas only **37% of the time**, worse than
chance — so instead of building a list from that noise, min's list was borrowed
wholesale and tested. It won on **17 of 17** measurements (−0.80 pooled), then on
**16 of 16** in a follow-up spanning three different evaluation periods (−0.86
pooled, positive in every period cluster). Because the list was *selected* on min's
data and only then *tested* on cheap2h, there is no selection bias on this target —
it is a genuinely out-of-sample result, which is why it replicated so cleanly.

**Both directions are now closed for these 15 columns.** A follow-up audit
(2026-08-05, 512 walk-forwards) asked the two complementary questions:

- *Did the prune throw something away?* All 36 excluded columns were added back —
  grouped into 7 physically coherent blocks plus 8 individually, since adding one
  of ~50 correlated columns is nearly invisible while a whole block is not.
  **Nothing replicated.** The only sign-consistent result was the 8 price/market
  lags, which came back **+1.00 (harmful, 0 of 16 measurements favourable)**.
- *Is any of the 15 dead weight?* Leave-one-out on all 15. **Every one is KEEP or
  INCONCLUSIVE** (and INCONCLUSIVE means keep — the burden of proof is on removal).
  The most load-bearing are `mean_wind_stockholm` (+1.36 if removed) and `max_wind`
  (+0.51), both harmful to remove at every one of 16 measurements.

So this is a local optimum in both directions, not merely an improvement over what
came before. Three findings worth flagging, because all three are counter-intuitive:

- **The prune removes each trough target's own price lag** — `price_se4_min_lag1`
  for min, `price_se4_cheap2h_lag1` for cheap2h — despite each being its target's
  single highest-importance feature (~0.23–0.25). Feature *importance* measures
  in-sample usage, not marginal value: with 51 correlated columns XGBoost splits on
  whatever is convenient, and the horizon-honest evaluation freezes those lags stale
  across each test window anyway. Removing them pushes the model onto
  per-day-forecastable signals that stay valid at d+2…d+7.
- **avg and max were tested the same way and keep everything.** avg's best candidate
  came back a stable null (pooled +0.001 over 18 measurements, mixed sign); max's
  candidates leaned actively harmful. A prune that works for one target is not
  evidence for another — which is exactly why cheap2h had to be *measured* on min's
  list rather than assumed to share it.
- **15 columns is not 15 independent signals.** `residual_load` is itself a
  composite of nine of the excluded columns (temperature, five wind series, four
  radiation series), and `co2_price_lag1` is exactly recoverable from the two kept
  fuel columns (`gas_marginal_cost` − `ttf_price_lag1`, ÷ 0.35). The prune removes
  redundant *encodings* far more than it removes information.

**One property this does not fix:** min and cheap2h are still independent models, so
their predictions can violate `min ≤ cheap2h` — measured at 31 of 60 held-out days
after the change, versus 29 of 60 before, though the worst violation shrank from
+27.9 to +8.9 EUR/MWh. The coherence clamp that would enforce it was tested and
made cheap2h *worse* (see [Features Tested and Rejected](#features-tested-and-rejected)).

Full evidence trail: [FEATURE_REVALIDATION_PLAN.md](FEATURE_REVALIDATION_PLAN.md)
and [experiments/](experiments/).

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

### How changes are validated

**Standing practice (2026-08): no feature or model change is adopted on anything
but a drift-free A/B result, and the README quotes A/B deltas — not
before/after headline MAE — as the evidence for a change.**

Why this is a rule and not just a preference:

- **Headline MAE is period-dominated.** Between two evaluation windows a few weeks
  apart, the two models that did *not* change at all moved **+1.28** (`avg`) and
  **−1.76** (`max`) — larger than any change we have ever adopted, and in opposite
  directions. A before/after comparison across runs cannot separate the change from
  that, which is what makes it misleading rather than merely imprecise.
- **A/B removes exactly that.** `BASELINE` and `CANDIDATE` are fit on identical
  slices, so the delta isolates the change. Running several `shift`s then samples
  different day-alignments of the same history, and re-running on a
  separately-fetched snapshot additionally covers data revisions.
- **Feature importance is not evidence either.** It measures in-sample usage, not
  marginal value. `min` measurably improved when its highest-importance feature was
  removed (see [Per-Target Feature Sets](#per-target-feature-sets)).

The bar for adoption:

| Verdict | Meaning | Action |
|---------|---------|--------|
| `REAL` | sign-consistent across shifts, and abs(mean) ≥ spread | adopt |
| `BORDERLINE` | sign-consistent but smaller than the spread | replay on a **separately-fetched** snapshot; adopt only if the sign holds |
| `NOISE` | sign flips across shifts | reject |
| `NO_CHANGE` | identical to baseline | reject |

Additional standing requirements: the per-window **std must not inflate**, and
priority order is **cheap2h → min → avg** (`max` is not a priority).

**The A/B verdict is the gate (changed 2026-08-05).** A change counts as done once
it clears the bar above; it no longer waits on a confirming Actions run. The
previous rule required both, which in practice meant a validated improvement sat
unshipped for a day to be re-measured by a *weaker* instrument — a single rolling
run whose headline MAE moves ±1.3 with the period alone (see
[Current MAE Baseline](#current-mae-baseline)). An A/B across several shifts and
snapshots is strictly more evidence about accuracy than one production run is.

What that trade gives up, stated plainly so nobody has to rediscover it: the
Actions run was never a good *accuracy* check, but it was the only end-to-end
exercise of the parts the A/B harness never touches — the live fetches, the
EUR→SEK conversion, and the Home Assistant push. A backtest cannot catch a
NaN exchange rate blanking the payload (which has happened; see the fixed
currency bug in IMPROVEMENT_PLAN.md). So the requirement is replaced, not dropped:

- **Accuracy** → the A/B verdict, before commit.
- **Integration** → run `train` → `predict` → `get_feature_importance` against a
  cached snapshot locally before commit, asserting feature counts and finite
  predictions. Cheap (a couple of minutes) and it catches the shape and wiring
  errors a feature-set change can actually introduce.
- The next Actions run is still where a live-fetch or push regression would
  surface — **watch it, but don't block the change on it.**

**For *ablations* (testing whether to remove an existing feature) use
`ab/verdict.py::classify_ablation` instead of `classify`.** This is the
exception, not the everyday path — day-to-day A/B work is almost always testing
an *addition* (a new feature, a new source, a model change), where `classify`'s
"unproven → keep the status quo" is exactly right, because rejecting an unproven
addition already leaves the simpler model. That logic inverts for a removal:
"unproven → keep" then means *keep the feature*, so a genuinely worthless
feature — whose ~zero effect flips sign purely from fitting noise — gets
classified NOISE and never leaves. `classify_ablation` fixes this by also
weighing the *magnitude* of the effect (scaled to the target's own baseline
MAE), distinguishing dead weight (never matters, safe to drop) from a feature
that's large but genuinely regime-dependent (matters a lot sometimes, keep it) —
see the function's docstring for the full verdict table. It is **not** wired
into `ab_test.py run` / `run_ab()` — that entrypoint always uses `classify`,
since most CANDIDATEs are additions. Reach for `classify_ablation` explicitly
(`from ab.verdict import classify_ablation`) only when the CANDIDATE actually
removes a column, typically during a periodic feature-set audit like the one
that produced [Per-Target Feature Sets](#per-target-feature-sets) — not for
routine feature-addition testing.

**Re-opening a rejected verdict requires a stated mechanism, written before the
run.** The table below is long, and per-measurement noise is a few tenths — so if
you re-test enough rejected entries, some will "pass" by chance alone. That is
searching noise for a favourable answer, not validation. Before re-running
anything from [Features Tested and Rejected](#features-tested-and-rejected), write
down *why the answer would now differ*: what changed in the model, the data or the
measurement that caused the original rejection. "A lot has changed, maybe it
behaves differently now" is not a mechanism, and if you cannot state one, that is
your answer. Worked example: after the 2026-08 prune cut the priority targets from
51 columns to 15, exactly four rejected items had a mechanism (IMPROVEMENT_PLAN.md
"Round 13") and five were explicitly declined with reasons.

The same discipline covers any test with many arms — a hyperparameter sweep, a
per-feature ranking. Picking the best of N on the grid you also validate on is
selection on noise (this is what made round 2's per-feature ranking unusable, at
37% sign reproduction). **Screen on one period cluster, confirm the winner on the
others**, and fix the confirmation bar before looking at the screen results.

When updating the MAE table above, quote the run it came from and the snapshot it
was measured on, and record adopted changes as their **A/B deltas** in the table
beside it. Full worked example of the practice: the feature re-validation program in
[FEATURE_REVALIDATION_PLAN.md](FEATURE_REVALIDATION_PLAN.md) and
[experiments/](experiments/).

### How it works

**Key insight — one shift axis.** `walk_forward_validate` derives its 52-window grid
backwards from the *end* of the data (`min_train = n - iterations*step`). So
truncating the last `s` rows of the merged frame reproduces the eval exactly as it
would have run `s` days earlier — window placement **and** the training tail both
move together, the same way a real earlier run would differ. One knob (`shift`)
captures the whole between-day axis; running shifts `0..5` gives six simulated
run-days from one fetch.

**The shift axis is not a regime axis — don't read a NOISE verdict as "not
regime-dependent."** All six shifts cover nearly the same ~365 days, offset by
0-5 and overlapping 6/7 with their neighbour, so a sign flip across shifts means
"unstable to the exact day-of-week window boundary," not "helps in winter, hurts
in summer." Regime structure (calm vs. volatile periods) lives **within** one
shift's 52 test windows, which span a full year and can range 3-9 EUR/MWh in
calm stretches to 30+ in a cold snap or supply shock — averaging that into one
per-shift MAE hides it. If a feature's mean effect looks like noise but you
suspect it's actually large-and-regime-dependent (helps in some conditions,
hurts in others, cancelling out on average), that needs a different analysis:
keep the *per-window* deltas from one shift (not just their mean) and correlate
them against window-level descriptors (price level, volatility, wind, etc.)
instead of comparing across shifts.

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
   iterating. `--days N` fetches a longer/shorter window than the default ~3
   years; a longer one auto-routes to `ab_cache/long/` (`--root` to override)
   so it can't be picked up as "newest" by a routine run against the normal
   snapshots — see IMPROVEMENT_PLAN.md "Round 15" for why a longer window is
   sometimes wanted.
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
     **Getting the shift right on the new snapshot matters — the default `0-5`
     range can silently duplicate windows you've already tested.** `build_training_data`
     is a *fixed-length* sliding window (not accumulating), so a snapshot fetched
     `N` days later is the same length, just shifted `N` rows forward. That means
     `apply_shift(newer_data, N)` reproduces the older snapshot's shift-0 windows
     **exactly** — the one shift value that gives a genuine "same test,
     independently re-fetched" comparison. Any other shift on the new snapshot
     either duplicates a window the old snapshot's `0-5` sweep already covered, or
     (for extending coverage rather than replaying) needs checking against every
     shift used so far, not just the immediately preceding snapshot. Verify
     programmatically before trusting the result — don't assume: compare each
     candidate shift's resulting last `date` against the older run's tested dates,
     and assert no overlap (or, for a same-window replay, assert the dates match
     exactly). Getting `N` wrong doesn't error, it just quietly reuses data and
     inflates apparent replication.
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

### Measurement grids: tail-truncated vs sliding (read before designing a run)

`ab.harness.apply_shift` truncates the **tail** only, so a shifted point also
loses training history: on a 3-year snapshot `min_train` falls 721 → 541 → 361
across the NOW / −6M / −12M clusters. That confounds "different period" with
"less data", and it is not a theoretical worry — it produced three wrong or
grumbled readings (round 12's seasonal "gain", round 13.1's r = −0.6, and a
verdict that nearly removed the cheap2h hurdle; see the rejected table below).

**Prefer a sliding constant-length window when a long snapshot is available.**
`ab_test.py fetch --days 1825` caches ~5 years under `ab_cache/long/`; a window
of `rows[n-L-s : n-s]` then holds `min_train = L-364` fixed at every shift, so
each point is "production as it would actually have run on that date".
`experiments/run_round15_long_window.py` has the worked implementation
(`window()`) and a four-cluster grid reaching −21M with zero calendar overlap
against NOW.

**Especially important for any component that fits something internally** — a
classifier, cross-validation, an ensemble. Tail truncation starves it: the
hurdle's negative-price classifier sees ~21% positives, so 361 training rows
leave ~76 positives across a 5-fold split, and it measured as worthless when it
is in fact worth +0.250 EUR/MWh.

### Which verdict function to call

| Situation | Function | Notes |
|---|---|---|
| Addition, shift grid (`ab_test.py run`) | `classify` | REAL / BORDERLINE / NOISE / NO_CHANGE |
| Removal, shift grid | `classify_ablation` | needs `baseline_mae`; not wired into `run` |
| Addition, **period-cluster grid** | `classify_clustered` | one vote per period |
| Removal, **period-cluster grid** | `classify_ablation_clustered` | needs per-point deltas, refuses cluster means |

`classify`'s magnitude test is `|mean| >= spread`, and spread is a **range**
statistic — it grows with the number of measurement points while a real
standard error shrinks, so the rule gets *stricter* the more you measure
(measured: P(REAL) 96% at 2 points → 0% at 8, at a constant −0.234 effect). On
a 16-point grid it calls the cheap2h hurdle NOISE. Use the `_clustered`
variants whenever the grid has period clusters: they judge sign-consistency
across NOW / −6M / −12M (…) with one unweighted vote each, since the points
*within* a cluster overlap ~97% and would otherwise win on count alone.
`MIN_CLUSTER_EFFECT = 0.10` is **provisional** — the historical ledger cannot
discriminate 0.10 from 0.20; the only hard constraint is that it must stay
below ~0.29 so the hurdle keeps passing. Self-check: `python ab/check_verdict.py`.

**Always check `corr(delta, min_train)` before reading a cluster table.** A real
effect should not care about training-set size; if it does, the grid is telling
you about data volume, not period.

## Features Tested and Rejected

Keep this list updated — it prevents re-testing things that didn't work.

| Feature | Reason rejected |
|---------|----------------|
| Per-target hyperparameters (`max_depth`/`learning_rate`/`min_child_weight`/`colsample_bytree`), min & cheap2h | Round 13.1, 2026-08-05. `min`: no config even directionally consistent on the screen (best −0.031, 5/8). `cheap2h`: `colsample_bytree=0.6` and a `+min_child_weight=5` combo confirmed on 16 points — both NOISE, killed by −12M (combo NOW −0.234 / −6M −0.190 / −12M +0.001). Deltas correlate r=−0.6 with `min_train`, so the far clusters may understate them; recorded, not adopted. |
| Longer training window (4y / 4.4y / 4.8y vs 3y) | Round 15a/15c/15d, 2026-08-06, on a 5-year snapshot. Longer won 24/24 at the NOW period (cheap2h −0.73, min −0.61, avg −0.61) but NOISE across three periods — the gain tracks how crisis-heavy the added year is (mean TTF 92 → 124 → 129). |
| Shorter training window (2.0y / 2.5y vs 3y) | Round 15d, 2026-08-06. NOISE on all four period clusters for both priority targets, sign-flipping (cheap2h 2.0y: NOW +0.404 but −12M −0.467). Together with the row above this closes window length in BOTH directions: it is not a lever, and 15c's single-period 24/24 was a period effect. Keep `TRAINING_DAYS = 1095`. |
| Removing the cheap2h negative-price hurdle | Round 15b, 2026-08-06 — **tested and REJECTED, the hurdle stays.** Removal costs **+0.250 EUR/MWh**, positive in all four period clusters → `KEEP_LOAD_BEARING`. An earlier reading (13.1b/c) called it worthless; that was a `min_train` artefact — the old tail-truncated grid trained the far clusters on 361–541 rows, starving the hurdle's classifier of negative-price days. |
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
| Conservative feature prune for `cheap2h` (drop `day_of_year_cos`, `hdd_cold_boost`, `mean_wind`, `radiation_midday`, `reservoir_sweden_gwh` — rated dead weight by single-feature leave-one-out) | Combined-drop A/B across 3 independently-fetched snapshots: mean +0.08 / +0.24 (replay) / +0.02 — never negative, leans harmful, never NOISE→REAL. Not adopted. cheap2h's own per-feature evidence is separately unreliable (37% sign-agreement between independent fetches, worse than chance), so no refined candidate was pursued from this route (2026-08). |
| Feature-set prune for `avg` (drop 14: `co2_price_lag1`, `gas_marginal_cost`, `day_of_year_cos`, `hdd_linear`, `max_temp`, `mean_radiation_dk1`/`dk2`, `mean_temp`, `mean_wind`, `month_cos`/`sin`, `reservoir_sweden_gwh`, `wind_night`, `wind_variability`) | Combined-drop A/B: NOISE on the original snapshot (+0.03) but consistently **harmful** on the next one (all 4 shifts positive, mean +0.31) — traced to the `co2`/`gas` columns: an active 2026-07 TTF/gas price rise had made them newly load-bearing, something the original (calmer) snapshot didn't show. Re-tested with those fuel-linked columns excluded (12 features): resolved to a genuine null across **4** independently-fetched snapshots (+0.11 / +0.02 / −0.12 / −0.03, sign mixed every time, pooled ≈0) — not confidently dead weight, not harmful either. **Not pruned; do not re-test this exact 12-feature set** (2026-08). |
| Feature-set prune for `max` (drop 19, then a fuel-excluded 15-feature refinement) | Same program as `avg`, same fuel-price mechanism: the 19-feature drop was net-harmful (all-shifts-positive on 2 of 3 snapshots, up to +1.07). The fuel-excluded 15-feature refinement reduced but did not eliminate the harm (7 of 11 pooled measurements still positive). `max` is not a priority target; not pruned (2026-08). |
| `price_se4_cheap2h_lag1` on the pruned `cheap2h` model | The lag that originally justified giving cheap2h its own column list at all, and cheap2h's highest-importance feature — but **actively harmful** once the rest of the list is pruned: worse on **13 of 13** directly-comparable measurements. Same mechanism as `price_se4_min_lag1` on min: a target's own lag looks essential by in-sample importance while being frozen stale across the forecast window, crowding out per-day-forecastable signal (2026-08). |
| Adding back any of the 36 columns excluded from `TROUGH_FEATURE_COLUMNS`, for `cheap2h` | Tested as 7 physically coherent blocks + 8 singles over 16 measurements spanning three evaluation periods (2026-08-05). **Nothing replicated** — the best candidate was −0.07 with sign flips between period clusters. The 8 price/market lags (`price_de_lag1`, `price_dk2_lag1`, `price_se4_avg_lag1/2/7`, `price_se4_min_lag1`, `price_momentum`, `price_volatility_7d`) were the only sign-consistent block and were **harmful: +1.00, 0 of 16 favourable**. Do not re-add these individually hoping for a different answer; the block test is more sensitive than a single-column test, not less. |
| Removing any of the 15 `TROUGH_FEATURE_COLUMNS` from `cheap2h` | Leave-one-out over the same 16 measurements: every column classified KEEP or INCONCLUSIVE under `classify_ablation` (2026-08-05). Most load-bearing: `mean_wind_stockholm` (+1.36 if dropped) and `max_wind` (+0.51), both harmful to remove at all 16 measurements. One column leaned droppable but did not clear the bar — `price_se4_max_lag1` (−0.20, 12 of 16 favourable, negative in 2 of 3 period clusters) — **closed in round 12 (2026-08-05) reusing this same measurement: verdict `KEEP_SCENARIO`** (sign flips but the effect is large enough somewhere to be regime-dependent rather than dead weight). Kept, not removed. |
| Solar/intraday-trough features on `cheap2h` **as a year-round addition** | `radiation_midday`, `residual_load_range` and the 5-column solar block each help in the light half of the year and hurt in the dark half by about as much, so pooled over a year they cancel to ~0 and read as noise (2026-08-05). Measured LIGHT−DARK contrast: `radiation_midday` **−0.48**, replicated in all three period clusters and 14 of 16 measurements. The open hypothesis — that they need an annual-cycle feature alongside them so the model can condition on season — was tested in round 12 and also rejected; see the next row. |
| Seasonal interaction (`radiation_midday`/`B3_solar`/`B4_intraday` + `day_of_year_sin/cos`) for `cheap2h`, round 12 (2026-08-05) | Same 16-point grid as the two rows above. All three configs `NOISE` — sign flips between period clusters even though the LIGHT/DARK contrast still points the same direction as the plain-addition test above. A post-hoc **oracle-ceiling check** (candidate's per-window MAE in LIGHT windows, base15's in DARK — an upper bound no real gate can beat, since a real gate can't pick per window) landed at only −0.06 to −0.16 EUR/MWh, below this grid's resolution floor (`LOO_price_se4_max_lag1` above needed −0.20 to only reach 12/16). Worse: the apparent LIGHT gain correlates with `min_train` (r = +0.45 to +0.67 across the three configs) — it's a **data-starvation artefact**, not a seasonal effect. The −6M/−12M clusters (541/361 training rows) show the gain; the NOW cluster (721 rows, closest to production's ~1085) shows a loss or flat. Extrapolated to production's training size all three configs are net harmful (+0.4 to +0.7). Closed; do not re-test the day_of_year combination or the plain addition again on `cheap2h`. |
| Splitting the negative-price hurdle's classifier/regressor feature lists for `cheap2h` (`_fit_hurdle_model` currently uses ONE list for both), round 12 (2026-08-05) | Classifier-on-51-cols/regressor-on-15-cols (the live question, motivated by "will it go negative" being a coarser question than "how low"): `NOISE`, mean +0.12, no clear gain. Classifier-on-15/regressor-on-51 (reverse-split control): consistently worse, all 16 measurements positive (mean +0.94) — confirms the test harness (the control behaves as expected) but doesn't make the case for the primary split either. Closed; the single unified 15-column list (production, unchanged) stays for both halves of the hurdle. |

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
