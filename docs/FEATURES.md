# Features and Data Sources

The catalogue of every input the pipeline fetches and every column
`features.py` engineers from them. **Which model uses which columns** is a
separate question, answered in [MODEL.md](MODEL.md#per-target-feature-sets) —
the lists are deliberately different per target.

Where a column was added, changed or removed and why, see
[DECISIONS.md](DECISIONS.md). Columns that were tried and dropped are in
[REJECTED.md](REJECTED.md).

## Data Sources

| Source | What it provides | Auth |
|--------|-----------------|------|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | SE4/DE/DK2 hourly day-ahead prices, SE3 nuclear outages (A77), Sweden reservoir fill (A72) | `ENTSO_E_TOKEN` env var |
| [Open-Meteo](https://open-meteo.com) | Hourly weather archive (to ~today−5), recent-analysis top-up for the days the archive has not reached, and an 8-day forecast — SE4 (Malmö) and 5 regional locations | None |
| [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | TTF natural gas futures (`TTF=F`), EU ETS carbon allowances (`CO2.L`) | None |
| [NVE](https://biapi.nve.no/magasinstatistikk) | Norwegian hydro reservoir fill levels + 20-year min/max/median by week | None |
| [Nordpool](https://www.nordpoolgroup.com) | Published SE4 prices (used to exclude already-known days from predictions) | None |

## Features (51 engineered + 1 cheap2h-specific)

The catalogue below is the full engineered set (`FEATURE_COLUMNS`), used in its
entirety only by the **max** model. **Not every model uses every
feature** — see [Per-Target Feature Sets](MODEL.md#per-target-feature-sets):

| Model | Features used |
|-------|---------------|
| max | all 51 |
| **avg** | **a validated 42-feature subset** (`AVG_FEATURE_COLUMNS`) — `FEATURE_COLUMNS` minus the 9-column price/market lag family |
| **cheap2h** | **a validated 15-feature subset** (`TROUGH_FEATURE_COLUMNS`), plus `neg_price_proba` at fit/serve |
| **min** | **the same list minus `price_se4_max_lag1`** (`MIN_FEATURE_COLUMNS`, 14) |

(cheap2h's model also gets a `neg_price_proba` input computed at fit/serve time by
the negative-price hurdle classifier — see [Negative-Price Hurdle](MODEL.md#negative-price-hurdle-cheap2h-only).
It's not a column in `FEATURE_COLUMNS` below, since it's model-internal rather
than fetched/engineered by `features.py`.)

Note that **36 of the 51 columns below are not used by the two priority targets**
(37 for `min`), and 9 of them are also not used by `avg`. They are not dead code —
`max` uses all of them — but if you are reading this to understand what drives the
cheap-price or average-price forecast, the lists in
[Per-Target Feature Sets](MODEL.md#per-target-feature-sets) are what matters.

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
- `price_de_lag1`, `price_dk2_lag1` — previous day's prices in neighbouring zones.
  **Used only by `max`** — never in the trough lists, and dropped from `avg` in 2026-08-21
  with the rest of the price/market family (round 18 separately tested them alone for
  `min`/`cheap2h` and found them harmful there too, see
  [Features Tested and Rejected](REJECTED.md))
- Only lag-1 is valid: day-ahead auction clears all zones simultaneously

### SE4 Own Price Lags (autoregressive)
- `price_se4_avg_lag1` — was avg's strongest feature (~0.24 importance) until 2026-08-21.
  **No longer used by `avg`** — dropped along with the rest of the price/market family
  (see [Per-Target Feature Sets](MODEL.md#per-target-feature-sets)). **Used only by `max`** now.
- `price_se4_avg_lag2`, `price_se4_avg_lag7` — momentum and weekly seasonality. Same
  status: **used only by `max`** since 2026-08-21.
- `price_se4_min_lag1` — yesterday's min; historically the highest-importance feature for
  both min and cheap2h (~0.23–0.25). **No longer used by `min`, `cheap2h` or `avg`** —
  ablation testing showed removing it improves the trough targets (importance reflects
  in-sample usage, not marginal value, and the lag is frozen stale across each forecast
  window anyway), and the same family-level finding closed it out of `avg` too. **Used
  only by `max`** now. See [Per-Target Feature Sets](MODEL.md#per-target-feature-sets).
- `price_se4_max_lag1` — yesterday's max. **Used by `cheap2h` and `max`, not by `min` or
  `avg`** (2026-08-06 / 2026-08-21): on one 16-point / four-period grid, removing it is
  `REMOVE_HARMFUL` for min (−0.552, favourable in every period, per-window std also
  −1.07) but `KEEP_SCENARIO` for cheap2h (−0.439 on average, yet one period genuinely
  positive). Same column, same grid, opposite answers — which is why the two trough
  lists are no longer identical. It is the last SE4 price lag `min` carries, so **`min`
  now has no electricity-price feature at all**; `avg` lost it along with the rest of
  the price/market family.
- `price_se4_cheap2h_lag1` — yesterday's cheap2h. **No longer used by any model**
  (2026-08-05). It was built for the cheap2h model and was that model's highest-importance
  feature, but a direct A/B found it *actively harmful* once the rest of the list was
  pruned — worse on 13 of 13 directly-comparable measurements. The same lesson as
  `price_se4_min_lag1`: a target's own lag looks essential by importance and is frozen
  stale across the forecast window in practice. Still computed by `add_se4_price_lags`
  because the A/B experiment scripts use it as their reference arm.
- `price_momentum` — lag1 minus lag2 (rising vs falling trend). **Used only by `max`**
  since 2026-08-21 (dropped from `avg` with the rest of the price/market family).
- `price_volatility_7d` — rolling 7-day std (market regime stability). Same status:
  **used only by `max`** since 2026-08-21.

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

