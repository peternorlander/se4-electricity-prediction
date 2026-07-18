# SE4 Electricity Price Predictor

A machine learning pipeline that predicts day-ahead electricity prices for the Swedish SE4 bidding zone (Malmö/southern Sweden) and pushes the predictions to Home Assistant as a sensor.

## Purpose

The predictions are used in Home Assistant to make smart decisions about **schedulable energy consumption** — most importantly EV charging. If prices are expected to be lower in the coming days, charging can be deferred. If prices are expected to rise, the car charges now. The sensor exposes daily min/avg prices up to 8–10 days ahead, giving automation rules enough lead time to act.

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
    → Walk-forward validation: 35 windows × 7 days (evaluate.py)
         │
         ▼
Inference → EUR/MWh → SEK/kWh → Home Assistant sensor (ha_client.py)
```

## Current MAE Baseline

Evaluation uses 35-window walk-forward validation on 3 years of training data. Each target is reported **individually** — min, avg, max and cheap2h have different physical drivers, so a blended metric would hide the per-target movement that matters when tuning. `cheap2h` and `min` are the numbers to watch for scheduling; `max` is not a priority.

Post-harmonization baseline (first measured 2026-07, targets computed on hourly means):

| Target  | MAE (EUR/MWh) | Std |
|---------|--------------|------|
| min     | 13.66        | ±4.84 |
| avg     | 15.88        | ±6.16 |
| max     | 31.97        | ±13.88 |
| cheap2h | 13.56        | ±4.77 |

**Note:** min/max are not comparable to the pre-2026-07 figures (old min ~11.7). The harmonization changed the target itself — min is now the cheapest *hour* (hourly mean), removing the systematic high bias that inflated apparent accuracy on post-Oct-2025 windows. 13.66 is the honest error against the consistent target; compare future runs against this, not the old values.

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

## Features (47 shared + 1 cheap2h-specific)

### Local Weather (SE4/Malmö)
- `mean_temp`, `min_temp`, `max_temp` — daily temperature aggregates
- `mean_wind`, `max_wind` — daily wind speed (10m)
- `mean_radiation`, `max_radiation` — global horizontal irradiance (GHI W/m²)

### Regional Wind & Solar (5 locations, 100m hub-height)
Captures wind and solar generation in coupled markets that flow into SE4.
- **Karlskrona** — Baltic offshore wind patterns (strongest importance in this group)
- **DK2** — directly coupled to SE4 via Øresund (~1700 MW)
- **Stockholm** — SE3 load centre (also used for temperature gradient)
- **DK1** — western Denmark/Jutland
- **DE North** — northern Germany, Baltic Cable (~600 MW)

### Market Coupling
- `price_de_lag1`, `price_dk2_lag1` — previous day's prices in neighbouring zones
- Only lag-1 is valid: day-ahead auction clears all zones simultaneously

### SE4 Own Price Lags (autoregressive)
- `price_se4_avg_lag1` — strongest single feature (~0.14–0.16 importance)
- `price_se4_avg_lag2`, `price_se4_avg_lag7` — momentum and weekly seasonality
- `price_se4_min_lag1` — yesterday's min (direct signal for min model)
- `price_se4_max_lag1` — yesterday's max
- `price_se4_cheap2h_lag1` — yesterday's cheap2h (**cheap2h model only** via `CHEAP2H_FEATURE_COLUMNS`; kept out of min/avg/max to avoid perturbing their baseline with a correlated lag)
- `price_momentum` — lag1 minus lag2 (rising vs falling trend)
- `price_volatility_7d` — rolling 7-day std (market regime stability)

### Residual Load
Engineered composite feature: demand proxy minus weighted wind/solar supply.
- Demand proxy: `15 - mean_temp` (heating-based)
- Wind: cubic power curve `(v/13)³` applied per location, weighted by interconnection capacity to SE4
- Solar: `radiation / 500` per location, same weights
- Also exposed as `residual_load_lag1` for momentum

### Heating Degree Days
- `hdd_linear` — `max(0, 17 - temp)`: standard Nordic HDD (**consistently top-3 feature**)
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

## Known Limitations

- **3-year training window**: Unusual market periods (e.g. energy crisis 2021–2022) have outsized weight. Reservoir features will become more valuable as more data accumulates.
- **Daily resolution**: The model predicts daily aggregates, not 24 hourly prices. Hour-level predictions would be more actionable for EV scheduling but require significantly more feature engineering. The `cheap2h` target partially addresses this: it predicts what a ~2h charging session picking the day's cheapest hours would pay, which is the number the "charge today or wait" decision needs.
- **Forecast horizon**: All forecast days (1–8) use the same features. Day+8 weather forecasts are less accurate than day+1 but the model doesn't distinguish between them. A `forecast_horizon` feature (1–8) is a candidate improvement.
- **Max prediction accuracy** (~39 EUR/MWh MAE): Intentionally not optimized. Max prices are driven by rare spike events that are hard to predict from daily features.
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
- `mae_min` / `mae_avg` / `mae_max` / `mae_cheap2h` — current per-target MAE values from walk-forward validation
- `feature_importance_min` / `feature_importance_avg` / `feature_importance_cheap2h` — top features per model (for debugging)

For charging decisions ("charge today or wait for cheaper days"), compare `cheap2h` across days — it is the expected price of a ~2h session picking the day's cheapest hours, which is what the schedule can actually achieve.

The addon value is fetched live from Home Assistant each run, so distribution cost changes take effect immediately without redeploying.
