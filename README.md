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
    ├── Open-Meteo   → Weather archive + recent-analysis top-up + 8-day forecast (SE4/Malmö + 5 regional locations)
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
    → per-target feature sets: `max` uses all 51, `avg` runs a validated
      42-column subset (AVG_FEATURE_COLUMNS), and the two trough targets
      (cheap2h, min) run validated pruned subsets of 15 / 14 columns
      (TROUGH_FEATURE_COLUMNS, MIN_FEATURE_COLUMNS)
    → price_cheap2h additionally gets a negative-price hurdle: an XGBoost
      classifier's P(price_min < 0) feeds the regressor as an extra feature
    → price_cheap2h also gets a conformally calibrated 80% interval from two
      quantile regressors on the same features (cheap2h_low / cheap2h_high)
    → Walk-forward validation: 52 windows × 7 days (evaluate.py)
         │
         ▼
Inference → EUR/MWh → SEK/kWh → Home Assistant sensor (ha_client.py)
```

## Running it

```bash
pip install -r requirements.txt
```

| What | Command | Notes |
|---|---|---|
| Production run (fetch → train → predict → push to HA) | `python predict.py` | Needs `ENTSO_E_TOKEN`, `HA_URL`, `HA_TOKEN`. This is what GitHub Actions runs. |
| Cache a data snapshot for experiments | `python ab_test.py fetch` | Needs `ENTSO_E_TOKEN`. `--days 1825` caches ~5 years under `ab_cache/long/` for the sliding grid. |
| List cached snapshots | `python ab_test.py list` | Reports each snapshot's **weather tail**; snapshots with different tails must not share one measurement grid. |
| Run an A/B | `python ab_test.py run` | `--snapshot YYYY-MM-DD`, `--shifts 0-5`. Compares `BASELINE` against whatever `CANDIDATE` in `ab/variants.py` says. |
| Self-check the verdict rules | `python ab/check_verdict.py` | |

**Pin `OMP_NUM_THREADS=4` before comparing any two MAE numbers.** XGBoost's
floating-point reduction order depends on the thread count, so the same
unchanged model lands a few hundredths apart at 4 vs 16 threads — the size of a
small real effect.

The full walkthrough for running an experiment, including what to put in
`CANDIDATE` and which verdict function to call, is
[docs/AB_TESTING.md](docs/AB_TESTING.md#how-to-run-an-experiment-agent-playbook).

## Documentation map

| File | What it holds | Read it when |
|---|---|---|
| **README.md** (this file) | Purpose, architecture, how to run, documentation rules, setup, Home Assistant contract, known limitations | Starting out, or wiring the pipeline up |
| [docs/MODEL.md](docs/MODEL.md) | Targets, model configuration, per-target feature lists, the hurdle and the interval, current MAE baseline | You need to know what the model is today |
| [docs/FEATURES.md](docs/FEATURES.md) | Every data source and every engineered column, with its formula and purpose | You need to know what a column means |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Ledger of **adopted** changes: when, the evidence, the mechanism | You are wondering why something is built this way |
| [docs/REJECTED.md](docs/REJECTED.md) | Ledger of **rejected** changes, grouped by what they touched | Before trying anything — it is probably in here |
| [docs/AB_TESTING.md](docs/AB_TESTING.md) | The A/B harness, measurement grids, verdict functions, the adoption bar, the agent playbook | Before measuring anything |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Dated write-ups of what the model structurally cannot do and why | You are about to re-derive something the project already learned |
| [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) | Open items carried from the 2026-08 round | Looking for what to work on |
| [IMPROVEMENT_PLAN_2026-09.md](IMPROVEMENT_PLAN_2026-09.md) | Open items from the September 2026 review, in priority order | Looking for what to work on — start here |
| [SLOT_PREDICTION_PLAN.md](SLOT_PREDICTION_PLAN.md) | Spec for the per-slot targets (round 20), planned, not implemented | Working on slot prediction |

## How changes are documented

Every experiment ends in **exactly one** of two ledgers. Nothing is left only in
a commit message, a chat transcript, or a script under `experiments/`.

| Outcome | Where it goes |
|---|---|
| A/B validated, shipped | An entry in [docs/DECISIONS.md](docs/DECISIONS.md), plus whatever it changes in [docs/MODEL.md](docs/MODEL.md) / [docs/FEATURES.md](docs/FEATURES.md). Delete the item from the improvement plan. |
| A/B rejected | A row in [docs/REJECTED.md](docs/REJECTED.md), in the group it belongs to, tagged with its round. Delete the item from the improvement plan. |
| Either, **and** the round explained something beyond its own verdict | The ledger row or entry, *and* a dated section in [docs/FINDINGS.md](docs/FINDINGS.md) that the row links to. |
| A change to how we measure | [docs/AB_TESTING.md](docs/AB_TESTING.md). |
| Not done yet | The improvement plan only — never a ledger. |

Three rules that make this work:

1. **An entry must be self-contained.** Grid, snapshot, arms, deltas, how many
   measurements were favourable — written in the entry itself, in EUR/MWh. A
   pointer to a script is a courtesy, never the record.
2. **Quote A/B deltas, not before/after headline MAE.** Headline MAE moves with
   the evaluation period more than with the model; two unchanged models once
   moved +1.28 and −1.76 between two windows a few weeks apart. See
   [docs/AB_TESTING.md](docs/AB_TESTING.md#how-changes-are-validated).
3. **Tag the round.** `experiments/` scripts are named `run_round21_*.py`, so a
   round number in the ledger row is what connects a verdict to the run that
   produced it for as long as that scratch directory survives.

## Working with `experiments/`

`experiments/` is **scratch, deliberately untracked** (see `.gitignore`). It holds
the analysis scripts, the PowerShell runners and the raw per-window JSON/JSONL for
each round — working material for one investigation, not documentation.

**Do not document anything there.** The condensed result is the record, and it
belongs in the ledgers above. Concretely, if you are working in that directory:

- Write the verdict into [docs/REJECTED.md](docs/REJECTED.md) or
  [docs/DECISIONS.md](docs/DECISIONS.md) **with its numbers**, so the row survives
  the scripts.
- Anything reusable — a measurement grid, a window helper, a parser trap, a data
  quirk that cost you a day — goes into [docs/AB_TESTING.md](docs/AB_TESTING.md)
  or [docs/FINDINGS.md](docs/FINDINGS.md). If it is worth writing down for the
  next round, it was never scratch.
- Do not add a README to `experiments/`, and do not leave "pending run" notes
  there. Open work lives in the improvement plan, which is the one place that is
  read when deciding what to do next.
- Raw result files are expensive to regenerate but are not a substitute for a
  written verdict. Assume they are gone on the next machine.

## Known Limitations

- **3-year training window**: Unusual market periods (e.g. energy crisis 2021–2022) have outsized weight. Reservoir features will become more valuable as more data accumulates.
- **Daily resolution**: The model predicts daily aggregates, not 24 hourly prices. Hour-level predictions would be more actionable for EV scheduling but require significantly more feature engineering. The `cheap2h` target partially addresses this: it predicts what a ~2h charging session picking the day's cheapest hours would pay, which is the number the "charge today or wait" decision needs.
- **Forecast horizon**: All forecast days (1–8) use the same features and a single model that doesn't distinguish horizon. Horizon-aware modelling was **evaluated and shelved** (2026-07): the anchor-staleness sensitivity showed the true stale-lag cost is only ~1 EUR/MWh for min/cheap2h, so a `forecast_horizon` feature has little headroom. The scary-looking per-horizon curve is a weekday artifact of the step-7 walk-forward (see [Current MAE Baseline](docs/MODEL.md#current-mae-baseline)), not real horizon decay. The price-lag anchor is kept fresh regardless, since that was a free win.
- **Weather in evaluation**: walk-forward uses archive weather as a stand-in for the forecast, so results are optimistic on the weather axis — real day+7 weather forecasts are worse than archive. This optimism grows at far horizons and is not captured by the anchor-staleness or per-horizon metrics.
- **The eval does not model the training tail.** `walk_forward_validate` fits right up to its test window. Until 2026-08-23 production fitted ~5 days behind (the archive lag), so every baseline recorded before that date is optimistic by roughly the numbers in [Closing the weather-archive lag](docs/DECISIONS.md#closing-the-weather-archive-lag-round-19a). The top-up closes the gap to ~1 day; the residual is measured at **+0.155 (cheap2h) / +0.258 (min) / +0.591 (avg)** (the `gap1` arm), and the recent-analysis product's own error against the archive is on top of that and still unmeasured.
- **Max prediction accuracy** (~34 EUR/MWh MAE, the highest of the four targets): Intentionally not optimized. Max prices are driven by rare spike events that are hard to predict from daily features.
- **EUR/SEK rate**: Derived daily from Nordpool vs ENTSO-E prices. If data is unavailable, the rate may be stale.
- **FIXED 2026-07-22 — NaN exchange rate had blanked the whole HA payload.**
  Symptom: every min/avg/max/cheap2h prediction was pushed to Home Assistant as
  `NaN`. Root cause: `currency.calculate_eur_to_sek_rate` filtered the ENTSO-E
  frame to `date.today()` and averaged it; when that day had no ENTSO-E rows yet
  the mean was NaN, so `rate = nordpool_mean_sek / NaN`. Two things made the day
  empty in practice — a midnight rollover during the run (the function
  re-derived `date.today()` independently of `predict.main()`'s single `today`,
  so a run straddling midnight computed the rate for a day not yet fetched), and
  a structural split where live Nordpool data for today can exist before
  ENTSO-E's day-ahead prices for the same day are published. **Fix constraint
  worth remembering if this is ever touched again: do NOT "just use the latest
  available ENTSO-E day"** — the rate is `SEK_mean / EUR_mean` for the *same
  delivery day* (Nordpool applies a daily ECB fixing), so pairing Nordpool-today
  with ENTSO-E-yesterday gives a **wrong** rate, not merely a stale one. The fix
  takes the single `today` from `predict.main()` and walks back up to
  `_RATE_LOOKBACK_DAYS` (7) to the most recent delivery day present in **both**
  the ENTSO-E frame and Nordpool, computing the rate on that shared day; if no
  common day exists within the window it raises `ValueError` instead of
  silently returning NaN.

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
- `predictions_raw` — EUR-to-SEK converted prices, indexed by date (min/avg/max/cheap2h per day, plus `cheap2h_low` / `cheap2h_high`)
- `predictions_with_addon` — prices adjusted by `input_number.electricity_price_addon` (distribution costs etc.) with a 5% markup
- `mae_min` / `mae_avg` / `mae_max` / `mae_cheap2h` — current per-target MAE values from horizon-honest walk-forward validation
- `mae_by_horizon` — per-target MAE broken down by forecast horizon (day+1 … day+7); shows how accuracy decays with forecast distance
- `cheap2h_interval` — the band's own calibration for this run: `coverage_raw`, `coverage_calibrated`, `nominal_coverage`, `correction_eur_mwh`, `n_holdout`. **Watch `coverage_calibrated` against `nominal_coverage`** — if it drifts away from 0.80 the band has stopped meaning what it says
- `feature_importance_min` / `feature_importance_avg` / `feature_importance_cheap2h` — top features per model (for debugging)

For charging decisions ("charge today or wait for cheaper days"), compare `cheap2h` across days — it is the expected price of a ~2h session picking the day's cheapest hours, which is what the schedule can actually achieve.

`cheap2h_low` / `cheap2h_high` are an **80% band around that day's own prediction**, in the same SEK/kWh units and carried through the addon like everything else. `cheap2h` is always inside them. Use them as a guard rail on the *defer* decision, not as a replacement for the ranking: the cost of waiting when the price is about to rise is much larger than the cost of charging now when it was about to fall, so a rule like "do not defer if `cheap2h_high` for the later day is above today's known cheap price" is the asymmetry the point estimate cannot express. Ranking the days by `cheap2h_high` instead of `cheap2h` was tested and is **not** replicated — see [Prediction interval for cheap2h](docs/DECISIONS.md#prediction-interval-for-cheap2h-round-19c).

The addon value is fetched live from Home Assistant each run, so distribution cost changes take effect immediately without redeploying.
