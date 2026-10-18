# SE4 Electricity Price Predictor

A machine learning pipeline that predicts day-ahead electricity prices for the Swedish SE4 bidding zone (Malmö/southern Sweden) and pushes the predictions to Home Assistant as a sensor.

## Purpose

The predictions are used in Home Assistant to make smart decisions about **schedulable energy consumption** — most importantly EV charging. If prices are expected to be lower in the coming days, charging can be deferred. If prices are expected to rise, the car charges now. The sensor exposes daily min/avg/cheap2h prices for the six days after the ones already published (D+2 … D+7 from the run date: the weather forecast covers 8 days and today and tomorrow are dropped as known), giving automation rules enough lead time to act.

The pipeline runs via GitHub Actions, triggered from Node-RED/Home Assistant, and updates a Home Assistant sensor with predictions in SEK/kWh, including a configurable distribution cost addon.

## Architecture

```
Data sources (API)
    │
    ├── ENTSO-E      → SE4/DE/DK2 day-ahead prices, SE3 nuclear outages, Sweden reservoir (A72)
    ├── Open-Meteo   → Weather archive + recent-analysis top-up + 8-day forecast (SE4/Malmö + 8 regional locations)
    ├── Yahoo Finance → TTF natural gas futures, EU ETS carbon allowance (EUA)
    ├── NVE          → Norwegian hydro reservoir levels + 20-year seasonal median
    └── Nordpool     → Published prices (to skip days already known)
         │
         ▼
Feature engineering (features.py)
    → 51 engineered features (+2 northern-wind columns for min/avg/cheap2h), daily resolution, 3-year training window
         │
         ▼
Model training (model.py)
    → 4 separate XGBoost regressors: price_min, price_avg, price_max, price_cheap2h
    → per-target feature sets: `max` uses all 51, `avg` runs a validated
      44-column list (AVG_FEATURE_COLUMNS: a 42-column subset + 2 northern-wind
      columns), and the two trough targets (cheap2h, min) run validated pruned
      lists of 17 / 16 columns (TROUGH_FEATURE_COLUMNS, MIN_FEATURE_COLUMNS)
    → min and cheap2h get a coupling hurdle: a linear logistic P(the trough
      prices at continental level), from weather + Nordic reservoirs, feeds both
      regressors as p_couple
    → price_cheap2h additionally gets a negative-price hurdle: an XGBoost
      classifier's P(price_min < 0) feeds the regressor as an extra feature
    → min and cheap2h are made coherent: where predicted min exceeds predicted
      cheap2h, both become their mean (model.enforce_coherence)
    → price_cheap2h also gets a band: cheap2h_low (conformal q10), cheap2h_high
      (a two-state warning light driven by the coupling probability) and
      cheap2h_risk (that probability)
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
| Cache a data snapshot for experiments | `python ab_test.py fetch` | Needs `ENTSO_E_TOKEN`. Caches ~5 years under `ab_cache/long/` — the only shape that can train every test week on production's own 1088 rows. `--days 1095` caches a production-shaped one under `ab_cache/` for the integration check. |
| List cached snapshots | `python ab_test.py list` | Lists `ab_cache/long/`; `--root ab_cache` for the production-shaped ones. Reports each snapshot's **weather tail**; snapshots with different tails must not share one measurement grid. |
| Run an A/B | `python ab_test.py run` | `--snapshot YYYY-MM-DD`, `--shifts 0-5`. Trains each test week on a rolling 1088 rows like production; `--expanding` for the pre-2026-10 geometry. Compares `BASELINE` against whatever `CANDIDATE` in `ab/variants.py` says. |
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
| Either, **and** the lesson would change what someone does on a *different* item | The ledger row or entry, *and* a dated section in [docs/FINDINGS.md](docs/FINDINGS.md) that the row links to. If it only explains this verdict, it is row content — compress it into the row instead. |
| A change to how we measure | [docs/AB_TESTING.md](docs/AB_TESTING.md). |
| Not done yet | The improvement plan only — never a ledger. |

Five rules that make this work:

1. **A rejected row is four slots and about 60 words**, not a report:
   **what** was tested (precisely enough to recognise a re-proposal) · **the
   number and the instrument** (delta in EUR/MWh, which grid, how many
   measurements favourable — and whether the arm was leaky or horizon-honest,
   [why that matters](docs/AB_TESTING.md#horizon-honesty-measure-only-what-production-can-know))
   · **why** it failed, one sentence · **status**: closed, or what would reopen
   it. If it will not fit, the overflow is either unnecessary or a finding.
   This is a *length* rule on purpose: [REJECTED.md](docs/REJECTED.md) is read
   before trying something, and a ledger nobody finishes reading stops
   preventing the re-test it exists to prevent.
2. **One fact, one place.** A row states its verdict and links to the finding;
   it does not restate it. The same number in two files is the failure mode
   that makes both unreadable.
3. **Quote A/B deltas, not before/after headline MAE.** Headline MAE moves with
   the evaluation period more than with the model; two unchanged models once
   moved +1.28 and −1.76 between two windows a few weeks apart. See
   [docs/AB_TESTING.md](docs/AB_TESTING.md#how-changes-are-validated).
4. **Tag the round.** `experiments/` scripts are named `run_round21_*.py`, so a
   round number in the ledger row is what connects a verdict to the run that
   produced it for as long as that scratch directory survives.
5. **A change to `min` or `cheap2h` reports the band too.** `cheap2h_low` /
   `cheap2h_high` / `cheap2h_risk` are what the automation warns on, and MAE
   cannot see them. Run `python ab_test.py band` and put its scorecard — and any
   `BREACH` with the decision taken on it — in the ledger entry or row
   ([docs/AB_TESTING.md](docs/AB_TESTING.md#scoring-the-band)).

**Do not record**: how you got there, what you tried first, run times, per-point
or per-window tables, script names beyond the round tag, or a restatement of the
standing rules. None of it changes a later decision. `experiments/` is where the
working material lives, and it is expected to disappear.

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

- **The model sees exactly three years, and that is fixed.** `TRAINING_DAYS = 1095` is a *sliding* window, not an accumulating one, so history does not build up: every fit sees 1095 rows and any regime older than that is invisible. Both longer (4y/4.4y/4.8y) and shorter (2.0y/2.5y) windows were tested and rejected — window length is not a lever (see [REJECTED.md](docs/REJECTED.md)). The binding consequence is the **prediction ceiling**: XGBoost cannot extrapolate past its training targets, so a price the last three years never reached is unreachable — see [The 2026-08-17 regime break](docs/FINDINGS.md#the-2026-08-17-regime-break).
- **Daily resolution**: The model predicts daily aggregates, not 24 hourly prices. Hour-level predictions would be more actionable for EV scheduling but require significantly more feature engineering. The `cheap2h` target partially addresses this: it predicts what a ~2h charging session picking the day's cheapest hours would pay, which is the number the "charge today or wait" decision needs.
- **Forecast horizon**: All forecast days (1–8) use the same features and a single model that doesn't distinguish horizon. Horizon-aware modelling was **evaluated and shelved** (2026-07): the anchor-staleness sensitivity showed the true stale-lag cost is only ~1 EUR/MWh for min/cheap2h, so a `forecast_horizon` feature has little headroom. The scary-looking per-horizon curve is a weekday artifact of the step-7 walk-forward (see [Current MAE Baseline](docs/MODEL.md#current-mae-baseline)), not real horizon decay. The price-lag anchor is kept fresh regardless, since that was a free win.
- **Weather in evaluation**: walk-forward uses archive weather as a stand-in for the forecast, so results are optimistic on the weather axis. Measured 2026-10-11 on forecast vintages (Open-Meteo Previous Runs, 5 caches): scoring the same models on the forecast issued L days earlier costs **+1.63 cheap2h / +1.69 min / +2.81 avg EUR/MWh at production's leads** (D+2…D+7), almost all of it from D+5 (cheap2h +1.0…+2.1 at L5, +3.0…+4.7 at L7) — see [FINDINGS.md](docs/FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b). The headline MAE in [MODEL.md](docs/MODEL.md) and every weather feature's archive-weather A/B gain should be read with that in mind; the one weather candidate re-scored so far kept 40–75 % of its gain.
- **The headline eval trains on less than production does.** `walk_forward_validate` expands: its 52 test weeks have to come out of the same 1088-row frame production fits on, so the first trains on 724 rows and the last on 1081 — ~900 on average against production's 1088. Measured on the current feature set, production-length training scores **~0.36 better** on `min` in the most recent period, so the headline is mildly pessimistic there, though the sign is period-dependent. Production cannot avoid this (three years of fetch is exactly one evaluation year short), so the A/B grid was fixed instead and the headline was not — see [Production-length training in the A/B grid](docs/DECISIONS.md#production-length-training-in-the-ab-grid).
- **The eval does not model the training tail.** `walk_forward_validate` fits right up to its test window. Until 2026-08-23 production fitted ~5 days behind (the archive lag), so every baseline recorded before that date is optimistic by roughly the numbers in [Closing the weather-archive lag](docs/DECISIONS.md#closing-the-weather-archive-lag-round-19a). The top-up closes the gap to ~1 day; the residual is measured at **+0.155 (cheap2h) / +0.258 (min) / +0.591 (avg)** (the `gap1` arm), and the recent-analysis product's own error against the archive is on top of that and still unmeasured.
- **Max prediction accuracy** (the highest MAE of the four targets by a wide margin — see [MODEL.md](docs/MODEL.md#current-mae-baseline)): Intentionally not optimized. Max prices are driven by rare spike events that are hard to predict from daily features.
- **EUR/SEK rate**: Derived daily from Nordpool vs ENTSO-E prices. If data is unavailable, the rate may be stale. A NaN rate once blanked the entire Home Assistant payload — [FINDINGS.md](docs/FINDINGS.md#the-nan-exchange-rate-that-blanked-the-payload-2026-07-22) has the postmortem and the constraint to respect if `currency.py` is ever touched again.

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
- `predictions_raw` — EUR-to-SEK converted prices, indexed by date (min/avg/max/cheap2h per day, plus `cheap2h_low` / `cheap2h_high`, and `cheap2h_risk` — a probability 0–1, not converted)
- `predictions_with_addon` — prices adjusted by `input_number.electricity_price_addon` (distribution costs etc.) with a 5% markup
- `mae_min` / `mae_avg` / `mae_max` / `mae_cheap2h` — current per-target MAE values from horizon-honest walk-forward validation. **Monitoring only — never quote these in documentation**: each run measures whatever window that day ends on, so the value moves with the evaluation period more than with the model ([docs/AB_TESTING.md](docs/AB_TESTING.md#how-changes-are-validated))
- `mae_by_horizon` — per-target MAE broken down by forecast horizon (day+1 … day+7); shows how accuracy decays with forecast distance
- `cheap2h_interval` — the band's own calibration for this run: `coverage_calibrated` vs `nominal_coverage` (the low half), `calm_high_holdout_coverage` (should sit near 0.90), `coupled_level_ratio`, `coupled_share_training` and `warning_threshold`. A drift in the last three is where a new price regime shows first — see [docs/MODEL.md](docs/MODEL.md#the-cheap2h-band)
- `feature_importance_min` / `feature_importance_avg` / `feature_importance_cheap2h` — top features per model (for debugging)
- `eur_to_sek_rate` — SEK per EUR this run converted with, so a logged SEK/kWh value can be turned back into the model's EUR/MWh
- `git_sha` — the commit the run executed (`GITHUB_SHA`); absent on a local run

For charging decisions ("charge today or wait for cheaper days"), compare `cheap2h` across days — it is the expected price of a ~2h session picking the day's cheapest hours, which is what the schedule can actually achieve.

`cheap2h_low` / `cheap2h_high` / `cheap2h_risk` are the **warning** next to the ranking, in the same SEK/kWh units as everything else except the risk (a probability). `cheap2h` is always inside `[low, high]`. **`cheap2h_high` has two states**: when `cheap2h_risk` is 0.3 or more, it jumps to the price the day would clear at if SE4 couples to the continent (about 1.3 × the gas plant's marginal cost — roughly 140 EUR/MWh in October 2026); otherwise it is a high for a calm day, typically a third to a half of that. So a rule like "do not defer charging to a day whose `cheap2h_high` is above today's known cheap price" now fires on the days that can turn expensive and stays quiet on the ones that cannot; `cheap2h_risk` says how sure. Over the last year it warned on ~46 % of days, ~69 % of them correctly, and caught ~80 % of the expensive days; an expensive day it misses ends well above a low `high`. Rank days by `cheap2h`, not by `high` — that was tested and is **not** replicated. See [docs/MODEL.md](docs/MODEL.md#the-cheap2h-band).

A Node-RED flow in Home Assistant copies `predictions_raw` (plus the rate and commit above) into the MariaDB table `electricity_price_predictions_log` after every run, one row per target day, so what was served can later be scored against realised prices ([plan 2.2](IMPROVEMENT_PLAN_2026-09.md#22-production-prediction-log-and-realised-scorecard)). The flow lives in Home Assistant, not in this repo.

The addon value is fetched live from Home Assistant each run, so distribution cost changes take effect immediately without redeploying.
