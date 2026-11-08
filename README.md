# SE4 Electricity Price Predictor

A machine learning pipeline that predicts day-ahead electricity prices for the Swedish SE4 bidding zone (Malmö/southern Sweden) and pushes the predictions to Home Assistant as a sensor.

## Purpose

The predictions are used in Home Assistant to make smart decisions about **schedulable energy consumption** — most importantly EV charging. If prices are expected to be lower in the coming days, charging can be deferred. If prices are expected to rise, the car charges now. The sensor exposes daily min/avg/cheap2h prices for the six days after the ones already published (D+2 … D+7 from the run date: the weather forecast covers 8 UTC days, today and tomorrow are dropped as known, and so is the Swedish date at the end of the forecast that only gets its first one or two hours), giving automation rules enough lead time to act.

The pipeline runs via GitHub Actions, triggered from Node-RED/Home Assistant, and updates a Home Assistant sensor with predictions in SEK/kWh, including a configurable distribution cost addon.

## Architecture

```
Data sources (API)
    │
    ├── ENTSO-E      → SE4/DE/DK2 day-ahead prices, SE3 nuclear outages, Sweden reservoir (A72)
    ├── Open-Meteo   → Weather archive + recent-analysis top-up + 8-day forecast (SE4/Malmö + 8 regional locations);
    │                  the forecast once more as a three-model mean for the blend shadow (plan 2.3b)
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
| Run an A/B | `python ab_test.py run` | `--snapshot YYYY-MM-DD`, `--shifts 0-5` or `--grid` (NOW / −6M / FAR clusters); `--targets cheap2h` (or `avg`, …) fits only what the screen needs — `min` and `cheap2h` always together. Trains each test week on a rolling 1088 rows like production; `--expanding` for the pre-2026-10 geometry. Compares `BASELINE` against whatever `CANDIDATE` in `ab/variants.py` says. Every fit is scored twice and the report labels both: **ARCHIVE** weather (a perfect forecast — the standing gate) and **FORECAST** weather (what production pays). The forecast vintages come from Open-Meteo (public, no token) and are cached under `ab_cache/vintages/`; `--no-vintage` skips them. |
| Fetch / check the forecast vintages | `python ab_test.py vintages` | Warms `ab_cache/vintages/` for a snapshot's evaluated days and fails if any hour lacks a vintage. `run` does this itself; use it to fetch ahead of a long run. |
| Run the tests | `python -m pytest` | `pip install -r requirements-dev.txt` first. Offline; the snapshot checks run once per cache under `ab_cache/long/` and are skipped where there is none (the Actions runner). Runs on every push and pull request (`tests.yml`). How to add one: [docs/TESTING.md](docs/TESTING.md). |
| Self-check the verdict rules | `python ab/check_verdict.py` | |
| Self-check forecast-vintage scoring | `python ab/check_vintage.py` | Offline mechanics; add `--snapshot YYYY-MM-DD` for the bit-identity check against the pre-vintage harness. |

**Pin `OMP_NUM_THREADS=4` before comparing any two MAE numbers.** XGBoost's
floating-point reduction order depends on the thread count, so the same
unchanged model lands a few hundredths apart at 4 vs 16 threads — the size of a
small real effect. **And compare on one platform**: Windows and Linux fit different models
from the same data with the same pinned versions (MAE up to ±0.3 on `min`/`cheap2h`, every row
different), so a local number never reproduces production's to the bit — see
[FINDINGS.md](docs/FINDINGS.md#windows-and-linux-fit-different-models-from-the-same-data-round-31).
Install exactly `requirements.txt`: the numeric stack is pinned, and a change of pin moves numbers.

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
| [docs/TESTING.md](docs/TESTING.md) | What the test suite guards, when a test is added, and the rules a test follows | Fixing a defect or adding a contract |
| [docs/AB_TESTING.md](docs/AB_TESTING.md) | The A/B harness, measurement grids, verdict functions, the adoption bar, the agent playbook | Before measuring anything |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Dated write-ups of what the model structurally cannot do and why | You are about to re-derive something the project already learned |
| [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) | Open work only: every open item; in section 4 what is ready now (in order), what waits for what, and the register of shadow arms running in production | Looking for what to work on — start here; "the next step" is the first item of its 4.1, and if 4.1 is empty there is nothing to do yet (waiting items are never started early or in part) |
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
| A defect fixed | A test in `tests/` that fails on the code before the fix ([docs/TESTING.md](docs/TESTING.md)), plus a ledger entry if the fix moves numbers. |
| Not done yet | The improvement plan only — never a ledger. Ready work in its 4.1, work that waits on a date, a cache or a decision in 4.2 with the condition. |
| A shadow arm running in production | A row in the plan's [shadow register (4.3)](IMPROVEMENT_PLAN.md#43-shadow-arms-in-production), from the commit that adds it to the one that removes it. Its item stays in the plan until then. |

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
- **Weather in evaluation**: walk-forward uses archive weather as a stand-in for the forecast, so results are optimistic on the weather axis. Measured on forecast vintages (Open-Meteo Previous Runs): scoring the same models on the forecast issued L days earlier costs the current model **+1.8 cheap2h / +1.8 min / +2.7 avg EUR/MWh on the six sent days** (six caches, 2026-10-22; +2.7 on cheap2h in the most recent year, +3.4 on the newest window), almost all of it from D+5 — see [FINDINGS.md](docs/FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b). The headline MAE in [MODEL.md](docs/MODEL.md) and every weather feature's archive-weather A/B gain should be read with that in mind; of the two weather candidates re-scored so far, the northern wind points kept 40–75 % of their gain and the MW-scale residual load none of it on the days production sends.
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

For the tests: `pip install -r requirements-dev.txt`, then `python -m pytest`.

### GitHub Actions

The pipeline is defined in `.github/workflows/daily_predict.yml` (install the pinned requirements → `predict.py`; a failure before `predict.py` is reported to the run-status sensor by its last step) and runs on `workflow_dispatch` — triggered via the GitHub API from a Node-RED flow in Home Assistant (no fixed schedule). Secrets are configured in the repository settings matching the environment variables above. The test suite has its own workflow, `.github/workflows/tests.yml`, on every push and pull request; it needs no secrets.

## Home Assistant Integration

Every run also writes **`sensor.electricity_price_predictions_run`** (`run_status.py`), success or not: state `ok` or `failed`, with `finished_at`, `last_success_at`, `error` (on a failure), `stage` (`workflow` when the run stopped before `predict.py`, at install or tests), `git_sha` and `run_url`. The predictions sensor below is only written by a run that succeeds, so a failed run leaves the previous prediction in place — on purpose: the automation would rather plan on a day-old prediction than on none (it rode out a two-day API outage that way). The Node-RED flow already notices a missing run by waiting for a new predictions state after it dispatches the workflow. This sensor adds why a run failed (`error`, `stage`) and how old the prediction being kept is (`last_success_at`); nothing requires the automation to read it. A run that never reached Python (the dispatch did not arrive, the runner never started) updates nothing. `predict.py` also checks the fetched training data before fitting (`data_checks.py`: wind in km/h, `min ≤ cheap2h ≤ avg ≤ max` on every row, no non-finite value) and fails the run rather than push predictions trained on broken inputs.

The pipeline creates/updates `sensor.electricity_price_predictions` with attributes:
- `predictions_raw` — EUR-to-SEK converted prices, indexed by date (min/avg/max/cheap2h per day, plus `cheap2h_low` / `cheap2h_high`, and `cheap2h_risk` — a probability 0–1, not converted)
- `predictions_with_addon` — prices adjusted by `input_number.electricity_price_addon` (distribution costs etc.) with a 5% markup
- `cheap2h_interval` — the band's own calibration for this run: `coverage_calibrated` vs `nominal_coverage` (the low half), `calm_high_holdout_coverage` (unwarned days, should sit near 0.90) with `calm_high_correction_eur_mwh` / `calm_high_holdout_days` per risk bin, `coupled_level_ratio`, `coupled_share_training` and `warning_threshold`. A drift in the last three is where a new price regime shows first — see [docs/MODEL.md](docs/MODEL.md#the-cheap2h-band)
- `feature_importance_min` / `feature_importance_avg` / `feature_importance_cheap2h` — top features per model (for debugging)
- `eur_to_sek_rate` — SEK per EUR this run converted with, so a logged SEK/kWh value can be turned back into the model's EUR/MWh
- `git_sha` — the commit the run executed (`GITHUB_SHA`); absent on a local run
- `predictions_shadow` — **not for the automation**: arms predicted next to the served one, as `{variant: [rows in the predictions_raw structure]}`. The key is written verbatim into the prediction log's `variant` column, so a new production A/B arm is logged without touching the flow ([how to add one](#adding-a-shadow-arm-to-the-prediction-log)). Today one arm, `blend`: the same model and band fed the hourly mean of three weather models (ECMWF IFS, ICON, GFS) instead of the served forecast ([plan 2.3b](IMPROVEMENT_PLAN.md#23-the-weather-axis-price-it-then-improve-it)). An arm whose computation failed is absent, which never stops the served push; with no arm the attribute is left out

For charging decisions ("charge today or wait for cheaper days"), compare `cheap2h` across days — it is the expected price of a ~2h session picking the day's cheapest hours, which is what the schedule can actually achieve.

`cheap2h_low` / `cheap2h_high` / `cheap2h_risk` are the **warning** next to the ranking, in the same SEK/kWh units as everything else except the risk (a probability). `cheap2h` is always inside `[low, high]`. **`cheap2h_high` has three levels in practice**: when `cheap2h_risk` is 0.3 or more it is the price the day would clear at if SE4 couples to the continent (1.3 × the gas plant's marginal cost, ~150 EUR/MWh in October 2026, the same on every warned day because the gas price is frozen over the forecast); below 0.3 it is a high that about 9 in 10 days at that risk stay under — low on calm days (risk < 0.1), with headroom on uncertain ones (0.1–0.3). So a rule like "do not defer charging to a day whose `cheap2h_high` is above today's known cheap price" fires on the days that can turn expensive and stays quiet on the calm ones; `cheap2h_risk` says how sure, and is the number to use for grading days within a warned week. Over the last year it warned on ~46 % of days, ~69 % of them correctly, and caught ~80 % of the expensive days. Rank days by `cheap2h`, not by `high` — that was tested and is **not** replicated. See [docs/MODEL.md](docs/MODEL.md#the-cheap2h-band).

A Node-RED flow in Home Assistant copies `predictions_raw` (plus the rate and commit above) into the MariaDB table `electricity_price_predictions_log` after every run, one row per target day, so what was served can later be scored against realised prices ([plan 2.2](IMPROVEMENT_PLAN.md#22-production-prediction-log-and-realised-scorecard)). It writes every arm of `predictions_shadow` the same way, under its key as `variant` (served rows are `'served'`); a bad arm is skipped, never the served rows. The flow lives in Home Assistant, not in this repo.

### Adding a shadow arm to the prediction log

To run something in production *next to* the served prediction, without the automation seeing it, and have it logged so it can be scored against realised prices later (a forward A/B):

**Use it sparingly, and only for a question a backtest cannot answer.** A shadow is not free: a forward comparison needs weeks of logged rows on *unchanged* code, so while one is running, every change to what the served or the shadow prediction reads — the feature lists, the model, the weather fetch, the freeze — either waits or restarts its clock, and an arm left running ties up main for as long as it runs. Each arm also adds fetches and run time to every production run. The default instrument is still the A/B harness ([docs/AB_TESTING.md](docs/AB_TESTING.md)); reach for a shadow only when the thing to measure exists only in production (the real forecast at serve time, as for the 2.3b blend), with a written bar and an end date, and delete the arm when its verdict is in.

1. In `predict.main`, compute the arm's predictions in the same structure as `predictions_raw`: `{date: {min, avg, max, cheap2h, cheap2h_low, cheap2h_high, cheap2h_risk}}`, raw SEK/kWh, through `convert_predictions_to_sek` with the run's own `eur_to_sek_rate`, and for the same days as the served prediction.
2. Add it as `predictions_shadow["<name>"] = ...`. That is the whole change: the attribute is pushed and the flow writes one row per day with `variant = '<name>'`, with the run's rate and commit.
3. **The name** is the `variant` value forever, so pick it to be readable in a query a year from now. It must match `ha_client.SHADOW_VARIANT_NAME`: lower-case letters, digits and `_`, starting with a letter, **at most 16 characters** (the column is `VARCHAR(16)` and part of the key), never `served`. A name that does not match is dropped with a warning in the run log; the Node-RED node applies the same rule.
4. **A shadow must never cost the served push.** Wrap its computation so a failure returns `None` and the arm is simply left out (`predict_blend_shadow` is the pattern). A missing arm in a run is a missing row, not an error.
5. **Register it** in the same commit — whoever adds the arm, agent or person, writes the row once: a row in the plan's [shadow register (4.3)](IMPROVEMENT_PLAN.md#43-shadow-arms-in-production) with the `variant`, the item it serves (number and short title), what it predicts, the date, its bar and when it ends. Nothing is filled in later: the arm logs from the first production run after the commit reaches main, and the forward comparison counts from the first row with its `variant` in the log. The item that owns the arm stays in the plan for as long as the arm runs.

Removing an arm is deleting its line in `predict.py` and its register row, in one commit, once its verdict is in the ledgers; its rows stay in the table under their name. Rerunning on the same day replaces each variant's rows for that day, but an arm that failed on the rerun leaves the earlier run's rows in place — the scorer drops a shadow row older than the served row of the same `prediction_date`.

The addon value is fetched live from Home Assistant each run, so distribution cost changes take effect immediately without redeploying.
