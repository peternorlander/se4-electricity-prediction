# The Model

What the model is today: its targets, its configuration, the per-target feature
lists, the components bolted onto `cheap2h`, and the accuracy it currently
measures. This file describes the **current state**. Why it got that way — the
evidence behind each adopted change — is [DECISIONS.md](DECISIONS.md); what was
tried and did not survive is [REJECTED.md](REJECTED.md).

The column catalogue itself is [FEATURES.md](FEATURES.md).

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

## Model Configuration

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
| `max` | `FEATURE_COLUMNS` | 51 |
| `avg` | `AVG_FEATURE_COLUMNS` — `FEATURE_COLUMNS` minus the 9-column price/market lag family | **42** |
| `cheap2h` | `TROUGH_FEATURE_COLUMNS` — plus `neg_price_proba`, appended by the hurdle at fit/serve time | **15** (+1) |
| `min` | `MIN_FEATURE_COLUMNS` — the same list minus `price_se4_max_lag1` | **14** |

The 15 kept features (cheap2h's list):

`max_wind`, `mean_wind_de_north`, `mean_wind_stockholm`, `price_se4_max_lag1`,
`residual_load`, `residual_load_min`, `temp_gradient_se3_se4`, `radiation_variability`,
`ttf_price_lag1`, `gas_marginal_cost`, `reservoir_norway_deviation`,
`reservoir_sweden_gwh`, `reservoir_sweden_change`, `is_workday`, `dow_sin`

**`min` runs 14 of those — everything except `price_se4_max_lag1` (2026-08-06).**
The two targets shared one list for a single day, because cheap2h had borrowed
min's; the first feature audit ever run *on min* then measured that column and got
the opposite answer for each target on the same grid: **REMOVE_HARMFUL for min**
(−0.552 EUR/MWh, favourable in all four evaluation periods, 14 of 16 measurements,
and per-window std *falling* 1.07 on 15 of 16) versus **KEEP_SCENARIO for cheap2h**
(−0.439 on average, but one period genuinely positive, and not from a single freak
window). Corroborated on a second data vintage: −0.241 (7 of 8) for min, −0.319
(7 of 8) for cheap2h, on the four independently fetched caches.

Note what this leaves: **`min` now has no electricity-price feature at all** — it
is purely weather / fuel / hydro / calendar. That is the logical end of a result
this repo has reproduced three times (a target's own frozen price lag hurts it),
and it is a fair thing to be uneasy about — the model has no idea what power
currently costs.

**This was tested and closed 2026-08-21 (round 18).** Every price feature
rejected for these targets is in `FORECAST_FROZEN_FEATURES`, so
`apply_forecast_freeze` pins it at its last known value for the whole 7-day
window — so the obvious hypothesis was that what's harmful is **a stale scalar
the model treats as current**, not price information as such (note that min
still keeps `ttf_price_lag1` and `gas_marginal_cost`, which are also lagged
prices, and leave-one-out says keep both). Three candidates tested this
directly: `price_vs_30d` (a stationary ratio rather than a level),
`days_since_price_anchor` alongside the anchor (explicitly tells the model how
stale it is), and the neighbouring-zone lags `price_de_lag1` / `price_dk2_lag1`
alone. **All three were harmful, on both trough targets, in every evaluation
period, confirmed on three independent measurements** (the confound-free
sliding grid, a 5-cache vintage ladder, and a fresh 5-year snapshot) — see
[Features Tested and Rejected](REJECTED.md). The staleness
hypothesis is specifically refuted: `days_since_price_anchor`, the arm that
tells the model exactly how stale the anchor is, was the *worst* of the three
on `min`. Reading: it is price-*level* information these targets reject, not
its freshness — they are trough targets driven by weather → residual load, and
any price level crowds that out, however it's encoded. **Closed — do not
re-open without a mechanism that is not about staleness.**

**How these lists were arrived at** — the 2026-07/08 per-target ablation
program, `min` adopting its prune first, `cheap2h` borrowing that list and
winning on it out-of-sample, and the audit that closed both directions — is
recorded in [DECISIONS.md](DECISIONS.md#per-target-feature-lists-for-min-and-cheap2h).
`avg`'s 42-column list comes from a separate block-level re-audit, recorded in
[DECISIONS.md](DECISIONS.md#avg-drops-the-pricemarket-lag-family-round-14b).

**One property this does not fix:** min and cheap2h are still independent models, so
their predictions can violate `min ≤ cheap2h` — measured at 31 of 60 held-out days
after the change, versus 29 of 60 before, though the worst violation shrank from
+27.9 to +8.9 EUR/MWh. The coherence clamp that would enforce it was tested and
made cheap2h *worse* (see [Features Tested and Rejected](REJECTED.md)).

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
didn't clear the strict A/B bar, and a 2026-08 re-test on the confound-free
four-period grid came back NOISE again, so it stays on a plain regressor.

### Prediction Interval (cheap2h only)

`model.train_interval` fits two `reg:quantileerror` regressors (α = 0.10 / 0.90)
on cheap2h's own feature list and hyperparameters, then applies split conformal
calibration on a 180-day holdout, so `predict()` can return `cheap2h_low` /
`cheap2h_high` — an 80 % band around that day's own point estimate. It is an
addition: `model.predict()` without an `interval` argument returns exactly what
it always did, so `evaluate.py` and the A/B harness are untouched.

The band is **information and a guard rail, never a gate** — MAE cannot measure
it, and ranking days by `q90` instead of the point estimate was tested and
rejected. Full record, including the serve-time ordering rules and what the band
said during the 2026-08-17 break:
[DECISIONS.md](DECISIONS.md#prediction-interval-for-cheap2h-round-19c).

## Current MAE Baseline

Evaluation uses **52-window (1-year) walk-forward validation** on 3 years of training data. Each target is reported **individually** — min, avg, max and cheap2h have different physical drivers, so a blended metric would hide the per-target movement that matters when tuning. `cheap2h` and `min` are the numbers to watch for scheduling; `max` is not a priority.

The evaluation is **horizon-honest**: each test window's price/market/fuel/reservoir lags are frozen to their last-known value, exactly as `build_forecast_features` does in production for all 8 forecast days. A validation that instead fed each test day its *true* previous-day price as `lag1` — knowledge the live model only has for day+1 — would make days 2–8 look far more accurate than they are. This baseline is the accuracy Home Assistant actually receives.

Current, measured on the **2026-08-21** data snapshot with the production code as
it stands (negative-price hurdle on cheap2h; `cheap2h` on the pruned 15-feature
trough list, `min` on the 14-column `MIN_FEATURE_COLUMNS` since 2026-08-06; `avg`
on the 42-column `AVG_FEATURE_COLUMNS` since 2026-08-21):

| Target  | MAE (EUR/MWh) | Std |
|---------|--------------|------|
| min     | 14.59        | ±6.77 |
| avg     | 18.88        | ±7.94 |
| max     | 37.40        | ±19.94 |
| cheap2h | **15.04**    | ±6.17 |

**These numbers are measured in the `gap0` condition** — the evaluator fits right
up to its test window, which it can only do because the whole evaluation year
sits in one contiguous frame. Production cannot: its frame ends where the
weather source ends. Until 2026-08-23 that was ~5 days back, making this table
optimistic by **+0.37 (cheap2h) / +0.59 (min) / +1.70 (avg)**. Since the
[archive-lag top-up](DECISIONS.md#closing-the-weather-archive-lag-round-19a) production fits
to yesterday, and the remaining optimism is the `gap1` row below:

| target | eval overstated by, before (gap5) | after (gap1) |
|---|---|---|
| cheap2h | +0.372 | **+0.155** |
| min | +0.586 | **+0.258** |
| avg | +1.695 | **+0.591** |

So the fix moved the deployed model *toward* this table rather than changing the
table. The table itself has not been re-measured since.

Measured with `OMP_NUM_THREADS=4`. **Pin the thread count before comparing this
table with anything.** XGBoost's floating-point reduction order depends on it, so
the same unchanged model lands a few hundredths apart at 4 vs 16 threads — which
is the size of a small real effect.

Each of the last three adoptions was checked the same way: re-measure the *same*
snapshot before and after, and require the untouched targets to reproduce exactly.

- **`min` dropping `price_se4_max_lag1` (2026-08-06):** min 15.680 → 15.516
  (−0.165) while `avg`, `max` and `cheap2h` were **bit-identical per window**.
- **cheap2h adopting the trough list (2026-08-05):** 16.85 → 15.94 (−0.91), the
  other three unchanged.
- **avg dropping the 9-column price/market family (2026-08-21):** 18.81 → 18.88
  (**+0.07**, essentially flat) while `min`, `max` and `cheap2h` were
  **bit-identical**. Unlike the two rows above, this single snapshot happened to
  land on a *calm* evaluation period for the mechanism this change targets (see
  [Per-Target Feature Sets](#per-target-feature-sets)) — a reminder that a
  same-snapshot before/after only checks the wiring, never the evidence.

That exact reproduction of the untouched targets is what makes such a before/after
meaningful — same snapshot, same windows, same code path, one thing changed. It is
**not** the evidence for any of these changes (that is the A/B: −0.55, −0.86 and
−1.2 to −2.0 respectively); it is a consistency check on the wiring. Note also how
much a single period can misstate the A/B in *either* direction: min's −0.165 here
against −0.55 pooled over four evaluation periods (understated), and avg's +0.07
here against a clmean of −1.22 to −1.97 on the two independent 5-year fetches
(reversed in sign) — the avg case is the sharper warning, since a single-snapshot
reading would have looked like a regression.

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
| `min` pruned feature list | **−0.64 EUR/MWh** | 16 measurements, **4 evaluation periods**, all negative. (The originally recorded −1.01 came from 14 measurements taken partly on the list's own selection windows; this is the confound-free re-measurement.) |
| `min` drops `price_se4_max_lag1` | **−0.55 EUR/MWh** | 16 measurements, **4 evaluation periods**, all negative; per-window std also −1.07. Corroborated at −0.24 (7 of 8) on a second data vintage |
| `cheap2h` pruned feature list | **−0.86 EUR/MWh** | 16 measurements, **3 evaluation periods**, all negative (after 17/17 in an earlier round) |
| Negative-price hurdle (cheap2h) | **−0.44 EUR/MWh** | 6 measurements, 1 snapshot, all negative |
| `avg` drops the 9-column price/market family | **−1.2 to −1.97 EUR/MWh** (clmean) | 32 measurements, **4 evaluation periods**, on **two independently-fetched 5-year snapshots** 15 days apart — all 4 clusters negative both times. Magnitude is regime-dependent (see [Per-Target Feature Sets](#per-target-feature-sets)), so read this as a range, not a point estimate |

See [How changes are validated](AB_TESTING.md#how-changes-are-validated) for why this project
reports it that way.

**Per-horizon MAE (`mae_by_horizon`) is weekday-confounded — do not read it as pure horizon decay.** The walk-forward steps by 7 days with 7-day windows, so horizon ≡ weekday (d+1 is always the same weekday as the window start). The curve mixes horizon and day-of-week and is non-monotonic. The clean measure of horizon/lag-staleness cost is the anchor-staleness sensitivity reported by the walk-forward (~1 EUR, see [DECISIONS.md](DECISIONS.md#price-lag-anchor-freshening)), **not** the per-horizon spread. This is why horizon-aware modeling (a `forecast_horizon` feature / per-horizon models) was evaluated and **shelved**: the true stale-lag headroom is ~1 EUR, and the bulk of the ~17 EUR error is regime-driven (winter cold-snap volatility, spring solar/negative-price ramp), not horizon-driven.

Feature importance is reported for **min, avg and cheap2h models only** — including max would dilute the signal for what actually matters for scheduling decisions.
