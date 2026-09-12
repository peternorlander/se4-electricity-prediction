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
[archive-lag top-up](#closing-the-weather-archive-lag-round-19a) production fits
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

See [How changes are validated](#how-changes-are-validated) for why this project
reports it that way.

**Intraday trough features (2026-07):** the daily min/cheap2h is set at a specific intraday trough (overnight wind or midday solar) that the daily-mean `residual_load` diluted. Exposing the trough directly — chiefly `residual_load_min` (daily minimum of hourly residual load) — cut cheap2h ~0.66 EUR/MWh and held min, with `residual_load_min` landing as a top-4 feature in all three priority models. See the [Intraday Trough Features](#intraday-trough-features) section.

**Price-lag anchor freshening (2026-07):** the SE4 price lags — the dominant min/cheap2h features — are now frozen at the freshest *known* ENTSO-E price (`se4_prices_daily`), not the training frame's last row. The training frame ends ~`WEATHER_ARCHIVE_LAG_DAYS` behind because it inner-joins prices with the lagging weather archive, so production was previously anchoring the price lags ~5 days stale (the DE/DK2 lags were already fresh — this closes the same gap for SE4's own lags). The walk-forward reports an **anchor-staleness sensitivity** (fresh d0 vs stale d5 ≈ old pipeline). Measured saving from freshening: **~1.1 EUR/MWh (min/cheap2h), ~2.3 (avg)** — real and free, but modest, because yesterday's min and six-days-ago min are similar.

**Closing the weather-archive lag (2026-08-23):** the Open-Meteo *archive* only
publishes to about today−5, so `build_training_data` produced a frame ending
~5 days back and **the model was fitted that far behind the first forecast day**.
The price-lag freshening above fixed the *anchor*; the *fit* stayed stale, and
that had never been measured because `evaluate.walk_forward_validate` fits right
up to its test window — it models the production anchor faithfully and the
production training tail not at all. `fetch_data` now tops the archive up from
Open-Meteo's forecast endpoint (`past_days`, `sources.open_meteo.fetch_recent` /
`fetch_international_wind_recent`) so the frame reaches **yesterday**. See
[Closing the weather-archive lag](#closing-the-weather-archive-lag-round-19a)
for the measurement, the two splice rules that make it safe, and what it does
*not* fix.

**Negative-price hurdle (2026-07-24/25, cheap2h only):** an XGBoost classifier predicts P(tomorrow's `price_min` < 0 EUR/MWh) — a distinct physical regime (renewable oversupply + low/weekend demand) a plain regressor otherwise has to infer implicitly from the same weather/calendar features. Its out-of-fold probability feeds into the `cheap2h` regressor as an extra feature (`neg_price_proba`, now the #1 feature for cheap2h at ~0.25 importance). Validated via `ab_test.py` on a real cached snapshot, shifts 0–5: **REAL, all 6 shifts improved, mean −0.44 EUR/MWh** — the largest, cleanest single-run win recorded for the top-priority target; local reproduction via `evaluate.walk_forward_validate` matched the A/B exactly (14.74 vs the same run's min/avg/max, which were unaffected). **Confirmed on a real Actions run and committed 2026-07-25**; the table above now includes it. `min` showed the same direction on 5 of 6 shifts but didn't clear the strict sign-consistency bar (one near-zero flip) — **not productionized for `min`** — re-tested on the confound-free four-period grid in 2026-08 and rejected again (NOISE: −0.10 on average but *worse* in the period closest to production, and min's own data refutes the obvious explanation — that period has the most negative-price days to learn from, not the fewest).

**Per-horizon MAE (`mae_by_horizon`) is weekday-confounded — do not read it as pure horizon decay.** The walk-forward steps by 7 days with 7-day windows, so horizon ≡ weekday (d+1 is always the same weekday as the window start). The curve mixes horizon and day-of-week and is non-monotonic. The clean measure of horizon/lag-staleness cost is the anchor-staleness sensitivity above (~1 EUR), **not** the per-horizon spread. This is why horizon-aware modeling (a `forecast_horizon` feature / per-horizon models) was evaluated and **shelved**: the true stale-lag headroom is ~1 EUR, and the bulk of the ~17 EUR error is regime-driven (winter cold-snap volatility, spring solar/negative-price ramp), not horizon-driven.

Feature importance is reported for **min, avg and cheap2h models only** — including max would dilute the signal for what actually matters for scheduling decisions.

## The 2026-08-17 regime break

The motivating case for round 19 and for the cross-border work of round 21.
Worth keeping because it is
the clearest evidence this project has for what the model structurally *cannot*
do, and because the obvious fixes were all tested and all failed.

### What happened

| date | avg | cheap2h | cheap2h/avg |
|---|---|---|---|
| 2026-08-16 | 20.1 | 6.0 | 0.30 |
| 2026-08-17 | 138.1 | 20.2 | 0.15 |
| 2026-08-18 | 163.7 | **127.9** | **0.78** |
| 2026-08-19 | 164.0 | **128.0** | **0.78** |
| 2026-08-20 | 142.8 | 81.2 | 0.57 |

`cheap2h = 128.0` is the **all-time maximum of the 3-year training window**
(n = 1096, median 13.1, p99 100.5). The previous August record across three
Augusts was 85.7.

**The shape broke, not just the level.** `cheap2h/avg` = 0.78 against a 3-year
median of 0.28 — there was no cheap window at all — and the trough moved from
night to midday: on 08-16 the cheap hours were 00–15, on 08-18 they were 13–14
(solar) and the night was the expensive part. A night-charging automation would
have paid ~160 instead of 128. Days with `cheap2h/avg > 0.70` **and** `avg > 100`
number 57 of 1096 and are almost all Nov–Feb.

### What caused it

SE4 re-coupled to the continent: the |SE4 − DE| gap collapsed from ~90 to ~15
EUR/MWh and the 30-day correlation went 0.33 → 0.79. 2026-07-25 → 08-11 had been
an 18-day run with SE4 − DE < −50 (mean −85.5), the longest in the whole history.

**It was not Baltic Cable returning**, which is what the press and an early read
of this data suggested. ENTSO-E A78 carries a forced outage on that link,
`available_mw = 0`, reason *"Trip of the BC-link"*, 2026-06-21 → 2026-09-18, and
the DE_LU physical flow was 0.0 MW every day from 2026-07-01 through 08-22. The
real mechanism has two layers:

* **Background:** crippled export capacity kept SE4's surplus trapped all summer
  (total exports ~780–1040 MW in Jul/Aug against 1400–2200 across Oct–Jan), which is
  why it ran so cheap and so decoupled.
* **Trigger:** the surplus vanished. `mean_wind_stockholm` — the trough models'
  most load-bearing feature — fell from the 70th percentile (08-16) to the
  **3.9th** (08-18) and **1.7th** (08-20). SE4 flipped from exporting to
  importing ~1960 MW from SE3 and repriced to import parity.

So the model *has* the wind. The obvious reading was that what it lacks is the
capacity state the wind has to be conditioned on. **That was tested in round 21
and it is not the answer** — see
[Cross-border capacity and flows](#cross-border-capacity-and-flows-round-21).
Capacity × calm fired on exactly the right days and changed nothing, because the
same state in September 2024 came with a cheap2h of 6.6: the conditioner that
separates the two is the northern supply balance, not the interconnector.

### What the model predicted, and why it is structural

Replaying the forecast production actually issued each morning (fit ≤ D−5, i.e.
under the archive lag), cheap2h at d+1:

| date | actual | predicted |
|---|---|---|
| 2026-08-17 | 20.2 | 20.4 |
| 2026-08-18 | **127.9** | 34.4 |
| 2026-08-19 | **128.0** | 30.5 |
| 2026-08-20 | 81.2 | 40.8 |

MAE 4.32 before the break, **57.75** from 08-17 on, bias −55.5. The highest
number production issued anywhere that week was **49.1**.

Four reasons this is not a tuning failure. Each was measured, and together they
close the obvious routes:

1. **No feature carries the cause.** `TROUGH_FEATURE_COLUMNS` has no continental
   price feature at all, and nothing anywhere encodes transmission capacity.
2. **Adding the continental price back does not fix it.** Arms on the same
   slices: production 52.84 spike MAE, `+price_de_lag1/price_dk2_lag1` 50.84,
   `+coupling gap` 51.63, and an **oracle given the same-day German price**
   (impossible in production) 51.30 — predicting **41.6** for a realised 127.9.
   DE sat at 100–170 through the entire cheap fortnight; only the coupling
   changed, so the price was never the missing information.
3. **The model cannot output the number.** XGBoost interpolates; it cannot
   extrapolate past its training targets. Fitted in-sample on the full window,
   cheap2h's maximum output is **108.8** all-year and **95.2** on May–Sep rows,
   against a realised 127.9. **No feature set makes 128 reachable from a 3-year
   window.** This is the single most durable constraint on future work here.
4. **Reformulating the target does not rescue it.** Predicting `cheap2h − avg`
   or `cheap2h/avg` and reconstructing gives 51.56 / 51.89 against 52.84 — the
   *shape* (0.78) is as unprecedented as the level, so decomposing just moves
   the out-of-distribution problem, and it costs 6.7 EUR/MWh of pre-break
   accuracy.

A 5-year training window raises the ceiling (cheap2h targets top out at 450.2
instead of 107.9) and does help on the break — 48.52 against 52.84 — but costs
three times the pre-break accuracy (7.98 against 2.53). Regime insurance with a
real premium, consistent with round 15's closure of window length rather than a
challenge to it.

## Closing the weather-archive lag (round 19a)

Adopted 2026-08-23. Measured with local experiment scripts under `experiments/`,
which is **not committed** — the numbers below are the record.

`fetch_data.WEATHER_ARCHIVE_LAG_DAYS = 5` — the Open-Meteo archive publishes to
about today−5, so the merged frame ended ~5 days back and **the model was fitted
five days behind the first forecast day**. The
[price-lag freshening](#current-mae-baseline) had fixed the *anchor* in 2026-07;
the *fit* was never addressed, and the gap was invisible because
`evaluate.walk_forward_validate` fits right up to its test window — it models
production's anchor faithfully and production's training tail not at all.

`fetch_data._splice_recent` now tops the archive up from Open-Meteo's forecast
endpoint (`sources.open_meteo.fetch_recent` / `fetch_international_wind_recent`,
`past_days`) so the frame reaches yesterday. **Two rules in that function are
load-bearing** — the archive wins wherever it has published, and the result is
cut at Swedish midnight so the newest day is never a partial one. Both are
explained where they are enforced; read the docstring before touching it.

Measured cost of the lag, on the round-15b sliding grid (four period clusters,
`min_train` constant), replicated on 28 measurements across seven independently
fetched caches:

| target | eval overstated production by, before (gap5) | after the fix (gap1) |
|---|---|---|
| `cheap2h` | +0.372 | **+0.155** |
| `min` | +0.586 | **+0.258** |
| `avg` | +1.695 | **+0.591** |

Both are REAL in all four clusters, and monotone in the gap (gap1 < gap3 < gap5
on every target — the dose-response a real mechanism shows and noise does not).
**The residual `gap1` column is the number that still matters**: the headline
baselines remain optimistic by that much, because the eval fits one day closer
than production ever can. Note the first day carries 35–44% of the whole
five-day cost — recency is steeply non-linear, which is what makes reaching
*yesterday* rather than today−2 worth the Swedish-midnight cut.

**The mechanism is the fit, not stale features.** A `gap5_fitonly` arm (fit
short, rolling anchors fresh) reproduces `gap5`; a `gap5_rollonly` arm (fit
full, only `wind_variability` / `radiation_variability` five days staler) is ~0
and NOISE on both priority targets. Not having the last five days of *targets*
costs everything; freezing the rolling regime signals earlier costs nothing.
That is why the fix extends the frame rather than re-seeding anchors.

**Replication.** The grid above varies period on one snapshot. Re-run varying
the *fetch*: all five 3-year caches at shifts 0–1 (**10/10 positive on every
target**, mean +0.820 cheap2h / +1.005 min / +2.226 avg) and a second 5-year
snapshot fetched 15 days earlier (4/4 positive, +0.573 / +0.668 / +1.632).
Pooled over all three grids — **28 measurements on seven independently fetched
caches** — positive on 25/28 (cheap2h), 26/28 (min), 28/28 (avg). The ladder
runs about double the clustered mean because every ladder point sits in the NOW
period, which the primary grid also puts at +0.777 / +1.053 / +2.075; read
+0.8 / +1.0 / +2.2 as the current-regime effect.

**What it did on the break week.** Production as it actually ran (fit ≤ D−5)
against the gap-closed condition, cheap2h at d+1:

| date | actual | production (gap5) | gap closed |
|---|---|---|---|
| 2026-08-18 | 127.9 | 34.4 | 42.7 |
| 2026-08-19 | 128.0 | 30.5 | 53.7 |
| **2026-08-20** | **81.2** | **40.8** | **104.6** |
| 2026-08-21 | 62.3 | 34.3 | 60.5 |

The 08-20 row is the finding in one line. By then the break was three days old
and both 08-18 and 08-19 had cleared at 128 — but production had not been
*fitted* on either, because the archive had not published their weather. The
chase lag is not one day, it is closer to six.

Caveat carried forward: those figures were measured with archive weather on both
sides of the comparison. Production fills the gap with the forecast product
instead, which carries its own error against the archive (windspeed matched to
0.000 MAE at four of five locations, temperature +0.75 °C, radiation ~6–17%), so
the realised gain is smaller than the measured cost by an amount nobody has
measured. The clean confirmation — a candidate-vs-baseline A/B whose candidate
frame is built from spliced weather — is still outstanding.

## Prediction interval for cheap2h (round 19c)

Adopted 2026-09-05. Measured with local experiment scripts under
`experiments/`, which is **not committed** — the numbers below are the record.

`cheap2h_low` / `cheap2h_high` are a **conformally calibrated 80% band around
each day's own `cheap2h` prediction**, from two `reg:quantileerror` regressors
(α = 0.10 / 0.90) on the same feature list and hyperparameters as the point
model. This is an addition: `model.predict()` without an `interval` argument
returns exactly what it always did, so `evaluate.py` and the A/B harness are
untouched.

Why it exists: a point estimate cannot say "I don't know". On 2026-08-18 the
model predicted 42.7 against a realised 127.9 with the same outward confidence
it predicts 6.0 with on a calm day. The band can — and its width is informative,
not decorative: the point estimate's MAE runs 7.26 → 21.50 across band-width
quartiles.

The **raw** band is over-confident (a nominal 80% interval covers 54%), so
`model.train_interval` applies split conformal on a 180-day holdout and refits on
all data. Measured 0.542 → 0.788 pooled, in every period cluster. Serve-time
order is sort → widen → clamp open around the point estimate; each step has a
measured reason, documented on `model.IntervalModel`.

**What the band said on the break week.** cheap2h q90 at d+1: −1.5 (08-16) →
34.4 (08-17) → 70.2 (08-18) → 78.1 (08-19) → 116.4 (08-20). As a *level* it
still badly understates 127.9 — the ceiling in
[The 2026-08-17 regime break](#the-2026-08-17-regime-break) binds the quantile
models too. As an *alarm* it is far more legible than the point estimate's
1.7 → 16.5 → 42.7, and it is the honest reason to ship the band: not that it
predicts a break, but that it stops claiming confidence it does not have.
Coverage during the break was 0.14, against 0.83 in the days before it.

### How this interacts with A/B testing

- **cheap2h MAE stays the gate** for feature and model changes. The quantile
  models are fitted on `TARGETS["cheap2h"]`'s feature list, so anything that
  improves the point model's inputs improves theirs.
- **MAE cannot measure the band, ever.** It only looks at one number per day.
  The conformal calibration moved coverage 0.594 → 0.806 and MAE by *exactly
  zero*. So the band gets a **guard rail, not a gate**: every run pushes
  `cheap2h_interval` with `coverage_calibrated`; after adopting any change,
  check it is still near `nominal_coverage`.
- **Changing the interval itself** — quantile levels, conformal window or α,
  separate hyperparameters — is not measurable by MAE at all. Score it on
  **pinball loss and coverage**, **per period cluster**, never pooled.

## Cross-border capacity and flows (round 21)

Closed 2026-09-12. This was IMPROVEMENT_PLAN item 0, the last route the
[2026-08-17 break](#the-2026-08-17-regime-break) left open: ENTSO-E physical
flows (A11) and transmission unavailability (A78) for the five SE4 borders,
cached under `ab_cache/crossborder/`. Scripts and raw per-window results:
`experiments/run_round21_*.py`, `experiments/results/round21_*.jsonl`, arms and
bar fixed in advance in `experiments/ROUND21_PREREGISTRATION.md`.

**For `cheap2h` and `min` the route is closed.** Round-15b sliding grid, 14
points, four period clusters, `classify_clustered`. Every arm — realised SE3
import, available southbound export capacity, export headroom, and the
pre-registered primary `capacity × calm-in-Stockholm` interaction — is NOISE on
both targets; export headroom is REAL-and-*harmful* on cheap2h (+0.144, 2 of 14
points favourable). The decisive arm is the control: a **leaky** per-day
capacity built from outage records production could not have had is NOISE too
(+0.033 / −0.050), so even perfect foreknowledge of interconnector capacity has
no year-average value for the trough targets. On the break week itself no arm
lifts the 08-18/19 day-ahead prediction above ~48 against a realised 128.

**One open candidate, on `avg`:** `export_headroom_lag1` — A78-available
southbound export capacity minus realised net southbound export, on the run
day, frozen across the horizon.

| snapshot | clmean | NOW | −6M | −12M | −21M | favourable | Δstd |
|---|---|---|---|---|---|---|---|
| `long/2026-08-21` | **−0.423** | −0.785 | −0.329 | −0.133 | −0.447 | 11/14 | −0.209 |
| `long/2026-08-06` (independent fetch) | **−0.218** | −0.524 | −0.144 | −0.056 | −0.148 | 11/14 | −0.266 |

REAL on both, every period cluster favourable both times, per-window std
*falling*. Three checks behind it: the A11-only encodings (export flow alone,
capacity inferred from whether a link flowed) are NOISE, so the A78 capacity
half is what carries it; the round-19e price-derived congestion feature is
**not** a substitute (REAL on one snapshot with std +0.23, NOISE on the other,
and adding it on top of headroom destroys the gain — round 18 again); and a
vintage-honest rebuild, per window from only the records published by that run
date, stays favourable on all four NOW points (−0.17 to −0.54). Read the effect
as **−0.2 to −0.4 EUR/MWh on `avg`**. The gain is concentrated where the avg
price/market prune's was: windows with Baltic Cable closed improve −1.14
against −0.23 elsewhere, the worst-MAE quartile −1.18, the calmest quartile
+0.46.

**Not adopted.** `avg` was not the pre-registered target and ~12 arms were tried
on it, so this is a screen that replicated, not a verdict. What it still needs:
the serve-time encoding (production knows tomorrow's *scheduled* exchanges and
the outages posted for it, not realised physical flows), a fresh cross-border
fetch with the fixed parser, and a re-check after the residual-load rebuild,
since `residual_load` is in avg's list and
[verdicts are scoped to the model](#how-changes-are-validated).

### If you touch `ab_cache/crossborder/`, read this first

Four properties of A78 that cost a day to find. The fetch script's parser was
fixed on 2026-09-12; `experiments/round21_xbfeat.py` re-parses `raw/` for caches
fetched before that.

* **`docStatus = A09` means cancelled.** 632 of 4487 period rows in the
  2026-08-23 cache are cancellations of outages that never happened, and they
  were ~64% of that cache's "link down" days. Filter them.
* **There is no vintage before 2025-11.** The API returns only each record's
  current revision (1795 of 2428 are rev ≥ 2), and *every* record for an event
  before November 2025 was re-published that month. So "what was knowable on
  date D" cannot be reconstructed for the older clusters at all, and a
  forward-looking A78 feature cannot be backtested honestly on this data. Nord
  Pool's UMM API (`publicationDate` + `version`) or daily A78 snapshots are the
  only ways to change that.
* **DK2 rows describe one 400 kV cable, not the border.** 69 days show
  `available_mw = 0` while the border keeps flowing. Never min() them into a
  border capacity; DK2 was held at nominal in every round-21 arm.
* **A78 and A11 do agree, once the cancellations are gone** — on the HVDC links
  the element *is* the border: Baltic Cable 135 of 135 outage days show zero
  flow, PL 202 of 218, LT 45 of 57. And A78 sees what flows cannot: partial
  reductions, such as Baltic Cable at 210 MW from 2025-10-15 to 2026-02-02.

One correction to the record: the DE_LU outage running 2026-08-17 → 11-08 that
IMPROVEMENT_PLAN flagged as a suspected revision is Breared–Söderåsen, an
SE4-internal line, and it is **cancelled**. The Baltic Cable trip record is
revision 3, created 2026-08-17 10:32 — its end date was posted on the break day.

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
| [Open-Meteo](https://open-meteo.com) | Hourly weather archive (to ~today−5), recent-analysis top-up for the days the archive has not reached, and an 8-day forecast — SE4 (Malmö) and 5 regional locations | None |
| [Yahoo Finance](https://finance.yahoo.com) via `yfinance` | TTF natural gas futures (`TTF=F`), EU ETS carbon allowances (`CO2.L`) | None |
| [NVE](https://biapi.nve.no/magasinstatistikk) | Norwegian hydro reservoir fill levels + 20-year min/max/median by week | None |
| [Nordpool](https://www.nordpoolgroup.com) | Published SE4 prices (used to exclude already-known days from predictions) | None |

## Features (51 engineered + 1 cheap2h-specific)

The catalogue below is the full engineered set (`FEATURE_COLUMNS`), used in its
entirety only by the **max** model. **Not every model uses every
feature** — see [Per-Target Feature Sets](#per-target-feature-sets):

| Model | Features used |
|-------|---------------|
| max | all 51 |
| **avg** | **a validated 42-feature subset** (`AVG_FEATURE_COLUMNS`) — `FEATURE_COLUMNS` minus the 9-column price/market lag family |
| **cheap2h** | **a validated 15-feature subset** (`TROUGH_FEATURE_COLUMNS`), plus `neg_price_proba` at fit/serve |
| **min** | **the same list minus `price_se4_max_lag1`** (`MIN_FEATURE_COLUMNS`, 14) |

(cheap2h's model also gets a `neg_price_proba` input computed at fit/serve time by
the negative-price hurdle classifier — see [Negative-Price Hurdle](#negative-price-hurdle-cheap2h-only).
It's not a column in `FEATURE_COLUMNS` below, since it's model-internal rather
than fetched/engineered by `features.py`.)

Note that **36 of the 51 columns below are not used by the two priority targets**
(37 for `min`), and 9 of them are also not used by `avg`. They are not dead code —
`max` uses all of them — but if you are reading this to understand what drives the
cheap-price or average-price forecast, the lists in
[Per-Target Feature Sets](#per-target-feature-sets) are what matters.

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
  [Features Tested and Rejected](#features-tested-and-rejected))
- Only lag-1 is valid: day-ahead auction clears all zones simultaneously

### SE4 Own Price Lags (autoregressive)
- `price_se4_avg_lag1` — was avg's strongest feature (~0.24 importance) until 2026-08-21.
  **No longer used by `avg`** — dropped along with the rest of the price/market family
  (see [Per-Target Feature Sets](#per-target-feature-sets)). **Used only by `max`** now.
- `price_se4_avg_lag2`, `price_se4_avg_lag7` — momentum and weekly seasonality. Same
  status: **used only by `max`** since 2026-08-21.
- `price_se4_min_lag1` — yesterday's min; historically the highest-importance feature for
  both min and cheap2h (~0.23–0.25). **No longer used by `min`, `cheap2h` or `avg`** —
  ablation testing showed removing it improves the trough targets (importance reflects
  in-sample usage, not marginal value, and the lag is frozen stale across each forecast
  window anyway), and the same family-level finding closed it out of `avg` too. **Used
  only by `max`** now. See [Per-Target Feature Sets](#per-target-feature-sets).
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
[Features Tested and Rejected](#features-tested-and-rejected). The staleness
hypothesis is specifically refuted: `days_since_price_anchor`, the arm that
tells the model exactly how stale the anchor is, was the *worst* of the three
on `min`. Reading: it is price-*level* information these targets reject, not
its freshness — they are trough targets driven by weather → residual load, and
any price level crowds that out, however it's encoded. **Closed — do not
re-open without a mechanism that is not about staleness.**

**How this list was arrived at (2026-07 → 2026-08).** A systematic per-target
ablation program tested every one of the 51 columns against all four targets, then
tested combined drop sets per target. `min` adopted the result first: **−1.01
EUR/MWh pooled across 14 measurements spanning four independently-fetched
snapshots**, every measurement negative, per-window std *decreasing* (−0.19).

*Correction to that number (2026-08-06).* Those 14 measurements were taken partly
on the same windows the list was selected from, so −1.01 is optimistic. Re-measured
on a confound-free grid spanning four evaluation periods — including one with zero
calendar overlap with the selection data — the prune is worth **+0.635 EUR/MWh**
(i.e. the 51-column list is that much worse), positive in every period but smallest
in the zero-overlap one (+0.345). The prune is confirmed; the magnitude was
inflated by selection, as the provenance predicted.

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

**`avg` (2026-08-21): the same family-of-frozen-lags finding, on a fourth
target.** `avg` was declared closed after the 2026-07/08 program on a
single-column leave-one-out — dropping `price_se4_avg_lag1` alone is absorbed
by `price_se4_avg_lag2`/`lag7`/`price_de_lag1`/`price_dk2_lag1`, so a
single-column test reads near-zero *whether or not the family as a whole is
load-bearing*. A block-level re-audit (dropping all 9 price/market columns as
a unit) found it **REMOVE_HARMFUL**, replicated on two independently-fetched
5-year snapshots 15 days apart: clmean **−1.97** and **−1.22 EUR/MWh**, all
four evaluation periods negative both times (12 to 16 of 16 measurements
favourable). Physical control blocks (wind, calendar, fuel/carbon) reproduced
almost to the decimal across both fetches, which corroborates the harness
itself and not just this one finding.

The mechanism is quantified, not just directional: across 879 deduplicated
calendar weeks, **corr(baseline MAE, delta) = −0.72**. The frozen
`price_se4_avg_lag1` anchor — previously avg's #1 feature at ~0.24 importance
— is a decent predictor in a stable price regime and catastrophically wrong
during a regime shift, because `apply_forecast_freeze` pins it at its
last-known value for the entire 7-day forecast horizon. Weeks where the old
model's MAE exceeds 40 EUR/MWh (3.6% of weeks) improve by −24.1 EUR/MWh on
average; calm weeks (96.4%) still improve on median, just by much less
(−0.76 mean). **This means the realized MAE delta will visibly vary between
production runs** depending on how many regime-shift weeks the evaluation
window happens to contain — a single-snapshot before/after landed at +0.07
(see [Current MAE Baseline](#current-mae-baseline)) purely because that one
window was calm; that is expected, not a contradiction of the A/B.

Full evidence trail: the run scripts and raw per-window results under
[experiments/](experiments/).

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

## A/B Backtest Flow

`ab_test.py` is the tool for evaluating any candidate feature or model change before
committing it. It exists because MAE effects on the priority targets are small
(often < 0.3 EUR/MWh) and **period-dependent**: the same change can look like an
improvement on one day's data and a regression on the next. Historically that forced
re-running an experiment across several real calendar days to tell signal from noise
— one change every few days. The A/B flow simulates those different run-days from a
single data snapshot, so a verdict takes one sitting.

### Snapshot generations — check the weather tail before building a grid

`python ab_test.py list` reports a **weather tail** per snapshot: how many days
behind its own `today` the weather stops. `5d` means archive-only (fetched
before the 2026-08-23
[archive-lag top-up](#closing-the-weather-archive-lag-round-19a)); `1d` means
topped up to yesterday. `*` marks a value derived on read rather than recorded
at save time — same number, so snapshots predating the field need no migration.

**Snapshots with different tails must not share one measurement grid.** A 1d
frame is ~4 rows longer, and the shift arithmetic derives its windows from the
end of the frame, so a grid spanning both would silently compare different
windows. A vintage ladder mixing generations varies the fetch *and* the tail at
once. Grid scripts that assert a row count will trip on the first post-top-up
fetch — that assertion is doing its job; update the constant, do not delete it.

**Do not delete or backfill old snapshots.** An A/B is paired, so the tail is
shared by both arms and cancels out of the delta. Measured rather than assumed:
one snapshot built both ways, same known-harmful candidate (the 8-column
price/market block on cheap2h, +1.00 in the ledger below), four shifts each —
the 1d frame gave mean delta +1.372 and the 5d frame +0.735, a gap of 0.638
against a within-frame shift-to-shift spread of up to 1.510. Inside the noise
the shift sweep already samples. (`classify` did label the two frames REAL and
NOISE, but on one 5d shift landing at −0.008 — the documented
sign-consistency brittleness, not the tail. Drop that point and both read REAL.)
Backfilling also cannot
supply a source the cache never fetched, which is the case that actually blocks
work. If you need a 1d-tail cache, fetch one — the reason to keep fetching is a
more recent evaluation period, which is what snapshots were always for.

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
NaN exchange rate blanking the payload (which has happened; see
[Known Limitations](#known-limitations) for the fixed bug). So the requirement
is replaced, not dropped:

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
51 columns to 15, a full re-audit of every rejected entry found exactly four items
with a stated mechanism for why the prune could change the answer (per-target
hyperparameters, solar-capacity scaling, time-decay weights on min/cheap2h, and the
min≤cheap2h coherence check) and five explicitly declined with reasons.

The same discipline covers any test with many arms — a hyperparameter sweep, a
per-feature ranking. Picking the best of N on the grid you also validate on is
selection on noise (this is what made round 2's per-feature ranking unusable, at
37% sign reproduction). **Screen on one period cluster, confirm the winner on the
others**, and fix the confirmation bar before looking at the screen results.

**A validated change's verdict is scoped to the model it was measured on —
re-check *adopted* changes too when a feature list changes materially, not
just rejected ones.** Found 2026-08-05: the cheap2h negative-price hurdle was
validated at −0.437 EUR/MWh on the old 52-column feature list; once cheap2h
moved to the pruned 15-column list, the hurdle's marginal value on *that*
model had never actually been measured (re-measuring found it had shrunk to
−0.026, no cluster consistency) — nothing in the process flagged that the
prune had silently invalidated an earlier verdict about a different
component. The ledger records verdicts, not the configuration each was
measured against, so a stale one can sit there looking valid. (This was
ultimately resolved, not just flagged: a later re-measurement on the
confound-free sliding grid — see the time-decay/hurdle discussion above —
found the hurdle **is** load-bearing on the pruned list after all; the
apparent shrinkage was the old grid starving its classifier of negative-price
days at the far clusters. The methodological point stands regardless of how
that particular case resolved.)

When updating the MAE table above, quote the run it came from and the snapshot it
was measured on, and record adopted changes as their **A/B deltas** in the table
beside it. Full worked example of the practice: the per-target feature re-validation
program, whose scripts and raw per-window results are under
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
  **`BASELINE` reads `model.TARGETS` directly**, so never wire a validated change
  into production while further A/B confirmations on that target are still
  pending — doing so silently collapses `BASELINE == CANDIDATE` and the next run
  measures nothing. Finish the measurements, then ship. (Corollary: after
  shipping, comparing against the *old* configuration needs an explicit `targets`
  override, not `BASELINE`.)
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
   snapshots. A longer window is wanted when a candidate needs the round-15b
   sliding grid (constant `min_train`, four evaluation periods) rather than
   the normal tail-truncation shift grid, which confounds period with
   training-set size at large shifts.
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
| Ranking forecast days by the interval's `q90` instead of the point estimate, for the charging decision | Round 19c, 2026-09-05. Looked like −0.8 EUR/MWh of realised charging cost **pooled over 14 grid points**, and **sign-flips per period cluster** (cheap2h H=7: NOW −2.82, −6M +0.25, −12M −0.51, −21M −1.12; min flips at both horizons). A NOW-cluster effect wearing a bigger `n` — the exact failure the clustered verdict rules exist to catch. The band ships as information and as a guard rail on the *defer* decision; the day **ranking** stays on the point estimate. Decision metrics must be scored per cluster, never pooled. |
| Anchored target for `min`/`cheap2h` (regress `y − known price lag`, add the anchor back) | Round 19b, 2026-08-22. **+2.4 to +48 EUR/MWh harmful**, 0 of 4 favourable in every period cluster, on both targets and all three anchors. The anchor is frozen across the horizon while SE4 troughs swing 1.5 → 50 → 8 within a week, so re-basing injects the anchor's whole day-to-day variance. With round 18 (price signals as *features*) this closes both routes: the prediction ceiling cannot be raised from inside the model. |
| Market-coupling / congestion features for `min`/`cheap2h` (`coupled_frac_dk2`, `relgap_dk2`, 1-day and 7-day) | Round 19e, 2026-08-22. Carries no price level (a 5 and a 200 EUR/MWh day score identically) and does track the regime (relgap monthly 0.055 Feb-26 → 0.493 Aug-26), but **NOISE on all four arms, both targets**. Best was `cpl_gap7` on cheap2h (clmean −0.268, 11/14 favourable) killed by −6M at +0.027. The state is not learnable from a price-derived lag; ENTSO-E cross-border capacity/flows are the remaining route (IMPROVEMENT_PLAN item 0). |
| 5-year training window as regime insurance (re-opened only for the ceiling, not for headline MAE) | Round 19, 2026-08-22. The 3-year window's cheap2h targets top out at 107.9 so 127.9 is unreachable; a 5-year window tops out at 450.2 and does score better on the 2026-08-17 break (spike MAE 48.52 vs 52.84) — but **three times worse on the calm days before it** (7.98 vs 2.53). Regime insurance with a real premium; consistent with round 15a/15c/15d's closure of window length rather than a challenge to it. Not a lever. |
| Cross-border flows and capacity for `min`/`cheap2h` (`se3_import_lag1`, `export_cap_lag1`, `export_headroom_lag1`, the pre-registered `capacity × calm` interaction, and all of them as a block) | Round 21, 2026-09-12, ENTSO-E A11 + A78 on the 14-point four-cluster grid. **All NOISE on both targets**; `export_headroom_lag1` is REAL-and-harmful on cheap2h (+0.144, 2 of 14 favourable). The control is what closes it: a *leaky* per-day capacity schedule built from outage records production could never have had is NOISE too (+0.033 cheap2h / −0.050 min), so the ceiling is not "we lack the capacity state". `min` liked capacity and the interaction in the NOW cluster only (−0.25) and in none of the three far ones — the round-19c "NOW-only effect wearing a bigger n" pattern. On the break week no arm lifts the 08-18/19 d+1 prediction above ~48 against 128. The interaction fired correctly on exactly those days; the same state in Sep 2024 came with cheap2h 6.6, so it is not learnable from this history. See [Cross-border capacity and flows](#cross-border-capacity-and-flows-round-21). **Kept open for `avg` only.** |
| Price-derived congestion (`relgap_dk2_lag1`, round 19e) on **`avg`** — the cheap substitute for the cross-border feature | Round 21, 2026-09-12. The one cell round 19e never measured. REAL −0.413 on `long/2026-08-21` **but with per-window std inflated +0.23**, and **NOISE (+0.083) on `long/2026-08-06`**, an independently fetched snapshot. Adding it alongside `export_headroom_lag1` turns that REAL result into NOISE. A price gap is not a substitute for the MW quantity, and this is a clean example of a single-snapshot REAL that vintage replication kills. |
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
| Removing any of the 15 `TROUGH_FEATURE_COLUMNS` from `cheap2h` | Leave-one-out over the same 16 measurements: every column classified KEEP or INCONCLUSIVE under `classify_ablation` (2026-08-05). Most load-bearing: `mean_wind_stockholm` (+1.36 if dropped) and `max_wind` (+0.51), both harmful to remove at all 16 measurements. One column leaned droppable but did not clear the bar — `price_se4_max_lag1` (−0.20, 12 of 16 favourable, negative in 2 of 3 period clusters) — closed in round 12 (2026-08-05) as `KEEP_SCENARIO`. **Re-measured for cheap2h on the confound-free sliding grid in round 17 (2026-08-06): still `KEEP_SCENARIO`** (−0.439 and 14 of 16 favourable, but the −6M period comes back +0.019 on one genuinely bad point — three of its windows at +9.7/+9.7/+5.5, so not a single-window artefact). Kept for cheap2h. **Removed for `min`**, where the same measurement on the same grid gives −0.552 and is favourable in all four periods — see [Per-Target Feature Sets](#per-target-feature-sets). |
| Adding back any of the 36 columns excluded from the trough list, for `min` | Round 14a, 2026-08-06 — the same 7 blocks + 8 singles, on the confound-free sliding grid (16 points, four evaluation periods, constant `min_train`). **Nothing adopted.** Only `B6_fuel_co2` (`ttf_rolling_7d`, `co2_price_lag1`, `co2_rolling_7d`) was sign-consistent at −0.187, and its gain tracks how much fuel-price turbulence the *training window* holds: NOW −0.058 (window TTF std 8) → −12M −0.307 (std 48). Production runs the NOW condition. Held pending a re-read once the far periods train on post-crisis windows (~2026-10), not adopted. Confirmations of harm: min's own frozen lag `price_se4_min_lag1` **+0.435** (1 of 16 favourable) and the whole 8-column price/market block **+0.738** (2 of 16), both harmful in all four periods. |
| Removing any of the remaining 14 `MIN_FEATURE_COLUMNS` from `min` | Round 14a leave-one-out on the same grid: every other column KEEP or INCONCLUSIVE (both mean keep — the burden of proof is on removal). Most load-bearing: `mean_wind_stockholm` **+1.015** (0 of 16 favourable) and `reservoir_norway_deviation` **+0.297**. Only `price_se4_max_lag1` cleared the bar and it has been removed (see [Per-Target Feature Sets](#per-target-feature-sets)); the list is otherwise a local optimum in both directions. |
| Negative-price hurdle on `min` | Rejected twice. 2026-07-24 on a 6-shift grid (5 of 6 favourable, one near-zero flip → NOISE), then re-opened with a stated mechanism and re-tested in round 16, 2026-08-06, on the confound-free four-period grid: **NOISE again**, clmean −0.104 with the period closest to production at **+0.079**. The obvious explanation — too few negative-price days to fit the classifier, which is what had distorted the *cheap2h* reading — is refuted here by min's own data: that period has the **most** negative-price days (198 of 731 training rows, 27%) and is exactly where the hurdle performs worst. `HURDLE_TARGETS` stays `{"cheap2h"}`; treat this as closed, not parked. |
| Solar/intraday-trough features on `cheap2h` **as a year-round addition** | `radiation_midday`, `residual_load_range` and the 5-column solar block each help in the light half of the year and hurt in the dark half by about as much, so pooled over a year they cancel to ~0 and read as noise (2026-08-05). Measured LIGHT−DARK contrast: `radiation_midday` **−0.48**, replicated in all three period clusters and 14 of 16 measurements. The open hypothesis — that they need an annual-cycle feature alongside them so the model can condition on season — was tested in round 12 and also rejected; see the next row. |
| Seasonal interaction (`radiation_midday`/`B3_solar`/`B4_intraday` + `day_of_year_sin/cos`) for `cheap2h`, round 12 (2026-08-05) | Same 16-point grid as the two rows above. All three configs `NOISE` — sign flips between period clusters even though the LIGHT/DARK contrast still points the same direction as the plain-addition test above. A post-hoc **oracle-ceiling check** (candidate's per-window MAE in LIGHT windows, base15's in DARK — an upper bound no real gate can beat, since a real gate can't pick per window) landed at only −0.06 to −0.16 EUR/MWh, below this grid's resolution floor (`LOO_price_se4_max_lag1` above needed −0.20 to only reach 12/16). Worse: the apparent LIGHT gain correlates with `min_train` (r = +0.45 to +0.67 across the three configs) — it's a **data-starvation artefact**, not a seasonal effect. The −6M/−12M clusters (541/361 training rows) show the gain; the NOW cluster (721 rows, closest to production's ~1085) shows a loss or flat. Extrapolated to production's training size all three configs are net harmful (+0.4 to +0.7). Closed; do not re-test the day_of_year combination or the plain addition again on `cheap2h`. |
| Giving `min`/`cheap2h` a non-stale price signal — `price_vs_30d` (ratio to a 30-day rolling mean), `days_since_price_anchor` (anchor age, added alongside `price_se4_avg_lag1`), or `price_de_lag1`/`price_dk2_lag1` alone — round 18 (2026-08-21) | Prompted by the obvious objection to round 14a: after its prune `min` has **no electricity-price feature at all**. The pre-registered mechanism was that what had been shown harmful is not "price information" but *a stale scalar the model treats as current*, so a signal that is either a stationary ratio, or explicitly age-tagged, might survive. **It does not — all three arms are harmful, on both targets, in every period.** Confound-free sliding grid, 16 points / four periods: `min` +0.444 / +0.610 / +0.319, `cheap2h` +0.226 / +0.347 / +0.212 (`classify_clustered` → REAL-and-harmful for 5 of the 6 target×arm cells, NOISE for the sixth). Corroborated on a 10-point vintage ladder across five independently-fetched caches (`min` **0 of 10 favourable on all three arms**) and again on a fresh 5-year snapshot anchored 15 days later (`min` +0.673/+0.535/+0.520). The staleness hypothesis is specifically refuted: `days_since_price_anchor`, the arm that *tells the model how stale the anchor is*, is the **worst** of the three on `min`. Reading: it is price-level information itself these targets reject, not its freshness — they are trough targets driven by weather → residual load, and a price level (however encoded) crowds that out. Closed; do not re-open without a mechanism that is not about staleness. |
| Splitting the negative-price hurdle's classifier/regressor feature lists for `cheap2h` (`_fit_hurdle_model` currently uses ONE list for both), round 12 (2026-08-05) | Classifier-on-51-cols/regressor-on-15-cols (the live question, motivated by "will it go negative" being a coarser question than "how low"): `NOISE`, mean +0.12, no clear gain. Classifier-on-15/regressor-on-51 (reverse-split control): consistently worse, all 16 measurements positive (mean +0.94) — confirms the test harness (the control behaves as expected) but doesn't make the case for the primary split either. Closed; the single unified 15-column list (production, unchanged) stays for both halves of the hurdle. |

## Known Limitations

- **3-year training window**: Unusual market periods (e.g. energy crisis 2021–2022) have outsized weight. Reservoir features will become more valuable as more data accumulates.
- **Daily resolution**: The model predicts daily aggregates, not 24 hourly prices. Hour-level predictions would be more actionable for EV scheduling but require significantly more feature engineering. The `cheap2h` target partially addresses this: it predicts what a ~2h charging session picking the day's cheapest hours would pay, which is the number the "charge today or wait" decision needs.
- **Forecast horizon**: All forecast days (1–8) use the same features and a single model that doesn't distinguish horizon. Horizon-aware modelling was **evaluated and shelved** (2026-07): the anchor-staleness sensitivity showed the true stale-lag cost is only ~1 EUR/MWh for min/cheap2h, so a `forecast_horizon` feature has little headroom. The scary-looking per-horizon curve is a weekday artifact of the step-7 walk-forward (see Current MAE Baseline), not real horizon decay. The price-lag anchor is kept fresh regardless, since that was a free win.
- **Weather in evaluation**: walk-forward uses archive weather as a stand-in for the forecast, so results are optimistic on the weather axis — real day+7 weather forecasts are worse than archive. This optimism grows at far horizons and is not captured by the anchor-staleness or per-horizon metrics.
- **The eval does not model the training tail.** `walk_forward_validate` fits right up to its test window. Until 2026-08-23 production fitted ~5 days behind (the archive lag), so every baseline recorded before that date is optimistic by roughly the numbers in [Closing the weather-archive lag](#closing-the-weather-archive-lag-round-19a). The top-up closes the gap to ~1 day; the residual is measured at **+0.155 (cheap2h) / +0.258 (min) / +0.591 (avg)** (the `gap1` arm), and the recent-analysis product's own error against the archive is on top of that and still unmeasured.
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

`cheap2h_low` / `cheap2h_high` are an **80% band around that day's own prediction**, in the same SEK/kWh units and carried through the addon like everything else. `cheap2h` is always inside them. Use them as a guard rail on the *defer* decision, not as a replacement for the ranking: the cost of waiting when the price is about to rise is much larger than the cost of charging now when it was about to fall, so a rule like "do not defer if `cheap2h_high` for the later day is above today's known cheap price" is the asymmetry the point estimate cannot express. Ranking the days by `cheap2h_high` instead of `cheap2h` was tested and is **not** replicated — see [Prediction interval for cheap2h](#prediction-interval-for-cheap2h-round-19c).

The addon value is fetched live from Home Assistant each run, so distribution cost changes take effect immediately without redeploying.
