# Decisions and Adopted Changes

Every change that shipped, newest first: what it was, when it was adopted, the
evidence that justified it, and the mechanism behind it. This is the positive
half of the ledger — its twin is [REJECTED.md](REJECTED.md), and a change
belongs in exactly one of the two.

**Each entry must be readable without `experiments/`.** That directory is
scratch and is not committed (see the README's "Working with `experiments/`"),
so an entry carries its own numbers: grid, snapshot, arms, deltas, how many
measurements were favourable. A script name is a courtesy, never the record.

| Change | Adopted | A/B delta (EUR/MWh) |
|---|---|---|
| [Prediction interval for cheap2h](#prediction-interval-for-cheap2h-round-19c) | 2026-09-05 | not MAE-measurable — coverage 0.542 → 0.788 |
| [Closing the weather-archive lag](#closing-the-weather-archive-lag-round-19a) | 2026-08-23 | recovers +0.155 cheap2h / +0.258 min / +0.591 avg of the ~5-day lag's cost |
| [`avg` drops the price/market lag family](#avg-drops-the-pricemarket-lag-family-round-14b) | 2026-08-21 | −1.2 to −1.97 (clmean) |
| [`min` drops `price_se4_max_lag1`](#per-target-feature-lists-for-min-and-cheap2h) | 2026-08-06 | −0.55 |
| [Per-target feature lists (min 14, cheap2h 15)](#per-target-feature-lists-for-min-and-cheap2h) | 2026-08-05/06 | −0.64 (min), −0.86 (cheap2h) |
| [Negative-price hurdle on cheap2h](#negative-price-hurdle-on-cheap2h) | 2026-07-25 | −0.44 |
| [Price-lag anchor freshening](#price-lag-anchor-freshening) | 2026-07 | ~−1.1 min/cheap2h, ~−2.3 avg |
| [Exposing the intraday trough](#exposing-the-intraday-trough) | 2026-07 | ~−0.66 cheap2h |

The same deltas, with their replication counts, are tabulated in
[MODEL.md](MODEL.md#current-mae-baseline).

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
[The 2026-08-17 regime break](FINDINGS.md#the-2026-08-17-regime-break) binds the quantile
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

## Closing the weather-archive lag (round 19a)

Adopted 2026-08-23. Measured with local experiment scripts under `experiments/`,
which is **not committed** — the numbers below are the record.

`fetch_data.WEATHER_ARCHIVE_LAG_DAYS = 5` — the Open-Meteo archive publishes to
about today−5, so the merged frame ended ~5 days back and **the model was fitted
five days behind the first forecast day**. The
[price-lag freshening](#price-lag-anchor-freshening) had fixed the *anchor* in 2026-07;
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

## `avg` drops the price/market lag family (round 14b)

Adopted 2026-08-21.

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
(see [Current MAE Baseline](MODEL.md#current-mae-baseline)) purely because that one
window was calm; that is expected, not a contradiction of the A/B.

## Per-target feature lists for min and cheap2h

Adopted 2026-08-05 (`cheap2h`, 15 columns) and 2026-08-06 (`min`, the same list
minus `price_se4_max_lag1`, 14 columns). The lists themselves are in
[MODEL.md](MODEL.md#per-target-feature-sets).

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

## Negative-price hurdle on cheap2h

Adopted 2026-07-25. The component itself — the classifier, the out-of-fold
training and why the leak-safety matters — is documented in
[MODEL.md](MODEL.md#negative-price-hurdle-cheap2h-only).

**Negative-price hurdle (2026-07-24/25, cheap2h only):** an XGBoost classifier predicts P(tomorrow's `price_min` < 0 EUR/MWh) — a distinct physical regime (renewable oversupply + low/weekend demand) a plain regressor otherwise has to infer implicitly from the same weather/calendar features. Its out-of-fold probability feeds into the `cheap2h` regressor as an extra feature (`neg_price_proba`, now the #1 feature for cheap2h at ~0.25 importance). Validated via `ab_test.py` on a real cached snapshot, shifts 0–5: **REAL, all 6 shifts improved, mean −0.44 EUR/MWh** — the largest, cleanest single-run win recorded for the top-priority target; local reproduction via `evaluate.walk_forward_validate` matched the A/B exactly (14.74 vs the same run's min/avg/max, which were unaffected). **Confirmed on a real Actions run and committed 2026-07-25**; the table above now includes it. `min` showed the same direction on 5 of 6 shifts but didn't clear the strict sign-consistency bar (one near-zero flip) — **not productionized for `min`** — re-tested on the confound-free four-period grid in 2026-08 and rejected again (NOISE: −0.10 on average but *worse* in the period closest to production, and min's own data refutes the obvious explanation — that period has the most negative-price days to learn from, not the fewest).

Re-tested after the 15-column prune, because
[a verdict is scoped to the model it was measured on](AB_TESTING.md#how-changes-are-validated):
removal costs **+0.250 EUR/MWh**, positive in all four period clusters
(round 15b, 2026-08-06). The hurdle stays.

## Price-lag anchor freshening

**Price-lag anchor freshening (2026-07):** the SE4 price lags — the dominant min/cheap2h features — are now frozen at the freshest *known* ENTSO-E price (`se4_prices_daily`), not the training frame's last row. The training frame ends ~`WEATHER_ARCHIVE_LAG_DAYS` behind because it inner-joins prices with the lagging weather archive, so production was previously anchoring the price lags ~5 days stale (the DE/DK2 lags were already fresh — this closes the same gap for SE4's own lags). The walk-forward reports an **anchor-staleness sensitivity** (fresh d0 vs stale d5 ≈ old pipeline). Measured saving from freshening: **~1.1 EUR/MWh (min/cheap2h), ~2.3 (avg)** — real and free, but modest, because yesterday's min and six-days-ago min are similar.

## Exposing the intraday trough

**Intraday trough features (2026-07):** the daily min/cheap2h is set at a specific intraday trough (overnight wind or midday solar) that the daily-mean `residual_load` diluted. Exposing the trough directly — chiefly `residual_load_min` (daily minimum of hourly residual load) — cut cheap2h ~0.66 EUR/MWh and held min, with `residual_load_min` landing as a top-4 feature in all three priority models. See the [Intraday Trough Features](FEATURES.md#intraday-trough-features) section.
