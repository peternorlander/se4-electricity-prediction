# Decisions and Adopted Changes

Every change that shipped, newest first: what it was, when it was adopted, the
evidence that justified it, and the mechanism behind it. This is the positive
half of the ledger — its twin is [REJECTED.md](REJECTED.md), and a change
belongs in exactly one of the two.

**Each entry must be readable without `experiments/`.** That directory is
scratch and is not committed (see the README's "Working with `experiments/`"),
so an entry carries its own numbers: grid, snapshot, arms, deltas, how many
measurements were favourable. A script name is a courtesy, never the record.

An entry that adds or changes a feature also states **how it was made
horizon-honest** — frozen across the forecast window, recomputed per day, or
explicitly leaky as an upper bound. Without that, a future reader cannot tell
whether the number was ever achievable in production
([AB_TESTING.md](AB_TESTING.md#horizon-honesty-measure-only-what-production-can-know)).

| Change | Adopted | A/B delta (EUR/MWh) |
|---|---|---|
| [`min ≤ cheap2h` coherence](#min--cheap2h-coherence-round-25) | 2026-10-12 | coherence for Home Assistant (the pair crossed on 40–50 % of days); `min` −0.17 … −0.23 clmean (every cluster, 4 of 5 caches), `cheap2h` −0.08 … 0.00 (not harmful); NOW slightly weaker |
| [Northern wind points](#northern-wind-points-round-25) | 2026-10-12 | cheap2h −1.04 … −1.18, min −0.71 … −0.81, avg −0.32 … −0.42 (REAL on 5/5 caches, 60/60 points); on forecast-vintage weather ~40–75 % of the cheap2h gain, ≈ 0 at a 7-day lead |
| [A03 forward-fill of ENTSO-E price curves](#a03-forward-fill-of-entso-e-price-curves) | 2026-10-10 | correctness fix, harm-checked — cheap2h −0.029 / min +0.014 / avg +0.038 clmean against the corrected targets, none REAL-harmful; also fixes the EUR/SEK rate on 30 % of 2026 days |
| [Production-length training in the A/B grid](#production-length-training-in-the-ab-grid) | 2026-10-10 | no delta of its own — it changes what every later measurement means (every test week now trains on production's 1088 rows) |
| [One day-ahead price per hour, chosen deterministically](#one-day-ahead-price-per-hour-chosen-deterministically) | 2026-10-10 | data integrity, not accuracy — `price_de` was SDAC or the German 15-min intraday auction, re-drawn per hour and per fetch; now the SDAC curve (sequence 1), 100 % equal to SMARD 2021–2026 |
| [Residual-load wind-curve unit fix](#residual-load-wind-curve-unit-fix-round-23) | 2026-09-17 | cheap2h −0.11 to −0.16 clmean (REAL on 3/4 independently-fetched snapshots); min favourable on 4/4 but sub-threshold |
| [Prediction interval for cheap2h](#prediction-interval-for-cheap2h-round-19c) | 2026-09-05 | not MAE-measurable — coverage 0.542 → 0.788 |
| [Closing the weather-archive lag](#closing-the-weather-archive-lag-round-19a) | 2026-08-23 | recovers +0.155 cheap2h / +0.258 min / +0.591 avg of the ~5-day lag's cost |
| [`avg` drops the price/market lag family](#avg-drops-the-pricemarket-lag-family-round-14b) | 2026-08-21 | −1.2 to −1.97 (clmean) |
| [`min` drops `price_se4_max_lag1`](#per-target-feature-lists-for-min-and-cheap2h) | 2026-08-06 | −0.55 |
| [Per-target feature lists (min 14, cheap2h 15)](#per-target-feature-lists-for-min-and-cheap2h) | 2026-08-05/06 | −0.64 (min), −0.86 (cheap2h) |
| [Negative-price hurdle on cheap2h](#negative-price-hurdle-on-cheap2h) | 2026-07-25 | −0.44 |
| [NaN instead of zero for uncovered price sources](#nan-instead-of-zero-for-uncovered-price-sources) | 2026-08-06 | data integrity, not accuracy — no-op on a 3-year window |
| [Price-lag anchor freshening](#price-lag-anchor-freshening) | 2026-07 | ~−1.1 min/cheap2h, ~−2.3 avg |
| [Exposing the intraday trough](#exposing-the-intraday-trough) | 2026-07 | ~−0.66 cheap2h |

The same deltas, with their replication counts, are tabulated in
[MODEL.md](MODEL.md#current-mae-baseline).

## `min ≤ cheap2h` coherence (round 25)

Adopted 2026-10-12, as `model.enforce_coherence` — called by `model.predict`,
`evaluate.walk_forward_validate` and `ab.harness.run_walk_forward`. Measured with local
experiment scripts under `experiments/` (not committed); the numbers below are the record.

**The problem.** `min` (the cheapest hour) and `cheap2h` (the mean of the two cheapest
hours) are independent regressors, and the true gap between them is tiny — over the last
year the mean of the two cheapest hours sits a median 0.4 EUR/MWh (mean 1.3) above the
cheapest hour. Their predictions therefore cross constantly: **predicted `min` exceeded
predicted `cheap2h` on 40–50 % of held-out days** on every cache (46 % on `2026-10-10`
with the northern wind points in), an average excess of 2.0–2.4 EUR/MWh over all days
(about 5 on the days it happens), and on a live forecast taken the day it shipped 7 of 9
days were crossed. A pair that violates the
identity is incoherent for the Home Assistant rule and, because both are noisy estimates of
nearly the same number, loses information.

**The fix.** Where predicted `min` > predicted `cheap2h`, replace **both** by their mean
("split"); other days, `avg` and `max` are untouched, and `cheap2h_low/high` are built around
the served `cheap2h`. Post-processing only — no input changes, so there is nothing to
freeze: horizon honesty is untouched.

**Arms, rolling 1088-row grid, five caches, models as shipped (with the northern wind
points); change in MAE vs the raw pair, EUR/MWh, negative = better.** Per-cache cluster
means NOW / −6M / FAR:

| cache | `min`, split | `cheap2h`, split | `min`, lower |
|---|---|---|---|
| `2026-10-10` | +0.03 / −0.13 / −0.41 | −0.33 / −0.06 / +0.14 | +0.31 / −0.14 / −0.67 |
| `2026-09-16` | −0.13 / −0.13 / −0.41 | −0.12 / −0.01 / +0.14 | −0.08 / −0.17 / −0.67 |
| `2026-08-21` | −0.06 / −0.15 / −0.41 | −0.22 / −0.05 / +0.14 | +0.08 / −0.18 / −0.67 |
| `2026-09-12` | −0.05 / −0.09 / −0.41 | −0.27 / −0.09 / +0.14 | +0.16 / −0.07 / −0.67 |
| `2026-08-06` | −0.05 / −0.22 / −0.41 | −0.28 / −0.04 / +0.14 | +0.13 / −0.28 / −0.67 |

clmean: split `min` −0.17 … −0.23 (REAL — every cluster negative — on four of five),
split `cheap2h` −0.08 … +0.00 (NOISE, not harmful); "lower" (`min` = `min(min, cheap2h)`,
`cheap2h` untouched) `min` −0.17 … −0.31. FAR is the same calendar window in every cache,
so only NOW and −6M are independent between caches.

**Why split, and not the arm first recommended.** "Lower" looked best on the pooled number
and leaves the priority target alone, but its gain is carried by −6M and FAR: in **NOW, the
period closest to production, it is worse on four of five caches (+0.08 … +0.31)**. Split
improves `min` in all three clusters on four caches and `cheap2h` in NOW on all five
(−0.12 … −0.33), at a cost of +0.14 in FAR. When the two models disagree about the order
both are evidently off by about the same, and averaging halves both errors; "lower" trusts
the one model. The pre-registered rule — ship if not harmful on both targets (pooled clmean
≤ +0.02 and not REAL-harmful) on at least four of five caches — is met by both; the choice
between them rests on the per-cluster table above. A third arm, raising `cheap2h` to `min`,
gained nothing ([REJECTED.md](REJECTED.md)).

**Expect little, and period-dependent.** The freshest NOW cluster is the weak spot (`min`
+0.03 on `2026-10-10`); on that cache at shift 0 the shipped pair gives `min` 15.01 → 15.10
and `cheap2h` 15.28 → 14.83, at −6M `min` −0.04 / `cheap2h` −0.12, at FAR `min` −0.32 /
`cheap2h` +0.16. The payoff is a coherent pair, not an accuracy step.

**Verified locally before commit** (`experiments/integration_check_coherence.py`,
`OMP_NUM_THREADS=4`): the harness with the change equals the stored "split" arm to 1e-9 at a
NOW, a −6M and a FAR point of `2026-10-10`, with `avg` and `max` bit-identical to the same run
with coherence switched off; the acceptance test passes — `run_walk_forward(train_window=None)`
and `evaluate.walk_forward_validate` agree on mean and std for all four targets; and
`model.predict` on the live forecast gives `min ≤ cheap2h` on every day, untouched days equal the
raw predictions, crossed days equal the mean, `avg`/`max` are unchanged and the interval contains
the served `cheap2h`.

**Consequences.** Every A/B number measured through `ab.harness` from now on includes the
change, as does the headline eval and the `mae_*` values pushed to Home Assistant (monitoring
only; they move a little). The MAE table in [MODEL.md](MODEL.md#current-mae-baseline) predates it
and was not re-measured. A slot target set (round 20) would need the sibling identity
`min(slots) ≥ cheap2h` enforced the same way.

## Northern wind points (round 25)

Adopted 2026-10-12, in `sources/open_meteo.py` and `features.py`. Measured with local
experiment scripts under `experiments/` (not committed); the numbers below are the
record. The explanation and the full audit are in
[FINDINGS.md](FINDINGS.md#the-northern-wind-belt-is-worth-1-eurmwh-on-cheap2h-about-half-of-it-on-a-real-forecast-round-25).

**What changed.** Three Open-Meteo points were added to `WIND_LOCATIONS` —
Norrbotten (65.3, 20.9), Jämtland (63.2, 14.6) and Ostrobothnia (63.1, 21.6), the wind
fleets behind the SE3 import price that the grid had no point under
([FINDINGS.md §1.4](FINDINGS.md#14-the-weather-grid-has-no-point-under-the-northern-wind-fleet)).
`aggregate_intraday_features` turns them into two columns, `north_wind_power` (mean
over the sites of the m/s power curve, daily mean) and `north_wind_night` (00–06 local
mean), which are appended **last** to `TROUGH_FEATURE_COLUMNS` (→ `MIN_FEATURE_COLUMNS`
follows) and `AVG_FEATURE_COLUMNS`. Lists are now 17 / 16 / 44 (`max` unchanged at 51: it
was not measured, so the columns are deliberately not in `FEATURE_COLUMNS`). The order is
part of the change: it is the order that was measured, and XGBoost's column sampling is
index-based.

**A/B result.** Rolling 1088-row grid, NOW / −6M / FAR, 12 points on each of the five
`ab_cache/long/` caches (the two 5d-tail and three 1d-tail caches never share a grid),
`classify_clustered`, N1 − base in EUR/MWh:

| cache | cheap2h | min | avg |
|---|---|---|---|
| `2026-10-10` | −1.04 | −0.75 | −0.32 |
| `2026-09-16` | −1.18 | −0.71 | −0.33 |
| `2026-08-21` | −1.12 | −0.72 | −0.39 |
| `2026-09-12` | −1.14 | −0.74 | −0.36 |
| `2026-08-06` | −1.16 | −0.81 | −0.42 |

REAL on all fifteen cells, every point favourable on cheap2h and min (60 of 60),
per-window std 0.93–1.02 of base. The FAR cluster is the same calendar window in every
cache, so only NOW (−1.24 … −1.65) and −6M (−0.96 … −1.10) are independent between
caches; the table replicates the fetch, not five periods. Neighbours, on three caches:
each single site gives about half (cheap2h −0.55 … −0.71); a fourth site (Sundsvall)
adds nothing on cheap2h (+0.09 / +0.19 / +0.01) and was rejected
([REJECTED.md](REJECTED.md)); removing `mean_wind_stockholm` from the new model costs
+0.57 … +0.79 (cheap2h) and +0.27 … +0.51 (min), so it stays.

**Horizon honesty — and the weather axis.** Both columns are per-day weather quantities
production recomputes for every forecast day from the forecast, so neither is frozen
(the test rows carry their own day's value, as production does). The weather *source*
is the weak point: every eval weather input is ERA5 archive, a perfect forecast.
Re-scored with the models still trained on archive weather and the test rows fed the
forecast issued L days earlier (Open-Meteo Previous Runs, `best_match` — what the
production fetch resolves to — for all nine points, two caches × 12 points), the
cheap2h gain is **−0.50 / −0.41 / −0.44** (NOW / −6M / FAR, `2026-10-10`) and
**−1.23 / −0.59 / −0.44** (`2026-09-16`) against −1.24 / −0.96 / −0.92 and
−1.65 / −0.98 / −0.92 on archive: 40–75 % survives, negative in every cluster. It
decays with lead (L1–L3 ≈ −1.0, L5 −0.75, L6 −0.45, **L7 ≈ 0**), `min` keeps −0.30 …
−0.91 (−0.29 at L7), and `avg` keeps almost nothing (`2026-10-10`: +0.05 / −0.12 /
−0.10) — its archive gain is perfect foresight, shipped anyway because it is not
harmful and the columns serve the priority targets. Leak smell (archive gain at test
row 6 over row 0): 1.24 cheap2h, 1.60 min, 1.01 avg, flag at 2. **Expect the headline
MAE to move about half as far as the archive A/B suggests**, mostly at D+2…D+5.

**Verified locally before commit** (`experiments/integration_check_northwind.py`,
`OMP_NUM_THREADS=4`): the production frame built by `features.build_training_data` from
the grafted `long/2026-10-10` cache has the two columns identical to the experiment
frame and every shared production column identical; the shipped lists equal the old
lists plus the two columns appended last; the production variant's walk-forward on
`2026-10-10` shift 0 reproduces the stored N1 arm to six decimals (min 15.0085,
cheap2h 15.2795, avg 19.4838); `model.train` on production's 1088 rows → importance
sizes 16 / 44 / 18 (`north_wind_power` importance 0.052 min, 0.013 avg, 0.022 cheap2h);
and the live forecast path — Open-Meteo forecast for all eight locations →
`build_forecast_features` → `model.predict` with the interval — gives finite features
and predictions on every day.

**The long caches were extended in place, 2026-10-12.** `aggregate_intraday_features`
raises on a missing site column (a silent drop would shrink the frame through `dropna`),
so every cache taken earlier would no longer build. The three sites were grafted into
each cache's `wind_intl_hourly` — archive up to the cache's own `today − 5 d`, the recent
analysis after, the splice `fetch_data` uses — after checking that the method
reproduces a cached column exactly (fresh archive vs the cached Stockholm wind over
4344 h: max difference 0.0000) and asserting every existing column unchanged. Each
`meta.json` records it. Consequence for A/B work: `BASELINE` now includes the northern
columns; comparing against the pre-change configuration needs an explicit `targets`
override ([AB_TESTING.md](AB_TESTING.md#how-it-works)).

**Not done.** `max` was not measured and does not read the columns. The 2026-10-16 long
cache is the first independent NOW window for a replay.

## A03 forward-fill of ENTSO-E price curves

Adopted 2026-10-10, in `sources/entso_e.py`. A correctness fix that changes the
**targets themselves**, so it was evaluated differently from a feature and
gated by a harm check rather than by an improvement bar.

**The defect.** A44 price curves are `curveType` A03, "variable sized block": a
point is sent only when the price changes, and omitted positions repeat the
previous point to the end of the Period. The parser read point by point, so a
repeated price disappeared from `prices_hourly` and `market_prices_hourly` —
543 of 70,824 SE4 points over five years, 9 in 2021 rising to 246 in 2026 with
15-minute MTUs. `price_avg` was wrong on 286 of 1815 days (max 16.2 EUR/MWh),
`price_cheap2h` on 84 (max 14.1, biased upward on cheap days because a cheapest
price held for two hours counted once), and the EUR/SEK rate on 30 % of 2026
days. Full measurement:
[FINDINGS.md](FINDINGS.md#entso-e-omits-a-price-point-when-the-price-repeats-2026-10-10).

**The fix.** `_parse_period` takes the TimeSeries' curve type; for A03 it
expands the Period to every position 1..N, N = Period length / resolution, each
omitted position carrying the previous point. 23- and 25-hour DST days fall out
of the Period length; A01 curves and documents without a curve type are left
exactly as before. Offline self-test (synthetic documents: PT60M and PT15M,
omitted interior and tail positions, single-point days, 23/25-hour days,
multi-Period documents, A01 untouched, the DE ⋈ DK2 merge in the fetch path) —
`experiments/probe_a44_a03.py --selftest`, which also proves that filling an
already-parsed cache gives the parser's result exactly, on 71 random days
straddling DST changes and the PT60M → PT15M switch. That equivalence is what
allowed the measurement below on existing caches.

**How it was evaluated, and why.** A normal A/B scores each arm against its own
frame's labels, which here would compare two different questions. Instead both
pipelines were fitted once and scored against the **corrected** targets (the
decision), and again against the old ones (secondary):

* `old` — today's pipeline: unfilled frame for training and test features;
* `new` — SE4 and DE/DK2 filled;
* `new_se4` — SE4 filled only, fitted for `max` alone (the only target reading
  `price_de_lag1` / `price_dk2_lag1`; for the others it equals `new`).

Grid: `long/2026-10-10` (1815 rows), rolling 1088-row training, production
`model.TARGETS`, horizon-honest freeze, 12 points in NOW / −6M / −11M (shifts
0,1,4,7 / 180,183,186,189 / 326,329,332,335), `OMP_NUM_THREADS=4`. Harness
check: the `old` arm on old labels reproduced the
[calibration re-run's](#production-length-training-in-the-ab-grid) `base` roll
`min` at shift 0 bit-for-bit (mean 16.4136).

**Pre-registered bar** (agreed with Peter before the run): ship unless
`classify_clustered` on the corrected-label deltas is REAL *and harmful* on
cheap2h, min or avg. A fix that makes the training data match the source
should not have to prove an MAE gain; it only has to not cost one.

| target | NOW | −6M | −11M | clmean | verdict | favourable points |
|---|---|---|---|---|---|---|
| cheap2h | −0.107 | +0.066 | −0.044 | −0.029 | NOISE | 7 / 12 |
| min | −0.004 | +0.039 | +0.009 | +0.014 | NOISE | 1 / 12 |
| avg | +0.018 | +0.050 | +0.044 | +0.038 | BORDERLINE | 5 / 12 |
| max (SE4 + DE/DK2, approximate — see below) | +0.072 | +0.023 | −0.680 | −0.195 | NOISE | 7 / 12 |
| max (SE4 only) | +0.004 | +0.014 | −0.175 | −0.052 | NOISE | — |

Deltas are new − old in EUR/MWh, negative = new better. Scored against the old
labels the table is the same to within 0.002, and the label change alone moves
the old model's MAE by at most 0.02 per point (cheap2h −0.0002, min +0.001,
avg +0.003 on average) — the corrected days are too few and the corrections too
small for the *score* to notice; what moves is the fit. Per-window std did not
inflate (cheap2h +0.05 on average, range −0.09..+0.24; min −0.01).

**Verdict: ship.** Nothing is REAL-harmful. `avg` is the one to name: the same
sign in all three clusters, +0.038, below the 0.10 floor, with its 12 point
deltas spread from −0.234 to +0.322 and 5 of 12 favourable — the grid cannot
tell it from zero, and the bar was set before the run precisely so a
sub-threshold drift on a correctness fix would not decide it. The `max` DE/DK2
arm was approximate and is not a result: it post-hoc filled a joined market
frame whose DE still mixed curves, and the exact replay later showed 859 of the
1016 restored market hours had been omitted by one zone only — so the other
zone's real price was unknowable from that frame.

**Horizon honesty.** No new column, nothing frozen or unfrozen. Every changed
value — targets, SE4 lags, `price_volatility_7d`, DE/DK2 lags — is the same
quantity as before, now read completely; production reads it from the same
document at the same time.

**Verified locally before commit**: the filled `long/2026-10-10` frame cut to
production's 1088 rows (2023-10-18 → 2026-10-09; no production-shaped cache is
kept any more) → `model.train` → `train_interval` → `model.predict` →
`evaluate.get_feature_importance`: 66 columns, feature counts 14 / 42 / 51 / 15,
importance 14 / 42 / 16, all 42 predictions finite.

**Every long cache was corrected in place, 2026-10-11** — SE4 by the post-hoc
fill, DE/DK2 by replaying the saved raw A44 responses through the filling
parser; the checks and the exact DE/DK2 numbers (`price_de_lag1` /
`price_dk2_lag1` move on 392 days, max 25.4 / 21.3) are in
[FINDINGS.md](FINDINGS.md#entso-e-omits-a-price-point-when-the-price-repeats-2026-10-10).
The gappy originals were deleted once the repair was verified, so a result
measured on a cache before 2026-10-11 — including the A/B above, which ran on
the first 2026-10-10 fetch — no longer reproduces bit-for-bit; its ledger row
is the record. The headline table in
[MODEL.md](MODEL.md#current-mae-baseline) was not re-measured for this change
and will move slightly when it next is.

## Production-length training in the A/B grid

Adopted 2026-10-10. Not an accuracy change but a change to what every later
measurement *means*, so it has no A/B delta of its own.

**What was wrong.** `predict.py` fits on **1088 merged rows**: `TRAINING_DAYS`
= 1095 calendar days minus the 7 head rows whose 7-day lags and rolling means
cannot be computed (measured on `ab_cache/2026-09-11` — 1088 rows,
2023-09-19 → 2026-09-10). Both walk-forward loops *expand* instead: on the
canonical 1095-row window the first test week trains on 731 rows and the last on
1088, ~910 on average. So the average measurement saw **84 % of production's
training set**, and only the final week of the year matched production at all.
Round 15b's sliding grid fixed the *between-cluster* form of this and was
documented as giving "production as it would actually have run on that date" —
it did not: within each point the expansion remained, unmeasured, through every
round from 11 to 23.

**What changed.** `ab.harness.run_walk_forward` takes `train_window`. `None`
keeps the expanding geometry, so the acceptance test against
`evaluate.walk_forward_validate` still holds (verified bit-for-bit against the
previous implementation); an int trains every test week on exactly that many
rows. `ab_test.py run` passes `PRODUCTION_TRAIN_ROWS = 1088` and loads from
`ab_cache/long/` by default, printing the geometry it used; `--expanding`
restores the old one and is the only way to run against a production-shaped
snapshot. A production-faithful point costs `1088 + 364 + shift` rows, so the
harness now refuses a frame that cannot supply them — naming the numbers —
rather than quietly expanding, and `ab_test.py fetch` defaults to ~5 years and
prints which clusters its rows reach.

**Evidence that the geometry matters** (2026-09-16, `long/2026-09-12`, 216
walk-forwards, 7 arms picked from the ledger for known size sensitivity, both
geometries on the same 12 points across NOW / −6M / −11M): verdicts moved.
cheap2h + `radiation_midday` + `doy_sin/cos` went **NOISE (+0.016) → REAL
(−0.162)**, a mode effect of −0.178 that was sign-consistent in all three
clusters; adding `price_se4_max_lag1` back to `min` got more harmful
(+0.324 → +0.429); the `min` hurdle went REAL → NOISE. Four arms kept their
verdicts, including the +1.15 price-block control. Absolute MAE: rolling scored
**0.37 better at NOW** on both priority targets, ~0 at −6M and slightly worse at
−11M, so the headline eval is mildly pessimistic in the period closest to
production and the sign is period-dependent — consistent with round 15d closing
window length as a lever.

**Replicated on a second snapshot and the current feature set** (2026-10-10,
`long/2026-10-10`, 1815 rows, the same 12 points, another 216 walk-forwards, run
after the [wind-curve unit fix](#residual-load-wind-curve-unit-fix-round-23)
changed several of the columns involved). The seasonal arm behaves the same way:
exp **NOISE (+0.070) → roll REAL (−0.217)**, mode effect **−0.287**, negative in
all three clusters (−0.153 / −0.280 / −0.429) and larger than the first run's
−0.178. The base-MAE pattern repeats too — rolling better by **0.392** (cheap2h)
and **0.294** (min) at NOW, ~0 at −6M, 0.21–0.23 *worse* at −11M, so the
geometry's absolute effect is period-dependent in the same direction both times.

What did *not* replicate: the `min` + `price_se4_max_lag1` mode effect
(+0.106 REAL → −0.012 NOISE). That is what the noise estimate predicted for the
weaker arm — per-point noise is ±0.27, so one REAL mode effect among seven arms
is expected by chance, and the arm that survived two independent snapshots is
the one to believe. Three other verdicts moved between geometries in the second
run without a replicating mode effect (hurdle removal on cheap2h
KEEP_LOAD_BEARING → INCONCLUSIVE, the 13.1 combo BORDERLINE → NOISE, `min` +
fuel/CO2 NOISE → BORDERLINE); all three sit within ~0.1 of a verdict boundary,
which says the bar is close there, not that the geometry decided it.

**The decision still rests on the arithmetic, not on any one arm** — a
measurement that trains on 84 % of production's data is measuring a different
model — but it is no longer unsupported on the current model.

**Cost.** ~20 % more compute per point, and a narrower grid until the data
catches up: the canonical −12M cluster needs 1821 rows (a long fetch from about
2026-10-16), −21M needs 2098 (mid-2027), so a rolling grid runs NOW / −6M /
−11M — three clusters, satisfying `MIN_CLUSTERS`, but with ~10 % calendar
overlap against NOW where −12M had ~2 %. Rounds 11–23 remain comparable with
each other and are **not** comparable with rolling-geometry numbers; a new row
should say which geometry produced it. The headline eval stays expanding by
necessity: production fetches three years, exactly one evaluation year short of
being able to do anything else.

## One day-ahead price per hour, chosen deterministically

Adopted 2026-10-10, in `sources/entso_e.py`, in two steps the same day: a
deterministic rule in the morning, then — once the curves were identified — the
SDAC curve by sequence (item 2.15a, below). A data-integrity fix that affects
production as much as the backtest, so it is recorded here even though it has no
A/B delta.

**The defect.** An A44 response for DE/LU carries more than one price curve for
the same hour. The parser walked every `<Period>` in the document and finished
with `sort_values("timestamp").drop_duplicates("timestamp")`. That default sort
is an unstable quicksort, so which of two equal timestamps survived depended on
the whole array — and the yearly request chunks start on the fetch date, so the
draw changed per hour and per fetch even though the response order itself is
stable (two probe fetches listed the TimeSeries identically). Measured across the
7 production-shaped and 4 long caches (2026-09-17, overlapping hours only):
**43,696 of 46,105 hours — 95 % — carry exactly two distinct values**, and which
one a given cache holds was effectively random (11 hours had three). Gap between
the pair: median 9.74, mean 16.66, **max 695 EUR/MWh**; the daily mean
`price_de_lag1` moved by up to 99 EUR/MWh between fetches. It reproduced in
pairs fetched two days apart with identical code, both before and after the
2026-09-12 edits to this file, so it was neither a source revision nor a code
version. SE4 showed no conflicts in that measurement and DK2 none above float
rounding (its sequence-2 copy, below); both run through the same function.

**Consequences.** `predict.py` re-fetches three years through this path on every
daily run, so `max` — the only target whose feature list holds `price_de_lag1` /
`price_dk2_lag1` — trained on a slightly different German history every day, and
no A/B on a cached snapshot could reproduce what production did. The priority
targets never used the column. Every price-block arm in rounds 11, 14a and 18
measured a partly random column; all came back harmful, so the direction stands
and the magnitudes do not.

**What the two curves are (2.15a, probed 2026-10-10).** Every TimeSeries of six
DE windows (2022-02, 2023-06, 2024-09, the 2025-10 switch, 2025-11, 2026-09) was
scored against the SMARD/Bundesnetzagentur day-ahead price, and every hour the
caches disagreed on was checked against SMARD back to 2021-10:

| | sequence 1 | sequence 2 |
|---|---|---|
| matches SMARD / Nord Pool SDAC | **100 %** of points, every window | < 0.5 % |
| resolution | PT60M until 2025-09-30, PT15M after | PT15M throughout |
| hourly-mean corr with SDAC | — | 0.90–0.97, gap median 5–10 EUR/MWh |
| shape | — | first quarter of each hour +2.5…+7 above the hour mean until SDAC itself went to 15 min, then flat |

Both carry `contract_MarketAgreement.type = A01` and `auction.type = A01`, and
they differ **only** in `classificationSequence_AttributeInstanceComponent.position`.
Sequence 2 is most consistent with the German 15-minute intraday auction (the
sawtooth is that auction's signature); ENTSO-E does not label it, so that
identification is an inference. What is measured is that it is not SDAC. Before
2025-10 its `:15/:30/:45` rows vanished in `fetch_market_prices`' inner merge with
hourly DK2, so the caches held SDAC or the intraday `:00` quarter — the quarter
that carries the sawtooth's peak, which is why the "other" value ran 6–8 EUR/MWh
high in 2021–2023 and its |gap| to SDAC had a median of 20–37 there.

The morning's rule ordered unlabelled candidates by `(businessType, mRID,
price)`. `mRID` is just the TimeSeries' position in the document and sequence 2
is listed first, so the string sort (`"1" < "2"`, but `"10" < "9"`) picked the
**intraday curve on 14 of 15 days**: 6.7 % of points SDAC in the hourly era,
15–21 % after. Deterministic, and wrong — production's `max` trained on mostly
intraday German prices for the hours between the two commits.

**The fix as it stands.** (1) The DE/LU request is pinned to
`classificationSequence_AttributeInstanceComponent.position=1`, which removes the
second curve at the source; `contract_MarketAgreement.type=A01` does nothing (both
curves are A01). SE4 and DK2 are **not** pinned: they label SDAC with no sequence
element and the pinned request returns no data for them. (2) The parser keeps the
A01 contract and drops every curve labelled with a sequence other than 1 —
SDAC is "sequence 1 or unlabelled". That second half is not cosmetic: since
2026-01-23 DK2 also publishes an explicit sequence-2 *copy* of its SDAC prices
on most days (equal to within 3·10⁻⁵, float rounding), and a keep-only-sequence-1
rule emptied DK2's whole last year on the first fetch through it. (3) Anything
ambiguous is now a hard error (`AmbiguousPriceError`): two prices for one
timestamp after the selection, a document whose every curve is non-SDAC, two
request chunks disagreeing, or an empty chunk (every chunk lies in the past, so
empty means lost data; that is how the DK2 year vanished silently, the frame
shrinking 1815 → 1452 rows). A day-ahead model must never silently train on an
intraday price.

**Verified.** The new parser on every saved raw response — pinned and unpinned,
DE, SE4, DK2, 2022 to 2026 — gives 100 % SDAC, and a five-year fetch
(`ab_cache/long/2026-10-10`, 69,712 rows, no missing day) matches SMARD for
DE-LU and Energy-Charts for DK2 on **100 % of points in every year**. SE4 from
ENTSO-E is identical to Nord Pool. `experiments/probe_a44_curves.py --selftest`
covers the real DE shape (sequence 2 listed first, the `"9"/"10"` mRID trap), the
DK2 shape, the hard errors and the DE-only pin offline.

**The caches are corrected in place.** All five long caches in `ab_cache/long/`
now hold the SDAC series; the four older ones had their market prices replaced
from the clean fetch, verified column by column (see
[AB_TESTING.md](AB_TESTING.md#snapshot-generations--check-the-weather-tail-before-building-a-grid)).
The coin-flip draws earlier rounds measured are gone with them, so those rounds'
`max` and price-lag numbers live only in their ledger rows.

## Residual-load wind-curve unit fix (round 23)

Adopted 2026-09-17. Measured with local experiment scripts under
`experiments/`, which is **not committed** — the numbers below are the record.

`features._wind_power_curve` clips at `rated_speed=13.0` believing its input is
m/s (its own docstring said so). `sources/open_meteo.py` never requests
`wind_speed_unit`, so `windspeed_100m` arrives in Open-Meteo's default **km/h**.
13 km/h is 3.6 m/s — an ordinary breeze — so the curve was pinned at its rated
maximum on **~88% of hours**
([FINDINGS.md §1.1](FINDINGS.md#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated)),
collapsing the wind term of `residual_load`, `residual_load_min`,
`residual_load_range` and `wind_night` to a near-constant. Fixed by dividing
wind speed by 3.6 *inside* the curve before clipping — the raw `mean_wind*`
columns are untouched, so every feature reading them directly stays
bit-identical; only what feeds the curve changes.

A/B result, round-15b 16-point/four-cluster grid, run independently on **all
four** `ab_cache/long/` snapshots that existed at the time (two 5d-tail, two
1d-tail — never pooled into one grid across tails, per
[AB_TESTING.md](AB_TESTING.md#snapshot-generations--check-the-weather-tail-before-building-a-grid)):

| snapshot | tail | cheap2h clmean | verdict | min clmean | verdict |
|---|---|---|---|---|---|
| `2026-08-06` | 5d | −0.015 | NOISE | **−0.157** | **REAL** |
| `2026-08-21` | 5d | **−0.114** | **REAL** | −0.099 | NOISE |
| `2026-09-12` | 1d | **−0.136** | **REAL** | −0.088 | NOISE |
| `2026-09-16` | 1d | **−0.162** | **REAL** | −0.053 | NOISE |

cheap2h clears `classify_clustered` on 3 of 4 independently-fetched snapshots
(the fourth, `2026-08-06`, is inconsistent across its own four clusters, not
reversed — pooled mean still favourable at −0.015). min clears it on 1 of 4.
**All eight snapshot×target pooled means are favourable — the direction never
flips once across four independent fetches.** Per-window std never inflates
for either target on any snapshot. `avg`/`max` are members of the same
`residual_load` family (the shared `FEATURE_COLUMNS` list, not just the
trough list) so are **not** bit-identical either, and stay NOISE with mixed
sign on all four snapshots — no measured cost, and neither is a priority
target.

**Horizon-honesty.** No new column and no lag introduced. `residual_load` /
`residual_load_min` / `residual_load_range` / `wind_night` were already
correctly left unfrozen as per-day-forecastable features
([AB_TESTING.md](AB_TESTING.md#horizon-honesty-measure-only-what-production-can-know));
this changes an existing derivation's physics, not what production can know or
when — same freeze status as before.

**Verified locally before commit**
(`features.build_training_data` → `model.train` → `evaluate.get_feature_importance`
→ `model.predict`, against `ab_cache/2026-09-11`): 1088 rows, 66 columns;
`min`/`avg`/`cheap2h` feature counts unchanged (14/42/16); all predictions
finite; hourly SE4 wind-curve saturation measured at 6.8% post-fix (was
~88%).

**What was deliberately not done.** The raw `mean_wind*` columns and their
unit labels (docs/FEATURES.md still says "10 m"; the pre-fix docstring said
m/s) are left alone — that is a separate, purely-documentation commit. Arms S
(fixed per-zone MW capacities) and C (per-year calibrated from ENTSO-E A75/A65)
are the next, larger steps in `IMPROVEMENT_PLAN_2026-09.md` §2.1: the unit fix
alone does not cure the *scale* mismatch between the wind/solar terms and the
demand proxy — `residual_load` stays ≥0.998 correlated with temperature even
after this fix (FINDINGS.md §1.1) — so a materially bigger effect is expected
there, not here.

The general mechanism this closes (temperature-in-disguise, the scale
mismatch that remains) is [FINDINGS.md §1.1](FINDINGS.md#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated).

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

**Why quantile regressors rather than the point model's own residual spread.**
A dedicated q90 is a much better q90: pinball loss at α = 0.90 is **4.400**
against **6.963** for the point model used as if it were q90 (cheap2h), and
3.906 against 6.402 for min — better in every period cluster. The raw band also
widens in the right places: cheap2h width at NOW is 28.9 on low-price days
against 47.4 on top-decile days.

The **raw** band is over-confident (a nominal 80% interval covers 54%, and only
0.31 on top-decile days), so `model.train_interval` applies split conformal on a
180-day holdout and refits on all data. Measured 0.542 → 0.788 pooled, in every
period cluster, and top-decile coverage 0.307 → 0.510. **The price is a band
about 50 % wider** (29.8 → 44.7 mean width on cheap2h, 28.2 → 41.0 on min) —
that is the honest width, not a cost. Serve-time order is sort → widen → clamp
open around the point estimate; each step has a measured reason, documented on
`model.IntervalModel`.

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

**Which columns the splice actually perturbs**, measured end to end by rebuilding
the merged frame with the forecast product spliced over the last days and diffing
the model's own input columns, as a fraction of each feature's 3-year standard
deviation:

| column | disagreement (sd units) |
|---|---|
| `radiation_variability` | 0.489 |
| `residual_load_min` | 0.212 |
| `mean_wind_de_north` | 0.169 |
| `residual_load` | 0.118 |
| `temp_gradient_se3_se4` | 0.114 |
| the other 10 trough columns | 0.000 |

So splicing is close to free for wind, hydro, fuel and calendar, and imperfect
only for the radiation-derived columns. `radiation_variability` is the one to
watch: it is a 7-day rolling window, so a single day's radiation difference
propagates for a week. Wind is bit-identical at Malmö, DK1, DK2, Karlskrona and
Stockholm — which matters because wind is what the trough targets run on.

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

**Re-confirmed on a clean German series (round 2.15b, 2026-10-11).** Both 14b
fetches carried a `price_de_lag1` that was half intraday auction. Adding the
family back on the SDAC series, rolling 12-point grid (NOW / −6M / −11M): **REAL
harmful +1.080**, 0 of 12 favourable, against +0.932 on the same day's coin-flip
cache — the decision stands, and a cleaner anchor if anything hurts more
([REJECTED.md](REJECTED.md)).

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

**Negative-price hurdle (2026-07-24/25, cheap2h only):** an XGBoost classifier predicts P(tomorrow's `price_min` < 0 EUR/MWh) — a distinct physical regime (renewable oversupply + low/weekend demand) a plain regressor otherwise has to infer implicitly from the same weather/calendar features. Its out-of-fold probability feeds into the `cheap2h` regressor as an extra feature (`neg_price_proba`, now the #1 feature for cheap2h at ~0.25 importance). Validated via `ab_test.py` on a real cached snapshot, shifts 0–5: **REAL, all 6 shifts improved, mean −0.44 EUR/MWh** — the largest, cleanest single-run win recorded for the top-priority target; local reproduction via `evaluate.walk_forward_validate` matched the A/B exactly (14.74 vs the same run's min/avg/max, which were unaffected). **Confirmed on a real Actions run and committed 2026-07-25** — which the rule at the time required and the rule since 2026-08-05 does not; the A/B verdict is the gate and a production run is not evidence about accuracy. The headline table in [MODEL.md](MODEL.md#current-mae-baseline) includes it. `min` showed the same direction on 5 of 6 shifts but didn't clear the strict sign-consistency bar (one near-zero flip) — **not productionized for `min`** — re-tested on the confound-free four-period grid in 2026-08 and rejected again (NOISE: −0.10 on average but *worse* in the period closest to production, and min's own data refutes the obvious explanation — that period has the most negative-price days to learn from, not the fewest).

Re-tested after the 15-column prune, because
[a verdict is scoped to the model it was measured on](AB_TESTING.md#how-changes-are-validated):
removal costs **+0.250 EUR/MWh**, positive in all four period clusters
(round 15b, 2026-08-06). The hurdle stays.

## NaN instead of zero for uncovered price sources

Adopted 2026-08-06, in `features._warn_uncovered`. Not an accuracy change and
not A/B'd — a data-integrity fix — but it changes what a long snapshot looks
like, so it is recorded here.

`build_training_data` used to `fillna(0.0)` any day a price source did not
cover. A price of zero is not a neutral placeholder: it is a valid-looking
number XGBoost will happily split on, and it is physically wrong. Found via the
first 5-year fetch — EUA carbon data starts **2021-10-18**, so a window opening
earlier got 65 days of 0.00 EUR/t (the real price was ~60) propagated into
`gas_marginal_cost`, which is a `TROUGH_FEATURE_COLUMNS` member.

The gap is now left as NaN, which XGBoost handles natively by learning a default
direction, and a loud warning names the source and the date range. At serving
time these columns are never missing, so that branch is simply never taken.

**Two consequences worth knowing.** The normal ~3-year window is unaffected —
both sources cover it completely, so this is a no-op for production today; the
point is that the *next* gap announces itself instead of being trained on. And
the final `dropna` now drops those rows, so a 5-year snapshot starts where its
latest-starting source does: `long/2026-08-06` is **1747 rows from 2021-10-21**,
not 1815 from 2021-08-14. Grid scripts that assert a row count will trip on this;
the assertion is doing its job (see
[AB_TESTING.md](AB_TESTING.md#measurement-grids-training-geometry-and-period-clusters-read-before-designing-a-run)).

## Price-lag anchor freshening

**Price-lag anchor freshening (2026-07):** the SE4 price lags — the dominant min/cheap2h features — are now frozen at the freshest *known* ENTSO-E price (`se4_prices_daily`), not the training frame's last row. The training frame ends ~`WEATHER_ARCHIVE_LAG_DAYS` behind because it inner-joins prices with the lagging weather archive, so production was previously anchoring the price lags ~5 days stale (the DE/DK2 lags were already fresh — this closes the same gap for SE4's own lags). The walk-forward reports an **anchor-staleness sensitivity** (fresh d0 vs stale d5 ≈ old pipeline). Measured saving from freshening: **~1.1 EUR/MWh (min/cheap2h), ~2.3 (avg)** — real and free, but modest, because yesterday's min and six-days-ago min are similar.

## Exposing the intraday trough

**Intraday trough features (2026-07):** the daily min/cheap2h is set at a specific intraday trough (overnight wind or midday solar) that the daily-mean `residual_load` diluted. Exposing the trough directly — chiefly `residual_load_min` (daily minimum of hourly residual load) — cut cheap2h ~0.66 EUR/MWh and held min, with `residual_load_min` landing as a top-4 feature in all three priority models. See the [Intraday Trough Features](FEATURES.md#intraday-trough-features) section.
