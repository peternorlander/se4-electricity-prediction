# A/B Backtest Flow and Measurement Methodology

How a change is measured in this project, and the bar it has to clear. Read this
before running any experiment — it is the discipline the whole ledger rests on.

Companion files: [DECISIONS.md](DECISIONS.md) (what passed),
[REJECTED.md](REJECTED.md) (what did not), [MODEL.md](MODEL.md) (what is in
production now).

## Why this exists

`ab_test.py` is the tool for evaluating any candidate feature or model change before
committing it. It exists because MAE effects on the priority targets are small
(often < 0.3 EUR/MWh) and **period-dependent**: the same change can look like an
improvement on one day's data and a regression on the next. Historically that forced
re-running an experiment across several real calendar days to tell signal from noise
— one change every few days. The A/B flow simulates those different run-days from a
single data snapshot, so a verdict takes one sitting.

## Two shapes of snapshot, and which one a grid needs

`python ab_test.py fetch` caches **~5 years** under `ab_cache/long/` — that is
the default since 2026-10-10, because a production-shaped fetch cannot measure
production (see
[Measurement grids](#measurement-grids-training-geometry-and-period-clusters-read-before-designing-a-run)).
`--days 1095` still caches a **production-shaped** snapshot (~1088 rows, exactly
what `predict.py` fits on) under `ab_cache/`; its job is the local
`train` → `predict` → `get_feature_importance` integration check before
shipping, not measurement. `run` loads from `ab_cache/long/` and trains each
test week on a rolling `PRODUCTION_TRAIN_ROWS`-row window; it refuses, with the
numbers, on a frame too short to do that.

## Snapshot generations — check the weather tail before building a grid

`python ab_test.py list` reports a **weather tail** per snapshot: how many days
behind its own `today` the weather stops. `5d` means archive-only (fetched
before the 2026-08-23
[archive-lag top-up](DECISIONS.md#closing-the-weather-archive-lag-round-19a)); `1d` means
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
price/market block on cheap2h, +1.00 in [REJECTED.md](REJECTED.md)), four shifts each —
the 1d frame gave mean delta +1.372 and the 5d frame +0.735, a gap of 0.638
against a within-frame shift-to-shift spread of up to 1.510. Inside the noise
the shift sweep already samples. (`classify` did label the two frames REAL and
NOISE, but on one 5d shift landing at −0.008 — the documented
sign-consistency brittleness, not the tail. Drop that point and both read REAL.)
Backfilling also cannot
supply a source the cache never fetched, which is the case that actually blocks
work. If you need a 1d-tail cache, fetch one — the reason to keep fetching is a
more recent evaluation period, which is what snapshots were always for.

**What a cache holds that a fresh fetch cannot reproduce is narrower than it
looks — it was measured.** 2026-09-17, every consecutive pair of the 4 long and
7 production-shaped caches, raw sources and built frames, overlapping dates only
(see [FINDINGS.md](FINDINGS.md#cached-history-is-stable-except-for-one-column-2026-09-17)):
SE4 prices, weather deep in history, nuclear outages, TTF and the Swedish
reservoir series are **identical across fetches**. So "preserving the vintage"
is not, for this pipeline, a reason to keep a cache. Three exceptions are:

* the Norwegian reservoir series is restated deep in history (up to 117 of 259
  weeks, max 0.049 TWh — worth ≤0.001 on `reservoir_norway_deviation`);
* the last ~4 days of a 1d-tail cache are forecast-spliced weather that a later
  fetch replaces with archive actuals (~90 hourly rows, up to 2.8 °C);
* `price_de` was SDAC or the German intraday auction, re-drawn per hour and per
  fetch, until
  [the SDAC selection](DECISIONS.md#one-day-ahead-price-per-hour-chosen-deterministically)
  (2026-10-10). That draw is **no longer in any cache** — see below — so an
  older `max` or price-lag result cannot be reproduced bit-for-bit; its ledger
  row is the record.

**Every long cache in `ab_cache/long/` holds the SDAC series (corrected in place
2026-10-11).** `long/2026-10-10` is the clean five-year fetch (69,712 rows, 100 %
equal to SMARD/Energy-Charts in every year). Since SDAC prices are never revised,
the four older caches (`2026-08-06`, `-08-21`, `-09-12`, `-09-16`) had their
market prices replaced by that fetch over each cache's own window, and say so
in their `meta.json`. Checked per cache before the swap: same frame dates, only
`price_de_lag1` and `price_dk2_lag1` change, and both equal the clean fetch on
every common date; nothing else any experiment reads was touched, so every
other cached result reproduces. `price_dk2_lag1` moves on ~250–280 days (up to
13 EUR/MWh) because the hours SDAC omits as repeats are no longer filled from a
second curve — which is what the fixed parser loads.

**…and every repeated price is filled in (corrected in place 2026-10-11).**
ENTSO-E sends a price point only when the price changes (curve type A03), and
until the [A03 fill](DECISIONS.md#a03-forward-fill-of-entso-e-price-curves) the
parser dropped every repeat — from the targets as well as the lags. All five
long caches were filled: SE4 directly in the cached frame (provably identical to
the parser's fill), DE/DK2 by replaying the saved raw A44 responses
(`experiments/results/probe_a44/raw/debug_*.xml`, production's own ten requests
over 2021-10-11 → 2026-10-10) through the filling parser, which is exact where a
fill of the joined DE ⋈ DK2 frame is not. Checks and numbers are in
[FINDINGS.md](FINDINGS.md#entso-e-omits-a-price-point-when-the-price-repeats-2026-10-10).
**This one does change cached results:** `price_avg`, `price_cheap2h` and every
SE4 price lag move on some days, so a ledger number measured on a cache before
2026-10-11 will not reproduce bit-for-bit on it today, and the originals were
not kept — the ledger row is the record. Keep those raw responses: they are the
only way to rebuild DE/DK2 exactly without a token.

**…and every long cache carries the three northern wind sites (extended in place
2026-10-12).** The northern wind points ([DECISIONS.md](DECISIONS.md#northern-wind-points-round-25))
are read by `aggregate_intraday_features`, which raises on a missing site, so the five caches
were extended: `wind_intl_hourly` gained `windspeed_/radiation_/temperature_` for `norrbotten`,
`jamtland`, `ostrobothnia` from Open-Meteo (archive to the cache's own `today − 5 d`, the recent
analysis after, the same splice as `fetch_data`). Additive — every existing column is asserted
unchanged and each `meta.json` says so — so no earlier cached result moves. A cache fetched
before this date and not extended will fail to build; a fresh `ab_test.py fetch` carries the sites.

Retention, therefore: keep a cache while an experiment that cites it is still
open, keep `ab_cache/crossborder/` (A78 outage records were republished
wholesale in 2025-11 and that vintage really is unrecoverable), and otherwise
retire one when nothing cites it. The ledger row carries the numbers; the
directory never was the record.

## How changes are validated

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
  removed (see [Per-Target Feature Sets](MODEL.md#per-target-feature-sets)).

The bar for adoption:

| Verdict | Meaning | Action |
|---------|---------|--------|
| `REAL` | sign-consistent across shifts, and abs(mean) ≥ spread | adopt |
| `BORDERLINE` | sign-consistent but smaller than the spread | replay on a **separately-fetched** snapshot; adopt only if the sign holds |
| `NOISE` | sign flips across shifts | reject |
| `NO_CHANGE` | identical to baseline | reject |

Additional standing requirements: the per-window **std must not inflate**, and
priority order is **cheap2h → min → avg** (`max` is not a priority).

**A precondition that outranks the whole table: the candidate must only be able
to see what production can see.** A REAL verdict on an arm that was handed
information the live pipeline will not have is not a small error in the
measurement — it is a measurement of something else entirely, and it will not
reproduce when the change ships. Before reading any verdict, settle the questions
in [Horizon honesty](#horizon-honesty-measure-only-what-production-can-know)
below, and record the answer in the ledger entry.

**The A/B verdict is the gate (changed 2026-08-05).** A change counts as done once
it clears the bar above; it no longer waits on a confirming Actions run. The
previous rule required both, which in practice meant a validated improvement sat
unshipped for a day to be re-measured by a *weaker* instrument — a single rolling
run whose headline MAE moves ±1.3 with the period alone (see
[Current MAE Baseline](MODEL.md#current-mae-baseline)). An A/B across several shifts and
snapshots is strictly more evidence about accuracy than one production run is.

What that trade gives up, stated plainly so nobody has to rediscover it: the
Actions run was never a good *accuracy* check, but it was the only end-to-end
exercise of the parts the A/B harness never touches — the live fetches, the
EUR→SEK conversion, and the Home Assistant push. A backtest cannot catch a
NaN exchange rate blanking the payload (which has happened; see
[Known Limitations](../README.md#known-limitations) for the fixed bug). So the requirement
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
that produced [Per-Target Feature Sets](MODEL.md#per-target-feature-sets) — not for
routine feature-addition testing.

**Re-opening a rejected verdict requires a stated mechanism, written before the
run.** [REJECTED.md](REJECTED.md) is long, and per-measurement noise is a few tenths — so if
you re-test enough rejected entries, some will "pass" by chance alone. That is
searching noise for a favourable answer, not validation. Before re-running
anything from [Features Tested and Rejected](REJECTED.md), write
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

**Where documented numbers come from, and where they must never come from.**
Every number in the docs is a *measurement you can re-run*, not an observation of
a production run:

| what | where it comes from |
|---|---|
| the headline MAE table in [MODEL.md](MODEL.md#current-mae-baseline) | one named snapshot, re-measured deliberately, `OMP_NUM_THREADS=4` pinned, with the snapshot date and the code state quoted beside it |
| the evidence for any change | its **A/B delta** on the four-cluster grid — never a before/after of the headline table, and never a single run |
| the production `mae_min` / `mae_avg` / `mae_max` / `mae_cheap2h` attributes | **monitoring only.** Do not copy them into any document. |

The last row is the easy mistake. Those attributes are a real walk-forward result
and still useless as a record: each Actions run computes them on whatever
window that day happens to end on, and the headline moves ±1.3 EUR/MWh with the
evaluation period while the model is unchanged. A number lifted from one run
cannot be compared with the one before it or reproduced afterwards. Watch them for
drift and alarms; document the snapshot measurement.

So when you update the table, re-measure it: quote the snapshot it was measured on
and the code state it describes, and record the adopted change as its **A/B
delta** — in that table and as an entry in [DECISIONS.md](DECISIONS.md). The
fullest worked example of the practice is the per-target feature re-validation
program, written up in
[DECISIONS.md](DECISIONS.md#per-target-feature-lists-for-min-and-cheap2h).

## Horizon honesty: measure only what production can know

**This is the precondition for every number in the ledger, and the easiest thing
in the project to get silently wrong.** A verdict is only worth acting on if the
candidate was measured under production's own information constraints. A
candidate handed knowledge production will not have does not measure a better
model — it measures the leak, and it will not reproduce when it ships.

The canonical case. Suppose a candidate adds "yesterday's DE price". On a
walk-forward test window every row already carries *its own* true previous-day
value, so in the eval day+3 gets the real price from the day before day+3.
Production never has that: it runs once, before tomorrow's prices are published,
and issues d+1…d+7 in one shot. For d+3 it has two honest options — freeze the
last known value across the horizon, or predict the DE price too. If the test
rows take neither, the measured gain is partly "we told the model the answer",
and the further out the horizon the larger that share.

### How the harness handles it, and where it stops helping you

`features.apply_forecast_freeze` overwrites every column in
`FORECAST_FROZEN_FEATURES` on the test window with one frozen value, reproducing
what `build_forecast_features` does in production:

* **Lag-type columns** are frozen at the **first test row** — its lag-1 is the
  last training day's actual price, exactly what freshened production freezes,
  and it depends only on training-period data, so it leaks nothing.
* **Unshifted rolling columns** (`_FORECAST_FROZEN_ROLLING`: `wind_variability`,
  `radiation_variability`) are frozen at the **last training row** instead,
  because the first test row's window would include the test day itself.
* **Per-day columns are deliberately left alone** — weather, HDD, residual load,
  calendar, planned nuclear outages. Production legitimately recomputes these for
  every forecast day from the published forecast, so freezing them would make the
  eval *pessimistic* rather than honest.

**The default for anything new is "not frozen".** `Variant.frozen_features`
exists precisely so a candidate can freeze a column it invents without editing
the production list — but if you leave it `None`, your new lag-type column is
treated as per-day forecastable and the run measures a leak. Worse,
`apply_forecast_freeze` skips names that are not columns of the frame:

```python
for col in features_to_freeze:
    if col not in frozen.columns:
        continue
```

So a misspelled entry fails **silently and in the dangerous direction** — the run
looks frozen and is not. Assert, in the variant, that every name you added is
both a column of the frame and a member of the list you pass.

### The other half: can production serve the quantity you measured?

Horizon honesty asks whether the value was *knowable* at run time. This asks
whether the value production would actually read is the **same series** you
measured — and a feature can pass the first test and fail this one.

Round 21 is the worked case, and it cost three rounds of replication to notice.
`export_headroom_lag1` used the realised cross-border flow for day *R* = T+1.
That is knowable at run time, but only as the day-ahead *schedule*: the physical
flow has not happened yet. The two series correlate **0.56** on the feature, so
the A/B measured something production cannot serve — and the same feature built
from the freshest realised flow production does have (`lag2`) is NOISE. Three
REAL verdicts, on two independent snapshots, none of them transferable.

So before running a grid on a new external series, ask where production reads it
from and *when*: if the answer is "a different publication of roughly the same
thing", measure that publication instead, or at minimum measure how far the two
diverge first. It is one cheap probe against several days of replication.

### Three questions before measuring any new feature

1. **When is it published, relative to the run?** Production runs once, after
   tomorrow's day-ahead result and before delivery. A source that only ever
   describes d+1 cannot be a *forecast* feature for d+2…d+7 at all — it can still
   be valuable for calibration or as a quasi-static regime state, but say which.
2. **Does its value change per forecast day in production?** If yes, it must
   change in the eval too. Round 18's `days_since_price_anchor` is deliberately
   *not* frozen: the anchor's age varies by horizon and that variation is the whole
   feature. If no, freeze it — and freeze it from data that predates the test
   window.
3. **Is it revised after the fact?** A snapshot holds *today's* version of
   history, not what was knowable then. ENTSO-E A78 is the worked example: every
   outage record for an event before 2025-11 was re-published that month, so no
   vintage exists for the older clusters and a forward-looking feature built from
   it **cannot be backtested honestly at all** (see
   [FINDINGS.md](FINDINGS.md#if-you-touch-ab_cachecrossborder-read-this-first)).
   Prices and weather get revised too, which is why a cross-snapshot replay is
   part of the bar.

### The case worth knowing by heart: what "lag1" actually means

`export_headroom_lag1` (round 21, on `avg`) is the sharpest instance in the repo,
because the leak was not in the harness — it was in the *column name*.

The arm was REAL on three independently fetched snapshots (clmean −0.423 / −0.218
/ −0.233, every period cluster favourable, per-window std falling). Nothing in the
verdict looked suspect. But `lag1` means **day R**, the day before the first
forecast day — and at run time a realised A11 physical flow for day R **does not
exist yet**. Production can only know day R as a day-ahead *schedule*, from a
source the pipeline does not fetch. Re-encoded to `lag2`, the freshest realised
flow production could actually read, the same feature is **NOISE** (clmean
−0.092, −12M +0.014, same 14-point grid).

So a three-times-replicated REAL result rested entirely on one day of
unavailable freshness. The obvious repair — fetch the day-ahead schedule
production *can* read (ENTSO-E A09) and rebuild the column from that — was then
measured, and it closed the item on 2026-09-13 rather than rescuing it: the
servable version correlates only **0.56** with the realised-flow version the A/B
had measured. A09 does answer for all five SE4 borders, but a commercial schedule
is a different series from a physical flow — the raw daily southbound totals
correlate 0.71, and DK2 only 0.64, because Øresund carries loop flows no schedule
describes. See
[FINDINGS.md](FINDINGS.md#cross-border-capacity-and-flows-round-21) and the two
rows in [REJECTED.md](REJECTED.md).

Three lessons, all general:

* **A lag number is a claim about a publication calendar, not about an offset.**
  `lag1` is honest for SE4/DE/DK2 day-ahead prices, which are published the
  afternoon before delivery. It is dishonest for realised flows, actual
  generation, or anything settled after the fact. Ask *when the value exists*,
  every time, per source — not what the column is called.
* **A REAL verdict cannot detect this.** Both arms shared the same impossible
  freshness, so it cancelled out of every delta, on every snapshot, in every
  cluster. Replication does not help either: replicating a leak reproduces the
  leak. Only reading the source's publication schedule catches it, which is why
  it is question 1 above and why it happens before the run, not after.
* **Re-encoding to what production can know is a new experiment, not a haircut on
  the old one.** It is tempting to treat the honest version as "the same effect,
  somewhat smaller". It need not be the same effect at all: here the servable
  series correlates 0.56 with the measured one, so the re-encoded arm tests a
  different hypothesis and needs its own pre-registered bar. Budget for that
  before promising a result, and never carry the leaky arm's magnitude forward as
  an estimate of the honest one's.

### Three kinds of leak, each with a case in this repo

| leak | what it is | the case |
|---|---|---|
| **Per-day freshness** | test rows carry each day's own true lag; production freezes one value for the week, and for some sources cannot even know day R | the DE-price example above, and `export_headroom_lag1` — REAL on three snapshots, NOISE once re-encoded to what production can read |
| **Training tail** | the eval fits right up to its test window; production fits where its slowest source ends | the archive lag — invisible for a year, worth **+1.70 avg / +0.59 min / +0.37 cheap2h**, residual **+0.591 / +0.258 / +0.155** even after the fix ([DECISIONS.md](DECISIONS.md#closing-the-weather-archive-lag-round-19a)) |
| **Vintage / revision** | the cached history is the corrected one, not the one production saw | A78 outage records, re-published wholesale in 2025-11 |

The middle row is the one to take seriously as a warning. It was not a subtle
effect — for `avg` it was larger than any change ever adopted — and it sat
unmeasured through dozens of A/Bs, because every arm shared it so it cancelled
out of each delta. **A leak shared by both arms is invisible in the delta and
still wrong in production.** When a candidate changes *what production can know*
rather than just what the model does with it, the paired design stops protecting
you and the arm has to model the constraint explicitly.

### Build the leaky arm on purpose

The most useful thing to do with a leak is to measure it deliberately, as an upper
bound. If a candidate does not help **even when cheating**, no honest encoding of
it will, and the route closes properly:

* Round 21 built a per-day interconnector capacity schedule from outage records
  production could never have had. **NOISE** (+0.033 cheap2h / −0.050 min) — which
  is what actually closed cross-border capacity for the trough targets, far more
  firmly than the honest arm could have.
* Round 19 handed the model the **same-day** German price, impossible in
  production. Spike MAE 51.30 against production's 52.84, still predicting 41.6
  for a realised 127.9 — so the continental price was never the missing
  information.

Label such an arm as leaky in the ledger entry, with the honest arm beside it. An
upper bound is evidence; an upper bound mistaken for a result is not.

### Archive weather is a perfect forecast: re-score weather candidates on vintages

Every weather input in this repo's eval is ERA5 archive — a forecast with no error.
Production pays the error, and it is not small: scoring the same models on the
forecast issued L days earlier costs +1.63 cheap2h / +1.69 min / +2.81 avg EUR/MWh
at production's leads, +3…+5 at a 7-day lead and ~0 under D+3
([FINDINGS.md](FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b)).
That error is shared by both arms of a paired A/B, so it cancels from a *feature set*
comparison — and it does not cancel from a *weather feature*, whose whole value is
the forecast skill it carries. The measured case: three northern wind points scored
cheap2h −1.04 … −1.18 on archive weather on all five caches and **−0.4 … −1.2 on
forecast vintages** (40–75 % of it), falling to ~0 at a 7-day lead; the `avg`
version, −0.32 … −0.42 on archive, kept almost nothing.

So, for any addition whose columns derive from weather inputs:

1. Re-score the winner (not every arm) with the model still trained on archive
   weather and the test rows fed `*_previous_dayL` vintages for **every** weather
   point it reads, the new ones included (`experiments/run_round25_n1.py` is the
   worked implementation; fetch with `fetch_round25_vintage*.py`). Row j of the
   eval maps to lead min(j + 2, 7): the walk-forward's first row is production's
   first *sent* day, D+2.
2. Record both numbers in the ledger entry. The survival criterion written before
   the run for N1 was "negative in every cluster on cheap2h"; a gain that keeps its
   sign at a fraction of its size is a smaller true effect, not a failure.
3. Read the gain by lead. An honest weather feature decays with lead (here
   −1.0 at L1–L3 → −0.45 at L6 → 0 at L7); archive gain at test row 6 over row 0
   above ~2 would be the leak signature.

Vintages exist from 2024-03 (ECMWF IFS from 2024-04) for the 6 production points
and any other point; a candidate cannot be scored before that.

### Smells that mean "check the wiring before believing it"

* The gain **grows with horizon**. Honest features decay with horizon or hold
  flat; a leak gets stronger the further out you go, because that is where the
  information gap is widest.
* The effect is **larger than anything comparable in the ledger**. Real effects on
  the priority targets are a few tenths of a EUR/MWh. A candidate worth several is
  a wiring claim first and a finding second.
* The new column **dominates feature importance immediately**. That is what a
  column containing the answer looks like.
* The untouched targets **moved**. They must be bit-identical unless the change
  legitimately touches them; if they moved, the harness is not measuring what you
  think it is.

## How it works

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
- `ab/harness.py` — `run_walk_forward()`, a copy of the `evaluate.py` loop (both apply
  `model.enforce_coherence` to the predicted `min`/`cheap2h` pair since 2026-10-12, so they still agree
  and still describe what Home Assistant receives)
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

## How to run an experiment (agent playbook)

Follow this whenever testing a feature or model change from the improvement plan:

1. **Get a snapshot.** `python ab_test.py fetch` (needs `ENTSO_E_TOKEN`; runnable
   locally from VS Code). Reuse an existing one with `python ab_test.py list` if a
   recent snapshot is already cached — a snapshot a few days old is fine for
   iterating. The default is a **~5-year fetch into `ab_cache/long/`**, which is
   what a measurement needs: it is the only shape that can give every test week
   production's own 1088 training rows, and it carries the sliding four-cluster
   grid. The fetch prints which clusters its rows reach. `--days 1095` caches a
   production-shaped snapshot in `ab_cache/` instead — for the integration check
   in step 5, not for measuring.
2. **Express the change as `CANDIDATE`** in `ab/variants.py`. This is the only file
   you edit, usually a handful of lines. A `Variant` has:
   - `transform(data) -> data` — adds or modifies **columns** on the merged daily
     frame (e.g. scale a feature, add a derived feature). **Must not add, drop, or
     reorder rows** — `BASELINE` and `CANDIDATE` are compared on identical slices, and
     the harness asserts the row count and `date` column are unchanged.
   - `fit_fn(train_slice) -> models` — override to change training (objective, sample
     weights, hyperparameters). Default: `model._fit_models` (production).
     **Gotcha: `avg` is the one target with time-decay weighting**
     (`HALF_LIFE_DAYS["avg"] = 500`), looked up **by target name** inside
     `_fit_models`. A variant that renames the target key silently drops the
     weighting and measures a model production does not run. Assert the key at
     startup.
   - `targets` — a `{name: (target_col, feature_cols)}` dict; override to add a new
     feature column to the models' feature lists (a `transform` that only *creates* a
     column has no effect until the column is added to `feature_cols` here).
   - `frozen_features` / `frozen_rolling` — override only if the change adds a
     lag-type feature that must be frozen in the horizon-honest eval (default `None` =
     production lists). **Decide this before running anything** — the default
     leaves a new column per-day fresh, which is a leak for anything lag-like.
     See "Horizon honesty" above; it is the difference between a number that
     survives production and one that does not.
3. **Run it.** `python ab_test.py run` (add `--snapshot YYYY-MM-DD` to pick a
   specific one, `--shifts 0-5` to change the grid). It loads from
   `ab_cache/long/` and trains every test week on a rolling 1088 rows, printing
   the geometry it used; `--expanding` reverts to the pre-2026-10 geometry, and
   is the only way to run against a production-shaped snapshot. Read the
   per-target verdict table. **Priority order is cheap2h first, then min**;
   `avg` is welcome, `max` is not a priority.
4. **Decide by the verdict**, weighing cheap2h/min and checking std isn't inflated:
   - `REAL` → adopt: implement the change properly in the real modules, then revert
     `CANDIDATE` to the no-op `Variant(name="candidate")`.
   - `NOISE` / `NO_CHANGE` → reject: record it as a row in
     [REJECTED.md](REJECTED.md), in the group it belongs to and with its numbers,
     and revert `CANDIDATE`.
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
5. **Run the integration check, then ship.** Since 2026-08-05 the A/B verdict is
   the gate and an adopted change does **not** wait on a confirming Actions run
   (see "How changes are validated" above — this step used to say it did). What
   replaces it: run `train` → `predict` → `get_feature_importance` against a
   cached snapshot locally, asserting feature counts and finite predictions, then
   commit. Watch the next Actions run for live-fetch or push regressions, but do
   not block the change on its headline MAE.

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

## Measurement grids: training geometry and period clusters (read before designing a run)

### The training geometry: a production-shaped snapshot cannot measure production

**Production trains on 1088 rows. Until 2026-10 no A/B point did.**
`predict.py` fetches `TRAINING_DAYS` = 1095 calendar days; the first 7 cannot
have their 7-day lags and rolling means computed and are dropped, so the frame
`model.train()` fits on is **1088 merged rows** (measured on
`ab_cache/2026-09-11`: 1088 rows, 2023-09-19 → 2026-09-10). Both walk-forward
loops instead *expand*: `min_train = n − 52·7`, and week *i* trains on
everything before it. On the canonical 1095-row window that is **731 rows for
the first test week, 1088 for the last, ~910 on average** — only the final week
matches production, and the average point trains on 84 % of what production has.

A production-shaped snapshot cannot fix this, and neither can a longer *frame*
alone: one evaluation year has to come out of the same window, so a
production-faithful point costs

```
1088 (train) + 364 (evaluate) + shift  rows
```

which is 364 rows more than a ~3-year fetch has at all. That is the whole reason
long snapshots are now the default and `ab_test.py run` passes
`train_window=ab.harness.PRODUCTION_TRAIN_ROWS`: with it, every test week trains
on exactly the rows production would have had on that date. `--expanding` falls
back to the old geometry, which is what every grid before 2026-10 used.

Capacity, measured on the caches that exist:

| long cache | rows | rolling points fit at shifts | clusters reachable |
|---|---|---|---|
| `long/2026-09-12` | 1787 | 0–335 | NOW, −6M, (−11M at 326–335) |
| `long/2026-09-16` | 1791 | 0–339 | NOW, −6M, (−11M) |
| `long/2026-10-10` | 1815 | 0–363 | NOW, −6M, (−11M); −12M needs 369 |

The canonical −12M cluster (shifts 360–369) therefore needs **1821 rows**, which
a long fetch from about 2026-10-16 provides; −21M needs 2098 and is not
reachable before mid-2027 (the frame cannot start before 2021-10-21, see
[below](#how-long-a-snapshot-to-fetch-and-why-a-long-one-is-shorter-than-you-asked-for)).
Until then a rolling grid runs NOW / −6M / −11M, which satisfies
`MIN_CLUSTERS = 3` but buys less period separation than −12M did (~10 % calendar
overlap with NOW instead of ~2 %).

**What the switch changed when it was measured** (2026-09-16,
`long/2026-09-12`, 216 walk-forwards, 7 arms picked from the ledger for known
size sensitivity, both geometries on the same 12 points): the two geometries
disagree enough to move verdicts. `cheap2h` + `radiation_midday` + `doy_sin/cos`
went **NOISE (+0.016) → REAL (−0.162)**, with a mode effect of −0.178 that was
sign-consistent across all three clusters; adding `price_se4_max_lag1` back to
`min` got more harmful (+0.324 → +0.429); the `min` hurdle went REAL → NOISE.
Four other arms kept their verdicts, including the +1.15 price-block control.
**Re-run on `long/2026-10-10` after the wind-curve fix, the seasonal arm
replicated** — exp NOISE (+0.070) → roll REAL (−0.217), mode effect −0.287,
negative in all three clusters — while the `min` + `price_se4_max_lag1` mode
effect did not (+0.106 → −0.012). Per-point noise on a mode effect is ±0.27, so
one REAL among seven arms is a chance event and the arm that survives two
independently fetched snapshots is the one to believe. Three further verdicts
moved in the re-run without a replicating mode effect (hurdle removal
KEEP_LOAD_BEARING → INCONCLUSIVE, the 13.1 combo BORDERLINE → NOISE, `min` +
fuel/CO2 NOISE → BORDERLINE): all sit within ~0.1 of a bar. The decision rests
on the arithmetic above, not on any one arm. Absolute MAE moved as well, and the same way in both
runs: rolling scored **0.29–0.39 better** than expanding at NOW on the priority
targets, ~0 at −6M, and 0.2 *worse* at −11M, so the headline eval is mildly
pessimistic in the period closest to production and the sign is
period-dependent — consistent with [round 15d](REJECTED.md) closing window
length as a lever.

### Tail-truncated vs sliding windows

`ab.harness.apply_shift` truncates the **tail** only, so a shifted point also
loses training history: on a 3-year snapshot `min_train` falls 721 → 541 → 361
across the NOW / −6M / −12M clusters. That confounds "different period" with
"less data", and it is not a theoretical worry — it produced three wrong or
grumbled readings (round 12's seasonal "gain", round 13.1's r = −0.6, and a
verdict that nearly removed the cheap2h hurdle; see [REJECTED.md](REJECTED.md)).

**Prefer a sliding constant-length window when a long snapshot is available.**
`ab_test.py fetch` caches ~5 years under `ab_cache/long/`; a window of
`rows[n-L-s : n-s]` then holds `min_train = L-364` fixed at every shift, so no
cluster is starved relative to another. Note what this does **not** give you,
and what this page claimed until 2026-10: it equalises training size *across*
points, but within each point the training set still expands 731 → 1088 unless
`train_window` is set, so a sliding point is "the headline eval as it would have
run on that date", not "production as it would have run on that date". Pair it
with `train_window=PRODUCTION_TRAIN_ROWS` (the `run` default) for the latter.
The worked implementation, and **the canonical 16-point grid** — rounds 14a, 14b,
16, 17 and 18 all reused it verbatim, which is why their answers are directly
comparable and why re-using it burns no new day-coverage:

```python
def window(data, length, shift):
    """Constant-length window ending `shift` days before the frame's end."""
    n = len(data)
    end, start = n - shift, n - shift - length
    assert start >= 1, f"window(L={length}, shift={shift}) starts before row 1"
    return data.iloc[start:end].reset_index(drop=True)

L_3Y = 1095                      # production's TRAINING_DAYS
GRID = [("NOW",    0), ("NOW",    1), ("NOW",    4), ("NOW",    7),
        ("-6M",  180), ("-6M",  183), ("-6M",  186), ("-6M",  189),
        ("-12M", 360), ("-12M", 363), ("-12M", 366), ("-12M", 369),
        ("-21M", 637), ("-21M", 640), ("-21M", 643), ("-21M", 646)]
```

Round 15b's own run used two NOW points rather than four (14 points total), and
round 21 followed it; either is fine, four is the default. What the clusters buy,
measured on `long/2026-08-06`:

| cluster | shifts | evaluation year | calendar overlap with NOW |
|---|---|---|---|
| NOW | 0, 1, 4, 7 | 2025-07-28 → 2026-08-02 | — |
| −6M | 180, 183, 186, 189 | 2025-01-27 → 2026-02-03 | ~50 % |
| −12M | 360, 363, 366, 369 | 2024-07-31 → 2025-08-07 | ~2 % |
| −21M | 637, 640, 643, 646 | 2023-10-28 → 2024-11-03 | 0 % |

**Points within a cluster overlap 97–100 %.** Their evaluation years differ by
one to twelve days, so they agree with each other almost by construction —
sign-consistency *within* a cluster is close to free, and only agreement *across*
clusters is a finding. That is the whole reason `classify_clustered` gives each
cluster one unweighted vote instead of counting points. Asserting that the points
have distinct end dates is still worth doing, but distinctness alone is a weak
guarantee, not independence.

### The rolling grid in practice (round 25): NOW / −6M / FAR, and what five caches replicate

Round 25 used 12 points per cache, offsets from that cache's own end: NOW 0/1/4/7,
−6M 180/183/186/189 and **FAR at the largest shifts the *expanding* 1095-row frame
still fits** — `n − 1095 − 364 − {9, 6, 3, 0}`, i.e. 7 fewer than the rolling
geometry could reach, because a mode comparison (rolling vs expanding) needs both on
the same points. 5d-tail caches (`2026-08-06`, `-08-21`) and 1d-tail caches
(`-09-12`, `-09-16`, `-10-10`) each got their own grid and were never pooled.
Two properties of this set of caches decide what "replicated on all five" means:

* **FAR is the same calendar window in every cache.** The frame cannot start before
  2021-10-21, so the largest shift ends the evaluation year on 2025-10-11 whatever
  the cache date; its deltas are identical to the digit across caches (cheap2h N1
  −0.918 five times). It replicates nothing.
* **NOW overlaps 92–99 % across the three 1d caches** (4–28 days apart), and ~80 %
  with the two 5d caches (about two months earlier). Five caches therefore test *fetch*
  dependence — the cross-vintage replay the bar asks for — and give two independent
  periods (NOW, −6M), not five. `long/2026-10-16` onward is the first genuinely
  later NOW window.

The null arm that prices this grid: identical features, xgboost `random_state` 7
instead of 42, `min` and `cheap2h`. Its cluster-mean delta against the production seed, per cache: **cheap2h 0.010 – 0.115** (0.04 / 0.03 / 0.01 on the three 1d caches, 0.115 / 0.083 on the two 5d caches), **min 0.008 – 0.029**. So a cheap2h effect below ~0.15 on one cache is indistinguishable from refitting with another seed, and `MIN_CLUSTER_EFFECT = 0.10` sits inside that band; `min` resolves ~0.05. What carries a small verdict is agreement across caches and clusters, never the size of one clmean. (The 5d caches' larger floor is one cache each — read it as "up to ~0.1", not as a property of 5d caches.)

### How long a snapshot to fetch, and why a long one is shorter than you asked for

**`--days 1825` is the right length. You will get fewer rows than that, and that
is correct behaviour, not damage.** A long fetch reaches back past the start of
the shortest-history source: EU ETS carbon (`CO2.L` on Yahoo) begins **2021-10-18**
and nothing earlier exists to fetch. `_warn_uncovered` leaves those days NaN
instead of zero-filling them, the final `dropna` removes the rows, and the frame
therefore begins **2021-10-21** — the first day every column is real — whatever
start date the fetch window implies. Measured on the three long caches:

| cache | days requested | usable rows | lost at the head | frame starts | weather tail |
|---|---|---|---|---|---|
| `long/2026-08-06` | 1822 | 1747 | 75 | 2021-10-21 | 5d |
| `long/2026-08-21` | 1822 | 1762 | 60 | 2021-10-21 | 5d |
| `long/2026-09-12` | 1825 | **1787** | 38 | 2021-10-21 | **1d** |

Three things follow, and they are the whole practical answer:

* **The waste shrinks on its own and disappears 2026-10-20.** The floor is a
  fixed calendar date, so every week that passes, a 1825-day window starts closer
  to it. From 2026-10-20 a 5-year fetch loses nothing at all.
* **Do not fetch longer to compensate.** Asking for 2200 days buys zero extra
  usable rows — it just widens the head that gets dropped, and lengthens a fetch
  that is already the slow part. 1825 stays the right number.
* **Every long cache has a different length.** 1747 / 1762 / 1787 above are three
  legitimate row counts. A grid script that asserts the count from the cache it
  was written against will trip on the next one; that assertion is doing its job.
  Update the constant deliberately, and re-check the shifts — the clusters are
  offsets from the *end* of the frame, so they stay put, but a `window()` that
  reaches furthest back has less room (see
  [DECISIONS.md](DECISIONS.md#nan-instead-of-zero-for-uncovered-price-sources)
  for the guard itself).

A long cache needs no repair and none of the three above has any: the snapshot
stores the **raw fetched inputs**, not the built frame, so `build_training_data`
re-applies the current guard every time one is loaded. Even a cache fetched
before the guard existed rebuilds correctly today. The only conceivable "fix" —
carbon prices before 2021-10-18 — is data the source does not have.

**Weather tails still matter more.** `long/2026-08-06` and `long/2026-08-21` are
5d-tail caches from before the archive top-up; `long/2026-09-12` is the first 1d
one. Per the rule above they must not share one measurement grid — that is a real
incompatibility, unlike the head.

Two points per cluster would be enough for the cluster vote; four in the far
clusters buys a per-cluster sanity read at negligible extra compute. The
within-cluster points overlap ~97 %, which is why
[`classify_clustered`](#which-verdict-function-to-call) gives each *cluster* one
unweighted vote rather than counting points.

**Especially important for any component that fits something internally** — a
classifier, cross-validation, an ensemble. Tail truncation starves it: the
hurdle's negative-price classifier sees ~21% positives, so 361 training rows
leave ~76 positives across a 5-fold split, and it measured as worthless when it
is in fact worth +0.250 EUR/MWh.

## Practices that keep a run honest

Five habits every round in the ledger converged on. They cost minutes and have
each caught a real error.

**Re-run a previous round's base arm as a free harness check.** If your new run
uses the established grid, its baseline arm is *the same computation* an earlier
round already did — so it must reproduce those numbers to the last decimal. A
mismatch means something non-deterministic crept in (thread count is the usual
culprit) and no delta in the run can be trusted. Rounds 12 and 16 both did this;
assert it, don't eyeball it.

**Store per-window results raw, aggregate afterwards.** Write one JSONL line per
(point, config) holding all 52 windows' MAE *and* their dates. Every aggregation
— pooled mean, per cluster, a seasonal LIGHT/DARK split, a regime correlation —
is then a post-hoc question answered without re-running anything. Round 2 stored
only per-target means and had to be re-run to answer a follow-up; round 11 stored
per-window and answered three.

**Make the run resumable.** Skip (point, config) pairs already on disk. These
runs are tens of minutes; an interrupted one should continue, not restart. For a
multi-target run a point counts as done only once *every* target's result is
written.

**Screen and confirm in two separate invocations.** Run the screen, look at it,
*then* pass the surviving configs to the confirm run as an explicit argument.
Same discipline as pre-registering the bar, enforced by the tooling rather than
by memory — it makes it structurally awkward to adjust what counts as a winner
after seeing the screen.

**A shared feature-list constant means a removal touches two models.**
`TROUGH_FEATURE_COLUMNS` was shared by `min` and `cheap2h`, so round 14a's
removal of `price_se4_max_lag1` would have silently changed the top-priority
target too. Round 17 measured the same column on `cheap2h` on the identical grid
before anything was edited, got the opposite verdict, and the two lists were
split. Before editing a shared constant, measure every target that reads it —
this is the scoping rule above, in its most concrete form.

## What a run costs

Measured at `OMP_NUM_THREADS=4`, one walk-forward (52 windows):

| target | one walk-forward | note |
|---|---|---|
| `min` | ~7.6 s | plain regressor |
| `avg` | ~17.0 s | time-decay weights |
| `cheap2h` | ~17.9 s | ~2.4× `min` — the hurdle's 5-fold `cross_val_predict` |

All targets are fit together in one `run_walk_forward` call, so a point costs
about the sum, not the max. Rules of thumb from the ledger: the 16-point grid
with 2 configs is ~10 min on 4 workers; 4 configs ~27 min serial and ~7–10 min on
4 workers; a 13-config audit ~59 min serial and ~15–20 min on 4 workers. A
NOW-only screen (4 points) is roughly a quarter of the full grid. Budget a
feature-audit round in hours, not days — the expensive thing in this project has
always been designing the measurement, not running it.

## Which verdict function to call

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

## Regime structure lives inside a run, not across shifts

Worth stating separately, because it is the mistake most easily made twice: if a
candidate's mean effect looks like noise but you suspect it is really
large-and-regime-dependent, the shift grid cannot answer that. All shifts in one
sweep cover the same ~365 days offset by a few, each window overlapping 6/7 with
its neighbour, so a sign flip across shifts means "unstable to the exact window
boundary", not "helps in winter, hurts in summer".

The regime axis is the **52 windows within** one run, where production MAE ranges
from about 3.8 (a calm month) to 33.3 (a solar-ramp month) — a 9× spread that a
per-shift mean averages away. The analysis that answers the question keeps the
per-window deltas from one shift and correlates them against window-level
descriptors (price level, volatility, negative-price share, wind, residual load).
Single-window deltas are noisy, so trust the correlation across the 52 windows
and the cross-shift stability of that correlation, never an individual window.

## Replaying the same windows on an independently fetched snapshot

`build_training_data` is a *fixed-length* sliding window, so two snapshots
fetched `N` days apart build to the **same length** and the later one is the
earlier one shifted `N` rows forward. That is what makes
`apply_shift(newer, N) == older` at shift 0 an exact replay rather than an
approximation — verified on the 2026-07-24 / 2026-07-26 pair, where all 1083
overlapping dates matched row for row at `N = 2`.

Assert it in the script rather than trusting the arithmetic: compare the
resulting last `date` against the older run's tested dates and require either an
exact match (for a replay) or no overlap at all (for extending coverage). A wrong
`N` does not error — it silently re-tests data you already have and inflates
apparent replication.
