# SE4 Prediction — Improvement Plan: New Ideas

Written 2026-11-10. A review of how `cheap2h` and its band (`cheap2h_low`, `cheap2h_high`,
`cheap2h_risk`) are predicted, of what [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) has open and
of what [docs/REJECTED.md](docs/REJECTED.md) has closed, looking for gaps and for ideas that
are in neither. **Nothing in this file is validated and nothing is implemented.** It is a list
of candidates, ranked by how likely each is to give a result.

**How it relates to the main plan.** [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) stays the
to-do list. An idea from this file that is taken up moves there under the next free item
number (2.22 onwards) with a State line, and is cut from here. Until then the ideas carry
their own ids, `NI-1` … `NI-11`, always written with their title. The standing rules of the
main plan apply to every one of them: a stated mechanism, a bar written before the run,
clustered verdicts on the rolling grid, forecast-vintage scoring, and the band scorecard for
anything that touches `min` or `cheap2h`.

**What the numbers in this file are.** To rank the ideas, every one that could be screened
cheaply was screened on
`ab_cache/long/2026-10-16` (one snapshot, three evaluation years, archive weather and forecast
vintages; [the appendix](#appendix-the-screens-behind-the-ranking) has the instrument and its
limits). A screen is a hint about where to look. It is one cache, one run and one seed, and
it is not a verdict: nothing below may enter a ledger from this file.

---

## 1. How the prediction is made today

Per day, from `se4/model.py` and `predict.py`:

1. **Coupling probability.** A linear logistic regression predicts `p_couple`, the probability
   that the day's trough prices at continental level (`cheap2h / gas_marginal_cost > 0.6`),
   from eight columns: Stockholm, Malmö and north-German wind, temperature, a 7-day radiation
   variability, workday, and the Norwegian and Swedish reservoirs.
2. **Negative-price probability.** An XGBoost classifier predicts P(`price_min` < 0) from
   `cheap2h`'s 17 regressor columns plus `p_couple`.
3. **Point estimate.** An XGBoost regressor (squared error, 500 trees, depth 5) predicts
   `cheap2h` from the 17 columns plus the two probabilities. The 17 are weather (five wind
   columns, two residual-load columns that are temperature in practice, a temperature
   gradient, radiation variability), fuel (TTF, gas marginal cost), three reservoir columns,
   two calendar columns and one price lag (`price_se4_max_lag1`). Fuel, reservoirs, the price
   lag and the variability column are frozen at the run day for all six sent days.
4. **Coherence.** Where the separate `min` model predicts above `cheap2h`, both become their
   mean.
5. **`cheap2h_low`.** A q10 quantile regressor on the same columns, lowered by a split-conformal
   correction from a 180-day holdout. The correction is the one computed for the old symmetric
   80 % band: one number from misses on *both* sides.
6. **`cheap2h_risk`** is `p_couple` from the classifier fitted on the whole window.
7. **`cheap2h_high`.** If the risk is 0.3 or more: 1.32 × the frozen gas marginal cost (the
   75th percentile of the ratio on coupled training days), the same number on every warned
   day. Otherwise: a q90 fitted on decoupled days plus one conformal correction per risk bin
   (below 0.1, and 0.1–0.3), about +11 and +54 EUR/MWh in October 2026.

Three properties of this design matter for everything below.

- **Every calibration is done on archive weather.** The holdout that sets the corrections for
  `low` and `high`, and the classifier's probabilities, see the weather that actually happened.
  Production serves them on a forecast two to seven days old.
- **One calibration serves all six days.** D+2 and D+7 get the same corrections and the same
  trust in the risk.
- **The classifier's eight inputs were fixed in round 24**, four days after the northern wind
  points shipped, and have not been revisited except to try nuclear inputs (rejected).

## 2. Where the error is

Screen numbers, six sent days (D+2…D+7), EUR/MWh, three evaluation years ending 2026-10-15
(NOW), 2026-04-18 (−6M) and 2025-10-20 (−12M). Where the docs have the same quantity the
screen reproduces it (weather penalty in NOW +3.50 against +3.35 for the same cache's NOW
cluster in FINDINGS; warning rate, precision and recall on archive weather 46 % / 69 % / 79 %
against the README's 46 / 69 / 80).

| | NOW | −6M | −12M |
|---|---|---|---|
| `cheap2h` MAE, archive weather | 13.5 | 12.0 | 12.2 |
| `cheap2h` MAE, forecast weather (what production pays) | 17.0 | 13.6 | 13.3 |
| … of which D+2…D+4: archive → forecast | 10.8 → 12.0 | 10.7 → 10.7 | 13.6 → 14.2 |
| … of which D+5…D+7: archive → forecast | 16.2 → 21.9 | 13.2 → 16.5 | 10.8 → 12.4 |
| share of days that coupled | 40 % | 38 % | 20 % |
| share of the absolute error on those days | 54 % | 56 % | 37 % |
| MAE on coupled days / on the others (forecast) | 23.0 / 12.9 | 20.2 / 9.7 | 24.7 / 10.5 |
| regret of "charge on the day predicted cheapest of six", archive → forecast | 4.4 → 7.5 | 2.3 → 5.5 | 4.1 → 5.5 |

Two blocks carry most of it.

- **The far days on a real forecast.** In the newest year the forecast costs +5.8 EUR/MWh at
  D+5…D+7 and +1.2 at D+2…D+4. In decision terms it is larger still: the regret of picking the
  cheapest day is 1.3 to 2.4 times what the archive eval reports.
- **The bimodal coupled days.** On coupled days the point estimate is low by 7 to 25, on the
  others high by 3 to 9. That is the hedge a squared-error regressor makes between "about 10"
  and "about 120", and no feature removes it.

Three cheap attacks on the first block were screened and did nothing (section 6): a linear
recalibration of the forecast weather, averaging predictions over several forecast vintages,
and training on forecast weather for the far days. The honest reading is that the point
estimate at D+5…D+7 is limited by what a wind forecast knows five to seven days out. What can
be improved for certain is how truthfully the band says so, and that is where the top of the
ranking sits.

## 3. Gaps found in the plan, the ledgers and the docs

1. **`cheap2h_low` is calibrated with the wrong score.** Since round 24 only the low half of
   the symmetric band is served, but its correction is still the two-sided one, driven by
   upside misses on coupled days. In the newest year it pushes `low` down by 12 EUR/MWh for
   no reason: the outcome ends below `low` on 1 % of days against a nominal 10 %, and the
   served `low` averages about 0 EUR/MWh next to a point estimate of 36. → NI-1.
2. **The band is calibrated and scored on archive weather only.** `ab_test.py band` does not
   score forecast vintages (AB_TESTING.md says so), and the conformal holdout does not use
   them. On forecast weather the promise "not warned means about 1 day in 10 ends above
   `high`" does not hold at D+5…D+7. → NI-2.
3. **The warning numbers in the README are archive-weather numbers.** "~69 % of warnings
   correct, ~80 % of expensive days caught" becomes 63 % and 69 % on forecast weather in the
   same year. Not an idea, a correction to make when NI-2's instrument exists.
4. **The coupling classifier never got the northern wind.** FINDINGS says the northern supply
   balance is what separates a coupled day from a decoupled one; the regressor reads the two
   northern columns, the classifier that produces `cheap2h_risk` and the warning does not. → NI-3.
5. **Decision regret has been measured once** (round 19c, archive weather, before the coupling
   hurdle), and the plan's next measurement waits for months of production log. Forecast
   vintages allow it now, for two years, including for the automation's own rule. → NI-4.
6. **2.3c Train on forecast weather is blocked by the 2.3b verdict** on the argument that the
   verdict tells whether the loss is in the forecast or in the model. The blend recovers
   about a quarter, so three quarters remain either way; and the cheapest form of 2.3c did
   nothing in the screen (section 6). Suggest lowering its expectation rather than its
   priority; if it is run, it needs the per-lead design, not a shortcut.
7. **The ECMWF ensemble has no forecast archive**, so every day it is not stored is lost for
   2.3d Weather-uncertainty band and for anything else that wants ensemble members. → NI-7.
8. **The MAE table in MODEL.md** is from the 2026-08-21 snapshot and predates five shipped
   changes. Known and stated there; listed for completeness.

## 4. The ranking

"Likely" means likely to produce a result that passes a bar of the kind this repo writes, not
likely to be large.

| # | Idea | Improves | Screen evidence | Likelihood | Effort |
|---|---|---|---|---|---|
| 1 | **NI-1 `cheap2h_low` calibrated on its own side** | `low` | below-`low` 1–2 % → 4–14 %, `low` 5–12 EUR/MWh closer to the point in the two recent years | Very likely | Hours |
| 2 | **NI-2 Band calibrated and scored on forecast weather, per lead group** | `high`, `risk`, `low` | unwarned days above `high` at D+5…D+7: 9 % on archive, 23 % on forecast (NOW) | Very likely | 1–2 days |
| 3 | **NI-3 Northern wind and the recent coupling rate in the coupling classifier** | `risk`, `high`, maybe the point | log loss −0.03 … −0.08 in the two recent years on forecast weather, flat to slightly worse in the oldest | Likely for the band, uncertain for the point | Half a day + A/B |
| 4 | **NI-4 Decision-regret scorecard on forecast vintages** | what the automation pays; a guard rail for every change | a `high` guard on deferral was worse than the plain point rule in all three years | Very likely to give a usable answer; it is an instrument, not a model change | 1 day |
| 5 | **NI-5 A coupled level that depends on the day** | `high` on warned days | cross-validated R² 0.25 for the coupled ratio from temperature, midday radiation and workday | Likely, modest | Half a day |
| 6 | **NI-6 Recency-weighted conformal corrections** | `low`, `high` through a regime change | the correction a year needs differs 5 → 12 EUR/MWh between the three years | Uncertain | Half a day |
| 7 | **NI-7 Store the ensemble forecast now; predict through its members later** | `high`, `risk`, far-day point | none possible (no archive) | The log is certain and cheap; the gain is uncertain | 2 hours for the log |
| 8 | **NI-8 An hourly first stage for the trough** | point, slot targets | not screened | Uncertain; the largest possible gain on the point | 3–5 days |
| 9 | **NI-9 Training rows weighted by regime similarity, from a longer pool** | point in a regime shift, the ceiling | not screened | Uncertain | 2–3 days incl. a longer fetch |
| 10 | **NI-10 Forward power prices as the coupled level** | `high`, point on coupled days | not screened; source not probed | Long shot | 2 hours to probe, then 2 days |
| 11 | **NI-11 Pooling SE3 and DK2 rows with SE4** | point | not screened | Long shot | 2 days |

The first three are band changes or feed the band, and that is deliberate. The repo's own
history says the point estimate moves by tenths and often not at all, and six point-estimate
ideas screened for this document came back as noise or harmful (section 6). The band has
measurable defects with a known fix.

---

## 5. The ideas

### NI-1 `cheap2h_low` calibrated on its own side

**Mechanism.** `train_interval` computes one correction from `max(q10 − y, y − q90)` at
α = 0.20 and subtracts it from the q10. Most of that score comes from outcomes above the q90
on coupled days, which have nothing to do with how low a day can go. The q90 it is paired
with is no longer served at all. A one-sided split-conformal correction for the low side
alone (score `q10 − y` on the holdout, its 90th percentile, allowed to be negative) targets
what `low` claims: about one day in ten ends below it.

**Screen** (archive weather, D+2…D+4 / D+5…D+7):

| | NOW | −6M | −12M |
|---|---|---|---|
| outcome below `low`, today | 1.3 % / 0.6 % | 1.9 % / 1.9 % | 7.1 % / 13.5 % |
| the same, one-sided | 14.1 % / 9.6 % | 4.5 % / 6.4 % | 5.8 % / 12.2 % |
| mean `cheap2h − low`, today | 31.4 / 40.9 | 24.5 / 26.2 | 20.7 / 16.0 |
| the same, one-sided | 19.0 / 28.4 | 19.6 / 21.3 | 21.3 / 16.6 |

In the newest year the served `low` averages −0.1 EUR/MWh against 12.4 for the one-sided
version and 36.0 for the point. A floor that is always about zero tells the automation
nothing.

**Why this is not the rejected asymmetric band.** The round-24 row tested one correction per
side on the *symmetric* band and was rejected for what it did to the high side (a one-mode
band cannot say "either 10 or 120"). The high side has since been replaced. Its low-side
number (3 % → 15 %) is the same effect as here and overshot in the autumn weeks, which is why
the bar below has an upper limit.

**What to build.** In `train_interval`, a separate low-side correction; `IntervalModel`
keeps the clamp `low ≤ cheap2h`. The `coverage_calibrated` diagnostic loses its meaning and
is replaced by the low side's holdout miss rate. Band only: no point model changes.

**Bar (suggested).** `below_low` between 5 % and 15 % in every cluster and in the last 56
days, on archive and on forecast weather (NI-2's instrument), and mean `cheap2h − low` not
larger than today in any cluster.

**If it fails.** If the miss rate swings outside 5–15 % between clusters, the 180-day uniform
holdout is the problem and NI-6 Recency-weighted conformal corrections is the next step.

### NI-2 Band calibrated and scored on forecast weather, per lead group

**Mechanism.** The corrections for `high` and the probabilities behind the warning are
calibrated on days whose weather is known exactly. At D+5…D+7 the served wind is a forecast
that has lost much of its skill, so a day the classifier calls safe is less safe than the
calibration believes. Conformal calibration fixes this by construction if the holdout is
scored the way the band is served: on the forecast issued L days earlier.

**Screen** (same fits, test rows on archive weather against forecast vintages at production's
lead map):

| D+5…D+7 | NOW | −6M | −12M |
|---|---|---|---|
| unwarned days that ended above `high`: archive → forecast | 9.0 % → 22.9 % | 16.1 % → 19.8 % | 7.4 % → 11.0 % |
| expensive days warned (recall) | 0.83 → 0.71 | 0.67 → 0.64 | 0.57 → 0.52 |
| warnings that came true (precision) | 0.73 → 0.64 | 0.71 → 0.72 | 0.60 → 0.55 |
| Brier score of `cheap2h_risk` | 0.156 → 0.210 | 0.191 → 0.211 | 0.078 → 0.089 |

At D+2…D+4 the difference is small and mixed (unwarned above `high` 8.8 % → 5.7 %,
8.9 % → 13.0 %, 14.8 % → 18.1 %). The risk is over-confident at both ends on far days: over the
three years, days served a risk below 0.1 at D+5…D+7 coupled 11 % of the time (5 % on archive
weather) and days served 0.1–0.3 coupled 31 %.

**What to build**, in this order:

1. *The instrument.* `ab.band.band_walk_forward` scores each window's test rows a second time
   on the vintage frames, as `ab.harness.run_walk_forward_errors` already does for MAE, and
   `score_band` reports per cluster × {D+2…D+4, D+5…D+7} × {archive, forecast}. Rows are
   weekdays in a step-7 grid, so compare archive against forecast *within* a lead group, never
   one lead group against the other.
2. *The calibration.* In `train_band`, score the 180 holdout days on forecast-vintage
   features, once per lead, and compute the risk-bin corrections per lead group. Each holdout
   day has a vintage at every lead, so a cell gets three leads × its days and does not starve.
3. *The risk.* A one-parameter-per-lead-group recalibration of the classifier's probability
   on the same holdout (it pulls far-day probabilities toward the base rate), applied before
   the 0.3 threshold.
4. *Production.* The Previous Runs request for the last ~190 days at the nine weather points,
   one request per point, kept in the source cache under its own version
   ([DATA_CACHE.md](docs/DATA_CACHE.md)).

**Bar (suggested).** On forecast-vintage scoring: `above_high_unwarned` between 7 % and 13 %
in both lead groups in every cluster; recall not lower by more than 5 points; `lowrisk_high`
at D+2…D+4 not higher by more than 5 EUR/MWh. The price is known in advance and should be
written into the entry: `high` on unwarned far days will be visibly higher.

**If it fails.** If a calibrated far-day `high` is so high that it no longer separates days,
that is the finding: the band cannot support a deferral to D+5…D+7, and the README's Home
Assistant section should say so.

### NI-3 Northern wind and the recent coupling rate in the coupling classifier

**Mechanism.** Two things the classifier cannot see. (a) The northern supply balance:
FINDINGS records that with Stockholm wind calm, a calm north gave 73.5 EUR/MWh and a windy
north 41.5. `north_wind_power` is in the regressor and not in `COUPLING_FEATURE_COLUMNS`.
(b) How often coupling happens now: the second of the three reasons the autumn-2026 miss was
"it under-rated how often coupling had become". A trailing 14-day share of coupled days
(`cpl_share14`, the label through the last published day, frozen across the forecast) lets a
linear model move its base rate within days instead of waiting for the reservoir terms.

**Screen** (classifier alone, weekly origins, change in log loss against production's eight
columns; negative is better):

| arm | NOW | −6M | −12M |
|---|---|---|---|
| forecast weather: + `north_wind_power` | −0.048 | −0.031 | +0.018 |
| forecast weather: + `cpl_share14` | −0.029 | −0.032 | −0.012 |
| forecast weather: + both + `radiation_midday` | −0.079 | −0.050 | +0.021 |
| archive weather: + both + `radiation_midday` | −0.092 | −0.084 | +0.002 |

AUC on forecast weather in the newest year 0.80 → 0.86 for the combined arm; at the 0.3
threshold recall 0.69 → 0.76 and precision 0.63 → 0.67 (9 fewer missed expensive days and 4
fewer false alarms in 312 days). The combined arm was run with `radiation_midday` in it;
that column alone and `gas_marginal_cost` alone do nothing, so the arms to measure are the
two columns named in the title. The oldest year, where coupling was rare, does not gain from the northern column.

**Why this is not round 19e.** That round put price-derived coupling lags into the *tree
regressor* as features and got noise. This puts a base-rate signal into the *linear
classifier*, which is the component FINDINGS showed can use a state variable the trees cannot.
It is still a price-derived lag, so it gets a permuted control and the freeze.

**What to build.** Three arms: `+north_wind_power`; `+cpl_share14` (added to
`FORECAST_FROZEN_FEATURES`); both. Production computes the share through D+1, which is
published when it runs.

**Bar (suggested).** `ab_test.py run --targets cheap2h` on the rolling grid with vintages:
`min` and `cheap2h` not REAL-harmful in any cluster (the regressor already reads the northern
columns, so a point gain is not expected). `ab_test.py band`, on NI-2's forecast scoring:
recall and precision not worse in any cluster and better by at least 3 points in NOW.
The change moves the point models, so it restarts the 2.3b Three-model weather blend shadow's
clock; take it after that verdict.

**If it fails.** If −12M-type years get worse and the recent ones better, the gain is a
regime effect and the column belongs behind a regularisation choice (`C`), not in the list.

### NI-4 Decision-regret scorecard on forecast vintages

**Mechanism.** The automation is judged on what it pays, and regret can be computed for any
rule from predictions the harness already makes. Two things make it worth doing now. The
weather axis costs more in regret than in MAE (section 2). And a rule can matter more than
any model change in the ledger.

**Screen.** One decision per test week: charge now at the known price (the cheaper of D+0 and
D+1) or wait for the day predicted cheapest; regret against the best of all eight days;
forecast weather:

| rule | NOW | −6M | −12M |
|---|---|---|---|
| never wait | 24.9 | 9.7 | 7.9 |
| wait if the predicted `cheap2h` is below the known price | 6.9 | 4.0 | 3.7 |
| the same, and only if that day's `cheap2h_high` is also below the known price | 12.6 | 4.6 | 4.2 |
| the same, and only if that day's risk is below 0.3 | 8.3 | 5.2 | 3.7 |

The `high` guard was worse on the mean in all three years and bought no real protection in
the tail (90th percentile of regret 31.9 against 21.2 in the newest year, the worst week
unchanged). Adding a margin to the comparison, or a handicap on the far
days, did not help either. **This is a simplified rule on a weekly grid, not the automation's
rule**, so it says the README's advice ("do not defer to a day whose `cheap2h_high` is above
today's known cheap price") should be scored, not that it is wrong.

**What to build.** `ab_test.py regret`: the rolling-origin geometry already designed for
2.18 A near-horizon model (seven origins per week), archive and forecast weather, per cluster,
mean / 90th percentile / worst. Rules: pick-cheapest-of-six, wait-or-charge, and the Node-RED
rule as Peter states it. Report it next to the band scorecard for every `min` / `cheap2h`
change. A later product step, if the rule turns out to want it: serve the probability that a
day ends below a given price, from `risk` and the two quantile models.

**Bar.** None; it is instrumentation, like 2.2 Production prediction log and realised
scorecard, which it complements: two years of backtest now against a few months of log later.

### NI-5 A coupled level that depends on the day

**Mechanism.** On every warned day `high` is one number, 1.32 × the frozen gas cost. But the
ratio on coupled days is not one number: over the 321 coupled days of the newest training
window its mean is 1.07 and its standard deviation 0.30, which is ±31 EUR/MWh at the current
gas cost. Part of that is predictable from the day itself: it
correlates −0.36 with temperature and −0.30 with midday radiation (a sunny midday pulls the
continental trough down even when SE4 is coupled), and weekends sit at 1.00 against 1.08 on
workdays. A linear model of the ratio on coupled days reaches a cross-validated R² of 0.25.

**What to build.** In `train_band`: a small linear model of `cheap2h / gas_marginal_cost` on
the coupled training days (workday, `mean_temp`, `radiation_midday`, north-German wind), and
the coupled level = its prediction plus the 75th percentile of its residuals, times the gas
cost. All inputs are per-day forecastable. `high` then differs between the warned days of a
week, in EUR/MWh.

**Bar (suggested).** Band scorecard on forecast scoring: `above_high` on warned days and
`missed_cost` not worse in any cluster, mean `high` on warned days that did not couple lower.

**If it fails.** The level stays a constant and the README's advice to grade warned days by
`cheap2h_risk` stands.

### NI-6 Recency-weighted conformal corrections

**Mechanism.** The holdout is 180 days with equal weight, so a correction reacts to a new
regime over months. The screen shows how much the right correction moves: the symmetric
correction averages 5.3, 9.3 and 12.2 EUR/MWh in the three years (oldest to newest), the
one-sided low correction 5.9, 4.4 and −0.4. During the August break coverage fell to 0.14.

**What to build.** Weight the conformity scores by `0.5 ** (age / 45 days)` and take the
weighted quantile; same holdout, no new state. Later, with the prediction log, the step size
can react to served misses (adaptive conformal inference).

**Bar (suggested).** `below_low` and `above_high_unwarned` closer to nominal in every cluster
*and* in the last 56 days, on forecast scoring. Take it after NI-1 and NI-2, which change
what is being calibrated.

**If it fails.** The effective sample is too small (a 45-day half-life leaves about 65 days
of weight) and the uniform holdout was the better trade.

### NI-7 Store the ensemble forecast now; predict through its members later

**Mechanism.** At D+5…D+7 the model is given one wind forecast and treats it as the truth.
The better input is the distribution: run the fitted models on each of the 51 ECMWF ensemble
members and average the *predictions* (not the weather; the price is not linear in wind, and
with a coupling state it is close to a step). The mean is a better point, the mean of the
members' coupling probabilities a better risk, and their spread a day-specific width for the
band. This is the full form of 2.3d Weather-uncertainty band, which only conditions the band
on the wind spread.

**The gap that is costing days.** Open-Meteo keeps no archive of ensemble forecasts, so none
of this can be backtested, and every day not stored cannot be recovered. Storing the *inputs*
is not a shadow arm: it predicts nothing, freezes nothing on main, and any later model can be
replayed on it.

**What to build now.** One fetch per run of the ensemble (nine points, three variables, eight
days, 51 members), written as one compressed file per day. The source cache is the wrong
place (GitHub evicts it after seven idle days); a workflow artifact with a periodic download,
or a data-only branch, are the candidates and the choice is Peter's. About 260 000 values a
day, on the order of 1 MB before compression.

**Evidence, such as it is.** The three-model blend recovers a quarter of the far-day penalty.
Averaging predictions over three successive vintages of the *same* model did nothing in the
screen (section 6), which lowers the prior for the point; the band and the risk are the more
likely winners.

**Bar.** For the log: none. For the propagation, written when three months of members exist:
scored on the log against the served prediction on the same days, per lead group.

### NI-8 An hourly first stage for the trough

**Mechanism.** The trough is formed at an hour: night wind against night demand, or midday
sun. The daily model sees daily means and has 1088 rows to learn which of the two it is,
which is where the seasonal interactions (round 12, round 25) and the solar features failed
for lack of data. An hourly model of the price on hourly weather has 26 000 rows and learns
hour × season × wind × sun directly. Its predicted curve gives two daily columns for the
existing model: the mean of its two cheapest predicted hours and the hour they fall in. The
slot targets of 2.6 Slot targets fall out of the same curve.

**Why this is not round 29's calibrated residual.** That first stage predicted the physical
residual load, which weather explains well, and what was left over carried the price. This one
predicts the price at the resolution where the trough is set; the information it adds is the
within-day alignment of wind, sun and demand that daily aggregation removes.

**What to build.** Hourly frame from the snapshot's hourly weather and prices; the day's slow
state (fuel, reservoirs, `p_couple`) repeated per hour; out-of-fold predictions in whole-week
blocks; two daily columns into `cheap2h` and `min`; forecast-vintage frames exist hourly, so
vintage scoring works unchanged.

**Bar (suggested).** `classify_clustered` REAL on `cheap2h` on archive weather, negative in
every cluster on forecast vintages at production's map, both net of a two-column permuted
control (two noise columns cost about +0.1), and the band scorecard without a breach.

**If it fails.** Daily resolution is not what limits `cheap2h`, the README's limitation can
say so, and 2.6 Slot targets should expect slot models no better than the daily one.

### NI-9 Training rows weighted by regime similarity, from a longer pool

**Mechanism.** Window length is closed in both directions for a reason the ledger states: a
longer window helps when the added year resembles the present and hurts when it does not, and
a five-year window raises the prediction ceiling from 108 to 450 at three times the calm-day
error. Time decay is closed too, because recent is not the same as similar. Weights by
*similarity of state* take the old rows only when they are relevant: full weight for the last
three years (so the calm case is production), and for older rows a kernel on the distance in
`gas_marginal_cost` and `reservoir_norway_deviation` to the run day's frozen values. One
weight vector per run, because the state is frozen across the six days.

**What to build.** A snapshot longer than five years (the rolling grid leaves the oldest
cluster no older pool otherwise), the weight function, `sample_weight` through the existing
`HALF_LIFE_DAYS` path. Score the August break weeks apart, as 2.19 Time index for `avg` does.

**Bar (suggested).** `classify_clustered` on `cheap2h`, not REAL-harmful on `min`, the
calm-week MAE before 2026-08-17 not worse, and the band without a breach.

**If it fails.** The ceiling stays where FINDINGS left it: not raisable from inside the model.

### NI-10 Forward power prices as the coupled level

**Mechanism.** Every price signal tried on the trough targets was a *past* spot price, and the
rejection reads "these targets reject a price level however it is encoded". A week-ahead power
future is a different object: the market's forecast for the delivery week, with the outages,
the hydrology and the weather outlook already in it. The most direct use avoids the regressor
entirely: the coupled level in `cheap2h_high` is today 1.32 × a frozen gas cost, and a German
week future times the usual trough-to-base ratio is a better estimate of what a coupled day
costs. It also carries what the row on news and oil in REJECTED asked for: a source that TTF
does not already contain.

**What to do first.** Probe for a free daily settlement series with at least four years of
history (German week or month base; the Nordic system price and the SE4 area differential).
If none exists, close it there. Two hours.

**Bar.** Written after the probe. As a band input: NI-5's bar. As a regressor feature it
needs the full grid and should be expected to fail like the other price levels.

### NI-11 Pooling SE3 and DK2 rows with SE4

**Mechanism.** The models are short of rows, not of columns: eleven noise columns cost +0.33,
a reseed moves `cheap2h` by 0.13, and more years do not help because older years are another
regime. Rows from *other zones in the same years* are the same regime. SE4's trough correlated
0.89 with DK2's in 2026. Training one model on SE4, DK2 and SE3 rows with a zone column
triples the rows for the weather-to-trough mapping.

**What to build.** SE3 day-ahead prices (one new ENTSO-E request), the same feature frame per
zone, a zone indicator, prediction for SE4 only.

**Bar (suggested).** `classify_clustered` on `cheap2h`. **If it fails**, the zones' price
formation differs more than their troughs' correlation suggests.

---

## 6. Screened for this document and dropped

So that they are not proposed again without a new mechanism. All on forecast weather, six sent
days, NOW / −6M / −12M, change in `cheap2h` MAE against production unless stated.

| What was tried | Screen result | Reading |
|---|---|---|
| Monotone constraints on the regressors (wind and reservoirs down, fuel, workday and `p_couple` up) | +0.56 / −0.14 / +0.52; `min` +0.29 / +0.09 / +1.53 | The trees want the non-monotone parts |
| `asinh` transform of the target | +0.33 / +0.48 / −0.94 | Sign flips between years |
| Explicit mixture: (1 − risk) × a regressor trained on decoupled days + risk × mean coupled ratio × gas cost | +4.5 / +3.6 / +0.6 | The tree with `p_couple` as a feature is far better than the linear mix; the hedge is not fixed this way |
| Linear recalibration of each forecast weather column per lead, fitted on the training rows' vintages | +0.03 / +0.12 / −0.37; 25, 23 and 31 of 52 weeks favourable | Noise |
| Averaging the prediction over three successive forecast vintages | D+5…D+7 −0.13 / +0.89 / +0.22 | Noise; older forecasts add error, not diversity |
| Training on forecast weather for the far days (the frame's weather replaced by the lead-6 forecast from 2024-05; older rows keep archive weather), scored at D+5…D+7 | +0.65 / +0.14 / −0.12; 23, 28 and 23 of 52 weeks favourable; with a column flagging the forecast rows +0.19 / +0.18 / +0.66 | The cheapest form of 2.3c does nothing; see gap 6 |
| Residual anchor: the model's own error on the last three known days, added with a weight that decays over the horizon | D+2…D+4 +1.40 / +0.20 / +0.30 | The last day's error correlates 0.25 with D+2's and ~0 beyond: one more confirmation that the trough targets reject recent price information |
| A handicap on the far days, or a margin, in the decision rule | no improvement in any year | The far days still rank usefully |
| `radiation_midday` or `gas_marginal_cost` as a coupling-classifier input | `radiation_midday` within ±0.007 log loss; gas −0.022 in one year, ±0.007 in the others | Nothing |

## 7. Looked at and not proposed: the ledger already closes it

Checked against [docs/REJECTED.md](docs/REJECTED.md) so that this file adds nothing that is
closed there: price lags and price levels on the trough targets in any encoding; the anchored
target and the spread decomposition; window length in both directions; time decay and the
time index on `min` / `cheap2h`; seed ensembling; per-target hyperparameters; absolute-error
loss; neighbour holidays; cross-border flows and capacity; nuclear outages, clean or
decomposed; week-ahead and realised load; the solar-capacity index and the seasonal
interactions; the MW-scale and the calibrated residual load on all horizons; ranking days by
the q90. NI-3, NI-8, NI-9 and NI-10 each sit next to one of these and say in their own text
why the mechanism is different.

## 8. Order of work against the open plan

- Nothing here is wired in before the November run (2.19 Time index for `avg`, 2.20 Nuclear
  column as production sees it, 2.18 A near-horizon model; from 2026-11-13) has its verdicts:
  a change on main makes a waiting result stale.
- **NI-1 and NI-2 are band-only.** They do not move a point prediction, so they do not restart
  the 2.3b Three-model weather blend shadow, whose bar is on `cheap2h` and `min`. They do
  change what the prediction log holds for `low` and `high`; the log's `git_sha` column
  separates before from after.
- **NI-4 touches no model** and can be built at any time, together with the rolling origins
  2.18 needs.
- **NI-7's log** is independent of everything and loses a day for every day it waits.
- **NI-3 moves the point models** and waits for the 2.3b verdict.
- NI-5 and NI-6 follow NI-2, which gives them their instrument. NI-8 to NI-11 are taken only
  when the main plan's 4.1 is empty.

---

## Appendix: the screens behind the ranking

**Instrument.** `ab_cache/long/2026-10-16` (read-only), frame built by the unmodified
`features.build_training_data`; shifts 0, 180 and 360 (evaluation years ending 2026-10-15,
2026-04-18 and 2025-10-20; these are NOW, −6M and −12M, not the standard NOW / −6M / FAR
grid); 52 weekly windows per shift, rolling 1088 training rows, production's
`apply_forecast_freeze`; `min` and `cheap2h` fitted by `model._fit_models`, coherence applied,
the band by `model.train_band`. Every window scored on archive weather and on the cached
forecast vintages (`ab_cache/vintages`, nothing fetched) at production's lead map: sent row j
on the forecast issued j + 2 days earlier. Only the six sent rows are reported. Windows,
xgboost 3.3.0, `OMP_NUM_THREADS=4`.

**What makes these hints and not verdicts.**

- One snapshot, one seed, one run per arm. The repo's seed floor on `cheap2h` is 0.13 and the
  noise between clusters several tenths; a screen delta under about 0.5 in one year means
  nothing by itself.
- Three years from one cache are three periods, not replication of the fetch.
- Lead groups are weekday groups in a step-7 grid (row 0 was a Friday, a Sunday and a Tuesday
  in the three years). Archive against forecast within a group is a clean comparison; one
  group against the other is not.
- The decision screens take one decision per week from a fixed weekday.
- The classifier arms were scored on the classifier alone (log loss, AUC, warnings at 0.3),
  not through the regressors.
- The band numbers (NI-1, NI-2) are the exception in kind: they are miss rates of the shipped
  band against its own nominal rates, over 156 to 312 days per cell, and the archive figures
  reproduce the README's. They are still one cache.

**Scripts.** Scratch, written outside the repository for this session and not kept. The
descriptions above and in each idea are the specification; every number in this file is a
screen number unless it cites a ledger.
