# Slot prediction for cheap2h — implementation and evaluation plan (round 20)

**Status: PLANNED, not started. Nothing in this file is implemented.**

Goal: alongside `cheap2h` (the mean of the day's two cheapest hours), predict
*when* in the day those hours fall, so the Home Assistant automation can tell
"cheaper tomorrow, but only at midday when the car is away" apart from
"cheaper tomorrow night". This is an **addition** in the same sense as the
round-19c interval: the headline `cheap2h` number is unchanged, three numbers
are added next to it.

Read first: docs/AB_TESTING.md ("A/B Backtest Flow", "How changes are validated",
"Measurement grids", "Which verdict function to call", and "Prediction interval
for cheap2h (round 19c)" — the last one is the closest precedent for how an
addition is scored and shipped here. `IMPROVEMENT_PLAN.md` carries the standing
rules (state a mechanism, screen on one cluster and confirm on the others,
pre-register the bar). This file assumes all of that.

---

## 1. Decision already made (do not re-open without new evidence)

### 1.1 Represent slots as three regression targets, not a classifier

| Target column | Definition | Local-time hours |
|---|---|---|
| `price_cheap2h_night` | mean of the 2 cheapest hourly means within the slot | 00:00–07:59 |
| `price_cheap2h_day` | same | 08:00–15:59 |
| `price_cheap2h_evening` | same | 16:00–23:59 |

Computed in `features.aggregate_prices_daily` on the **same hourly resample and
the same Europe/Stockholm date** as `price_cheap2h`, so all four share one
target definition across the 15-minute MTU boundary (docs/MODEL.md "Target
Definition").

The slot label (`night`/`day`/`evening`) and the ambiguity ("is it close?") are
**derived** from the three values, in Home Assistant or at the very end of
`predict.py` — never modelled directly. Reasons, all measured in the 2026-09-05
exploration (section 6):

- **The two representations are equal in the decision that matters.** In the
  deadline simulation ("charging must finish by 08:00 on day+1"), a classifier
  gate and a direct per-slot regression landed at 2.7 vs 2.6 EUR/MWh mean
  regret. The classifier wins on label accuracy (77 % vs 72 %) but accuracy
  weights a 0.5 EUR miss the same as a 40 EUR miss; ~⅓ of days have the
  runner-up slot within 5 EUR/MWh, so most "misses" are free.
- **Regression plugs into the existing gate.** `model.TARGETS` → `evaluate.py`,
  `ab/` and the verdict rules all work in EUR/MWh MAE. A classifier would need
  its own evaluation path, its own verdict rule, and a new noise floor
  (52 weeks give ±2.3 pp standard error on accuracy).
- **Ambiguity becomes a number, not a class.** Days where night and day troughs
  are equally deep are common (see 6.2) but days where the *whole* curve is
  flat are rare (2 %); a "stable / all day" class would almost never fire.
  With three values, "stable" is `second_best − best < threshold`, chosen by
  the consumer.
- **Which side of midnight is unpredictable, and regression handles that
  honestly.** When the cheapest pair is at 22–23 the reachable 00–07 hours are
  in median 13 EUR/MWh dearer, so the distinction is real for a deadline; but
  both classifier and regression recall the `evening` class at ~25 % (base
  rate). A regression says "night 35, evening 38" and lets the consumer use
  the reachable one, instead of forcing a coin-flip label.

### 1.2 Calendar-day slots, boundaries at 00 / 08 / 16 — fixed, a contract

An earlier idea in the same exploration (night = 22–07, wrapping midnight)
scores better on label accuracy but is **wrong for the use case**: it would put
"Saturday 22–23" (after the car has left) and "Saturday 00–07" (before) in the
same bucket. Calendar-day slots keep every slot on one side of any deadline.

The boundaries are a contract with the Home Assistant automation. Changing them
later invalidates every stored MAE and every A/B that reported the slot
targets. Fix them now; if a different granularity is ever wanted, add targets,
do not move these.

### 1.3 Priority stays cheap2h → min → avg

The slot targets are reported alongside, never above, `cheap2h`. See section
4.3 for exactly how they participate in future A/Bs.

---

## 2. Implementation steps (production wiring)

Do these in order. Each step has a check; do not proceed on a failed check.

### 2.1 Targets in `features.aggregate_prices_daily`

Add the three columns to the `groupby("date").agg(...)`. The hour for the slot
cut is the **local** hour of the resampled hourly timestamp (the same
`_to_swedish_date` conversion, extended to expose the hour). Implementation
note: the current code converts to a date only; compute the local hour on the
same tz-converted series rather than on the UTC timestamp.

Checks:
- `min(night, day, evening) >= price_cheap2h` holds on **every** row (equality
  when both cheapest hours sit in one slot, strict when they are split across
  slots — ~12 % of days, mostly 23:00 + 00:00). This is an identity of the
  definition; a violation is a bug.
- DST transition days (23- or 25-hour days in local time): the night slot has
  7 or 9 hours. `nsmallest(2)` still works; assert no NaN is produced.
- `build_training_data` ends with `dropna()`; the three new columns must not
  introduce NaN rows (they cannot, if the identity above holds).

### 2.2 `model.TARGETS`

```python
"cheap2h_night":   ("price_cheap2h_night",   TROUGH_FEATURE_COLUMNS),
"cheap2h_day":     ("price_cheap2h_day",     TROUGH_FEATURE_COLUMNS),
"cheap2h_evening": ("price_cheap2h_evening", TROUGH_FEATURE_COLUMNS),
```

- Same list object as `cheap2h`, deliberately — see 4.3 for what that does and
  does not imply.
- `HALF_LIFE_DAYS`: add explicit `None` entries for the three (the `.get`
  default already yields None; make it visible).
- `HURDLE_TARGETS` stays `{"cheap2h"}` in this step. The hurdle on slots is a
  step-3 experiment (section 3), not a default.
- No slot-specific price lags (`price_se4_cheap2h_night_lag1` etc.) in this
  step. The trough targets' own frozen lags were measured harmful on
  `min`/`cheap2h` (see `features.py` comments around `MIN_FEATURE_COLUMNS`);
  for days 2–8 a lag is frozen anyway. Adding one is a step-3 experiment with
  a stated mechanism, not a default.

Check: **untouched targets must reproduce bit-identically.** Run
`evaluate.walk_forward_validate` on `ab_cache/2026-09-05` before and after,
`OMP_NUM_THREADS=4` both times, and compare the per-window MAE arrays for
`min`/`avg`/`max`/`cheap2h` with `np.array_equal`. Each target is its own
regressor with a fixed `random_state`, so adding keys to the dict must not
move them. If they move, something else changed — stop and find it.

### 2.3 `evaluate.py`

- The per-iteration print line hard-codes the four names; generalise it to
  iterate `TARGETS` (or print the slots on a second line). `mae_by_horizon`
  and the returned `mae_<name>` dicts already iterate `TARGETS` and need no
  change.
- `_IMPORTANCE_TARGETS` stays `("min", "avg", "cheap2h")` — do not push three
  more importance dicts to the HA sensor.

### 2.4 `predict.py`, `currency.py`, `ha_client.py`

`model.predict` iterates `TARGETS`, and `convert_predictions_to_sek` /
`apply_addon` iterate every key of every day, so the three numbers flow to
`predictions_raw` / `predictions_with_addon` **with no code change**, in
SEK/kWh, addon applied like everything else.

**Pitfall:** both conversion loops multiply every value. Do **not** put a
string slot label into the predictions dict — it would crash the conversion.
Two acceptable options, pick one and document it in the README HA section:

1. Push numbers only; derive the label in Home Assistant with a template
   (`min` over the three keys). Simplest, and keeps the payload uniform.
2. Add `cheap2h_slot` (string) and `cheap2h_slot_margin` (EUR→SEK-converted
   difference between the second-best and best slot) **after** conversion,
   inside `push_predictions`' `_to_list`, or in a small helper called from
   `predict.main` on the converted dicts.

Option 1 is recommended for the first release; the margin is one subtraction
in a template.

No interval for the slot targets in this step. If wanted later it is the
19c recipe on a different target and gets its own coverage guard rail.

### 2.5 Local integration check (the "A/B is the gate" trade-off)

Per docs/AB_TESTING.md "How changes are validated": no Actions run is required, but run
the local `train → predict → get_feature_importance` smoke test against a
cached snapshot (~2 min) and assert: seven keys per day, all finite, the three
slot values are ≥ the `cheap2h` prediction *only in truth, not necessarily in
prediction* (see 5.3), and the SEK conversion of a fabricated payload with the
new keys does not blank anything. The conversion path is exactly what no
backtest covers.

### 2.6 Documentation to update on shipping

- README "Architecture" (4 → 7 regressors); docs/MODEL.md "Target Definition" (the three
  definitions + the identity), "Home Assistant Integration" (new keys, the
  deadline rule in 5.2 as the worked example, the derivation of label/margin),
  "Model" / "Per-Target Feature Sets" (why they share the trough list), and
  the "Current MAE Baseline" table (three new rows, measured on the same
  snapshot and thread count as the existing rows — and **do not** compare
  them against the exploration numbers in section 6, which used a different
  model, different features and actual weather).
- `IMPROVEMENT_PLAN.md`: add the step-3 follow-ups (section 3) as open items,
  and note that item 3 (`min ≤ cheap2h` coherence) now has a sibling identity
  (5.3).
- `experiments/ROUND20_FINDINGS.md`: the scoring run results (section 4),
  same shape as `ROUND19_FINDINGS.md`.

---

## 3. What to evaluate before shipping (scoring run, round 20)

This is a **scoring run, not an A/B** (there is no baseline slot model to
delta against), exactly like round 19c. Pre-register the bar below *before*
running anything.

### 3.1 Grid

The round-15b sliding constant-length window on a **long** snapshot, four
period clusters NOW / −6M / −12M / −21M, `OMP_NUM_THREADS=4`. Use
`experiments/run_round15_long_window.py::window()` and the cluster map in
`experiments/analyze_round19c_conformal.py::CLUSTER_OF`.

Snapshot: prefer a fresh `python ab_test.py fetch --days 1825` (needs
`ENTSO_E_TOKEN`; Peter runs this locally). `ab_cache/long/2026-08-21` is
usable if a fresh one is not available, but check the weather tail first
(docs/AB_TESTING.md "Snapshot generations"). Slot share drifts strongly with solar build-out
(section 6.3), so the NOW cluster is the one that describes what production
will see; the older clusters describe robustness, not expected accuracy.

### 3.2 Per-window raw output

Write per-day rows like `run_round19c_quantiles.py` does
(`round20_raw_<shift>.csv`: `window, h, date, actual_cheap2h, pred_cheap2h,
actual_night, pred_night, actual_day, pred_day, actual_evening, pred_evening`).
Every metric below is computed from these files by an `analyze_round20.py`,
so the expensive fits run once.

### 3.3 Metrics and the pre-registered bar

**A. Per-slot MAE vs three naive references, per cluster.** For each slot:

| Reference | What it tests |
|---|---|
| "use the `cheap2h` prediction for every slot" | does a slot model add anything over the number we already have? |
| persistence (yesterday's actual slot value, frozen across the horizon) | the same reference every trough target is held to |
| monthly climatology of the slot value | the seasonal prior |

Bar: the slot model beats **all three** on its own MAE in **every** cluster.
Failing the first one for a slot means that slot target is not worth pushing.
Expect `evening` to be the worst (its cheap hours are 22–23 only, ~25 % of
night-trough days); that is the physics, not a defect.

**B. Decision regret under the deadline policy, per cluster.** Reproduce the
policies from the exploration on real harness predictions, for horizons
h = 1 and h = 2..7 separately (day+1 is known from Nordpool at 13:00 in
production, so the value of the slot lives at h ≥ 2):

| Policy | Rule |
|---|---|
| E never wait | charge today at today's known `cheap2h` |
| A cheap2h only | wait for day D if predicted `cheap2h(D)` < today's known cheap2h; pay D's actual **night** value (the only reachable slot before an 08:00 deadline) |
| C slot regression | wait if predicted `cheap2h_night(D)` < today's known cheap2h; pay D's actual night value |
| D slot as gate | as A, but only if predicted `night(D) − min(slots(D)) < 5` |

Regret = paid − best reachable actual. Bar, judged with the
`classify_clustered` vocabulary (one vote per cluster, deltas C−A and C−E):
**C must have lower mean regret than both A and E in every cluster.** The
exploration (section 6.4) found A *worse than never waiting*; if the harness
confirms that on horizon 2–7, it is the single strongest argument for the
feature and should be quoted in docs/MODEL.md.

Score decision metrics **per cluster, never pooled** — the 19c q90 rule was a
pooled artefact (`experiments/ROUND19_FINDINGS.md`).

**C. Sanity on the label derived from the values.** Argmin accuracy and the
confusion matrix per cluster, reported but **not gated**. Expect ~70 %
three-class in the NOW cluster, with `evening` recall near base rate. This is
recorded so nobody later "discovers" the midnight coin-flip and tries to fix
it with a classifier.

**D. Guard rails.** Untouched targets bit-identical (2.2). Per-window std of
the slot MAEs reported. Min-of-slots vs `cheap2h` coherence rate (5.3).

### 3.4 Compute

Three more regressors per window: +75 % on every walk-forward and A/B run
from now on. Acceptable; if it becomes a problem, give `run_walk_forward` /
`run_ab` a targets filter rather than removing the slots from `TARGETS`.

---

## 4. Future A/B work: does cheap2h evidence carry over to the slots?

Peter's question, answered here so the rule is written down.

### 4.1 Mechanically: partly automatic

The three slot targets reference the same `TROUGH_FEATURE_COLUMNS` list, so a
list edit validated for `cheap2h` reaches them without further wiring. The
hurdle does not (it is keyed by `HURDLE_TARGETS`), and neither does a
`HALF_LIFE_DAYS` change.

### 4.2 Evidentially: no

`cheap2h` is a **mixture**: on ~43 % of days its hours are night (wind/demand
driven), on ~41 % midday (solar driven), and the mix drifts by year. A feature
that helps `cheap2h` on average can help `day` and hurt `night`, or the other
way round, and the pooled `cheap2h` delta hides that. The project has already
recorded this failure mode once — the cheap2h prune silently expired the
hurdle's justification (docs/MODEL.md "Per-Target Feature Sets"). Verdicts are scoped
to the model they were measured on.

### 4.3 The rule

- **`cheap2h` remains the gate and the priority.** A change is adopted on its
  `cheap2h` (then `min`) verdict, as today.
- **The slots are reported in every A/B for free** — `run_ab` defaults to
  `model.TARGETS.keys()`, so once wired they appear in the verdict table. Read
  them as a **guard rail, like the interval's coverage**: a change adopted on
  `cheap2h` must not be `REAL`-harmful on a slot target. If it is, the change
  still ships for `cheap2h` but the harmful slot gets its **own** feature
  list (the same per-target divergence `min` and `cheap2h` already have), with
  the split measured, not assumed.
- **Slot-specific changes need a slot-primary A/B.** A feature added because
  of a `day`-slot mechanism (e.g. a radiation *level* feature — note the trough
  list carries only `radiation_variability`, no level and no season term) is
  judged on `cheap2h_day` first and must not harm `cheap2h`.
- **Do not wire a slot experiment into `TARGETS` mid-A/B.** `ab.variants.BASELINE`
  mirrors `model.TARGETS`; a candidate that edits the production dict collapses
  BASELINE == CANDIDATE. Use the variant's `targets` override, as for every
  feature-list experiment.

### 4.4 Step-3 experiments worth a stated mechanism (open items after shipping)

1. **Hurdle on `cheap2h_day`.** Mechanism: negative prices are a midday-solar
   regime in summer and a night-wind regime in winter; `P(price_min < 0)` is
   a mixture of both. A slot-specific hurdle (`P(slot value < 0)`) is the
   same idea, cleaner. Start with `day`; `night` second.
2. **A radiation-level / season feature for `cheap2h_day` only.** Mechanism:
   the day trough *is* the solar trough; the trough list has no radiation
   level and only `dow_sin` for calendar. This is where a genuinely new
   mechanism exists for one slot and not the others.
3. **Time-decay on the slot targets.** Mechanism: slot share moved 31 % → 65 %
   day in two years, so the *relative* level of the day slot has a trend the
   uniform 3-year fit averages away. Decay was NOISE on `cheap2h`; the mixture
   argument in 4.2 says the slot could differ. Measure, do not assume.
4. **Slot-specific price lag.** Only with a mechanism for why a frozen lag
   should help here when it hurt the trough targets. Default expectation:
   harmful.

---

## 5. Things to keep in mind

### 5.1 Horizon
Production runs once tomorrow's Nordpool prices are published, so day+1 is
never predicted. The slot value lives on days +2..+8, where every price lag is
frozen — the slot models are effectively weather + calendar models on those
days. Report `mae_by_horizon` for the slots; expect a flatter curve than for
`cheap2h` because the frozen lags carry less for them.

### 5.2 The Home Assistant rule this enables
"Charge before deadline T unless some day D before T has a predicted slot
value, **in a slot that ends before T and when the car is home**, lower than
the best known price before T." Write this into the README HA section with a
concrete template; it is the reason the feature exists.

### 5.3 Coherence, and the clamp lesson
`min(night, day, evening) ≥ price_cheap2h` is an identity in truth. In
predictions it can be violated (independent regressors). Measure the
violation rate in the scoring run and record it, but **do not clamp** without
an A/B: the avg-anchored clamp was rejected because clipping toward a wrong
neighbour moves away from truth (IMPROVEMENT_PLAN item 3). Treat it exactly
like `min ≤ cheap2h`.

### 5.4 Regime drift
Slot share by year (best slot on the calendar-day cut): 2024 day 31 % /
night 50 %; 2025 47 % / 40 %; 2026 65 % / 28 %. Month is decisive: Nov–Feb
night ≥ 75 %, Apr–Jun day 75–88 %. Never evaluate label accuracy pooled over
three years; per cluster, and read the NOW cluster as the production estimate.

### 5.5 What is fixed vs derived
Fixed: the three definitions and boundaries. Derived (consumer-side, cheap to
change): the label, the margin, any "stable" threshold. Keep it that way.

### 5.6 Thread count
`OMP_NUM_THREADS=4` for every measurement, as for every other table in the
docs.

---

## 6. Exploration record (2026-09-05, not reproducible from the repo)

Numbers below come from a scratch script over
`ab_cache/2026-09-05/prices_hourly.pkl` (1075 days, 2023-09-06 → 2026-09-05,
resampled to hourly means, Europe/Stockholm dates and hours) and a
GradientBoosting toy model on seven weather/calendar features plus yesterday's
slot values, using **actual** archive weather (optimistic), horizon day+1,
last 52 weeks walk-forward, one period. They justify the design decisions in
section 1; they are **not** baselines for anything in section 3.

### 6.1 Where the two cheapest hours fall
Hour histogram (2150 hours): 22–05 → 1212, 12–15 → 850, 16–21 → 25, 06–11 →
63. Best slot on the 00/08/16 cut: night 43 %, day 41 %, evening 15 % (almost
all at 22–23). Both hours in one slot 88 %; split 12 %, mostly 23:00 + 00:00.

### 6.2 Ambiguity (best-2h within slot, best vs runner-up slot)
gap ≤ 2 EUR/MWh 16 %; ≤ 5 31 %; ≤ 10 47 %; median 11. Runner-up is night↔day
in the large majority of close cases. Whole-day range ≤ 10 EUR/MWh on 2.2 %
of days only.

### 6.3 Drift
See 5.4. Persistence (yesterday's best slot) 61 % over three years, 71 % in
2026.

### 6.4 Toy models, last 52 weeks, 00/08/16 cut

| | Classifier | Per-slot regression (argmin) |
|---|---|---|
| Label accuracy | 77 % | 72 % |
| Recall night / day / evening | 0.80 / 0.86 / 0.25 | 0.69 / 0.87 / 0.17 |
| Regret, free slot choice (EUR/MWh) | 4.6 | 5.9 |
| Regret, deadline policy | 2.7 | 2.6 |

Per-slot toy MAE: night 18.6, day 18.2, evening 22.4; toy `cheap2h` on the
same features 14.5; min of the three slot predictions vs true cheap2h 13.7.

Deadline policy (today known, day+1 predicted, must finish by 08:00 on
day+1), mean regret / p90 / share of days with regret > 10 EUR/MWh:

| Policy | mean | p90 | > 10 |
|---|---|---|---|
| A cheap2h only (current logic) | 9.4 | 35.6 | 24 % |
| E never wait | 3.7 | 15.5 | 13 % |
| B cheap2h + classifier gate | 2.7 | 9.1 | 9 % |
| C predicted night vs today | 2.6 | 6.2 | 8 % |
| D cheap2h + regression gate (night within 5) | 3.4 | 10.6 | 10 % |

When the truth is `evening` (22–23), the reachable night hours are dearer by
a median 13.4 EUR/MWh (p75 40); when the truth is `night`, the evening hours
are dearer by a median 17.9. The side-of-midnight question is real for a
deadline and not learnable from daily features.

### 6.5 Core computation, for reference

```python
hourly = prices.set_index("timestamp")["price_eur_mwh"].resample("1h").mean().dropna()
local = hourly.index.tz_convert("Europe/Stockholm")
df = pd.DataFrame({"p": hourly.values, "date": local.date, "hour": local.hour})
df["slot"] = pd.cut(df["hour"], [-1, 7, 15, 23], labels=["night", "day", "evening"])
per_slot = df.groupby(["date", "slot"], observed=True)["p"].apply(lambda s: s.nsmallest(2).mean()).unstack()
cheap2h  = df.groupby("date")["p"].apply(lambda s: s.nsmallest(2).mean())
assert (per_slot.min(axis=1) >= cheap2h - 1e-9).all()
```
