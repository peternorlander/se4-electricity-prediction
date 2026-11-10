# SE4 Prediction — Improvement Plan

The to-do list for model accuracy: only what is **not yet done**, which includes work that
has been started and paused. What is open, in the order it should be done, is
[section 4](#4-what-to-do-next): what is ready now, what is blocked and by what, and the
shadow arms running in production. **"The next step" is the first item of
[4.1](#41-ready-now-in-order); if 4.1 is empty, there is nothing to do yet** — blocked items
are never started early or in part.

**Item numbers are stable.** They are the September review's, and the docs, ledgers and
commits refer to items by number ("plan 2.19", "2.1-C"), so a closed item leaves a gap
(2.1, 2.4, 2.7–2.9, 2.12 and 2.15–2.17 are closed) and nothing is renumbered; a new item takes the
next free number. "Old item N" — a single number — means an item of the 2026-08 list, as
it does in older commit messages that say "`IMPROVEMENT_PLAN.md` item N". **A number alone
says nothing to a reader:** wherever an item is named outside its own heading — section 4, a
ledger row, a commit message, a report to Peter — give it with its short title ("2.19 Time
index for `avg`").

Everything settled lives under [docs/](docs/): the current model in
[MODEL.md](docs/MODEL.md), shipped changes in [DECISIONS.md](docs/DECISIONS.md), the
rejected ledger in [REJECTED.md](docs/REJECTED.md), the dated write-ups in
[FINDINGS.md](docs/FINDINGS.md) and the methodology in [AB_TESTING.md](docs/AB_TESTING.md).
Read AB_TESTING.md in full first — the harness, "How to run an experiment", "How changes
are validated" (the adoption bar), "Measurement grids" and "Which verdict function to
call" — and REJECTED.md before trying anything. This file assumes that context and does
not repeat it.

**Priority: `cheap2h` first, then `min`. `avg` improvements are welcome; `max`
is explicitly not a priority.**

**Standing rules for everything below.**

- Re-opening a rejected idea, or trying a new one, needs a stated mechanism for why it
  should work — not "we changed a lot, maybe it's different now." If you can't state
  one, that's the answer.
- Screen on one period cluster, confirm on the others, and pre-register the
  confirmation bar before looking at the screen.
- Measure on a long snapshot (`ab_test.py fetch --days 1825`) on the rolling grid —
  NOW / −6M / FAR today, see
  [Measurement grids](docs/AB_TESTING.md#measurement-grids-training-geometry-and-period-clusters-read-before-designing-a-run)
  — and judge with `classify_clustered` / `classify_ablation_clustered`, not the plain
  `classify` / `classify_ablation`
  ([which one](docs/AB_TESTING.md#which-verdict-function-to-call)). `OMP_NUM_THREADS=4`.

**How this file is maintained.** A measured item is cut back to what is still open — or
deleted outright — as soon as its verdict is in the ledgers:
[DECISIONS.md](docs/DECISIONS.md) if it shipped, [REJECTED.md](docs/REJECTED.md) if it did
not, plus a section in [FINDINGS.md](docs/FINDINGS.md) when the round explained something
([README](README.md#how-changes-are-documented) has the rules). A half-done item keeps only
what it still owes and where it stands. The ledgers are the record; this file is the
to-do list. Section 4 is kept current in the same commit as the change that moves an item:
ready or blocked (with its blocker), and the shadow register while an arm runs.

**Pausing an item.** An item may be paused for any reason — a missing cache, a date, a
decision, or simply the end of a session — and what has been learned so far stays in the
item, so that a later run can continue it without redoing it. The working material under
`experiments/` is **not** that record: it is scratch, assumed gone on the next machine, so
whatever a restart needs (the arms, the grid, the bar, the selection rule, what was already
measured) is written here, in words. Script names may be given as a courtesy, never as the
only description.

Every item in section 2 opens with a **State** line that answers three questions:

1. *Where is it?* `Not started` · `In progress` · `Paused` · `Running in production` (a
   shadow arm, [4.3](#43-shadow-arms-in-production)) · `Parked` (nothing to build until
   something changes).
2. *Does anything stop the next step?* Exactly one of:
   **Not blocked — can be taken now**, and for a paused item *resume at* the concrete next
   action; or **Blocked by** *what* **because** *why*, **until** *when* (a date, a cache, a
   decision, another item's verdict). A block without a *because* is not allowed: a reader
   must be able to judge from the line whether the reason still holds. If nothing blocks, the
   line says so in those words, so that nobody waits for something that is not there.
3. *What exists already?* For a paused item: what was done and found so far, with the date
   and the commit it was measured against. Interim numbers are not a verdict and do not
   enter a ledger; they stay in the item until the verdict is in, then they are cut and the
   ledger entry takes over ([README](README.md#how-changes-are-documented)). A result
   measured on a model that has since changed on main is stale — say so, do not delete it.

Section 4 is the index of the State lines: *Not blocked* means 4.1, *Blocked* means 4.2 with
the blocker in a few words and the earliest date. The full *because* is written once, in the
item.

---

## 1. The findings the items are built on — in docs/FINDINGS.md

The September 2026 review's four findings (`residual_load` is temperature in disguise; the
regime the model was tuned on is moving away from it; which sources are reachable and what
they carry; the weather grid has no point under the northern wind fleet), the source probes
(Appendix A) and the reproduction recipes (Appendix B) are in
[docs/FINDINGS.md](docs/FINDINGS.md#1-what-this-review-measured-2026-09-08-snapshot-ab_cache2026-09-05).
Items 2.5, 2.6 and 2.14 lean on §1.2 and §1.3 in particular.

---

## 2. Items, in priority order

Each item: mechanism → what to build → arms and the pre-registered bar → what a
failure would mean. Effort is a rough guess of agent time excluding compute.

### 2.2 Production prediction log and realised scorecard

**State (2026-11-08):** In progress — the log is built and written by Node-RED on every run
since 2026-10-18; the scorer is not written. **Blocked by** four weeks of logged rows,
**because** a read before then covers only ~13 warned days and would say nothing the
backtest does not, **until 2026-11-15** (later if the node was not writing every run from
2026-10-18). Nothing else blocks it: on that date the scorer is the first thing to do.

**Mechanism.** Every accuracy number in docs/MODEL.md is a backtest with two known
optimisms that no backtest can remove: archive weather in place of the forecast
(measured 2026-10-11 on vintages: +1.63 cheap2h / +1.69 min / +2.81 avg at production's
leads, [FINDINGS.md](docs/FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b);
the log shows what it is in practice) and the gap1 fit tail (+0.155 / +0.258 /
+0.591). The only instrument that measures what Home Assistant actually
received is the record of what was sent, scored against what was realised.
Until 2026-10-18 there was no such record: `predictions_raw` is overwritten on every push.

**What the log holds, for the scorer.** The log is not in the repo and
not an Actions artefact. It lives in Peter's Home Assistant MariaDB (table
`electricity_price_predictions_log`), written by a Node-RED function node from the
sensor's `predictions_raw` right after each run — raw SEK/kWh, without the 5 % markup
and addon, which are a transform that only makes a logged value harder to score. One row
per (`prediction_date`, `target_date`, `variant`): `horizon_days` (computed from the two
dates, so it stays right if the first predicted day shifts), `min` / `avg` / `max` /
`cheap2h` (`*_sek_kwh`), `cheap2h_low` / `cheap2h_high` / `cheap2h_risk`, and two
run-level columns: `eur_to_sek_rate` (turns a logged SEK value back into the model's
EUR/MWh exactly) and `git_sha` (the commit the run executed, so a before/after read
groups rows by code instead of guessing from dates). The last two reach the sensor from
`ha_client._run_provenance`; rows written before that change carry NULL. A rerun on the
same day replaces that day's rows — what Home Assistant last saw. `variant` defaults to
`'served'`; the slot is for shadow rows (below). The Node-RED flow is not in the repo.

The older wide table `electricity_price_predictions` (d1…d6 min/avg/max, **with** addon,
from at least 2026-09-22) is kept as the baseline from before the round-24/25 changes and
is **to be brought into the scoring** (bullet below): its values have the addon baked in
(`ha_client.apply_addon`: raw × 1.05 + addon), so they cannot be scored until the addon
is known. It has no `cheap2h`, and the new
table starts after the 2026-10-12 and 2026-10-16 changes, so the log has **no before-period
for `cheap2h`**: production is judged against what the backtest predicted for it, not
against an earlier production.

**Why 2026-11-15.** That is four weeks after the log started, the earliest point at which
it is worth reading, so `score_production.py` and every step that reads the log (the 2.3b
forward replay, 2.3c, 2.3d, the 2.10 and 2.13 production checks) wait until then; the
scorer is not written earlier either. What it buys: at ~4 weeks a first, descriptive look at
MAE per horizon, at the band (it warns on ~46 % of days, so ~13 warned days — thin) and at
the decision regret; at ~13 weeks the optimism re-derivation below. The clock is four weeks
of daily rows from the day the Node-RED node is writing every run.

**Reading the data (decided 2026-10-18).** The database is the record. The scorer reads
a CSV export of the whole table (phpMyAdmin, dated file name) from
`ab_cache/prediction_log/`: gitignored with the rest of `ab_cache/`, ignored by
`list_snapshots` (no `meta.json`), and disposable because it is re-exported on demand.
The old wide table is exported at the same time and not before (decided 2026-10-18), together
with the addon value for its period.

**What remains.**

- `score_production.py`: joins the log to realised daily prices (ENTSO-E, already
  fetched) and reports MAE per target and per horizon; the warning-light band scored with `ab.band.score_band` on what
  was served (its threshold 0.3 and coupled level were chosen post hoc on a historical
  walk-forward with archive weather, so the log is the only honest check); and the
  *decision* regret of the charging rule the automation actually uses (README HA
  section) — the metric that the slot plan and the q90 rule were both scored on in
  backtests but never in production.
- **Bring the old log in.** Export `electricity_price_predictions` together with the new
  table and treat it as extra min/avg/max history from at least 2026-09-22 — about four more
  weeks than the new table has. Three things have to be recovered first:
  - *The addon.* raw = (value − addon) / 1.05, with the addon each run read live from
    `input_number.electricity_price_addon`. Peter has to state the value and the dates it
    changed, if it ever did: the HA recorder keeps states for its purge window only
    (typically 10 days) and an `input_number` has no long-term statistics, so the history
    may not be recoverable from HA.
  - *The target days.* Only `d1_date` is stored; d2…d6 are the following days, which holds
    because the forecast set is consecutive days. Horizon = `d1_date` − `prediction_date`
    + (k − 1).
  - *The rate.* No `eur_to_sek_rate` was stored; re-derive it as the pipeline does
    (`currency.calculate_eur_to_sek_rate`) to get EUR/MWh.
  There is no `git_sha`, so score it in blocks split at the known change dates (2026-10-12,
  2026-10-16) and never pool it blindly with the new table.
- Re-derive the two optimism figures empirically once ~13 weeks are logged:
  production MAE at D+2 minus backtest MAE at D+2 is the sum of the weather and
  gap1 effects, and the per-horizon slope is the weather part.
- A shadow arm for 2.3b. **Pipeline side built 2026-10-31**: `predict.predict_blend_shadow`
  re-predicts with the same fitted models, band and rate on the hourly mean of
  `sources.open_meteo.BLEND_MODELS` (all nine weather points) and pushes it as arm `blend`
  of `predictions_shadow` (`{variant: rows}`, the generic slot for any production A/B arm;
  the flow writes each key as its `variant`); any failure leaves the arm out and never
  blocks the served push (`experiments/integration_check_blend_shadow.py`). The Node-RED
  node writes every arm generically (done 2026-10-31, Peter). The arm is in the shadow
  register ([4.3](#43-shadow-arms-in-production)); the forward-replay clock for 2.3b starts
  on the first day blend rows are logged, not on 2026-10-18. The scorer must drop a shadow row whose
  `created_at` is older than the served row's for the same `prediction_date`: a same-day
  rerun whose arm failed upserts the served rows and leaves the earlier run's shadow rows.
- Not logged: the run's `cheap2h_interval` diagnostics (`coupled_level_ratio`,
  `calm_high_holdout_coverage`, …), which is where a new regime shows first. Add a JSON
  column if drift tracking needs them.
- **`horizon_days = 8` rows are not forecasts.** Until 2026-10-21 every run sent a
  seventh day, D+8: since the UTC switch of 2026-03-22 the 8-day forecast's last Swedish
  date holds only the 1–2 hours between Swedish and UTC midnight, and nothing dropped it
  ([DECISIONS.md](docs/DECISIONS.md#whole-swedish-days-only)). The 2026-10-17 test row and
  every run before the fix logged one. They carry no information and can be deleted from
  the table; the scorer should reject `horizon_days > 7` regardless, until a longer
  horizon is shipped on purpose (2.11). The old wide table never had the stub — its flow
  kept six days.

**Bar.** None — this is instrumentation. It is the prerequisite for reading
item 2.3 honestly and for the forward replays of 2.10 and 2.13. It cannot say whether the
two changes of 2026-10-12 helped by a before/after on `cheap2h` (no before exists, see
above); the question it can answer is whether served `cheap2h` lands near the backtest's
expectation.

**Effort.** Log: done. Scorer: a few hours. Zero model risk.

### 2.3 The weather axis: price it, then improve it

2.3a (what the archive weather flatters the eval by) was measured on 2026-10-11 and is
closed: [FINDINGS.md](docs/FINDINGS.md#archive-weather-flatters-the-eval-by-13-eurmwh-almost-all-of-it-beyond-d4-round-25-plan-23a23b).

**2.3b Multi-model blend at serve time — measured, missed its strict bar.**
**State (2026-11-08):** Running in production (shadow arm `blend`, [4.3](#43-shadow-arms-in-production)).
**Blocked by** the production scorer (2.2) and four weeks of logged `blend` rows, **because** a forward
comparison needs both arms on unchanged code and enough days to judge a −0.2 difference;
**until** four weeks after the first logged `blend` row, and not before 2026-11-15. The
item may not leave this plan while the arm is in `predict.py`. The
hourly mean of ECMWF, ICON and GFS recovers ~25 % of the penalty (cheap2h −0.37,
min −0.42, avg 0) and is negative in all 15 cache × cluster cells, but the
pre-registered per-lead bar was met on 0 of 5 ([REJECTED.md](docs/REJECTED.md)).
Next step is a *forward* replay, not another backtest: log the blend's prediction next
to the served forecast (a `variant = 'blend'` row in the 2.2 log; the pipeline pushes it
since 2026-10-31, the Node-RED write is open, see 2.2), and score both on the
days the log covers — not before 2.2's 2026-11-15 and not before four weeks of blend rows,
with a bar written for production's lead mapping (row j at
lead j + 2): production-mapped cheap2h and min negative in every cluster-equivalent
period and the pooled change ≤ −0.2. Cost at serve time: three model requests for
nine points.

**2.3c Train on what you serve.** **State:** Not started. **Blocked by** the 2.3b verdict,
**because** that verdict tells whether the loss is in the forecast (the blend recovers it) or
in the model (this item); **until** 2.3b has one. The headroom is now a number: +1 … +5 EUR/MWh on
cheap2h at L5–L7. Mechanism unchanged — the model learns price = f(analysis
weather) and is served f(forecast weather) — but vintages exist only from 2024-03,
so far-lead rows would train on ~2.6 years. Per-lead models or a stacked set with
a lead column; start only after the 2.3b replay shows whether input quality or
model mismatch carries the penalty. Bar as 2.3b.

**2.3d Ensemble spread as an interval conditioner.** **State:** Not started. **Blocked by**
the production scorer (2.2) and by a written bar plus its own shadow arm
([README](README.md#adding-a-shadow-arm-to-the-prediction-log)), **because** the ensemble has
no vintage archive, so it can only be scored live; **until** 2.2 exists. Mechanism: the conformal band
is one width for the whole forecast set, adjusted by a single correction. Weather
uncertainty varies day to day and is observable as the 51-member spread of
`wind_speed_100m`. Condition the conformity quantile on spread bins (or add spread
to the two quantile regressors). Scored on pinball loss and coverage per cluster,
never MAE. Live only — the ensemble has no vintage archive — so scored
prospectively via 2.2.

**Effort.** 2.3b replay: one day plus waiting for data; 2.3c three days; 2.3d one day.

### 2.5 The hurdle's label under regime drift — only the slot-specific arm is left

**State (2026-11-08):** Not started. **Blocked by** 2.6 (slot targets), **because** the
slot-specific hurdle needs the slot targets to exist; **until** 2.6 is built.

H5 (threshold 5 EUR/MWh) and HQ (trailing 10th-percentile label) were measured on
five caches and are NOISE ([REJECTED.md](docs/REJECTED.md)). What remains is
**HS**: `P(cheap2h_day < 0)` and `P(cheap2h_night < 0)` as two probabilities, the
slot-specific hurdle of `SLOT_PREDICTION_PLAN.md` §4.4, which needs the slot targets
from 2.6 to exist first. Bar: `classify_clustered` on cheap2h; do not re-test the
hurdle on `min` (closed twice).

### 2.6 Slot targets (round 20)

**State (2026-11-08):** Not started. **Blocked by** the November run (2.19, 2.20, 2.18),
**because** slot targets go into `model.TARGETS`, which `ab.variants.BASELINE` mirrors, and a
baseline must not move while another candidate is being measured; **until** that run's
verdicts are in. Nothing else blocks it.

`SLOT_PREDICTION_PLAN.md` is the spec. It used to wait for the residual-load
rebuild (old 2.1), which would have given the model a feature that tells a night-wind trough
from a midday-solar one; that item closed in round 29 (the weather reaches 11–21 % of what a
physical residual is worth, [FINDINGS.md](docs/FINDINGS.md#the-physical-residual-load-is-a-high-ceiling-that-the-weather-reaches-little-of-round-29)),
so the slot targets are built on the current features.

Do not wire them in mid-A/B. `ab.variants.BASELINE` mirrors `model.TARGETS`
(docs/AB_TESTING.md), so the slot targets must not be added while another candidate is
being measured, and vice versa.

### 2.10 `min ≤ cheap2h` coherence — shipped 2026-10-12, re-check in ~3 months

**State (2026-11-08):** Shipped; the re-check is not started. **Blocked by** a long cache
from mid-January 2027, **because** the question is whether the freshest year still pays and
only a cache with new NOW windows can answer it; **until 2027-01-15**.

Record and arms: [DECISIONS.md](docs/DECISIONS.md#min--cheap2h-coherence-round-25). Every
number there is a full evaluation year (52 weekly windows) and the three clusters are three
such years, so the pooled gain is a year-long effect — but it is modest, and the freshest
year is its weak spot: `min` in the NOW cluster is +0.03 on the newest cache and only
−0.05 … −0.13 on the others, while FAR and −6M carry most of the pooled −0.17 … −0.23.
That is worth confirming rather than assuming. **When a long cache from mid-January 2027
or later exists**, measure the change on and off (patch `ab.harness.enforce_coherence` to the
identity for the "off" arm, as `experiments/integration_check_coherence.py` does) on NOW and
−6M. Bar, written now: `min` NOW ≤ 0 and `cheap2h` NOW not REAL-harmful on at least two of
three later caches. If `min` NOW is positive on most of them, revert to the raw pair (the
"lower" arm is already rejected). The production log (2.2) is the second source for the
same question.

### 2.11 Longer horizon for the automation (product decision, cheap)

**State (2026-11-08):** Not started. **Blocked by** Peter's decision, **because** the model
cost is nil and the only question is whether the automation wants a 14-day view; **until**
he says so. Nothing technical blocks it.

Open-Meteo serves `forecast_days=16`; the pipeline requests 8 and sends **six**
predicted days (D+2..D+7): today and tomorrow have published prices, and the last
Swedish date of a UTC request is a 1–2 hour stub that `predict.py` drops (until
2026-10-21 it went out as D+8, [DECISIONS.md](docs/DECISIONS.md#whole-swedish-days-only)).
Extending to D+2..D+14 is a parameter change plus the nuclear-outage and calendar
windows (`forecast_end = today + 10`). Keep the whole-day filter, and request one day
more than the last day sent: Open-Meteo changes the last hours of a series with
`forecast_days` (de_north, 2026-10-18), so a longer request is not bit-identical on the
previous last day. Accuracy
beyond D+7 is unmeasured — the eval uses 7-day windows — so ship it only with
the interval, and report it separately in 2.2. Whether the automation wants a
14-day view is Peter's call; the model cost is nil.

### 2.13 Northern wind points — shipped 2026-10-12, replayed 2026-10-22

**State (2026-11-08):** Shipped and replayed; only the production check is open. **Blocked
by** the production log (2.2), **because** the check scores served `cheap2h` against the
backtest's expectation and needs four weeks of rows; **until 2026-11-15**.

Record: [DECISIONS.md](docs/DECISIONS.md#northern-wind-points-round-25). Hydro
precipitation and a fourth site (Sundsvall) were measured and rejected
([REJECTED.md](docs/REJECTED.md)). The replay on `long/2026-10-16` and the `min` vintage
re-score are done (round 27): it replicates on all six caches under the coupling hurdle,
forecast-weather gain on the newest window cheap2h −0.45 / `min` −0.52 clmean. One thing
is left:

- **The production log (2.2) is where the real gain shows** — not before 2026-11-15.
  There is no before/after for `cheap2h`: the log starts 2026-10-18, after the change
  (see 2.2), so score the served `cheap2h` against the forecast-weather expectation in
  DECISIONS.md (NOW cluster −0.7 … −1.2, of which −1.1 at D+2…D+4 and −0.7 at D+5…D+7).

Not tested, low prior: other fleets (NO3/NO4, northern Finland beyond Ostrobothnia) —
each single site gave ~half the three-site gain, so the regional average is the
signal and three points seem to saturate it.

### 2.14 Nord Pool UMM as the outage source

**State (2026-11-08):** Parked, nothing built. **Blocked by** a missing consumer,
**because** every use UMM had (the nuclear features of 2.7, the interconnector columns) has
been closed on its merits, so a source with no column to feed is cost without a question;
**until** a new per-day column from a revisioned document is proposed.

**Mechanism.** The open question in old item 0 / 2.4 was whether ENTSO-E A78
is usable as a *forward-looking* feature, because the API returns the current
version of a record and revisions are invisible, so a backtest could know
things production could not. **Round 21 settled that the A78 route cannot
answer it**: every record for an event before 2025-11 was re-published in
November 2025, so the cache holds no vintage at all before that date
(docs/FINDINGS.md "If you touch ab_cache/crossborder/"). With the trough targets
having rejected this data outright and the `avg` export-headroom candidate now
closed on its serve-time encoding, **the one consumer UMM had was the nuclear features of
item 2.7, and round 28 closed it** (the realised nuclear output, as a leaky ceiling, moves
`avg` by −0.03 and `min` by +0.18; [FINDINGS.md](docs/FINDINGS.md#the-nuclear-column-and-the-week-ahead-load-carry-nothing-a-model-can-use-round-28)).
**Parked 2026-10-23: build it only if a new per-day column from a revisioned document appears.**
The interconnector half has no consumer either. What follows is what was verified, kept
for that day. Nord Pool's UMM API is the upstream of those
records and keeps **every version of every message with its
`publicationDate`** — `GET /messages/{id}/{version}` returns the record as it
stood then — for production units (nuclear with installed and available MW per
period), transmission units (in/out area, available MW) and large consumers,
back to at least 2016 for the SE4 borders, without authentication. That is the
vintage information A77/A78 lack, and it answers the leak question by
construction: for any backtest date D, use the latest version of each message
with `publicationDate ≤ D`. Verified on the Baltic Cable trip
(`e911027b…`): v1 2026-06-21 end 06-29 → v2 06-23 end 08-31 → v3 08-17 end
09-18 → v4 09-11 end 09-14. Also in the feed for SE4 (Jun–Dec 2026): SE3→SE4
(40 messages), SE4↔PL (129), SE4↔DK2 (35), SE4↔LT (23), SE4↔DE-LU (13), plus
the SE2→SE3 and FI→SE3 cuts that set the import price.

**What to build.**

- `sources/umm.py`: fetch messages by `eventStartDate`/`eventStopDate` window
  (paginate with `skip`; `messageTypes=1` is production, `2` consumption,
  `3` transmission — not the order the names suggest; `areas=<SE4 EIC>`
  filters all three, 176 SE4 transmission messages for Jun–Dec 2026), keep
  every version, build two daily series
  *as of* a given date: unavailable nuclear MW in SE3 (replaces the event
  count in `nuclear_outage_se3`, old item 4 / 2.7). The same fetch also yields
  unavailable interconnector MW per SE4 border; parse it only if a feature
  needs it, since the one that would have is closed.
- Cache the full message history once under `ab_cache/umm/`; the daily
  production fetch only needs the current window (events overlapping
  today..today+16).
- Encode "as of run time" in the eval: `apply_forecast_freeze` freezes lag
  features; these are per-day features whose *value* depends on the run date,
  so the walk-forward must rebuild them per window from messages published
  before the window start. That is a small extension of the harness (a
  per-window feature builder). Round 25 got its forecast-vintage frames by building one
  frame per variant instead (`experiments/run_round25_vintage.py`), which does not
  generalise to a per-window as-of feature — so this is its own piece of machinery.

**Bar.** Whatever the new column's arms are, rebuilt as-of, against the same arms under the `AF` condition
of [AB_TESTING.md](docs/AB_TESTING.md#score-the-test-rows-as-production-could-know-them-the-af-condition-round-28);
`classify_clustered`.

**Effort.** One day for the source and cache, one day for the per-window
builder shared with 2.3a.

### 2.18 A near-horizon model that reads the price (added 2026-10-22)

**State (2026-11-08):** Not started; the design and three pre-registered bars are written
below. **Blocked by** a long cache fetched at least four weeks after `long/2026-10-16`,
**because** two of its three candidates (the `avg` `_mw` split and `CALRn`) were chosen after
their numbers were seen on the six existing caches, which share most of their NOW window, so
only a cache with genuinely new NOW windows can confirm them; **until 2026-11-13**. It is
evaluated in the same pass as 2.19 and 2.20 and is Peter's to take up.

**Idea (Peter's).** A frozen price lag is information about tomorrow's price and noise
about next week's. Today one model serves every day, so it either reads the price on
every day or on none. The price/market family was pruned from `min`, `cheap2h` and `avg`
because, frozen across all seven rows, it hurt: cheap2h +1.00, min +0.74, avg +1.08,
harmful in every period ([REJECTED.md](docs/REJECTED.md)). The question was never asked
for the near days alone. A separate *near model*, with the price family, for the first
N sent days, next to today's model for the rest, is a new mechanism. It is not the
round-18 price-level features or the shelved per-horizon models re-opened.

**Why now.** Since 2026-10-22 the anchor is D+1
([DECISIONS.md](docs/DECISIONS.md#price-anchor-on-d1)), so D+2's lag1 is exactly
yesterday's price, as in every training row: no train/serve mismatch on the first sent
day. Measured on that change: a fresher anchor helps cheap2h on D+2/D+3 (up to −0.44)
and nothing from D+4. That is the shape this item would exploit, on a model that barely
reads the price today.

**Design.**
- For each horizon h in the near set (D+2 = staleness 1, D+3 = staleness 2, …), train
  on the price family at the age that day will see. Use lag-h columns (`price_se4_*`
  shifted h days), not lag1 frozen and labelled as fresh. The near model for D+3 is
  trained on prices two days old.
- Score per eval row (1-based row k = D+(k+1), staleness k) against today's model on
  the same rows. Choose N as the last horizon where the near model wins in every
  cluster, and stop at the first horizon where it does not.
- Targets: `cheap2h` and `min` first (`ab_test.py band` as well, since both feed the
  band), `avg` after. `max` already reads the family.
- **Weather columns are near-horizon candidates too (round 27, post hoc).** On forecast
  vintages the MW-scale residual load is worth `avg` −0.1 … −0.4 at D+2…D+4 and nothing
  or worse at D+5…D+7, and the northern wind points on `avg` the same
  ([FINDINGS.md](docs/FINDINGS.md#a-wind-carrying-residual-load-pays-only-near-the-run-day-round-27)).
  A near model is where such a column could live. That makes it a hypothesis for this
  item with its own bar, scored on vintages, not a result to carry over.
  **Evaluate it in the same run as the price family (Peter, 2026-10-22).** Read off the
  stored round-27 errors (no new run), `avg` with `residual_load_mw` +
  `residual_load_min_mw` appended on D+2…D+k and today's model after, on forecast weather,
  six caches, NOW / −6M / FAR: k = D+3 −0.02 / −0.07 / −0.20, D+4 −0.05 / −0.15 / −0.20,
  **D+5 −0.14 / −0.24 / −0.36** (every cache and cluster negative, NOW −0.08 … −0.25),
  against ≈ 0 in NOW with the columns on all six days. The split was picked after seeing
  these numbers, so it is pre-registered here, before the 2.18 run: **near `avg` = AVG
  list + the two `_mw` columns for D+2…D+5, today's `avg` for D+6…D+7**. Bar: production-
  mapped forecast-weather delta negative in every cluster on ≥ 2 long caches with
  genuinely new NOW windows (one fetched mid-November 2026 or later), and the six-day
  total not worse. `min` and cheap2h do not get the columns at any horizon (worse on a
  forecast at every lead, round 27). Build it on the same per-horizon routing as the
  price-family near model; it is not worth a second avg model on its own (−0.14 in NOW
  is near the ~0.10–0.15 `avg` noise floor).

**A calibrated residual load is a third near-horizon candidate (round 29, post hoc).**
`CALRn`: two columns, `cal_resid_mean` and `cal_resid_min`, the prediction of the realised
eight-zone load − wind − solar (A75 + A65, GW) from the day's weather and calendar by an
XGBoost first stage (200 trees, depth 3, no time index, out-of-fold on the training rows),
for D+2…D+4 on `cheap2h` and `min`. Round 29, forecast vintages at production's map:
`cheap2h` D+2…D+4 **−0.36** (every cache and cluster), `min` −0.25, and ~0 at D+5…D+7; a gain
that exists on a forecast and not on the archive, chosen after the numbers were seen as the
better of two first-stage variants, and neutral in the newest four weeks. The table, the
size of the effect in percent of the base error and the traps are in
[FINDINGS.md](docs/FINDINGS.md#the-physical-residual-load-is-a-high-ceiling-that-the-weather-reaches-little-of-round-29).
**Bar, written 2026-10-25 before any replication exists:** on a
long cache fetched mid-November 2026 or later (with its own vintage fetch and the A75/A65
top-up, `experiments/fetch_round29_entsoe.py --since 2026-10-01`), the production-mapped
D+2…D+4 delta is negative in every cluster on `cheap2h` and not positive on `min`, the
archive delta is not REAL-harmful in any cluster, the six-day total is not worse, and
`ab_test.py band` shows no breach. Run it with `experiments/run_round29_cal.py --arms
base,CALRn` in the same pass as the price family. Cost if it holds: A75 + A65 for eight zones
fetched on every run (about five minutes with the token production already uses) and a first
stage refit per run; it would be a column of the near model, never of the all-horizon lists.
The `avg` version fails on vintages (NOW +0.62) and is not part of this.

**It is part of the horizon-split evaluation, not a separate run.** The near/far split N (the
"Measurement geometry" paragraph below) is chosen per horizon, and `CALRn` is scored in that
same run as one more candidate for the near model, next to the price family and the `avg`
`_mw` columns: per horizon D+2 … D+7 on forecast vintages, with the rolling origins, so that
the horizon where it stops paying decides where it is switched off. Until that run the
production model gets nothing from it.

**Bar** (write it down before the run). Per horizon, `classify_clustered` REAL and
favourable on ≥ 2 long caches with genuinely different NOW windows. The total over
D+2…D+7 must not get worse.

**Measurement geometry: decide before the run (Peter, 2026-10-24).** The step-7 grid is
right for all-horizon verdicts, but this item makes one per horizon, and two properties of
the grid then matter. Within one grid point, eval row k is always the same weekday, so a
per-horizon delta is also a per-weekday delta. And each horizon gets only 52 test days per
point. Consider rolling origins instead: fit once per test week as now, then issue a
forecast from each of the seven start days in that week, each with its own freeze anchor.
Every test day is then scored at every horizon and every horizon sees every weekday, for
about the cost in fits of the current grid. Both arms share the same fits, so the model
being 0–6 days old at an origin (production: ~1) does not bias the comparison. Three
things to settle in writing first: (1) the extra origins sharpen each point's per-horizon
estimate but are not replication, because the same day at D+3 and D+6 is strongly
correlated; the cluster stays the unit of the verdict
([AB_TESTING.md](docs/AB_TESTING.md)), and per-window spreads from overlapping origins are
optimistic. (2) Score D+2…D+7 only; row 7 (D+8) is not sent. (3) Run it on forecast
vintages as well as archive weather. On archive weather a day's D+3 and D+6 differ only by
the age of the frozen lags, while most of the real decay with horizon is weather forecast
error (+1.3 cheap2h / +1.5 min / +2.4 avg on the six sent days). The near/far split N is
the quantity most exposed to that.

**Cost.** One extra model per near horizon, plus routing in `predict.py` and
`model.predict` by horizon. The harness needs a per-row target, the same shape as
`run_round25_vintage.py`'s per-row scoring.

### 2.19 A time index for `avg` (added 2026-10-23, found as a control in round 28)

**State (2026-11-08):** Paused — prepared, not run. **Blocked by** a long cache fetched at
least four weeks after `long/2026-10-16`, **because** the six existing caches share most of
their NOW window and the gain there is partly the recent coupled weeks, and because the
six-cache half was once started alone (2026-10-31) and went stale when main changed; it is
therefore run together with the replication cache; **until 2026-11-13** (the fetch script
refuses earlier). *Done so far:* the arms below and the rule that picks the candidate arm are
scripted (`experiments/run_round30.py`, scratch), and its selfcheck reproduces round 28's `base`
and `TI` to the digit; the fetch (`experiments/fetch_round30.ps1`, with `-WithTopUps` for 2.20 and
2.18, `-Run` for the replication grid) is written. *Resume at:* fetch, then `run_round30.py run`
and `analyze` on the six old caches and the new one in one pass. If the scripts are gone, the
arms and the bar below are the whole specification.

**What was found.** Adding `t_index` (days since 2021-10-21 / 365) to the `avg` model is REAL on all
six long caches — six-cache mean **−0.28** (−0.18 … −0.43), negative in every cluster, per-window std
×0.96 – 1.00 — on top of the half-life-500 weights `avg` already has. The date is known for every
forecast day and nothing in the column is weather, so neither the forecast-vintage trap nor the AF condition
applies. On `cheap2h` and `min` the same column is REAL-harmful
([REJECTED.md](docs/REJECTED.md)). It is a control that turned out to be a result, found after the main round
was read and measured only on the six overlapping caches — a lead, not a change.

**Mechanism to test, not assume.** `avg` is the one target that wants recency. Half-life weights give recent
rows more weight; a time index lets the trees give the newest regime its own leaf, and every forecast row
lands in it. If the second is only a cheaper way to get the first, a shorter half-life reproduces the gain and the
index is redundant; if the index beats every half-life, it carries something weights cannot. The hazard is
the same property: through a regime break the model keeps predicting the newest leaf.

**Arms** (`avg` only; `min` and `cheap2h` are closed): `TI` alone; half-lives 250 and 365 without it
(production is 500); `TI` with each; a step version, whole years since 2021 (is the gain the slope or the
levels?); SEED7 as the null. Against production `avg`.

**Bar** (written now, before any run). On the six existing caches: `classify_clustered` REAL, six-cache mean
≤ −0.15, std ratio ≤ 1.05, favourable on ≥ 4 of 6. **And** on a cache fetched at least four weeks after
`long/2026-10-16` (mid-November 2026 or later), because the six caches share most of their NOW window and the
gain in NOW is partly the recent coupled weeks: NOW negative there, or the item stays a lead. **And** the four
weeks from 2026-08-17 (the regime break) are scored apart and must not be worse than `base`. `max` reads the
full list, so it is reported (a REAL-harmful `max` is Peter's call). `ab_test.py band` is not needed — `avg`
feeds no band.

**Effort.** Half a day; about one hour of compute (one target, 12 points per cache). No new fetch except the
replication cache.

The script also runs `max` with and without the index. The six old caches cost about two hours of
compute, which is why they are not run ahead of the replication cache.

### 2.20 The decomposed nuclear column, replayed as production sees it (added 2026-10-24, run with 2.19)

**State (2026-11-08):** Not started; scripts exist (`experiments/run_round28.py`). **Blocked
by** the same replication cache as 2.19, **because** both arms are leads picked as the best of
seven on caches that share their windows, and a parser change on a lead inside the noise floor
is not worth its risk; **until 2026-11-13**, in the same pass as 2.19.

**What is open.** Round 28 closed the nuclear column on its pre-registered bar (scored with every record
visible, HS). Scored as production actually serves it (AF: planned outages visible, forced ones only once
started — which is what `predict.py` fetches today), two arms read slightly better than today's column on
`avg`: `Rac` (distinct nuclear units + distinct non-nuclear units, replacing `nuclear_outage_se3`) −0.05,
negative on all six caches, and `Rall` (those two + unavailable nuclear MW + cancelled records) −0.07, negative
on five. About half of it is the old column losing more than the new ones when future forced outages are
invisible. It is inside the `avg` noise floor, the six caches share most of their windows, and the two arms
were the best of seven, so it is a lead picked after the fact, not a result
([FINDINGS.md](docs/FINDINGS.md#the-nuclear-column-and-the-week-ahead-load-carry-nothing-a-model-can-use-round-28)).

**When and how.** In the same run as 2.19, on the first long cache fetched at least four weeks after
`long/2026-10-16` (mid-November 2026 or later), plus a top-up of the A77 records to that date
(`experiments/fetch_round28_entsoe.py --only a77 --since <four weeks before the cache>`). Re-use
`experiments/run_round28.py` (`kind avg`, arms `base`, `Rac`, `Rall`; the data layer picks the newest
`ab_cache/entsoe_r28/` directory). Only the new cache's own grid counts; the six old caches are not re-run.

**Bar (written 2026-10-24, before the run).** For `Rac` and `Rall` separately, delta against `base` under AF on
the new cache: negative in each of NOW, −6M and FAR, cluster mean ≤ −0.05, and the HS delta not REAL-harmful.
If one arm passes, it is a parser change in `sources/entso_e.py` (keep unit, `psrType` and document status —
the guard in `experiments/results/nuclear_parser_fix.patch` is the starting point) and needs `max` reported and
the local integration check before it ships. If both pass, `Rac` (two columns, less to maintain) unless `Rall`
is ahead by more than 0.05. If neither passes, 2.20 is closed for good and the column stays.

**Effort.** No extra compute beyond 2.19's run (one target, two arms); the parser change about two hours if it
passes.

### 2.21 Dependency updates through Dependabot (added 2026-11-08)

**State (2026-11-08):** Not started. **Not blocked — can be taken now** (it is 4.1 item 2).
One timing rule: not on the day of a 4.2 run, since a version change on main makes a waiting
result stale.

**What and why.** `requirements.txt` has been pinned exactly since round 31. Nothing changes
without a commit, but nobody notices when a new release comes out either. Running unpinned
would not fix that: the runner would change versions on a random morning without anyone
seeing it, as it did with xgboost 3.4.1. Dependabot is GitHub's own update bot. It checks on a
schedule and opens a pull request with the new pins, and the owner is notified. Security
alerts for a pinned version with a known vulnerability come separately and at once,
whatever the schedule.

**How.**
1. `.github/dependabot.yml`: ecosystem `pip` on `/`, `schedule.interval: quarterly`, two
   groups. The *numeric stack* (pandas, numpy, xgboost, scikit-learn, scipy) moves numbers.
   The *fetch libraries* (yfinance, holidays, requests, pytz) talk to APIs that change:
   yfinance breaks when Yahoo changes, and a new release is usually the fix. Also ecosystem
   `github-actions` (checkout, setup-python), on the same schedule.
2. `tests.yml` already runs the suite on every pull request, so a Dependabot PR shows green
   or red before it is merged.
3. A committed version check. A draft exists in `experiments/run_round31_harness.py`
   (`points` + `compare`), but that directory is scratch: the deliverable is a script under
   version control, and the draft is only where to start. It runs the six long caches in the
   current environment and in one with the PR's pins. It reports, per target and cache, the
   MAE change, the largest row change and the fit time.
4. A README paragraph, "Updating dependencies". It covers what to do with each group's PR,
   and that a numeric-stack bump is merged in its own commit, never in the middle of a
   planned A/B series. Under rule 4.2, a version change on main makes a waiting result as
   stale as a model change does.

**Bar for merging a numeric-stack PR (written 2026-11-08).** Tests green. On the version
check, no target's MAE moves more than 0.05 on any cache. That is well inside the seed
floor (`cheap2h` 0.13), so a version change is never mistaken for a model change. Fit time
on Windows is no more than 1.5× today's. A PR that fails only the time bar stays open, with
the reason noted in it. That is xgboost 3.4.1's case today (2–3× slower on Windows). A
fetch-library PR merges on green tests.

**Effort.** About two hours: config, script, README. Afterwards, one PR per group per
quarter, and about 15 minutes for the version check on a numeric PR.

### 2.22 Warm-up days before the training window (added 2026-11-10)

**State (2026-11-10):** Not started. **Blocked by** the November pass over 2.19 / 2.20 / 2.18,
**because** it changes the rows production trains on and so moves the baseline those results
are measured against (the round 30 rule, section 4); **until** that pass is done, earliest
2026-11-13. Then take it at the first place in 4.1.

**Mechanism.** The window is fetched as exactly `TRAINING_DAYS = 1095` days, and every source
starts on its first day. The lagged features need days *before* it: `price_se4_avg_lag7` is NaN
for the window's first seven days, `ttf_rolling_7d` / `co2_rolling_7d` for the first three to
five (more after a weekend start: TTF and EUA have no kurs on weekends), the weather
variabilities and `price_volatility_7d` for the first two or three. `build_training_data`'s
`dropna()` removes every such row, which is why production fits on **1088 rows, not 1095**
(measured on the `2026-10-16` snapshot, 2026-11-10: all seven dropped rows are explained by
`price_se4_avg_lag7`; the fuel columns add none). A fetch that starts about ten days earlier
would have every lag defined from the window's first day, so the frame keeps all 1095 days.

**What it buys, honestly.** Seven more rows (0.6 %) at the oldest end of the window, and no
more TTF/EUA coverage than before: the fuel gap was already inside the seven dropped days, so
the "does not cover 3 of 1095 days" warnings are noise, not lost data. The expected effect on
MAE is far below the seed floor (`cheap2h` 0.13), so an A/B cannot show a gain; the case for it
is *data integrity* (production trains on the window it says it does), not accuracy.

**How.**
1. `fetch_training_inputs`: fetch every historical source from
   `today − TRAINING_DAYS − WARMUP_DAYS` (`WARMUP_DAYS = 10`: seven for the lag plus a weekend
   and a holiday of margin), and have `build_training_data` return the last `TRAINING_DAYS` rows
   after `dropna()` rather than whatever the window leaves. The production source cache
   ([DATA_CACHE.md](docs/DATA_CACHE.md)) sees a window that reaches further back than its files
   and fetches everything once; no version bump, because the request's meaning is unchanged.
2. `ab/harness.py` `PRODUCTION_TRAIN_ROWS = 1088` becomes 1095 in the same commit, and
   `docs/AB_TESTING.md` ("Production trains on 1088 rows") with it; otherwise the harness no
   longer trains on production's length.
3. The A/B snapshots start at their own window start, so none of the six long caches can test it:
   it needs one snapshot fetched with the extra days (`ab_test.py fetch`, 10 days earlier).

**Arms and bar (to pre-register before running).** A harm check in the A03 forward-fill's
design ([DECISIONS.md](docs/DECISIONS.md#a03-forward-fill-of-entso-e-price-curves)): the frame
with warm-up must equal today's frame **bit for bit on the 1088 rows both have**, and carry
seven extra leading rows with no NaN; then one walk-forward pass of both on the new snapshot
with no target REAL-harmful. Ship on that. A measured gain is not required, and not expected.

**Prior and what a failure means.** If an extra-row arm came out harmful on one target, the
seven oldest rows would be an outlier regime, not a defect of the change: the answer is to leave
the window as it is and close this as REJECTED with the number. Nothing else depends on it.

**Effort.** About an hour of code and tests (a frame-equality test on synthetic data, with the
warm-up days present and absent), plus one snapshot fetch and one pass.

## 3. Carried at low priority

Leads with a stated mechanism and a low prior. They are not blocked by anything, but they are
taken only when 4.1 has nothing before them, and not while a 4.2 run is due within a week.
What stays closed is in [REJECTED.md](docs/REJECTED.md), not here.

### `ttf_vs_30d` — a regime-abnormality ratio, low expected yield

**State (2026-11-08):** Not started. **Not blocked — can be taken now.**

**Mechanism.** World events do not hit SE4 prices directly; they transmit through fuel
markets (`ttf_price_lag1`, `ttf_rolling_7d`, `co2_price_lag1`, `gas_marginal_cost`),
which are already covered. The gap: trees split on absolute levels, so a level split
learned from 2023 data does not generalise to a 2026 shock (the motivating case was the
2026-07 Iran-war price spike). `ttf_vs_30d = ttf_price_lag1 / ttf_rolling_30d` (a new
30-day rolling mean next to the existing 7-day one) encodes "how far above normal" as a
stationary ratio, and should react before a shock fully lands in the electricity-price
lags.

**Prior.** Low. Its sibling `price_vs_30d` was rejected in round 18 (harmful on `min` and
`cheap2h`, three independent measurements, [REJECTED.md](docs/REJECTED.md)). This one is a
fuel-price ratio, not an own-price ratio, so the "own frozen price lag hurts the trough
targets" mechanism does not apply as such. Round 24's finding that the autumn-2026 misses
are a gas-price-and-coupling regime — shipped as the coupling hurdle
([DECISIONS.md](docs/DECISIONS.md#coupling-hurdle-and-the-warning-light-band-round-24)) —
is the stronger route to the same problem.

### The rest

**Also low priority, carried** (each **Not started, not blocked** unless stated):
`cpl_gap7` on cheap2h and the q90 decision rule at H=7 (round 19 follow-ups); and
`B6_fuel_co2` on `min`, **blocked by** far periods that train on post-crisis windows,
**because** its gain tracked how much fuel-price turbulence the training window held
([REJECTED.md](docs/REJECTED.md), the `min` add-back row); **until** the mid-November cache,
which is read with the time-index run.

**Structural note, not an item.** Nordic flow-based market coupling went live
2024-10-30, inside the training window. If a later analysis finds a step
change in SE4–SE3 coupling around that date, a single dated indicator is a
legitimate feature; do not add it speculatively — it would act as a time
index.

---

## 4. What to do next

Three lists, built from the State lines of section 2. **A request for "the next step" of this
plan means the first item of 4.1, and nothing else.**

- 4.1 holds every item whose State says *Not blocked*, in priority order — a paused item whose
  blocker has cleared is continued from its *resume at*, not restarted.
- 4.2 holds every item whose State says *Blocked*. Nothing in it is started — not partly, not
  early, not "the half that can run now", not a fetch script or a prepared run for it — until
  the blocker clears. A result measured on today's model goes stale with the next change to
  main (round 30, 2026-10-31: the six-cache half of 2.19 was started early and stopped for that
  reason). Pausing is different: stopping an item and writing down where it stands is always
  allowed, and it is what the State line is for.
- Before taking the first item of 4.1, check 4.2: an item whose blocker has cleared (its date
  has passed, its cache exists, its decision is made) moves into 4.1 at its place in the priority
  order, in the same commit that starts it, and its State line is rewritten to *Not blocked*.
- **If 4.1 is empty, the answer is that nothing is ready**: say so, list what 4.2 waits for and
  the earliest date, and stop. Do not pick a blocked item instead.
- 4.3 is the register of shadow arms running in production.

### 4.1 Ready now, in order

1. **Section 3, low-priority leads** — `ttf_vs_30d` (gas price against its 30-day mean),
   `cpl_gap7` (a coupling-gap lag for `cheap2h`), the q90 charging rule. Only with a stated
   mechanism, and not while a 4.2 run is due within a week (the same staleness rule).
2. **2.21 Dependency updates through Dependabot** — quarterly update PRs, a committed
   version check and a merge bar. Engineering, no model change; it can be done at any time,
   but not on the day of a 4.2 run.

### 4.2 Blocked — not started until the blocker clears

The index of the *Blocked* State lines. Only a few words of the blocker are repeated here;
the *because*, what is done so far and where to resume are in the item.

| Item | Blocked by | Earliest |
|---|---|---|
| **2.19 Time index for `avg`** (paused, prepared) · **2.20 Nuclear column as production sees it** · **2.18 Near-horizon model** | A long cache fetched four weeks after `long/2026-10-16`; one fetch day (`experiments/fetch_round30.ps1 -WithTopUps`), then one pass over all three | 2026-11-13 |
| **2.22 Warm-up days before the training window** | The November pass above: a change to production's training rows moves the baseline it measures against | after that pass, not before 2026-11-13 |
| **2.2 Production scorecard** · **2.13 Northern wind, production check** | Four weeks of the prediction log | 2026-11-15 |
| **2.3b Three-model weather blend** (shadow arm running, 4.3) | The scorecard (2.2) and four weeks of logged `blend` rows | four weeks after the first `blend` row, not before 2026-11-15 |
| **2.3c Train on forecast weather** | The 2.3b verdict | after 2.3b |
| **2.3d Weather-uncertainty band** | The scorecard (2.2), a written bar and its own shadow arm | after 2.2 |
| **2.6 Slot targets** · then **2.5 Hurdle label, slot-specific arm** | The November run above (a baseline must not move mid-measurement); 2.5 also needs 2.6 | after that run |
| **2.10 `min ≤ cheap2h` coherence, re-check** | A long cache from mid-January 2027 | 2027-01-15 |
| **2.11 Longer horizon** | Peter's product decision | — |
| **2.14 Nord Pool outage messages (UMM)** (parked) | A consumer: a new per-day column from a revisioned document | — |
| **`B6_fuel_co2` on `min`** (section 3) | Far periods that train on post-crisis windows | the mid-November cache |

### 4.3 Shadow arms in production

A shadow arm is a prediction pushed next to the served one in `predictions_shadow` and logged
under its own `variant` ([README](README.md#adding-a-shadow-arm-to-the-prediction-log)). It is
used only for a question a backtest cannot answer, because while it runs every change to what it
or the served model reads restarts its comparison. Rules:

- **Every arm in `predict.py` has a row here, written once** by whoever adds the arm (agent or
  person), in the same commit: its `variant`, the item it serves (number and title), what it
  predicts, the date, its bar and its end condition. Nothing in the row is updated afterwards;
  it changes only when the arm is extended (new end, with a reason) or removed (row deleted).
- **The item that owns an arm is not deleted from this plan while the arm is in `predict.py`.**
  Its verdict goes into the ledgers as usual; the arm is then removed from `predict.py` and its
  row from this table in one commit, and only then may the item go.
- An arm past its end condition with no verdict is a decision to take, not something to leave
  running: extend it here with a reason and a new date, or remove it.
- There is no start date to fill in: an arm logs from the first production run after its commit
  reaches main (normally the next day), and the scorer reads the first `prediction_date` with that
  `variant` from the log.

| `variant` | Item | What it predicts | Added | Bar | Ends |
|---|---|---|---|---|---|
| `blend` | 2.3b Three-model weather blend | The served models and band on the hourly mean of ECMWF IFS, ICON and GFS (`BLEND_MODELS`, all nine points) instead of `best_match` | 2026-10-31 | 2.3b: production-mapped `cheap2h` and `min` negative in every cluster-equivalent period, pooled ≤ −0.2 | When the 2.3b replay has its verdict, at the latest twelve weeks after the first logged row; adopt (blend becomes `served`) or remove |
