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

When updating the MAE table in [MODEL.md](MODEL.md#current-mae-baseline), quote
the run it came from and the snapshot it was measured on, and record adopted
changes as their **A/B deltas** — in that table and as an entry in
[DECISIONS.md](DECISIONS.md). The fullest worked example of the practice is the
per-target feature re-validation program, written up in
[DECISIONS.md](DECISIONS.md#per-target-feature-lists-for-min-and-cheap2h).

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

## How to run an experiment (agent playbook)

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
     production lists).
3. **Run it.** `python ab_test.py run` (add `--snapshot YYYY-MM-DD` to pick a
   specific one, `--shifts 0-5` to change the grid). Read the per-target verdict
   table. **Priority order is cheap2h first, then min**; `avg` is welcome, `max` is
   not a priority.
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

## Measurement grids: tail-truncated vs sliding (read before designing a run)

`ab.harness.apply_shift` truncates the **tail** only, so a shifted point also
loses training history: on a 3-year snapshot `min_train` falls 721 → 541 → 361
across the NOW / −6M / −12M clusters. That confounds "different period" with
"less data", and it is not a theoretical worry — it produced three wrong or
grumbled readings (round 12's seasonal "gain", round 13.1's r = −0.6, and a
verdict that nearly removed the cheap2h hurdle; see [REJECTED.md](REJECTED.md)).

**Prefer a sliding constant-length window when a long snapshot is available.**
`ab_test.py fetch --days 1825` caches ~5 years under `ab_cache/long/`; a window
of `rows[n-L-s : n-s]` then holds `min_train = L-364` fixed at every shift, so
each point is "production as it would actually have run on that date".
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
