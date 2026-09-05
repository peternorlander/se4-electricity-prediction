# SE4 Prediction — Outstanding Work

Working list of **not-yet-done** accuracy items only. Everything settled —
architecture, methodology, current MAE baseline, shipped changes, and the full
"Features Tested and Rejected" ledger — lives in `README.md`; read that first,
especially "A/B Backtest Flow" (how to run an experiment, which verdict
function to call, tail-truncated vs sliding grids) and "How changes are
validated" (the adoption bar). This file assumes that context and does not
repeat it.

**Priority: `cheap2h` first, then `min`. `avg` improvements are welcome; `max`
is explicitly not a priority.**

**Standing rule for everything below:** re-opening a rejected idea, or trying
a new one, needs a stated mechanism for why it should work — not "we changed
a lot, maybe it's different now." If you can't state one, that's the answer.
Screen on one period cluster, confirm on the others, and pre-register the
confirmation bar before looking at the screen. New A/Bs use the round-15b
sliding grid on a long snapshot (`ab_test.py fetch --days 1825`) — constant
`min_train`, four period clusters (NOW/−6M/−12M/−21M) — judged with
`classify_clustered` / `classify_ablation_clustered`, not the plain
`classify`/`classify_ablation` (see README "Which verdict function to call").

## Open items

### 0. Cross-border capacity / flows from ENTSO-E — data fetched, evaluation not started

**Status 2026-08-22: a 1-year probe fetch is on disk
(`ab_cache/crossborder/2026-08-22/`, via `experiments/fetch_entsoe_crossborder.py`).
Evaluation and implementation are a separate step and have not begun.**

Why this is item 0. Round 19 closed every route to regime-break handling that
runs off prices alone — round 18 (price signals as features), 19b (anchored
targets), 19e (coupling features derived from the SE4−DK2 price gap). All
harmful or NOISE. Cross-border data is the first genuinely *new* observable
since, and two of its documents publish *before* delivery, which is the property
no price-derived feature can have.

What came back, and what it says:

* **A11 physical flows** — works, all five borders (SE3, DK2, DE_LU, PL, LT),
  both directions, 136k rows for one year. Note the encoding: ENTSO-E only
  publishes points where flow is non-zero in that direction, so **a missing
  point is a zero**. Reindex to a full 15-minute grid and fill 0 before doing
  anything else; the absent rows are the signal, not a gap.
* **A78 unavailability of transmission infrastructure** — works, 626 events in
  one year (DE_LU 281, PL 177, SE3 71, DK2 52, LT 45). Carries
  `business_type` (A53 planned / A54 forced), `available_mw`, and a start AND
  end timestamp — i.e. posted ahead of delivery. `reason` is populated on
  143/626. This is the forward-looking document.
* **A61 forecasted transfer capacity — EMPTY on every border, and that is
  expected, not a bug.** The API resolves `A61` + `contract_MarketAgreement.Type=A01`
  to `FORECASTED_TRANSFER_CAPACITIES_EXPLICIT`, which only exists for
  explicitly-auctioned borders. Nordic borders are implicitly allocated through
  market coupling, so there is nothing to return. A78's `available_mw` is the
  substitute and it carries the same information in event form. Do not spend
  time "fixing" the A61 call without first probing which document types actually
  return data for these EIC pairs.

**What the probe already overturned.** The 2026-08-17 attribution in
`experiments/AUG21_SPIKE_POSTMORTEM.md` was wrong and the flow data says so
plainly: Baltic Cable did **not** come back. A78 carries a forced outage,
`available_mw = 0`, reason *"Trip of the BC-link"*, 2026-06-21 → 2026-09-18, and
`DE_LU` flow is 0.0 MW every single day from 2026-07-01 through 2026-08-22. What
changed on 08-17 was the Nordic supply balance, not interconnector availability:
SE4's import from SE3 stepped from ~1080 MW (08-16) to ~1960 MW (08-17) and
stayed there, while `mean_wind_stockholm` — the trough model's most load-bearing
feature — fell to the **3.9th percentile** (08-18) and **1.7th** (08-20) of its
three-year distribution. See the correction section in the post-mortem.

That refines the mechanism rather than killing it: the crippled export capacity
is the *background condition* that let SE4 run cheap all summer (total exports
~780–1040 MW in Jul/Aug against 1300–2200 in winter), and the near-record northern
calm is the *trigger* that removed the surplus and repriced SE4 to import parity.
The model has the wind; it does not have the capacity state it has to be
conditioned on. That interaction is the hypothesis to test.

**Discussion agenda for the dedicated thread** — the things that need a decision
rather than a measurement, roughly in the order they block each other:

1. **Is A78 usable as a FORWARD-looking feature at all?** This is the question
   the whole item hinges on. The document is published ahead of delivery, but
   the API returns the *current* record, not the one that was visible on a past
   date. If posted `end` dates get revised, a backtest built from today's
   snapshot knows things production could not have known, and every gain it
   measures is a leak. Two candidate resolutions: (a) start snapshotting A78
   daily from now on and only backtest on the frozen vintages, which is honest
   but means waiting months for enough history; (b) find a conservative
   encoding that revision cannot flatter — e.g. use only "an outage is active
   right now" rather than "it ends on date X". Option (b) is testable
   immediately and is probably where to start. The DE_LU record starting
   2026-08-17 and running to 2026-11-08 with 0 MW looks like exactly such a
   revision and is worth reading closely first.

2. **Which physical quantity is the feature?** Candidates, none tested:
   - `export_capacity_available` — sum over borders of nominal minus A78
     outage, forward-looking if (1) resolves favourably.
   - `se3_import_lag1` — realised SE3→SE4 flow. Backward-looking, but it was
     the variable that actually moved on 08-17 (~1080 → ~1960 MW).
   - `export_headroom` = capacity − realised export. The economically
     meaningful one: SE4 stays cheap while it has surplus it cannot ship.
   - The **interaction** of capacity with `mean_wind_stockholm`. This is the
     mechanism the 08-17 break actually demonstrates and the one with a stated
     causal story, so it should be pre-registered as the primary arm rather
     than discovered by sweeping.

3. **Does any of this survive round 18's finding?** Round 18 closed price-level
   features for min/cheap2h with the reading that "it is price-level
   information itself these targets reject — they are trough targets driven by
   weather → residual load, and a price level crowds that out." Flows and
   capacity are physical quantities in MW, not prices, so the mechanism is
   genuinely different. But `se3_import_lag1` is *close* to a price signal in
   disguise (flows are the market's response to prices), and it is a frozen
   lag like the ones round 18 rejected. State that objection before measuring,
   not after.

4. **Scope of the source module.** Five borders × three documents × 5 years is
   a lot of fetch for a pipeline that currently runs in a few minutes. Decide
   up front whether production fetches only what a validated feature needs
   (probably A78 for DE_LU/DK2 plus SE3 flow), and whether A11's volume needs
   daily pre-aggregation before it enters `fetch_training_inputs`.

5. **The one thing already decided:** do not spend time on A61. It resolves to
   `FORECASTED_TRANSFER_CAPACITIES_EXPLICIT`, which does not exist for
   implicitly-allocated Nordic borders. A78's `available_mw` is the substitute.

**What is NOT in scope here.** The archive-lag fix (README "Closing the
weather-archive lag") shipped 2026-08-23 and is unrelated. The conformalised
prediction interval from round 19c is a separate, independent item — see
`experiments/ROUND19_FINDINGS.md` §19c; it needs no new data and can proceed in
parallel.

Suggested order:

1. Re-run the fetch at `--years 5` so it spans `ab_cache/long`, and validate the
   flow series against the A78 events (an outage window should coincide with a
   zero-flow run; disagreement means one of the two is being parsed wrong).
2. Build candidate features and pre-register them before measuring. Obvious
   first set: `se3_import_lag1`, `export_capacity_available` (from A78, forward-
   looking), `export_headroom = capacity − realised export`, and the interaction
   of capacity with `mean_wind_stockholm`.
3. A/B on the round-15b sliding grid, `classify_clustered`, four period clusters
   — and score the 2026-08-17 week separately, since a five-day effect is
   invisible in a year-averaged MAE (`experiments/run_round19_spike.py` has the
   harness).
4. Only then decide on a production source module.

Open question to settle in step 1: A78 is published ahead of delivery, but the
**revision history is not** — the record on disk today is the *current* one, not
what was visible on 2026-08-16. A backtest built from today's A78 snapshot may
therefore be optimistic about what was knowable in advance. Check whether the
posted `end` dates moved (the 2026-08-17 → 2026-11-08 planned record on DE_LU
looks like exactly such a revision) before trusting any forward-looking claim.

### 1. Solar-capacity scaling for min/cheap2h — re-targeted, prior revised down

Idea: multiply radiation features (`mean_radiation`, `radiation_midday`) by an
installed-PV-capacity index, since SE4 solar has roughly doubled since 2023
and trees can't learn that monotonic buildout from cyclic calendar features
alone. A placeholder linear index was tested in 2026-07 and found no
replicable gain (see README rejected table) — but its own stated cause was
"already absorbed by `price_se4_min_lag1` / `residual_load_min`", and
`price_se4_min_lag1` is no longer in `min`'s feature list, so that absorber is
gone and the mechanism for re-opening still holds.

**The second leg of the case is weaker than it first looked.** A LIGHT/DARK
seasonal contrast in solar features is real and replicates, but adding
`day_of_year_sin/cos` alongside the solar features so the model could
condition on season came back net *harmful* once training-set size was
controlled for (the apparent gain was a data-starvation artefact of the old
tail-truncated grid, concentrated in the small-`min_train` clusters and
reversed at production's training size). "Helps in summer, hurts in winter"
nets to harmful once the model sees each season more than once.

**If run at all: screen and decide on the NOW cluster only** (closest to
production's training size) — the far clusters can manufacture a
training-size artefact that looks exactly like a seasonal win.

**Implementation warning, the part that makes this more work than it looks.**
Scaling `mean_radiation`/`radiation_midday` as a `transform` on the merged
daily frame is a **near no-op** for the trough targets: neither column is in
their feature lists, and `residual_load`/`residual_load_min`/
`radiation_variability` (which are) get computed from the hourly inputs
*before* a `transform` would run. The scaling has to move upstream — into the
hourly weather inputs or the solar term itself, inside
`add_residual_load` / `aggregate_intraday_features` / `add_weather_variability`
— or it will "confirm" the rejection for entirely spurious reasons.

Real ENTSO-E A68 per-zone installed capacity would be the theoretically
cleaner input (the current radiation features blend SE4+DK+DE, which grew at
different rates), but the first-order interaction test above showed nothing,
so the prior on this whole item is low. Do not invest in fetching A68 without
a more specific reason first.

### 2. Time-decay weights on `min`/`cheap2h` — re-test the measurement, not the model

`avg` keeps time-decay sample weighting (half-life 500) — a drift-free A/B
showed it helping in all three runs it was tested in. The same lever was
rejected for `min`/`cheap2h` in 2026-07, but on a materially weaker
instrument: three runs on different real days, before the shift/cluster grid
existed, with the effect (~±0.2 EUR/MWh) smaller than the between-run swing
(~0.4). The sliding four-cluster grid is a much sharper tool and has never
been pointed at this question.

**Honest caveat — this may still be unresolvable.** A separate confirmed
effect of similar size (−0.20 EUR/MWh) landed at only 12 of 16 point-level
sign-consistency on this same grid, so 0.2 sits right at its resolution
floor. Decide what counts as a pass *before* running, and be prepared for
`INCONCLUSIVE` to be the honest answer rather than a reason to keep pushing.

### 3. `min ≤ cheap2h` coherence

`min` and `cheap2h` are independent models, so nothing stops predicted
`cheap2h` (mean of the day's two cheapest hours) from coming in *below*
predicted `min` (the single cheapest hour) — a mathematical impossibility in
the real data (verified: holds on every held-out day checked). Last measured
(2026-08-06, before the two models' feature lists diverged further): 31 of 60
held-out days violated it, worst violation +8.9 EUR/MWh. **That number should
be treated as stale** — `min` and `cheap2h` now run different feature lists
(14 vs 15 columns, differing by `price_se4_max_lag1`) rather than the shared
list they had when this was measured, which likely changes how correlated
their errors are. Re-measure the violation rate before designing a fix.

This is a **different intervention from the already-rejected clamp**: what
was rejected was an *avg-anchored* `min ≤ cheap2h ≤ avg ≤ max` clamp, which
only ever touched cheap2h and made it worse (when predicted cheap2h exceeds
predicted avg, the incoherence means avg was too low, not cheap2h too high —
clipping cheap2h toward avg moves it away from truth). `min ≤ cheap2h` alone
is a different, narrower identity and doesn't inherit that failure mode.

Test three arms: raise cheap2h to min, lower min to cheap2h, split the
difference. Expect a small MAE effect at best — the real payoff is that Home
Assistant stops receiving a logically incoherent pair on a meaningful minority
of days.

### 4. Richer supply-side data (only if the above plateaus)

Genuinely new information rather than re-encoding what's already fetched,
roughly in priority order:

- **ENTSO-E day-ahead load forecast (A65)** and **wind/solar generation
  forecast (A69)** for SE4/DK/DE — actual TSO forecasts should beat the
  weather-proxy `residual_load` at short horizons.
- **Capacity-weighted nuclear outages** — A77 documents carry unit nominal
  power; weight by MW instead of counting each outage as 1 (a 1400 MW
  Oskarshamn-3 outage is not equivalent to a small unit).
- **SE3 / system price lag** — SE4 is tightly coupled northward, yet only
  DE/DK2 neighbouring-zone lags are in the feature set today.
- **Transmission (NTC) outages** on Baltic Cable / Öresund (A78) — drives SE4
  divergence from neighbouring zones; most parsing work of the four, so last.

### 5. `ttf_vs_30d` — regime-abnormality ratio, low expected yield

Motivated by the 2026-07 Iran-war energy-crisis price spike: world events
don't hit SE4 prices directly, they transmit through fuel markets
(`ttf_price_lag1`, `ttf_rolling_7d`, `co2_price_lag1`, `gas_marginal_cost`),
which are already covered. The gap: trees split on absolute levels, so a
level split learned from 2023 data doesn't generalize to a 2026 shock.
`ttf_vs_30d = ttf_price_lag1 / ttf_rolling_30d` (needs a new 30-day rolling
mean alongside the existing 7-day one) encodes "how far above normal" as a
stationary ratio instead, and should react before a shock fully lands in the
electricity-price lags.

**Its sibling, `price_vs_30d`, was tested and rejected** (round 18, 2026-08:
harmful on both `min` and `cheap2h`, confirmed on three independent
measurements — see README rejected table), which lowers the prior on this one
too, though it's a fuel-price ratio rather than an own-price ratio and so
isn't subject to the exact same "own frozen price lag hurts the trough
targets" mechanism. Gate on walk-forward MAE as usual; expect this to matter
less than items 1–3 above.

**Do not add news/geopolitical-event features** (event dummies, GPR-style
sentiment indices) on top of this — TTF prices the event in faster and more
quantitatively than any hand-built flag, a binary crisis dummy has only 1–2
occurrences in a 3-year window (unlearnable), and daily news indices at this
frequency mostly give XGBoost something to overfit. Oil/Brent isn't worth
adding either — gas, not oil, sets the European marginal cost, and gas is
already covered.
