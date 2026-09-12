# SE4 Prediction — Outstanding Work

Working list of **not-yet-done** accuracy items only, carried from the 2026-08
round. The newer list is [IMPROVEMENT_PLAN_2026-09.md](IMPROVEMENT_PLAN_2026-09.md)
— start there.

Everything settled lives under [docs/](docs/): the current model in
[MODEL.md](docs/MODEL.md), shipped changes in [DECISIONS.md](docs/DECISIONS.md),
the rejected ledger in [REJECTED.md](docs/REJECTED.md), and the methodology in
[AB_TESTING.md](docs/AB_TESTING.md) — read that one first, especially "How to
run an experiment", "Which verdict function to call", "Measurement grids:
tail-truncated vs sliding" and "How changes are validated" (the adoption bar).
This file assumes that context and does not repeat it.

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
`classify`/`classify_ablation` (see
[docs/AB_TESTING.md](docs/AB_TESTING.md#which-verdict-function-to-call)).

**How this file is maintained (2026-09-12).** An item is *deleted* from here
the moment its verdict is recorded in the ledgers — a row in
[docs/REJECTED.md](docs/REJECTED.md) or an entry in
[docs/DECISIONS.md](docs/DECISIONS.md), plus a section in
[docs/FINDINGS.md](docs/FINDINGS.md) when the round found something worth
explaining. This file is only what is left to do; the ledgers are the record.
Item 0 (cross-border capacity and flows) was closed that way on 2026-09-12: see
[docs/FINDINGS.md](docs/FINDINGS.md#cross-border-capacity-and-flows-round-21).

## Open items

### 1. Solar-capacity scaling for min/cheap2h — re-targeted, prior revised down

Idea: multiply radiation features (`mean_radiation`, `radiation_midday`) by an
installed-PV-capacity index, since SE4 solar has roughly doubled since 2023
and trees can't learn that monotonic buildout from cyclic calendar features
alone. A placeholder linear index was tested in 2026-07 and found no
replicable gain (see [docs/REJECTED.md](docs/REJECTED.md)) — but its own stated cause was
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
  DE/DK2 neighbouring-zone lags are in the feature set today. Round 21 raised
  the prior: the state that separates a cheap calm week (Sep 2024) from an
  expensive one (Aug 2026) is the northern supply balance, and SE3→SE4 net
  import halved during 2026 (quarterly mean 3182 → 1629 MW). It is still a
  price level, which round 18 closed for the trough targets, so the physical
  route (northern wind points, `IMPROVEMENT_PLAN_2026-09.md` §2.13) comes
  first.

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
measurements — see [docs/REJECTED.md](docs/REJECTED.md)), which lowers the prior on this one
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
