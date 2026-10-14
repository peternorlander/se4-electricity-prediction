# SE4 Prediction — Outstanding Work

Working list of **not-yet-done** accuracy items only, carried from the 2026-08
round. The newer list is [IMPROVEMENT_PLAN_2026-09.md](IMPROVEMENT_PLAN_2026-09.md)
— start there.

Everything settled lives under [docs/](docs/): the current model in
[MODEL.md](docs/MODEL.md), shipped changes in [DECISIONS.md](docs/DECISIONS.md),
the rejected ledger in [REJECTED.md](docs/REJECTED.md), and the methodology in
[AB_TESTING.md](docs/AB_TESTING.md) — read that one first, especially "How to
run an experiment", "Which verdict function to call", "Measurement grids:
training geometry and period clusters" and "How changes are validated" (the adoption bar).
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

**How this file is maintained (2026-09-12, updated 2026-10-14).** An item is *deleted* from here
the moment its verdict is recorded in the ledgers — a row in
[docs/REJECTED.md](docs/REJECTED.md) or an entry in
[docs/DECISIONS.md](docs/DECISIONS.md), plus a section in
[docs/FINDINGS.md](docs/FINDINGS.md) when the round found something worth
explaining. This file is only what is left to do; the ledgers are the record.
Closed from here so far: cross-border capacity and flows (2026-09-12), solar-capacity
scaling (2026-09-13; superseded by [IMPROVEMENT_PLAN_2026-09.md](IMPROVEMENT_PLAN_2026-09.md)
§2.1-C), time-decay on `min`/`cheap2h` (re-tested with a drift mechanism and rejected
2026-10-11, [REJECTED.md](docs/REJECTED.md)) and `min ≤ cheap2h` coherence (shipped
2026-10-12, [DECISIONS.md](docs/DECISIONS.md#min--cheap2h-coherence-round-25)). Everything else
that was open here has either moved to the newer plan or is below.

## Open items

### 1. Richer supply-side data — what is left

The 2026-08 list is mostly done or moved:

- **Capacity-weighted nuclear outages** and the **ENTSO-E week-ahead load forecast (A65)**
  are [IMPROVEMENT_PLAN_2026-09.md](IMPROVEMENT_PLAN_2026-09.md) §2.7 — where the nuclear
  half became a decomposition after the existing column turned out to count non-nuclear
  and cancelled records.
- **SE3 / system price lag.** The price route is closed for the trough targets (round
  18, [REJECTED.md](docs/REJECTED.md)); the physical route — northern wind points — shipped
  2026-10-12 ([DECISIONS.md](docs/DECISIONS.md#northern-wind-points-round-25)).
- **Day-ahead load (A65) and wind/solar generation forecasts (A69)** reach D+1 only, and
  D+1 is never predicted, so they cannot be forecast features at all (the horizon column
  of [FINDINGS.md §1.3](docs/FINDINGS.md#13-sources-verified-as-reachable-with-what-they-carry-appendix-a-has-the-probes));
  they are calibration data at best (2.1-C), and A75 actuals already serve that.

### 2. `ttf_vs_30d` — regime-abnormality ratio, low expected yield

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
less than the items in the newer plan. Round 24's finding that the autumn-2026
misses are a gas-price-and-coupling regime
([IMPROVEMENT_PLAN_2026-09.md](IMPROVEMENT_PLAN_2026-09.md) §2.17) is the stronger
route to the same problem.

**Do not add news/geopolitical-event features** (event dummies, GPR-style
sentiment indices) on top of this — TTF prices the event in faster and more
quantitatively than any hand-built flag, a binary crisis dummy has only 1–2
occurrences in a 3-year window (unlearnable), and daily news indices at this
frequency mostly give XGBoost something to overfit. Oil/Brent isn't worth
adding either — gas, not oil, sets the European marginal cost, and gas is
already covered.
