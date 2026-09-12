# Findings

Where the model's behaviour gets explained rather than just recorded. A round
that produced a number goes in the ledgers — [DECISIONS.md](DECISIONS.md) if it
shipped, [REJECTED.md](REJECTED.md) if it did not. A round that produced an
*understanding* — why a break was unpredictable, why a feature is degenerate,
what the data cannot support — goes here, and the ledger row links to it.

Everything below is dated and was true when measured. Before acting on one,
check whether the code it describes still looks that way.

| Finding | When | Why it still matters |
|---|---|---|
| [The 2026-08-17 regime break](#the-2026-08-17-regime-break) | 2026-08 | The clearest evidence of what the model structurally cannot do: a 3-year window cannot output 128 EUR/MWh, and four independent routes to fixing that all failed |
| [Cross-border capacity and flows](#cross-border-capacity-and-flows-round-21) | 2026-09-12 | Closes the interconnector route for the trough targets, including with leaked data; leaves one open `avg` candidate; documents six cross-border data traps |
| [1.1 `residual_load` is temperature in disguise](#11-residual_load-is-temperature-in-disguise-and-its-wind-term-is-saturated) | 2026-09-08 | The "physical trough driver" is ~−0.9997 correlated with temperature and its wind term is saturated — several rejected entries were measured on a degenerate column |
| [1.2 The regime is moving away from the model](#12-the-regime-the-model-was-tuned-on-is-moving-away-from-it) | 2026-09-08 | Negative-price share 25 % → 7 %, the trough moved from night to midday, DK2/DE coupling rising |
| [1.3 Sources verified as reachable](#13-sources-verified-as-reachable-with-what-they-carry-appendix-a-has-the-probes) | 2026-09-08 | What each candidate source carries, and the horizon column that decides whether it can be a forecast feature at all |
| [1.4 No weather point under the northern wind fleet](#14-the-weather-grid-has-no-point-under-the-northern-wind-fleet) | 2026-09-11 | The three northern candidates are each worth more than any existing site but Stockholm |

## The 2026-08-17 regime break

The motivating case for round 19 and for the cross-border work of round 21.
Worth keeping because it is
the clearest evidence this project has for what the model structurally *cannot*
do, and because the obvious fixes were all tested and all failed.

### What happened

| date | avg | cheap2h | cheap2h/avg |
|---|---|---|---|
| 2026-08-16 | 20.1 | 6.0 | 0.30 |
| 2026-08-17 | 138.1 | 20.2 | 0.15 |
| 2026-08-18 | 163.7 | **127.9** | **0.78** |
| 2026-08-19 | 164.0 | **128.0** | **0.78** |
| 2026-08-20 | 142.8 | 81.2 | 0.57 |

`cheap2h = 128.0` is the **all-time maximum of the 3-year training window**
(n = 1096, median 13.1, p99 100.5). The previous August record across three
Augusts was 85.7.

**The shape broke, not just the level.** `cheap2h/avg` = 0.78 against a 3-year
median of 0.28 — there was no cheap window at all — and the trough moved from
night to midday: on 08-16 the cheap hours were 00–15, on 08-18 they were 13–14
(solar) and the night was the expensive part. A night-charging automation would
have paid ~160 instead of 128. Days with `cheap2h/avg > 0.70` **and** `avg > 100`
number 57 of 1096 and are almost all Nov–Feb.

### What caused it

SE4 re-coupled to the continent: the |SE4 − DE| gap collapsed from ~90 to ~15
EUR/MWh and the 30-day correlation went 0.33 → 0.79. 2026-07-25 → 08-11 had been
an 18-day run with SE4 − DE < −50 (mean −85.5), the longest in the whole history.

**It was not Baltic Cable returning**, which is what the press and an early read
of this data suggested. ENTSO-E A78 carries a forced outage on that link,
`available_mw = 0`, reason *"Trip of the BC-link"*, 2026-06-21 → 2026-09-18, and
the DE_LU physical flow was 0.0 MW every day from 2026-07-01 through 08-22. The
real mechanism has two layers:

* **Background:** crippled export capacity kept SE4's surplus trapped all summer
  (total exports ~780–1040 MW in Jul/Aug against 1400–2200 across Oct–Jan), which is
  why it ran so cheap and so decoupled.
* **Trigger:** the surplus vanished. `mean_wind_stockholm` — the trough models'
  most load-bearing feature — fell from the 70th percentile (08-16) to the
  **3.9th** (08-18) and **1.7th** (08-20). SE4 flipped from exporting to
  importing ~1960 MW from SE3 and repriced to import parity.

So the model *has* the wind. The obvious reading was that what it lacks is the
capacity state the wind has to be conditioned on. **That was tested in round 21
and it is not the answer** — see
[Cross-border capacity and flows](#cross-border-capacity-and-flows-round-21).
Capacity × calm fired on exactly the right days and changed nothing, because the
same state in September 2024 came with a cheap2h of 6.6: the conditioner that
separates the two is the northern supply balance, not the interconnector.

### What the model predicted, and why it is structural

Replaying the forecast production actually issued each morning (fit ≤ D−5, i.e.
under the archive lag), cheap2h at d+1:

| date | actual | predicted |
|---|---|---|
| 2026-08-17 | 20.2 | 20.4 |
| 2026-08-18 | **127.9** | 34.4 |
| 2026-08-19 | **128.0** | 30.5 |
| 2026-08-20 | 81.2 | 40.8 |

MAE 4.32 before the break, **57.75** from 08-17 on, bias −55.5. The highest
number production issued anywhere that week was **49.1**.

Four reasons this is not a tuning failure. Each was measured, and together they
close the obvious routes:

1. **No feature carries the cause.** `TROUGH_FEATURE_COLUMNS` has no continental
   price feature at all, and nothing anywhere encodes transmission capacity.
2. **Adding the continental price back does not fix it.** Arms on the same
   slices: production 52.84 spike MAE, `+price_de_lag1/price_dk2_lag1` 50.84,
   `+coupling gap` 51.63, and an **oracle given the same-day German price**
   (impossible in production) 51.30 — predicting **41.6** for a realised 127.9.
   DE sat at 100–170 through the entire cheap fortnight; only the coupling
   changed, so the price was never the missing information.
3. **The model cannot output the number.** XGBoost interpolates; it cannot
   extrapolate past its training targets. Fitted in-sample on the full window,
   cheap2h's maximum output is **108.8** all-year and **95.2** on May–Sep rows,
   against a realised 127.9. **No feature set makes 128 reachable from a 3-year
   window.** This is the single most durable constraint on future work here.
4. **Reformulating the target does not rescue it.** Predicting `cheap2h − avg`
   or `cheap2h/avg` and reconstructing gives 51.56 / 51.89 against 52.84 — the
   *shape* (0.78) is as unprecedented as the level, so decomposing just moves
   the out-of-distribution problem, and it costs 6.7 EUR/MWh of pre-break
   accuracy.

A 5-year training window raises the ceiling (cheap2h targets top out at 450.2
instead of 107.9) and does help on the break — 48.52 against 52.84 — but costs
three times the pre-break accuracy (7.98 against 2.53). Regime insurance with a
real premium, consistent with round 15's closure of window length rather than a
challenge to it.

## Cross-border capacity and flows (round 21)

Closed 2026-09-12. This was IMPROVEMENT_PLAN item 0, the last route the
[2026-08-17 break](#the-2026-08-17-regime-break) left open: ENTSO-E physical
flows (A11) and transmission unavailability (A78) for the five SE4 borders,
cached under `ab_cache/crossborder/`. Arms and bar were fixed in advance, before
any result was read. The scripts, the pre-registration and the raw per-window
results (`experiments/run_round21_*.py`, `experiments/results/round21_*.jsonl`,
`experiments/ROUND21_PREREGISTRATION.md`) are scratch and uncommitted — the
numbers below are the record.

**For `cheap2h` and `min` the route is closed.** Round-15b sliding grid, 14
points, four period clusters, `classify_clustered`. Every arm — realised SE3
import, available southbound export capacity, export headroom, and the
pre-registered primary `capacity × calm-in-Stockholm` interaction — is NOISE on
both targets; export headroom is REAL-and-*harmful* on cheap2h (+0.144, 2 of 14
points favourable). The decisive arm is the control: a **leaky** per-day
capacity built from outage records production could not have had is NOISE too
(+0.033 / −0.050), so even perfect foreknowledge of interconnector capacity has
no year-average value for the trough targets. On the break week itself no arm
lifts the 08-18/19 day-ahead prediction above ~48 against a realised 128.

**One open candidate, on `avg`:** `export_headroom_lag1` — A78-available
southbound export capacity minus realised net southbound export, on the run
day, frozen across the horizon.

| snapshot | clmean | NOW | −6M | −12M | −21M | favourable | Δstd |
|---|---|---|---|---|---|---|---|
| `long/2026-08-21` | **−0.423** | −0.785 | −0.329 | −0.133 | −0.447 | 11/14 | −0.209 |
| `long/2026-08-06` (independent fetch) | **−0.218** | −0.524 | −0.144 | −0.056 | −0.148 | 11/14 | −0.266 |

REAL on both, every period cluster favourable both times, per-window std
*falling*. Three checks behind it: the A11-only encodings (export flow alone,
capacity inferred from whether a link flowed) are NOISE, so the A78 capacity
half is what carries it; the round-19e price-derived congestion feature is
**not** a substitute (REAL on one snapshot with std +0.23, NOISE on the other,
and adding it on top of headroom destroys the gain — round 18 again); and a
vintage-honest rebuild, per window from only the records published by that run
date, stays favourable on all four NOW points (−0.17 to −0.54). Read the effect
as **−0.2 to −0.4 EUR/MWh on `avg`**. The gain is concentrated where the avg
price/market prune's was: windows with Baltic Cable closed improve −1.14
against −0.23 elsewhere, the worst-MAE quartile −1.18, the calmest quartile
+0.46.

**Not adopted.** `avg` was not the pre-registered target and ~12 arms were tried
on it, so this is a screen that replicated, not a verdict. What it still needs:
the serve-time encoding (production knows tomorrow's *scheduled* exchanges and
the outages posted for it, not realised physical flows), a fresh cross-border
fetch with the fixed parser, and a re-check after the residual-load rebuild,
since `residual_load` is in avg's list and
[verdicts are scoped to the model](AB_TESTING.md#how-changes-are-validated).

### If you touch `ab_cache/crossborder/`, read this first

Six properties of this data that each cost hours to find. The fetch script's
parser was fixed on 2026-09-12; `experiments/round21_xbfeat.py` re-parses `raw/`
for caches fetched before that.

* **`docStatus = A09` means cancelled.** 632 of 4487 period rows in the
  2026-08-23 cache are cancellations of outages that never happened, and they
  were ~64% of that cache's "link down" days. The independent 2026-09-12 fetch
  reproduces the rate exactly (631 of 4575), so this is what the document type
  is, not one bad cache. Filter them.
* **There is no vintage before 2025-11.** The API returns only each record's
  current revision (1795 of 2428 are rev ≥ 2 in the 2026-08-23 cache), and
  *every* record for an event before November 2025 was re-published that month.
  The 2026-09-12 fetch confirms it from scratch: **0 of 4575 rows carry a
  `createdDateTime` before 2025-11, and 3633 of them fall in that single
  month.** So "what was knowable on date D" cannot be reconstructed for the
  older clusters at all, and a forward-looking A78 feature cannot be backtested
  honestly on this data. Nord Pool's UMM API (`publicationDate` + `version`) or
  daily A78 snapshots are the only ways to change that.
* **Outage history thins out fast before 2026, unevenly per border.** Distinct
  non-cancelled records per year in the 2026-09-12 fetch: DE_LU 0 / 22 / 13 / 7
  / 181 / 134 for 2021…2026, PL 52 / 109 / 109 / 131 / 87 / 155, SE3 37 / 110 /
  82 / 129 / 62 / 28. Days where A78 says the Baltic Cable is fully out:
  0 (2021), 5, 8, 17, 9, **96 (2026)**. So a capacity feature carries much less
  information in the far period clusters than in NOW — it is close to constant
  there — which is a caveat on any four-cluster verdict built on A78, including
  the open `avg` candidate above.
* **A11 range limits move.** The 365-day chunks that fetched five years of
  physical flows on 2026-08-22 returned **HTTP 400 on every historical chunk on
  2026-09-12**, while the same 365-day chunks of A78 were fine in the same run.
  27 minutes of fetching produced 78 rows. The fetcher now asks for the whole
  range and **halves it on refusal** down to a 16-day floor, so it finds
  whatever the current limit is instead of encoding a guess, and it logs
  ENTSO-E's own `Reason` text (which `requests`' `HTTPError` message drops —
  losing it is why the first failure was undiagnosable). `--into YYYY-MM-DD`
  re-fetches one document type into an existing cache directory without
  clobbering the others.
* **DK2 rows describe one 400 kV cable, not the border.** 69 days show
  `available_mw = 0` while the border keeps flowing. Never min() them into a
  border capacity; DK2 was held at nominal in every round-21 arm.
* **A78 and A11 do agree, once the cancellations are gone** — on the HVDC links
  the element *is* the border: Baltic Cable 135 of 135 outage days show zero
  flow, PL 202 of 218, LT 45 of 57. And A78 sees what flows cannot: partial
  reductions, such as Baltic Cable at 210 MW from 2025-10-15 to 2026-02-02.

One correction to the record: the DE_LU outage running 2026-08-17 → 11-08 that
IMPROVEMENT_PLAN flagged as a suspected revision is Breared–Söderåsen, an
SE4-internal line, and it is **cancelled**. The Baltic Cable trip record is
revision 3, created 2026-08-17 10:32 — its end date was posted on the break day.

## 1. What this review measured (2026-09-08, snapshot `ab_cache/2026-09-05`)

Four findings, each of which moves a priority. Numbers are reproducible with
the recipes in Appendix B.

### 1.1 `residual_load` is temperature in disguise, and its wind term is saturated

| quantity | value |
|---|---|
| corr(`residual_load`, `mean_temp`) | **−0.9997** |
| corr(`residual_load_min`, `max_temp`) | **−0.9988** |
| daily std of the demand term `15 − mean_temp` | 6.60 |
| daily std of the wind term `(min(v,13)/13)³` | 0.125 |
| daily std of the SE4 solar term `radiation/500` | 0.202 |
| share of *hours* with the SE4 wind term at 1.0 (rated) | **88 %** (DE-north 84 %, DK1 89 %, DK2 86 %, Karlskrona 85 %, Stockholm 83 %) |
| share of *days* with the SE4 daily wind term at 1.0 | **94 %** |

Two independent defects:

1. **Unit mismatch.** `sources/open_meteo.py` never passes `wind_speed_unit`, so
   Open-Meteo returns `windspeed_100m` in its default **km/h** (median 26.6 at
   Malmö, i.e. 7.4 m/s — a normal 100 m wind). `features._wind_power_curve`
   documents its input as m/s with `rated_speed = 13`. 13 km/h is 3.6 m/s, cut-in
   speed, so the cubic curve is at rated output on almost every hour and the
   "wind supply" term only distinguishes dead calm from everything else.
   Dividing by 3.6 inside the curve restores the intended shape: the daily
   term's std goes 0.125 → 0.266 and its share at 1.0 goes 94 % → 4 %.
   `wind_night`'s correlation with `price_cheap2h` roughly doubles
   (−0.116 → −0.218) from that one change.
2. **Scale mismatch, which the unit fix does not cure.** Even corrected, the wind
   term spans [0, 1] and the solar term [0, ~1.5] while the demand proxy is in
   °C and spans ~40. One unit of "full wind" is worth one degree of temperature.
   So `residual_load` and `residual_load_min` remain ≥ 0.998 correlated with
   temperature after the fix; the "physical trough driver" the README describes
   does not exist in the model. The trough targets get their wind information
   only through the raw speed columns `max_wind`, `mean_wind_stockholm`,
   `mean_wind_de_north` — no power curve, no regional weighting, no night
   window.

**Why this matters for the ledger.** Several rejected entries were measured on
this feature: the intraday trough block (`wind_night`, `radiation_midday`,
`residual_load_range`) as a year-round addition, the seasonal interaction round
12, and the solar-capacity scaling. Each was a test of a column whose wind
component was ~constant. That is a stated mechanism for re-opening them — after
the feature is rebuilt, not before. `residual_load_min`'s LOO verdict ("keep,
harmful to remove") is consistent with it being the trough list's only
temperature channel, not evidence that the trough physics is captured.

### 1.2 The regime the model was tuned on is moving away from it

| | 2023 (Sep–Dec) | 2024 | 2025 | 2026 (Jan–Sep) |
|---|---|---|---|---|
| share of days with `price_min < 0` | 26 % | 27 % | 23 % | **7 %** |
| share of days with `cheap2h ≤ 0` | 24 % | 25 % | 21 % | **7 %** |
| slot of the daily min: night 00–07 | 66 % | 45 % | 38 % | **26 %** |
| slot of the daily min: day 08–15 | 4 % | 31 % | 46 % | **63 %** |
| corr(SE4 cheap2h, DK2 cheap2h, same day) | 0.71 | 0.67 | 0.79 | **0.89** |
| corr(SE4 cheap2h, DE cheap2h, same day) | 0.44 | 0.47 | 0.66 | **0.73** |

Last 365 days: 30 negative-price days, 84 days with cheap2h < 5 EUR/MWh.

Three consequences: the hurdle classifier's positive class (`price_min < 0`) is
thinning in the NOW regime (item 2.5); the trough is now a midday-solar event
more often than a night-wind one, which is exactly the split the slot plan
formalises (2.6) and which the current blended features cannot express (2.1);
and SE4's trough is increasingly the DK2 trough, so anything that improves the
DK/DE side of the weather → supply mapping pays more every year (2.1, 2.3).

### 1.3 Sources verified as reachable, with what they carry (Appendix A has the probes)

| source | what it adds | horizon | history |
|---|---|---|---|
| Open-Meteo **Previous Runs API** | the forecast that was *issued* 1–7 days before each hour, per model | — | ICON / GFS from **2024-03-01**, ECMWF IFS from **2024-04-01** |
| Open-Meteo `models=` and **Ensemble API** | ECMWF / ICON / GFS / Météo-France side by side; 51 ECMWF ensemble members | 8 d (ensemble), **16 d** (`forecast_days=16` works) | live only |
| Energinet `Forecasts_Hour` | Energinet's own wind/solar forecasts for DK1/DK2, with the day-ahead vintage stored | D+1 only | from 2019-10-31 |
| Energinet `Transmissionlines` | hourly `ImportCapacity` / `ExportCapacity` / scheduled and physical exchange per DK border, incl. **DK2–SE4** | published with the day-ahead result | depth not verified (rate-limited during the probe) |
| Nordpool `DayAheadFlow` | tomorrow's *scheduled* day-ahead flow per SE4 border (SE3, SE3-SWL, DK2, PL, GER, LT), 15-min | D+1 | per-date query; depth not verified |
| ENTSO-E, not yet used | A65 **week-ahead** load (processType A31); A75 actual generation per type; A68 installed capacity per type; A77 with MW; A80 generation unavailability | see 2.7 | needs `ENTSO_E_TOKEN` |
| Open-Meteo **Nordic regional models** (`metno_seamless`, `dmi_seamless`) | MET Norway MEPS and DMI Harmonie (2.5 km), the reference short-range wind models for Scandinavia; "seamless" falls back to a global model beyond their range | 8 d | Previous Runs verified for 2025-06; earlier not probed |
| Nord Pool **UMM API** (`ummapi.nordpoolgroup.com/messages`) | every production / transmission / consumption unavailability message with **`publicationDate` and `version`**, installed and available MW per period, in/out area EICs, reason text; nuclear units carry installed 1081–1172 MW | published ahead of the event; **every earlier version retrievable** (`/messages/{id}/{version}`) | since 2015 (SE4 borders: 67 messages in March 2016 alone), ~115k messages, no auth |
| Additional Open-Meteo points, northern wind belt (1.4) | SE2 (Norrbotten, Jämtland) and Finnish Ostrobothnia wind — the fleets behind the SE3 import price | 8–16 d | archive to 1940 |

Note the horizon column. Production runs after tomorrow's prices are published,
so D+1 is never predicted and any source that only reaches D+1 is useless as a
*forecast* feature — it can still be valuable for **calibration** (2.1) and for
the **coupling state** (2.4), which is quasi-static across the week.

### 1.4 The weather grid has no point under the northern wind fleet

`mean_wind_stockholm` is the most load-bearing trough feature (LOO: +1.36 on
cheap2h, +1.02 on min if removed), yet Stockholm is a load centre with little
wind capacity. What it proxies is the Nordic surplus that reaches SE4 through
SE3: when the north is windy, SE3 clears low and SE4 imports at that price
(the 2026-08-17 break was the opposite — Stockholm wind at its 2nd percentile).
The fleets that create that surplus sit in SE2 (Markbygden in Norrbotten,
Jämtland, Västernorrland — Sweden's largest wind zone), SE1 and Finnish
Ostrobothnia, and none has a point in `WIND_LOCATIONS`. The southern side is
covered three times over (DK1, DK2, DE-north).

Screen, 2026-09-11: linear incremental R² on `cheap2h` from adding a candidate
point's daily and night-window power-curve terms to the existing six sites plus
radiation, temperature and calendar; 2023-09 → 2026-08 (n = 1091) and 2026
alone. In-sample and linear, so a ranking, not an effect size.

| candidate (lat, lon) | represents | ΔR² all | ΔR² 2026 | corr with nearest existing site |
|---|---|---|---|---|
| **Norrbotten** (65.3, 20.9) | SE2 north, Markbygden | **+0.041** | **+0.060** | 0.33 |
| **Ostrobothnia** (63.1, 21.6) | FI wind belt | **+0.037** | **+0.049** | 0.59 |
| **Jämtland** (63.2, 14.6) | SE2 inland | **+0.025** | **+0.056** | 0.49 |
| Sundsvall (62.4, 17.3) | SE2 coast | +0.017 | +0.039 | 0.50 |
| Bergen (60.4, 5.3) | NO hydro, precipitation | +0.006 | +0.018 | 0.44 |
| Gdańsk (54.4, 18.6) | PL Pomerania | +0.004 | +0.004 | 0.64 |
| Göteborg (57.7, 12.0) | SE3 west coast | +0.003 | +0.021 | 0.81 |
| Rostock (54.1, 12.1) | DE Mecklenburg / Baltic coast | +0.003 | +0.003 | 0.90 |
| Munich (48.4, 11.6), radiation | DE south solar | +0.002 | +0.022 | 0.84 |
| South Baltic offshore (55.0, 13.5) | Kriegers Flak / Arkona | +0.001 | +0.001 | 0.91 |
| German North Sea (54.3, 7.0) | offshore cluster | +0.000 | +0.009 | 0.75 |

For scale, removing an *existing* site from the same regression costs 0.002–0.007
except Stockholm at 0.049. The three northern points are each worth more than any
existing site but Stockholm, carry information the grid does not have
(correlation 0.33–0.59 with the nearest existing point), and matter more in 2026
than pooled. Every southern candidate is redundant with DK/DE-north at
correlation 0.75–0.91. DE-south solar is nil pooled and +0.022 in 2026 — the
rising solar coupling of 1.2 — and belongs with the day-slot work (2.1-C+, 2.6),
not with a wind-point addition.

## Appendix A — Source probes, 2026-09-08

All from this machine without credentials; ENTSO-E was not probed (token not
in the session).

**Open-Meteo Previous Runs**
`https://previous-runs-api.open-meteo.com/v1/forecast?latitude=55.6&longitude=13.0&hourly=wind_speed_100m_previous_day1,...&start_date=...&models=icon_seamless,ecmwf_ifs025,gfs_seamless`
Variables `wind_speed_100m`, `temperature_2m`, `shortwave_radiation` with
`_previous_day1..7` all return values. First month with data: **2024-03** for
`icon_seamless` and `gfs_seamless`, **2024-04** for `ecmwf_ifs025`; 2024-02 and
earlier return nulls for all three. Units default to km/h — pass
`wind_speed_unit=ms`.

**Open-Meteo forecast / ensemble**
`models=ecmwf_ifs025,icon_seamless,gfs_seamless,meteofrance_seamless` returns
one column per model; `forecast_days=16` returns 384 hours; the Ensemble API
(`ensemble-api.open-meteo.com/v1/ensemble`, `models=ecmwf_ifs025`) returns
`wind_speed_100m` plus `_member01.._member50` for 8 days.

**Energinet**
`https://api.energidataservice.dk/dataset/Forecasts_Hour` — PriceArea DK1/DK2,
ForecastType {Offshore Wind, Onshore Wind, Solar}, columns
`ForecastDayAhead / Forecast5Hour / Forecast1Hour / ForecastCurrent`,
earliest row 2019-10-31, latest HourUTC = tomorrow 23:00 (day-ahead only).
`.../dataset/Transmissionlines` — columns `PriceArea, ConnectedArea,
ImportCapacity, ExportCapacity, ScheduledExchangeDayAhead,
ScheduledExchangeIntraday, PhysicalExchangeNonvalidated,
PhysicalExchangeSettlement, CongestionIncomeEUR`. The DK2–SE4 filter hit the
rate limit (429, "try again in 110 seconds") — verify depth and whether
capacity is present for the *next* day with a single filtered call. Filter
syntax: `filter={"PriceArea":["DK2"],"ConnectedArea":["SE4"]}` (URL-encoded).

**Nordpool**
`https://dataportal-api.nordpoolgroup.com/api/DayAheadFlow?date=2026-09-08&deliveryArea=SE4`
returns 15-min `byConnectionArea` import/export/netPosition for SE3, SE3 (SWL),
DK2, PL, GER, LT, `status: Final`, updated 11:04 UTC the day before.
`DayAheadCapacities` with the same parameters returns 204 (empty) — the
endpoint exists but wants other parameters; not resolved.

**ENTSO-E cross-border cache** (`ab_cache/crossborder/2026-08-23`, 5 years):
302k A11 flow rows, 0 A61 rows (expected — explicit-auction document), 2478
A78 events (DE_LU 405 planned / 80 forced, DK2 304/33, LT 199/41, PL 688/99,
SE3 568/61). Planned-outage median duration 0.4 d (DE_LU, PL) to 3.7 d (SE3);
`available_mw` is populated (`nominal_mw` is not). Daily SE4 southbound net
export in 2026: 1856 MW (Jan) falling to 576 MW (Aug).

**Nord Pool UMM** (`https://ummapi.nordpoolgroup.com/messages`, probed
2026-09-11, no auth, `curl` works while Python `urllib` gets 403 — set a
browser-like `User-Agent`). Parameters that worked: `messageTypes` (**1
production, 2 consumption, 3 transmission** — a first probe misread 2 as
transmission and found nothing; use 3), `areas=<EIC>` (works for all three
types; SE4 transmission Jun–Dec 2026 = 176 messages),
`eventStartDate`/`eventStopDate`, `publicationStartDate`/`publicationStopDate`,
`limit` (≤ 100 returned per page), `skip`. Response
`{"items": [...], "total": N}`; each item has `messageId`, `version`,
`publicationDate`, `unavailabilityType` (1 unplanned / 2 planned),
`unavailabilityReason`, and `productionUnits[]` / `transmissionUnits[]` /
`consumptionUnits[]` with `installedCapacity` and `timePeriods[]` of
`availableCapacity`, `eventStart`, `eventStop`. SE3 nuclear example rows:
Ringhals 4 (1134 MW) available 0 from 2026-09-01, Forsmark 1 (1098 MW) 0 from
2026-09-06, Ringhals 3 (1081 MW) 0 to 2026-10-31 — all with version numbers
2–9. Total messages 114 758. Version history: `GET
/messages/{messageId}/{version}` returns each earlier version (HTTP 200) —
Baltic Cable message `e911027b-e42e-460b-b860-705a07ab5db7` v1–v4 as quoted in
2.14. Depth: SE4 transmission messages total 67 for March 2016, 35 for March
2019, 50 for March 2022 (one-month event windows; full-year windows time out
at 60 s — page by month). SE4-related transmission unit names seen in the
feed: `PL → SE4`, `SE4 → PL`, `SE3 → SE4`, `SE2 → SE3`, `FI → SE3`,
`SE4 → DK2`, `DK2 → SE4`, `LT → SE4`, `DE-LU → SE4`, `SE4 → LT`,
`SE4 → DE-LU`, `SE4 → SE3`, `SE3 → DK1`, `SE3 → NO1`.

**Location screen** (`scratchpad/screen_locations.py`, Open-Meteo archive
with `wind_speed_unit=ms`): base linear R² over the existing six sites +
radiation + temperature + calendar is 0.481 (cheap2h), 0.477 (min), 0.519
(avg). Full table in 1.4; per-candidate daily frames were saved as
`cand_<name>.pkl` next to the script during the session and are
re-fetchable in seconds.

**Naive references, last 364 days, cheap2h MAE** (for context when reading the
15.04 headline): persistence lag1 18.2, lag7 26.7, frozen 7-day anchor 24.5,
monthly climatology 23.9. Weekday means: Mon 40, Tue 49, Wed 47, Thu 38, Fri
34, Sat 23, Sun 20 EUR/MWh.

## Appendix B — Recipes

Reproduce 1.1 (needs a snapshot, no token):

```python
from ab.snapshot import load_snapshot
import features as F
inputs, meta = load_snapshot("2026-09-05")
base = F.build_training_data(**inputs)
print(base["residual_load"].corr(base["mean_temp"]))        # -0.9997
orig = F._wind_power_curve
F._wind_power_curve = lambda v, rated_speed=13.0: orig(v / 3.6, rated_speed)
fixed = F.build_training_data(**inputs)
F._wind_power_curve = orig
print(fixed["wind_night"].corr(fixed["price_cheap2h"]))    # -0.218 vs -0.116
```

Reproduce 1.2 from `ab_cache/2026-09-05/prices_hourly.pkl`: resample to hourly
means, Europe/Stockholm dates, `idxmin` per day for the slot of the minimum,
`(price_min < 0).mean()` per year; DK2/DE from `market_prices_hourly.pkl` with
the same `nsmallest(2).mean()` per day.
