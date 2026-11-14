"""
The day-level interval verdict: the gate since 2026-11-15. See docs/AB_TESTING.md,
"The adoption bar" and "The year grid".

WHY THIS REPLACED THE SIGN VOTES. classify() and classify_clustered() (ab/verdict.py) decide
on whether a handful of means share a sign. Neither uses an estimate of the noise, and the
review of 2026-11-14 measured what that costs (docs/FINDINGS.md, "What the rolling grid
holds"):

  * A refit with another seed reads REAL under the cluster rule (round 27, cheap2h +0.13),
    and an arm at -0.268 is rejected because one cluster mean is +0.027 (`cpl_gap7`).
  * The points of one cluster are not near-replicates. A one-column arm's four NOW points
    read +0.79 / +0.56 / -0.28 / +0.67 on cheap2h.
  * Shifts that differ by a multiple of 7 are the SAME fits on the days they share: with a
    rolling training window, the model for a test week is decided by the day the week
    starts on, not by the shift. So 0/1/4/7 samples three of the seven possible week
    alignments, and "12 points x 6 caches" is far fewer independent observations than it
    looks: 733 distinct test days, two independent years.

WHAT IT DOES INSTEAD. The unit is the calendar day. Every (week alignment, day) is counted
once however many points produced it, the paired difference |candidate error| - |baseline
error| is summed per day, and the uncertainty of the mean comes from a block bootstrap over
consecutive calendar blocks -- so the interval reflects how many distinct days stand behind
the number, which is the thing the sign votes could not see. The verdict is the interval,
plus a per-year check that no year pays for the other.

Weekly paired differences are uncorrelated from one week to the next (lag-1..4
autocorrelation -0.07 .. +0.06 over 21 arms, round 28's stored results), so a block of two
weeks is a conservative choice: it also covers the +-6 days over which the seven alignments
share a fit.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from se4.ab.harness import ARCHIVE, FIRST_SENT_LEAD, SENT_ROWS

BLOCK_DAYS = 14            # calendar days per bootstrap block
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261115  # fixed: the same errors always give the same interval
CONFIDENCE = 0.95

# How much worse a single year, or the upper end of a removal's interval, may be before it
# counts as a cost: a share of the target's baseline MAE on the gate rows, the same share
# ab.verdict.DEAD_WEIGHT_FRACTION has used for "no measurable effect" since 2026-08.
# About 0.2 EUR/MWh on cheap2h, above the measured seed floor (0.13). PROVISIONAL: revisit
# when the first forward confirmations exist (docs/AB_TESTING.md, "Forward confirmation").
TOLERANCE_FRACTION = 0.015

YEAR_DAYS = 364            # one evaluation year, 52 x 7
GATE_ALIGNMENTS = 7        # a gate verdict needs every weekday as a week start ...
GATE_MIN_DAYS = 700        # ... and two years of distinct test days

FORECAST = "forecast"      # condition names: which weather the sent rows are scored on


def point_dates(dates, shift: int, iterations: int = 52, step: int = 7) -> np.ndarray:
    """The (iterations, step) calendar days a point at `shift` evaluates, row for row with
    the arrays of harness.run_walk_forward_errors: row j of window i."""
    days = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]")
    n = len(days) - shift
    first = n - iterations * step
    if first < 0:
        raise ValueError(f"shift {shift} leaves {n} rows, fewer than {iterations * step} test days.")
    return days[first:n].reshape(iterations, step)


def sent_errors(errors: dict, name: str, condition: str) -> np.ndarray:
    """(iterations, SENT_ROWS) absolute errors of the rows production sends (D+2..D+7), from
    one run_walk_forward_errors result. FORECAST scores row j on the forecast issued j+2
    days ahead; ARCHIVE scores the same rows on archive weather."""
    if condition == ARCHIVE:
        return errors[ARCHIVE][name][:, :SENT_ROWS]
    return np.column_stack([errors[j + FIRST_SENT_LEAD][name][:, j] for j in range(SENT_ROWS)])


def day_table(dates, errors: dict, shifts, name: str, condition: str) -> pd.DataFrame:
    """
    One row per (week alignment, calendar day): the baseline's and the candidate's absolute
    error on a sent row. `errors` is {(shift, "base" | "cand"): run_walk_forward_errors
    result}. The alignment is the weekday the test week starts on. Two points that reach
    the same (alignment, day) fitted the same model on the same rows -- the rolling window
    makes the fit a function of the week start alone -- so they are averaged into one row
    and the largest disagreement between them is kept in .attrs["max_duplicate_gap"]
    (0 unless the fits were not in fact identical).
    """
    parts = []
    for shift in shifts:
        days = point_dates(dates, shift)[:, :SENT_ROWS]
        week_start = point_dates(dates, shift)[:, :1]
        alignment = ((week_start.astype("int64") + 3) % 7).repeat(SENT_ROWS, axis=1)  # 0 = Monday
        parts.append(pd.DataFrame({
            "date": days.ravel(), "alignment": alignment.ravel(),
            "base": sent_errors(errors[(shift, "base")], name, condition).ravel(),
            "cand": sent_errors(errors[(shift, "cand")], name, condition).ravel()}))
    rows = pd.concat(parts, ignore_index=True)
    grouped = rows.groupby(["date", "alignment"], sort=True)
    table = grouped[["base", "cand"]].mean().reset_index()
    spread = grouped[["base", "cand"]].max() - grouped[["base", "cand"]].min()
    table.attrs["max_duplicate_gap"] = float(spread.to_numpy().max()) if len(spread) else 0.0
    return table


def _block_sums(table: pd.DataFrame, block_days: int):
    """Per calendar block: the sum of paired differences and the number of rows."""
    day = table["date"].values.astype("datetime64[D]").astype("int64")
    block = (day - day.min()) // block_days
    delta = (table["cand"] - table["base"]).to_numpy()
    sums = np.bincount(block, weights=delta)
    counts = np.bincount(block)
    keep = counts > 0
    return sums[keep], counts[keep]


def bootstrap_interval(table: pd.DataFrame, block_days: int = BLOCK_DAYS, draws: int = BOOTSTRAP_DRAWS,
                       confidence: float = CONFIDENCE, seed: int = BOOTSTRAP_SEED) -> tuple:
    """
    (mean, low, high) of the paired difference candidate - baseline over `table`'s rows:
    the percentile interval of a block bootstrap that resamples consecutive calendar blocks
    of `block_days` with replacement. Fewer than two blocks gives (mean, nan, nan).
    """
    sums, counts = _block_sums(table, block_days)
    mean = float(sums.sum() / counts.sum())
    if len(sums) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return mean, float(low), float(high)


def split_years(table: pd.DataFrame) -> dict:
    """
    {label: rows} by evaluation year, counted back from the newest day in 364-day spans:
    "NOW" is the most recent, then "PREV", "PREV2" ... A leftover shorter than half a year
    joins the oldest full year (the year grid leaves five such days).
    """
    day = table["date"].values.astype("datetime64[D]").astype("int64")
    index = (day.max() - day) // YEAR_DAYS
    oldest = int(index.max())
    if oldest > 0 and len(np.unique(day[index == oldest])) < YEAR_DAYS // 2:
        index = np.minimum(index, oldest - 1)
    labels = ["NOW", "PREV"] + [f"PREV{k}" for k in range(2, int(index.max()) + 1)]
    return {labels[k]: table[index == k] for k in range(int(index.max()) + 1)}


def classify_interval(low: float, high: float, year_deltas: dict, tolerance: float,
                      unchanged: bool = False) -> str:
    """
    The verdict for an ADDITION (negative = the candidate is better):

    NO_CHANGE   the candidate's errors equal the baseline's on every row.
    REAL        the whole interval is below zero and no year is worse than `tolerance`.
    MIXED       the whole interval is below zero but a year is worse than `tolerance`:
                the gain is one period paying for another. Not adopted.
    HARMFUL     the whole interval is above zero.
    UNRESOLVED  the interval covers zero: these days cannot tell the change from nothing.
    """
    if unchanged:
        return "NO_CHANGE"
    if np.isnan(low) or np.isnan(high):
        return "UNRESOLVED"
    if high < 0:
        return "MIXED" if any(d > tolerance for d in year_deltas.values()) else "REAL"
    if low > 0:
        return "HARMFUL"
    return "UNRESOLVED"


def is_non_inferior(high: float, year_deltas: dict, tolerance: float) -> bool:
    """
    The bar for a REMOVAL or a simplification: the candidate is shown to cost less than
    `tolerance` (the interval's upper end is below it) and no single year costs more.
    "Not shown to help" is not enough to keep a column, and "not shown to hurt" is not
    enough to drop one; this asks for the cost to be bounded.
    """
    if np.isnan(high):
        return False
    return high < tolerance and all(d <= tolerance for d in year_deltas.values())


@dataclass
class DayVerdict:
    name: str
    condition: str
    delta: float               # mean of candidate - baseline over (alignment, day) rows
    low: float
    high: float
    years: dict                # {label: {"delta", "first", "last", "days"}}
    days: int                  # distinct calendar days
    rows: int                  # (alignment, day) rows
    alignments: int            # distinct week alignments
    base_mae: float
    cand_mae: float
    base_sd: float             # spread of the daily absolute error, per arm
    cand_sd: float
    tolerance: float
    verdict: str
    non_inferior: bool
    gate: bool                 # two years of days under all seven alignments
    max_duplicate_gap: float
    notes: list = field(default_factory=list)


def day_verdict(dates, errors: dict, shifts, name: str, condition: str, since=None) -> DayVerdict:
    """
    The interval verdict for one target. `condition` is FORECAST (the gate, when the run
    scored forecast vintages) or harness.ARCHIVE. `since` keeps only test days on or after
    that date -- the forward confirmation of a change adopted then.
    """
    table = day_table(dates, errors, shifts, name, condition)
    gap = table.attrs["max_duplicate_gap"]
    notes = []
    if since is not None:
        table = table[table["date"] >= np.datetime64(pd.Timestamp(since).date())]
        notes.append(f"test days from {pd.Timestamp(since).date()} only")
        if table.empty:
            raise ValueError(f"no test day on or after {since} in this run.")
    if gap > 1e-6:
        notes.append(f"points sharing a week alignment and a day disagree by up to {gap:.3g}: "
                     f"their fits are not identical (a transform that reads the frame's end?)")

    delta, low, high = bootstrap_interval(table)
    years = {}
    for label, rows in split_years(table).items():
        years[label] = {"delta": float((rows["cand"] - rows["base"]).mean()),
                        "first": str(rows["date"].min())[:10], "last": str(rows["date"].max())[:10],
                        "days": int(rows["date"].nunique())}
    year_deltas = {label: y["delta"] for label, y in years.items()}
    base_mae = float(table["base"].mean())
    tolerance = TOLERANCE_FRACTION * base_mae
    unchanged = bool((table["cand"].to_numpy() == table["base"].to_numpy()).all())
    days, alignments = int(table["date"].nunique()), int(table["alignment"].nunique())
    gate = (since is None and alignments >= GATE_ALIGNMENTS and days >= GATE_MIN_DAYS
            and len(years) >= 2)
    if not gate and since is None:
        notes.append(f"{days} days under {alignments} of 7 week alignments: a look, not a gate")
    return DayVerdict(
        name=name, condition=condition, delta=delta, low=low, high=high, years=years,
        days=days, rows=len(table), alignments=alignments,
        base_mae=base_mae, cand_mae=float(table["cand"].mean()),
        base_sd=float(table["base"].std(ddof=1)), cand_sd=float(table["cand"].std(ddof=1)),
        tolerance=tolerance,
        verdict=classify_interval(low, high, year_deltas, tolerance, unchanged=unchanged),
        non_inferior=(not unchanged) and is_non_inferior(high, year_deltas, tolerance),
        gate=gate, max_duplicate_gap=gap, notes=notes)


def format_verdict(v: DayVerdict, marker: str = "") -> list:
    """The report lines for one DayVerdict."""
    years = "  ".join(f"{label} {y['delta']:+.3f}" for label, y in v.years.items())
    head = (f"  {v.name:<8} delta {v.delta:+.3f}  [{v.low:+.3f}, {v.high:+.3f}]  "
            f"years: {years}  -> {v.verdict}{marker}")
    detail = (f"           base {v.base_mae:.2f}  cand {v.cand_mae:.2f}   tolerance {v.tolerance:+.2f}"
              f"   removal/fix bar (cost bounded below tolerance): {'PASS' if v.non_inferior else 'no'}"
              f"   sd of daily error {v.base_sd:.2f} -> {v.cand_sd:.2f}")
    return [head, detail] + [f"           NOTE: {note}" for note in v.notes]
