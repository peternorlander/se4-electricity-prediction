"""
Walk-forward loop for the A/B backtest flow. See docs/AB_TESTING.md.

Deliberately does NOT import or refactor evaluate.walk_forward_validate --
the production headline eval stays untouched so every recorded baseline
number stays comparable. This module carries its own
copy of the same ~25-line loop, parameterized by a Variant's fit_fn/targets/
freeze-lists so BASELINE and CANDIDATE run through identical slices. The two
are pinned together by an acceptance test: at shift=0 with the BASELINE
variant, run_walk_forward() must reproduce evaluate.walk_forward_validate()'s
per-window MAE exactly (same data, same seeds -> bit-identical fits).

`train_window` is the one deliberate divergence, and it defaults to None so
that acceptance test still holds. None reproduces the headline eval's expanding
geometry (731 -> 1088 rows across the year); an int makes every test week train
on the same number of rows production fits on. The headline eval cannot do the
latter -- production fetches three years, which is exactly one evaluation year
short of being able to -- so it stays expanding and the A/B grid does not.

The loop itself lives in run_walk_forward_errors, which keeps every test row's
absolute error instead of only the window mean, and can score the SAME fits a
second way: with the test rows' weather swapped for forecast vintages
(ab/vintage.py). run_walk_forward is a thin wrapper that returns the
archive-weather window means, computed from the same arrays with the same
operations as the loop it replaced, so its numbers are bit-identical
(ab/check_vintage.py asserts it).
"""
import numpy as np
import pandas as pd

from features import apply_forecast_freeze, FORECAST_FROZEN_FEATURES
from model import enforce_coherence


# Production's own training-set size, in merged rows. predict.py fetches
# fetch_data.TRAINING_DAYS (1095) calendar days, but the first LAG_WARMUP_ROWS of
# those cannot have their 7-day lags and rolling means computed, so
# build_training_data drops them: the frame production actually fits on is 1088
# rows (measured on ab_cache/2026-09-11, 1088 rows spanning 2023-09-19 ->
# 2026-09-10). This is the number a production-faithful measurement has to match,
# and it is what `train_window` is normally set to. Kept as literals here rather
# than imported from fetch_data (which pulls in the source modules and requests);
# ab_test.py asserts the two stay in step.
LAG_WARMUP_ROWS = 7
PRODUCTION_TRAIN_ROWS = 1088

# Lead mapping for forecast-vintage scoring. Production runs once and sends D+2..D+7
# (D+8 is no longer sent), anchoring its frozen price lags on D+1 (docs/DECISIONS.md,
# "Price anchor on D+1"). Eval row j (0-based) of a test week is therefore the sent day
# D+(j+2) and reads the forecast issued j+2 days ahead; rows 0..5 are what Home
# Assistant receives, row 6 is not sent and is left out of every production-mapped total.
ARCHIVE = "archive"
LEADS = tuple(range(1, 8))
FIRST_SENT_LEAD = 2
SENT_ROWS = 6


def parse_shift_range(spec: str) -> list:
    """
    Parse a shift spec into a sorted list of non-negative ints.

    Accepts 'lo-hi' (inclusive range), a comma list '0,2,4', or a single
    value '3'.
    """
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(x) for x in spec.split(","))


def apply_shift(data: pd.DataFrame, shift: int) -> pd.DataFrame:
    """
    Truncate the last `shift` rows of the date-sorted merged frame.

    walk_forward_validate derives its window grid backwards from the END of
    the data (min_train = n - iterations*step), so truncating the tail by
    `shift` days reproduces the eval exactly as it would have run `shift`
    days earlier -- window placement AND training tail move together, the
    same way a real rerun on an earlier day would. This single "shift" axis
    captures the whole between-day difference (see docs/AB_TESTING.md,
    "How it works").
    """
    if shift <= 0:
        return data
    return data.iloc[: len(data) - shift].reset_index(drop=True)


def column_equal(a: pd.Series, b: pd.Series) -> bool:
    """Exact equality of two columns, NaN == NaN (non-numeric compared as text)."""
    if pd.api.types.is_numeric_dtype(a):
        return bool(np.array_equal(a.values, b.values, equal_nan=True))
    return bool((a.astype(str).values == b.astype(str).values).all())


def swap_columns(archive: pd.DataFrame, vintage: pd.DataFrame, frozen) -> list:
    """
    The columns a test row takes from a vintage frame: every column that differs
    from the archive frame, except the frozen ones. Whatever differs IS
    weather-derived (the frames are built from the same inputs but for the weather
    columns), so no list of weather columns is needed. A frozen column that differs
    (residual_load_lag1, the rolling variability signals) stays at its archive
    value on purpose: production freezes those from data it already has, and
    apply_forecast_freeze does the same to the test rows.
    """
    frozen = set(frozen)
    return [c for c in archive.columns
            if c != "date" and c not in frozen and not column_equal(archive[c], vintage[c])]


def run_walk_forward_errors(
    data: pd.DataFrame,
    fit_fn,
    targets: dict,
    step: int = 7,
    iterations: int = 52,
    frozen_features=None,
    frozen_rolling=None,
    train_window=None,
    vintage_frames=None,
) -> dict:
    """
    The walk-forward loop, returning every test row's absolute error. Same
    arguments and semantics as run_walk_forward (documented there), plus:

        vintage_frames: None, or {lead: frame} -- the merged frame built on weather
                        carrying the forecast issued `lead` days before each hour
                        (ab.vintage.VintageSet.frames), row-for-row aligned with
                        `data` and already shifted/transformed the same way. The
                        model is fitted ONCE per window on archive weather; the
                        window's test rows are then scored once on archive weather
                        and once per lead, with the columns that differ between the
                        frames (minus the frozen ones, see swap_columns) taken from
                        that lead's frame. Fits, freeze and coherence are identical
                        across conditions, so the conditions differ in weather only.

    Returns:
        {condition: {target_name: np.ndarray (iterations, step) of |pred - actual|}}
        with condition ARCHIVE ("archive") and, if vintages were given, each lead
        as an int. Row j of a window is the j-th day after the training tail.
    """
    data = data.reset_index(drop=True)
    n = len(data)
    min_train = n - iterations * step

    if min_train < 1:
        raise ValueError(
            f"Not enough data for {iterations} iterations at step {step}: "
            f"need at least {iterations * step + 1} days, got {n}."
        )

    if train_window is not None:
        if train_window < 1:
            raise ValueError(f"train_window must be >= 1 row, got {train_window}.")
        if min_train < train_window:
            raise ValueError(
                f"A rolling training window of {train_window} rows needs at least "
                f"{train_window + iterations * step} rows in the frame, got {n}. A "
                f"normal ~3-year snapshot builds to ~{PRODUCTION_TRAIN_ROWS} rows and "
                f"cannot run a production-faithful grid at all -- fetch a long "
                f"snapshot (python ab_test.py fetch) and shift within it."
            )

    swaps = {}
    if vintage_frames:
        if step != 7:
            raise ValueError("forecast-vintage scoring maps test row j to lead j+2 and needs step == 7.")
        frozen_names = FORECAST_FROZEN_FEATURES if frozen_features is None else frozen_features
        vintage_frames = {lead: vf.reset_index(drop=True) for lead, vf in vintage_frames.items()}
        for lead, vf in vintage_frames.items():
            if len(vf) != n or not (vf["date"].values == data["date"].values).all():
                raise ValueError(f"vintage frame for lead {lead} is not row-aligned with the archive frame "
                                 f"({len(vf)} vs {n} rows) -- shift and transform it exactly as `data`.")
            swaps[lead] = swap_columns(data, vf, frozen_names)

    conditions = [ARCHIVE] + list(swaps)
    errors = {cond: {name: [] for name in targets} for cond in conditions}

    for iteration in range(iterations):
        train_end = min_train + iteration * step
        test_end = train_end + step

        train_start = 0 if train_window is None else train_end - train_window
        train_slice = data.iloc[train_start:train_end]
        raw_test = data.iloc[train_end:test_end]
        # Fit over the SAME `targets` used for prediction below, so a variant that
        # changes feature_cols (e.g. an ablation) trains and predicts on identical
        # feature lists. model._fit_models (the default fit_fn) accepts `targets`;
        # any custom fit_fn must too (signature: fit_fn(train_slice, targets)).
        models = fit_fn(train_slice, targets=targets)

        for cond in conditions:
            if cond == ARCHIVE:
                rows = raw_test
            else:
                rows = raw_test.copy()
                if swaps[cond]:
                    rows[swaps[cond]] = vintage_frames[cond].iloc[train_end:test_end][swaps[cond]].values
            frozen = apply_forecast_freeze(
                rows, train_slice,
                frozen_features=frozen_features, frozen_rolling=frozen_rolling,
            )

            # Same coherence post-processing as evaluate.walk_forward_validate and model.predict
            # (a no-op unless the variant predicts both min and cheap2h).
            preds = enforce_coherence({
                name: models[name].predict(frozen[feature_cols].values)
                for name, (_, feature_cols) in targets.items()
            })
            for name, (target_col, feature_cols) in targets.items():
                actual = raw_test[target_col].values
                errors[cond][name].append(np.abs(preds[name] - actual))

    return {cond: {name: np.array(rows) for name, rows in per_target.items()}
            for cond, per_target in errors.items()}


def window_means(row_errors: np.ndarray) -> np.ndarray:
    """Per-window MAE from an (iterations, step) error array: one 1-D mean per window,
    the operation the loop this replaced applied, so the result is bit-identical."""
    return np.array([float(row.mean()) for row in row_errors])


def run_walk_forward(
    data: pd.DataFrame,
    fit_fn,
    targets: dict,
    step: int = 7,
    iterations: int = 52,
    frozen_features=None,
    frozen_rolling=None,
    train_window=None,
) -> dict:
    """
    Walk-forward MAE per target, mirroring evaluate.walk_forward_validate's
    loop (horizon-honest freeze via apply_forecast_freeze) but parameterized by
    a variant's fit_fn/targets/freeze-list overrides, and by how much history
    each test week trains on (`train_window`). Archive weather only; for the same
    fits scored on forecast vintages too, use run_walk_forward_errors.

    Args:
        data:              Full merged daily frame, already shifted and
                           transformed by the caller. Must be date-sorted;
                           reset_index(drop=True) is applied here.
        fit_fn:            fit_fn(train_slice, targets) -> {target_name: fitted
                           model}. Receives the same `targets` used for prediction
                           so fitting and prediction share feature lists (see the
                           fit call below). model._fit_models is the default.
        targets:           {target_name: (target_col, feature_cols)}. Drives both
                           fitting (passed to fit_fn) and prediction.
        step:              Days per test window. Must stay 7 to match the
                           headline eval / production forecast horizon.
        iterations:        Number of windows. Must stay 52 to match the
                           headline eval (a full year).
        frozen_features:   Passed through to apply_forecast_freeze (None =
                           production FORECAST_FROZEN_FEATURES).
        frozen_rolling:    Passed through to apply_forecast_freeze (None =
                           production _FORECAST_FROZEN_ROLLING).
        train_window:      How much history each test week trains on.
                           None = EXPANDING, byte-for-byte what
                           evaluate.walk_forward_validate does: week i trains on
                           every row before it, so on a canonical 1095-row frame
                           the first week sees 731 rows and the last 1088. This
                           is the default so the acceptance test against the
                           headline eval keeps passing.
                           An int = ROLLING window of that many rows, so EVERY
                           week trains on as much history as production has
                           (PRODUCTION_TRAIN_ROWS) instead of only the last one.
                           The rolling form needs train_window + iterations*step
                           rows, which a normal ~3-year snapshot does not have;
                           use a long snapshot. Why this matters, and what it
                           changed when measured, is in docs/AB_TESTING.md
                           ("Measurement grids").

    Returns:
        {target_name: np.ndarray of per-window MAE, length `iterations`}
    """
    errors = run_walk_forward_errors(
        data, fit_fn, targets, step=step, iterations=iterations,
        frozen_features=frozen_features, frozen_rolling=frozen_rolling,
        train_window=train_window,
    )[ARCHIVE]
    return {name: window_means(rows) for name, rows in errors.items()}


def summarize_errors(errors: dict, name: str) -> dict:
    """
    The numbers a vintage-scored run reports for one target, from a
    run_walk_forward_errors result. All EUR/MWh, means over windows.

        archive          MAE on archive weather over ALL test rows -- the number
                         run_walk_forward and every ledger entry quote
        archive_sent     archive MAE over the SENT rows 0..5 only: the like-for-like
                         base of the penalty, because the forecast number below
                         leaves row 6 out
        archive_by_row   archive MAE per test row j (length step)
        forecast         production-mapped MAE on forecast vintages: rows 0..5, row j
                         scored on the forecast issued j+2 days ahead (D+2..D+7)
        forecast_by_row  that, per sent row j (length SENT_ROWS)
        penalty          forecast - archive_sent: what production pays that the
                         archive eval does not
        by_lead          {L: MAE of ALL rows scored on the forecast issued L days ahead}

    Without vintages only archive, archive_sent and archive_by_row are present.
    """
    archive = errors[ARCHIVE][name]
    out = {
        "archive": float(np.mean(window_means(archive))),
        "archive_sent": float(archive[:, :SENT_ROWS].mean()),
        "archive_by_row": archive.mean(axis=0),
    }
    leads = [c for c in errors if c != ARCHIVE]
    if leads:
        out["by_lead"] = {lead: float(errors[lead][name].mean()) for lead in sorted(leads)}
        out["forecast_by_row"] = np.array(
            [errors[j + FIRST_SENT_LEAD][name][:, j].mean() for j in range(SENT_ROWS)])
        out["forecast"] = float(out["forecast_by_row"].mean())
        out["penalty"] = out["forecast"] - out["archive_sent"]
    return out
