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
"""
import numpy as np
import pandas as pd

from features import apply_forecast_freeze
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
    each test week trains on (`train_window`).

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

    mae_lists = {name: [] for name in targets}

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

        frozen = apply_forecast_freeze(
            raw_test, train_slice,
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
            errors = np.abs(preds[name] - actual)
            mae_lists[name].append(float(errors.mean()))

    return {name: np.array(values) for name, values in mae_lists.items()}
