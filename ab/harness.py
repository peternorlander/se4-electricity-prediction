"""
Walk-forward loop for the A/B backtest flow. See the README "A/B Backtest
Flow" section.

Deliberately does NOT import or refactor evaluate.walk_forward_validate --
the production headline eval stays untouched so every recorded baseline
number (IMPROVEMENT_PLAN.md) stays comparable. This module carries its own
copy of the same ~25-line loop, parameterized by a Variant's fit_fn/targets/
freeze-lists so BASELINE and CANDIDATE run through identical slices. The two
are pinned together by an acceptance test: at shift=0 with the BASELINE
variant, run_walk_forward() must reproduce evaluate.walk_forward_validate()'s
per-window MAE exactly (same data, same seeds -> bit-identical fits).
"""
import numpy as np
import pandas as pd

from features import apply_forecast_freeze


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
    captures the whole between-day difference (see the README "A/B Backtest
    Flow" section, "Key insight").
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
) -> dict:
    """
    Walk-forward MAE per target, mirroring evaluate.walk_forward_validate's
    loop (expanding window, horizon-honest freeze via apply_forecast_freeze)
    but parameterized by a variant's fit_fn/targets/freeze-list overrides.

    Args:
        data:              Full merged daily frame, already shifted and
                           transformed by the caller. Must be date-sorted;
                           reset_index(drop=True) is applied here.
        fit_fn:            train_slice -> {target_name: fitted model}.
        targets:           {target_name: (target_col, feature_cols)}.
        step:              Days per test window. Must stay 7 to match the
                           headline eval / production forecast horizon.
        iterations:        Number of windows. Must stay 52 to match the
                           headline eval (a full year).
        frozen_features:   Passed through to apply_forecast_freeze (None =
                           production FORECAST_FROZEN_FEATURES).
        frozen_rolling:    Passed through to apply_forecast_freeze (None =
                           production _FORECAST_FROZEN_ROLLING).

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

    mae_lists = {name: [] for name in targets}

    for iteration in range(iterations):
        train_end = min_train + iteration * step
        test_end = train_end + step

        train_slice = data.iloc[:train_end]
        raw_test = data.iloc[train_end:test_end]
        models = fit_fn(train_slice)

        frozen = apply_forecast_freeze(
            raw_test, train_slice,
            frozen_features=frozen_features, frozen_rolling=frozen_rolling,
        )

        for name, (target_col, feature_cols) in targets.items():
            actual = raw_test[target_col].values
            errors = np.abs(models[name].predict(frozen[feature_cols].values) - actual)
            mae_lists[name].append(float(errors.mean()))

    return {name: np.array(values) for name, values in mae_lists.items()}
