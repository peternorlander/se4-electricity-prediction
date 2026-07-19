import numpy as np
import pandas as pd

from features import apply_forecast_freeze
from model import _fit_models, TARGETS


# Target number of walk-forward iterations for evaluation.
# The minimum training window is derived dynamically as n - EVAL_ITERATIONS * step,
# so the window expands each iteration and the final iteration trains on nearly
# all available data — closely matching what the production model uses.
_EVAL_ITERATIONS = 35

# Targets whose feature importance is reported. Max is excluded since
# prediction accuracy for max prices is not a priority.
_IMPORTANCE_TARGETS = ("min", "avg", "cheap2h")


def walk_forward_validate(training_data: pd.DataFrame, step: int = 7) -> dict:
    """
    Walk-forward validation with an expanding training window.

    Runs exactly _EVAL_ITERATIONS test windows. The first iteration uses a
    minimum training window (n - _EVAL_ITERATIONS * step days), and each
    subsequent iteration adds `step` more days to the training set. The final
    iteration trains on nearly all available data, closely matching the
    production model.

    This spreads evaluation across ~(EVAL_ITERATIONS * step) days of history,
    diluting the effect of any single unusual market period (e.g. geopolitical
    events) while progressively converging toward production model performance.

    Horizon-honest: each test window's price/market/fuel/reservoir lags are
    frozen to their last-known value via apply_forecast_freeze(), exactly as
    build_forecast_features() does in production. Without this the test rows
    carry each day's true previous-day price, which the live model only has for
    day+1 — so days 2–8 would look far more accurate than they really are. The
    headline number is therefore higher than a naive walk-forward, but it is the
    accuracy the Home Assistant automation actually receives.

    Weather features still come from the historical archive (a stand-in for the
    real forecast), so results remain slightly optimistic on the weather side —
    archive data is more accurate than a genuine multi-day forecast.

    Each target is reported individually — min, avg, max and cheap2h have
    different physical drivers, so a blended metric would hide the per-target
    movement that matters when tuning. A per-horizon (day+1 … day+N) breakdown
    is also reported, since the whole point of the freeze is to expose how
    accuracy decays with forecast distance.

    Args:
        training_data: Full cleaned daily DataFrame from build_training_data().
        step:          Number of days per test window (default 7, matching forecast horizon).

    Returns:
        Dict with mean and std of MAE for each target (EUR/MWh), plus a nested
        "mae_by_horizon" mapping target → {horizon: MAE}.
    """
    data = training_data.reset_index(drop=True)
    n = len(data)
    min_train = n - _EVAL_ITERATIONS * step

    if min_train < 1:
        raise ValueError(
            f"Not enough data for {_EVAL_ITERATIONS} iterations: "
            f"need at least {_EVAL_ITERATIONS * step + 1} days, got {n}."
        )

    # Per-window mean MAE per target.
    mae_lists = {name: [] for name in TARGETS}
    # Per-horizon absolute errors: target → horizon (1..step) → list of errors,
    # one per window.
    horizon_errs = {name: {h: [] for h in range(1, step + 1)} for name in TARGETS}

    print("\n=== Walk-forward validation ===")

    for iteration in range(_EVAL_ITERATIONS):
        train_end  = min_train + iteration * step
        test_end   = train_end + step

        train_slice = data.iloc[:train_end]
        raw_test    = data.iloc[train_end:test_end]
        models      = _fit_models(train_slice)

        frozen = apply_forecast_freeze(raw_test, train_slice)

        for name, (target_col, feature_cols) in TARGETS.items():
            actual = raw_test[target_col].values
            errors = np.abs(models[name].predict(frozen[feature_cols].values) - actual)
            mae_lists[name].append(float(errors.mean()))
            for i, err in enumerate(errors):
                horizon_errs[name][i + 1].append(float(err))

        start_date = pd.to_datetime(raw_test["date"].iloc[0]).strftime("%Y-%m-%d")
        end_date   = pd.to_datetime(raw_test["date"].iloc[-1]).strftime("%Y-%m-%d")
        print(f"  Iteration {iteration + 1:>2} ({start_date} – {end_date}):  "
              f"MAE min={mae_lists['min'][-1]:.2f}  avg={mae_lists['avg'][-1]:.2f}  "
              f"max={mae_lists['max'][-1]:.2f}  cheap2h={mae_lists['cheap2h'][-1]:.2f}"
              f"  [train={train_end} days]")

    mae_arrays = {name: np.array(values) for name, values in mae_lists.items()}
    horizon_mae = {
        name: {h: float(np.mean(errs)) for h, errs in per_h.items()}
        for name, per_h in horizon_errs.items()
    }

    print("\n  Mean MAE across windows (EUR/MWh):")
    for name in TARGETS:
        arr = mae_arrays[name]
        print(f"    {name:<8} {arr.mean():.2f} ± {arr.std():.2f}")

    print("\n  MAE by forecast horizon (EUR/MWh):")
    print("    horizon " + "".join(f"{name:>9}" for name in TARGETS))
    for h in range(1, step + 1):
        print(f"      d+{h:<4} " + "".join(f"{horizon_mae[name][h]:>9.2f}" for name in TARGETS))

    metrics = {
        f"mae_{name}": {
            "value": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
        }
        for name, arr in mae_arrays.items()
    }
    metrics["mae_by_horizon"] = {
        name: {str(h): round(mae, 4) for h, mae in per_h.items()}
        for name, per_h in horizon_mae.items()
    }
    return metrics


def get_feature_importance(models: dict) -> dict:
    """
    Compute feature importance for the min, avg and cheap2h models.

    Max model is excluded since prediction accuracy for max prices is not
    a priority. Each value represents the feature's relative contribution to
    prediction accuracy — higher means the model relies more heavily on that
    feature.

    Args:
        models: Dict of fitted models keyed by target name.

    Returns:
        Dict keyed by target name, each a {feature_name: importance}
        dict sorted by importance descending.
    """
    def _importance(model, feature_cols) -> dict:
        importance = {
            feature: round(float(imp), 4)
            for feature, imp in zip(feature_cols, model.feature_importances_)
        }
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return {
        name: _importance(models[name], TARGETS[name][1])
        for name in _IMPORTANCE_TARGETS
    }
