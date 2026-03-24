import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from features import FEATURE_COLUMNS
from model import _fit_models


# Target number of walk-forward iterations for evaluation.
# The minimum training window is derived dynamically as n - EVAL_ITERATIONS * step,
# so the window expands each iteration and the final iteration trains on nearly
# all available data — closely matching what the production model uses.
_EVAL_ITERATIONS = 35


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

    Uses historical archive weather as a stand-in for forecast weather — this
    means performance will be slightly optimistic compared to live predictions,
    since actual archive data is more accurate than a real forecast would be.

    Args:
        training_data: Full cleaned daily DataFrame from build_training_data().
        step:          Number of days per test window (default 7, matching forecast horizon).

    Returns:
        Dict with mean and std of MAE for min, avg, max and overall (EUR/MWh).
    """
    data = training_data.reset_index(drop=True)
    n = len(data)
    min_train = n - _EVAL_ITERATIONS * step

    if min_train < 1:
        raise ValueError(
            f"Not enough data for {_EVAL_ITERATIONS} iterations: "
            f"need at least {_EVAL_ITERATIONS * step + 1} days, got {n}."
        )

    mae_min_list, mae_avg_list, mae_max_list = [], [], []

    print("\n=== Walk-forward validation ===")

    for iteration in range(_EVAL_ITERATIONS):
        train_end  = min_train + iteration * step
        test_end   = train_end + step

        train_slice = data.iloc[:train_end]
        test_slice  = data.iloc[train_end:test_end]

        model_min, model_avg, model_max = _fit_models(train_slice)
        X_test = test_slice[FEATURE_COLUMNS].values

        mae_min = mean_absolute_error(test_slice["price_min"].values, model_min.predict(X_test))
        mae_avg = mean_absolute_error(test_slice["price_avg"].values, model_avg.predict(X_test))
        mae_max = mean_absolute_error(test_slice["price_max"].values, model_max.predict(X_test))

        mae_min_list.append(mae_min)
        mae_avg_list.append(mae_avg)
        mae_max_list.append(mae_max)

        start_date = pd.to_datetime(test_slice["date"].iloc[0]).strftime("%Y-%m-%d")
        end_date   = pd.to_datetime(test_slice["date"].iloc[-1]).strftime("%Y-%m-%d")
        print(f"  Iteration {iteration + 1:>2} ({start_date} – {end_date}):  "
              f"MAE min={mae_min:.2f}  avg={mae_avg:.2f}  max={mae_max:.2f}"
              f"  [train={train_end} days]")

    mae_min_arr = np.array(mae_min_list)
    mae_avg_arr = np.array(mae_avg_list)
    mae_max_arr = np.array(mae_max_list)
    mae_overall_arr = (mae_min_arr + mae_avg_arr + mae_max_arr) / 3

    print(f"\n  Overall MAE:  "
          f"min={mae_min_arr.mean():.2f} ± {mae_min_arr.std():.2f}  "
          f"avg={mae_avg_arr.mean():.2f} ± {mae_avg_arr.std():.2f}  "
          f"max={mae_max_arr.mean():.2f} ± {mae_max_arr.std():.2f}  "
          f"overall={mae_overall_arr.mean():.2f} ± {mae_overall_arr.std():.2f}")

    return {
        "mae_min":     {"value": round(float(mae_min_arr.mean()), 4), "std": round(float(mae_min_arr.std()), 4)},
        "mae_avg":     {"value": round(float(mae_avg_arr.mean()), 4), "std": round(float(mae_avg_arr.std()), 4)},
        "mae_max":     {"value": round(float(mae_max_arr.mean()), 4), "std": round(float(mae_max_arr.std()), 4)},
        "mae_overall": {"value": round(float(mae_overall_arr.mean()), 4), "std": round(float(mae_overall_arr.std()), 4)},
    }


def get_feature_importance(models: tuple) -> dict:
    """
    Compute average feature importance across the three models.

    Importances are averaged over min/avg/max models and sorted descending.
    Each value represents the feature's relative contribution to prediction
    accuracy — higher means the model relies more heavily on that feature.

    Args:
        models: Tuple of (model_min, model_avg, model_max).

    Returns:
        Dict of {feature_name: importance} sorted by importance descending.
    """
    model_min, model_avg, model_max = models
    avg_importance = (
        model_min.feature_importances_ +
        model_avg.feature_importances_ +
        model_max.feature_importances_
    ) / 3

    importance = {
        feature: round(float(imp), 4)
        for feature, imp in zip(FEATURE_COLUMNS, avg_importance)
    }
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
