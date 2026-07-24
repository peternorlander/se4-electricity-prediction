"""
Verdict classification and orchestration for the A/B backtest flow.
See the README "A/B Backtest Flow" section.
"""
import numpy as np
import pandas as pd

import model as model_module
from features import build_training_data
from ab.harness import run_walk_forward, apply_shift
from ab.variants import BASELINE, CANDIDATE

DEFAULT_TARGET_NAMES = tuple(model_module.TARGETS.keys())
PRIORITY_TARGETS = ("cheap2h", "min")
# Print/report order: priority targets first.
_REPORT_ORDER = ("cheap2h", "min", "avg", "max")


def _apply_variant(data: pd.DataFrame, variant) -> pd.DataFrame:
    transformed = variant.transform(data)
    if len(transformed) != len(data):
        raise ValueError(
            f"Variant '{variant.name}'.transform changed row count "
            f"({len(data)} -> {len(transformed)}); transforms may only add/modify columns."
        )
    if not (pd.Series(transformed["date"].values) == pd.Series(data["date"].values)).all():
        raise ValueError(f"Variant '{variant.name}'.transform reordered or changed rows.")
    return transformed


def _mean_mae(data: pd.DataFrame, variant, shift: int) -> dict:
    shifted = apply_shift(data, shift)
    transformed = _apply_variant(shifted, variant)
    per_window = run_walk_forward(
        transformed, variant.fit_fn, variant.targets,
        frozen_features=variant.frozen_features, frozen_rolling=variant.frozen_rolling,
    )
    return {name: float(np.mean(arr)) for name, arr in per_window.items()}


def classify(deltas: list) -> str:
    """
    REAL / NOISE / BORDERLINE / NO_CHANGE per the established replication
    rule (IMPROVEMENT_PLAN.md finding #4, method lesson under step 5):

    - NO_CHANGE: every delta is exactly zero (candidate == baseline).
    - NOISE:     the sign of the delta flips across shifts -- smaller than
                 the between-shift (between-simulated-day) variability.
    - REAL:      sign-consistent across all shifts AND |mean delta| >= the
                 between-shift spread (max - min).
    - BORDERLINE: sign-consistent but the effect is smaller than the spread
                 -- replay on a different-day snapshot before adopting.

    Args:
        deltas: candidate_MAE - baseline_MAE, one per shift (negative =
                candidate better).
    """
    deltas = np.array(deltas, dtype=float)
    if len(deltas) < 2:
        return "INSUFFICIENT"

    mean_delta = float(np.mean(deltas))
    spread = float(np.max(deltas) - np.min(deltas))

    if np.all(deltas == 0):
        return "NO_CHANGE"

    signs = np.sign(deltas)
    nonzero_signs = signs[signs != 0]
    if len(nonzero_signs) > 0 and not np.all(nonzero_signs == nonzero_signs[0]):
        return "NOISE"

    return "REAL" if abs(mean_delta) >= spread else "BORDERLINE"


def run_ab(inputs: dict, shifts: list, target_names=DEFAULT_TARGET_NAMES) -> dict:
    """
    Build training data once from a snapshot, then run BASELINE and
    CANDIDATE through run_walk_forward at each shift, printing a verdict
    table classified per the replication rule above.

    Args:
        inputs:       fetch_training_inputs()-shaped dict (from a snapshot).
        shifts:       List of shift values (see ab.harness.apply_shift).
        target_names: Which targets to report (default: all of model.TARGETS).

    Returns:
        {target_name: {"deltas": [...], "mean_delta": float, "spread": float,
                        "verdict": str}}
    """
    print("Building training data from snapshot...")
    data = build_training_data(**inputs)
    print(f"  -> {len(data)} days of merged data\n")

    base_by_shift = {}
    cand_by_shift = {}

    for shift in shifts:
        print(f"--- shift={shift} ---")
        base_by_shift[shift] = _mean_mae(data, BASELINE, shift)
        cand_by_shift[shift] = _mean_mae(data, CANDIDATE, shift)
        line = "  ".join(
            f"{name}: base={base_by_shift[shift][name]:.2f} cand={cand_by_shift[shift][name]:.2f} "
            f"delta={cand_by_shift[shift][name] - base_by_shift[shift][name]:+.2f}"
            for name in target_names
        )
        print(f"  {line}")

    print("\n=== Verdict (priority: cheap2h, then min) ===")
    results = {}
    report_order = [n for n in _REPORT_ORDER if n in target_names] + \
                   [n for n in target_names if n not in _REPORT_ORDER]
    for name in report_order:
        deltas = [cand_by_shift[s][name] - base_by_shift[s][name] for s in shifts]
        verdict = classify(deltas)
        mean_delta = float(np.mean(deltas))
        spread = float(np.max(deltas) - np.min(deltas))
        results[name] = {"deltas": deltas, "mean_delta": mean_delta, "spread": spread, "verdict": verdict}
        marker = " <-- priority" if name in PRIORITY_TARGETS else ""
        print(f"  {name:<8} mean_delta={mean_delta:+.3f}  spread={spread:.3f}  -> {verdict}{marker}")

    return results
