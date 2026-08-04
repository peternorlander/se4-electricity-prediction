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


# --- Ablation verdicts (candidate = feature REMOVED) -------------------------
#
# WHEN TO USE THIS (read before importing): only when CANDIDATE actually REMOVES
# a column -- i.e. a periodic feature-set audit, not routine A/B work. The
# everyday case (testing a new feature/source/model change) is an ADDITION, and
# classify() above is correct for that; this module is the deliberate exception.
# It is NOT wired into run_ab()/ab_test.py run, which always uses classify() --
# call classify_ablation() yourself on the per-shift deltas when you're
# specifically testing a removal (see README "How changes are validated" for the
# full rationale and FEATURE_REVALIDATION_PLAN.md / experiments/ for a worked
# example of the audit this was built for).
#
# classify() above answers "should we ADOPT this change?", where NOISE means
# "unproven -> don't change -> keep the status quo". For an ADDITION that is
# right: rejecting leaves the simpler model, and the burden of proof correctly
# falls on the new feature.
#
# For an ABLATION the same rule inverts: "don't change" means "keep the
# feature", so a genuinely worthless feature -- whose ~zero effect flips sign
# purely from fitting jitter -- is classified NOISE and kept forever. The rule
# can never retire dead weight.
#
# The missing dimension is MAGNITUDE, which classify() ignores entirely. A
# sign-flipping delta is ambiguous between two opposite situations:
#   (a) the effect is ~0 everywhere (dead weight)          -> remove
#   (b) the effect is LARGE but regime-dependent, helping   -> keep
#       in some periods and hurting in others
# Measured example of (b): dropping `price_momentum` from `max` has mean
# -0.017 (looks like nothing) but swings to 1.588 at one shift -- averaging to
# nothing is not the same as doing nothing.
#
# Thresholds are expressed as a FRACTION of the target's baseline MAE by
# default, because a fixed EUR/MWh cut-off is not comparable across targets:
# 0.25 EUR is ~1.6% of min's ~16 baseline but only ~0.7% of max's ~35 (with
# std ~19), which is why an absolute threshold finds "no dead weight" in max
# purely as an artefact of its larger scale.
DEAD_WEIGHT_FRACTION = 0.015   # |delta| below this share of baseline MAE = no measurable effect
SCENARIO_SWING_FRACTION = 0.035  # |delta| above this share somewhere = large regime-dependent effect


def classify_ablation(deltas: list, baseline_mae: float = None, replay_delta: float = None,
                       dead_threshold: float = None, swing_threshold: float = None) -> str:
    """
    Verdict for an ABLATION A/B, where the candidate has the feature(s) REMOVED
    and so a negative delta means removal LOWERED MAE.

    Sign-consistency is still the primary signal (it is the strongest evidence
    available); magnitude only arbitrates the sign-flipping cases that
    classify() lumps together as NOISE. See the module comment above.

    Args:
        deltas:          candidate(removed) - baseline, one per shift.
        baseline_mae:    that target's baseline MAE, used to scale the
                         thresholds (see DEAD_WEIGHT_FRACTION). Optional if
                         explicit thresholds are given.
        replay_delta:    optional delta from a different-snapshot replay of the
                         same window. Used as a confirmation gate: a replay that
                         meaningfully contradicts a sign-consistent verdict
                         downgrades it to INCONCLUSIVE, and dead weight must be
                         negligible on the replay too.
        dead_threshold:  absolute override for the dead-weight cut-off.
        swing_threshold: absolute override for the scenario-swing cut-off.

    Returns:
        NOT_APPLICABLE    - every delta exactly 0 (feature is not in this
                            target's feature list at all).
        REMOVE_HARMFUL    - removal helped at every shift -> drop it.
        REMOVE_DEADWEIGHT - sign flips AND the effect never reaches
                            dead_threshold anywhere (nor on the replay) -> no
                            measurable value, drop it for parsimony.
        KEEP_LOAD_BEARING - removal hurt at every shift -> the feature earns
                            its place.
        KEEP_SCENARIO     - sign flips but the effect reaches swing_threshold
                            somewhere -> genuinely regime-dependent; it helps
                            in some periods, so keep it.
        INCONCLUSIVE      - mid-sized flipping effect, or a replay that
                            contradicts the shift grid -> needs more data.
    """
    deltas = np.array(deltas, dtype=float)
    if len(deltas) < 2:
        return "INSUFFICIENT"

    if np.all(deltas == 0):
        return "NOT_APPLICABLE"

    if dead_threshold is None or swing_threshold is None:
        if baseline_mae is None:
            raise ValueError("classify_ablation needs baseline_mae, or explicit thresholds.")
        dead_threshold = DEAD_WEIGHT_FRACTION * baseline_mae if dead_threshold is None else dead_threshold
        swing_threshold = SCENARIO_SWING_FRACTION * baseline_mae if swing_threshold is None else swing_threshold

    nonzero = deltas[deltas != 0]
    sign_consistent = len(nonzero) > 0 and np.all(np.sign(nonzero) == np.sign(nonzero[0]))
    max_abs = float(np.max(np.abs(deltas)))

    if sign_consistent:
        helps_to_remove = np.sign(nonzero[0]) < 0
        # A replay that lands meaningfully on the other side contradicts the grid.
        if replay_delta is not None and abs(replay_delta) >= dead_threshold:
            if (replay_delta < 0) != helps_to_remove:
                return "INCONCLUSIVE"
        return "REMOVE_HARMFUL" if helps_to_remove else "KEEP_LOAD_BEARING"

    if max_abs >= swing_threshold:
        return "KEEP_SCENARIO"

    if max_abs < dead_threshold and (replay_delta is None or abs(replay_delta) < dead_threshold):
        return "REMOVE_DEADWEIGHT"

    return "INCONCLUSIVE"


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
