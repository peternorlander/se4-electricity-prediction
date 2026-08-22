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
    rule (see README "How changes are validated"):

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


# --- Cluster-level verdicts (for the 3-period measurement grid) --------------
#
# WHY THESE EXIST. classify() above decides magnitude with `|mean| >= spread`,
# where spread = max - min. That is a RANGE statistic: it grows with the number
# of measurement points, while a real standard error shrinks. The rule therefore
# gets STRICTER the more you measure -- backwards. Measured on round 13.1's data
# (2026-08-05), holding the effect fixed at -0.234 and subsampling k of 8 points:
#
#     k =  2   4   6   8
#     P(REAL) 96% 79% 46%  0%
#
# Same effect, same evidence quality; 96% -> 0% purely from point count. This is
# not hypothetical: the cheap2h hurdle, the largest confirmed win in this repo,
# cleared the old rule by |mean| 0.438 vs spread 0.430 -- a 2% margin on SIX
# shifts. The identical effect measured on the 16-point grid would very likely
# have been rejected as BORDERLINE.
#
# The fix is to stop treating the 16 grid points as 16 independent samples,
# which they are not: the 8 NOW points come from four snapshots whose evaluation
# periods overlap ~97% (see the ab-snapshot-overlap note and round 11). There
# are really THREE quasi-independent evaluation periods -- NOW, -6M, -12M -- so
# replication should be judged across those, with ONE VOTE EACH. Cluster means
# are averaged unweighted for exactly that reason: pooling over points would let
# NOW's 8 points outvote the two genuinely different periods 2:1 on count alone.
#
# RETROACTIVE VALIDATION (experiments/backtest_verdict_rule.py, 2026-08-05):
# replayed against every experiment ever run on this grid -- rounds 11, 12 and
# 13.1, 38 configs -- these functions reproduce ALL 38 existing verdicts.
#
# READ THIS BEFORE TRUSTING THAT NUMBER. It establishes specificity only. Not
# one of those 38 configs is cluster-sign-consistent AND negative (10 are
# consistently worse, 28 flip sign), so the entire testable ledger is null
# results and a rule that rejected everything would also have scored 38/38.
# Two consequences that must not be forgotten:
#
#   1. MIN_CLUSTER_EFFECT IS NOT VALIDATED. It never binds on any historical
#      data -- sweeping it from 0.01 to 0.80 leaves all 38 verdicts unchanged.
#      0.10 is a provisional placeholder chosen to sit just under the ~0.12
#      resolution floor implied by the between-cluster spread. The first change
#      that is cluster-sign-consistent and negative will be the first real test
#      of it; decide the number then, on a concrete case, rather than defending
#      this default.
#   2. Sensitivity is tested separately by round 13.1b
#      (experiments/run_round13_1b_hurdle_sensitivity.py), which re-measures the
#      known-good cheap2h hurdle on this grid. If that comes back as anything
#      other than REAL, these functions are too strict and should not be used.
#
# UNRESOLVED DESIGN QUESTION, recorded so it is not silently inherited: giving
# -12M an equal vote also gives it a VETO, and -12M trains on ~360 rows against
# production's ~1085. Round 12 established that this min_train confound can
# manufacture an apparent effect; round 13.1 found the mirror image, an effect
# that strengthens with training size (r = -0.6) and is therefore weakest in the
# cluster least like production. Whether a confounded cluster should be able to
# veto a change is a judgement call that has NOT been made. Until it is, check
# the correlation between per-point delta and min_train before reading any
# cluster table (round 12's standing instruction).
MIN_CLUSTER_EFFECT = 0.10  # PROVISIONAL -- see point 1 above; not validated
MIN_CLUSTERS = 3           # the standard grid: NOW / -6M / -12M


def _cluster_means(cluster_deltas: dict) -> dict:
    """
    {cluster_name: [per-point deltas]} -> {cluster_name: mean delta}.

    Accepts an already-averaged scalar per cluster too, so callers that have
    only cluster means can pass those directly.
    """
    means = {}
    for name, vals in cluster_deltas.items():
        if np.ndim(vals) == 0:
            means[name] = float(vals)
            continue
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            means[name] = float(arr.mean())
    return means


def classify_clustered(cluster_deltas: dict, min_effect: float = MIN_CLUSTER_EFFECT,
                        min_clusters: int = MIN_CLUSTERS) -> str:
    """
    Cluster-level verdict for an ADDITION. Same vocabulary as classify(), so the
    action mapping is unchanged (adopt on REAL, nothing else).

    Args:
        cluster_deltas: {cluster_name: [candidate - baseline, per point]}.
                        Negative = candidate better.
        min_effect:     magnitude floor on the mean of the cluster means.
                        See MIN_CLUSTER_EFFECT -- provisional, unvalidated.
        min_clusters:   evaluation periods required before concluding anything.

    Returns:
        NO_CHANGE    - every delta exactly zero.
        INSUFFICIENT - fewer than min_clusters periods measured.
        NOISE        - the cluster means disagree in sign (no replication).
        REAL         - one sign in every period AND |effect| >= min_effect.
        BORDERLINE   - one sign in every period but below min_effect. This is
                       where a genuine-but-small effect lands; it is NOT a
                       rejection so much as "the grid cannot resolve this".
    """
    means = _cluster_means(cluster_deltas)
    if len(means) < min_clusters:
        return "INSUFFICIENT"

    vals = np.array(list(means.values()), dtype=float)
    if np.all(vals == 0):
        return "NO_CHANGE"

    nonzero = vals[vals != 0]
    if not np.all(np.sign(nonzero) == np.sign(nonzero[0])):
        return "NOISE"

    return "REAL" if abs(float(vals.mean())) >= min_effect else "BORDERLINE"


def classify_ablation_clustered(cluster_deltas: dict, baseline_mae: float = None,
                                 min_effect: float = MIN_CLUSTER_EFFECT,
                                 min_clusters: int = MIN_CLUSTERS,
                                 dead_threshold: float = None,
                                 swing_threshold: float = None) -> str:
    """
    Cluster-level verdict for a REMOVAL (candidate = feature dropped), so a
    negative delta means removal LOWERED MAE. Mirrors classify_ablation's
    vocabulary and its inverted burden of proof: "unproven" must not mean
    "keep forever", or dead weight can never be retired (see the module comment
    above classify_ablation).

    WHICH TEST LOOKS AT WHICH DATA -- the subtle part, and one this function got
    wrong on the first attempt (caught by backtest_verdict_rule.py, which flipped
    5 round-11 verdicts to REMOVE at min_effect=0.20):

      * The REPLICATION test (sign consistency, and the min_effect floor) runs on
        CLUSTER MEANS. That is the whole point of this function -- one vote per
        evaluation period.
      * The MAGNITUDE tests (regime swing, dead weight) run on the RAW PER-POINT
        deltas. Two independent reasons. First, DEAD_WEIGHT_FRACTION and
        SCENARIO_SWING_FRACTION were calibrated against per-point spreads;
        averaging into cluster means shrinks magnitudes, so reusing those numbers
        on means silently reclassifies live features as dead weight. Second, the
        swing test is asking "does this feature ever matter anywhere", which
        averaging is precisely designed to destroy -- round 11's `price_momentum`
        on `max` has mean -0.017 but swings to 1.588 at a single point.

    Do NOT "simplify" this by running everything on one or the other.

    Args:
        cluster_deltas:  {cluster_name: [removed - baseline, per point]}.
        baseline_mae:    scales the dead-weight and regime-swing thresholds when
                         they are not given explicitly.
        min_effect:      magnitude floor for a sign-consistent verdict, applied
                         to the mean of the cluster means. Provisional -- see
                         MIN_CLUSTER_EFFECT.
        min_clusters:    evaluation periods required.
        dead_threshold:  absolute override for the dead-weight cut-off.
        swing_threshold: absolute override for the regime-swing cut-off.

    Returns:
        NOT_APPLICABLE    - every delta exactly zero (feature not in this list).
        INSUFFICIENT      - fewer than min_clusters periods measured.
        REMOVE_HARMFUL    - removal helped in every period, by >= min_effect.
        KEEP_LOAD_BEARING - removal hurt in every period, by >= min_effect.
        KEEP_SCENARIO     - periods disagree but some POINT swings hard: the
                            feature genuinely helps somewhere, so keep it.
        REMOVE_DEADWEIGHT - periods disagree and no point reaches
                            dead_threshold: no measurable value anywhere.
        INCONCLUSIVE      - sign-consistent but under min_effect, or a
                            mid-sized flipping effect. Keep, and say why.
    """
    means = _cluster_means(cluster_deltas)
    if len(means) < min_clusters:
        return "INSUFFICIENT"

    # Unlike classify_clustered, this function CANNOT work from cluster means
    # alone -- the swing and dead-weight tests need the raw per-point spread,
    # and means are systematically smaller than the points they average. Being
    # permissive here produced a silent wrong answer once already (the first
    # backtest run passed means and reclassified five live round-11 features as
    # REMOVE_DEADWEIGHT), so refuse scalars rather than quietly degrade.
    scalar_clusters = [k for k, v in cluster_deltas.items() if np.ndim(v) == 0]
    if scalar_clusters:
        raise ValueError(
            "classify_ablation_clustered needs the per-point deltas for each "
            f"cluster, but {scalar_clusters} were given as single numbers. The "
            "swing/dead-weight tests are calibrated on per-point spread and are "
            "meaningless on cluster means -- pass {cluster: [delta, ...]}."
        )

    per_point = np.array(
        [d for vals_in in cluster_deltas.values()
         for d in np.asarray(vals_in, dtype=float).ravel().tolist()],
        dtype=float,
    )

    if per_point.size and np.all(per_point == 0):
        return "NOT_APPLICABLE"

    if dead_threshold is None or swing_threshold is None:
        if baseline_mae is None:
            raise ValueError(
                "classify_ablation_clustered needs baseline_mae, or explicit "
                "dead_threshold/swing_threshold, to scale the magnitude tests."
            )
        if dead_threshold is None:
            dead_threshold = DEAD_WEIGHT_FRACTION * baseline_mae
        if swing_threshold is None:
            swing_threshold = SCENARIO_SWING_FRACTION * baseline_mae

    vals = np.array(list(means.values()), dtype=float)
    nonzero = vals[vals != 0]
    sign_consistent = len(nonzero) > 0 and np.all(np.sign(nonzero) == np.sign(nonzero[0]))
    effect = float(vals.mean())
    max_abs_point = float(np.max(np.abs(per_point))) if per_point.size else 0.0

    if sign_consistent:
        if abs(effect) < min_effect:
            return "INCONCLUSIVE"
        return "REMOVE_HARMFUL" if effect < 0 else "KEEP_LOAD_BEARING"

    if max_abs_point >= swing_threshold:
        return "KEEP_SCENARIO"
    if max_abs_point < dead_threshold:
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
