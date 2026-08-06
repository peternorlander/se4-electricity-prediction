"""Self-check for the verdict rules. Run: python ab/check_verdict.py

There is no pytest in this project; verification is by runnable scripts with
asserts (the same convention the plan's "synthetic-data smoke test" describes).
This one exists because the cluster rules shipped with two real bugs inside an
hour, both of which produced plausible-looking wrong answers rather than
crashes:

  1. min_effect was doing double duty as the dead-weight cut-off, so raising it
     made the ablation rule MORE eager to delete features.
  2. The magnitude tests were fed cluster MEANS instead of per-point deltas.
     Means are smaller than the points they average, so five live round-11
     features were reclassified as dead weight.

Both are locked down below. Neither would have been caught by reading the code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ab.verdict import (classify, classify_clustered,
                        classify_ablation_clustered)

BASE_MAE = 16.0          # -> dead_threshold 0.24, swing_threshold 0.56
checks = 0


def eq(actual, expected, label):
    global checks
    assert actual == expected, f"{label}: expected {expected}, got {actual}"
    checks += 1


def raises(fn, label):
    global checks
    try:
        fn()
    except ValueError:
        checks += 1
        return
    raise AssertionError(f"{label}: expected ValueError, none raised")


# --- classify_clustered: additions -------------------------------------------
eq(classify_clustered({"NOW": [-0.5], "-6M": [-0.4], "-12M": [-0.6]}),
   "REAL", "all clusters clearly better")
eq(classify_clustered({"NOW": [-0.05], "-6M": [-0.04], "-12M": [-0.06]}),
   "BORDERLINE", "consistent but under min_effect")
eq(classify_clustered({"NOW": [-0.5], "-6M": [+0.4], "-12M": [-0.6]}),
   "NOISE", "cluster signs disagree")
eq(classify_clustered({"NOW": [0.0], "-6M": [0.0], "-12M": [0.0]}),
   "NO_CHANGE", "all zero")
eq(classify_clustered({"NOW": [-0.5], "-6M": [-0.4]}),
   "INSUFFICIENT", "only two periods")
eq(classify_clustered({"NOW": [+0.5], "-6M": [+0.4], "-12M": [+0.6]}),
   "REAL", "consistently WORSE is also REAL -- callers must check direction")

# --- The motivating bug: the OLD rule is n-dependent -------------------------
# Same effect (~ -0.234), more measurement points, verdict degrades. This is a
# regression guard on the MOTIVATION for the cluster rules; if it ever stops
# holding, revisit whether they are still needed.
tight = [-0.24, -0.23, -0.25]
wide = [-0.227, -0.236, -0.268, -0.136, -0.223, -0.185, -0.181, -0.416]
eq(classify(tight), "REAL", "old rule on 3 points")
eq(classify(wide), "BORDERLINE", "old rule on 8 points, same mean effect")

# --- The fix: the cluster rule is invariant to points-per-cluster ------------
sparse = {"NOW": [-0.30, -0.10], "-6M": [-0.25, -0.15], "-12M": [-0.35, -0.05]}
dense = {c: v * 4 for c, v in sparse.items()}      # 8 points per cluster, same means
eq(classify_clustered(sparse), classify_clustered(dense),
   "verdict must not depend on how many points sit in each cluster")
eq(classify_clustered(sparse), "REAL", "sparse case resolves")

# --- classify_ablation_clustered: removals -----------------------------------
eq(classify_ablation_clustered({"NOW": [-0.5], "-6M": [-0.4], "-12M": [-0.6]},
                               baseline_mae=BASE_MAE),
   "REMOVE_HARMFUL", "removal helps in every period")
eq(classify_ablation_clustered({"NOW": [+0.5], "-6M": [+0.4], "-12M": [+0.6]},
                               baseline_mae=BASE_MAE),
   "KEEP_LOAD_BEARING", "removal hurts in every period")
eq(classify_ablation_clustered({"NOW": [-0.05], "-6M": [-0.04], "-12M": [-0.06]},
                               baseline_mae=BASE_MAE),
   "INCONCLUSIVE", "consistent but too small to act on")
eq(classify_ablation_clustered({"NOW": [0.0], "-6M": [0.0], "-12M": [0.0]},
                               baseline_mae=BASE_MAE),
   "NOT_APPLICABLE", "feature absent from this target's list")

# BUG 2 REGRESSION GUARD. Cluster means are all tiny (+0.05 / +0.025 / -0.025)
# and disagree in sign, but one POINT swings 1.6 -- far past the 0.56 swing
# threshold. The feature is regime-dependent, not dead. Judging this on cluster
# means would see max |mean| = 0.05 < 0.24 and delete it.
swingy = {"NOW": [+1.60, -1.50], "-6M": [+0.10, -0.05], "-12M": [-0.10, +0.05]}
eq(classify_ablation_clustered(swingy, baseline_mae=BASE_MAE),
   "KEEP_SCENARIO", "a big per-point swing must survive small cluster means")

# Genuinely dead: mixed signs and nothing anywhere reaches the dead threshold.
eq(classify_ablation_clustered({"NOW": [+0.02, -0.01], "-6M": [+0.03, -0.02],
                                "-12M": [-0.02, +0.01]}, baseline_mae=BASE_MAE),
   "REMOVE_DEADWEIGHT", "no measurable effect at any point")

# BUG 1 REGRESSION GUARD. min_effect gates only the sign-consistent branch; it
# must NOT make the dead-weight test hungrier. Raising it hard leaves the
# swingy case alone.
eq(classify_ablation_clustered(swingy, baseline_mae=BASE_MAE, min_effect=0.80),
   "KEEP_SCENARIO", "min_effect must not drive the dead-weight test")

# BUG 2, the loud version: scalars are refused rather than silently degraded.
raises(lambda: classify_ablation_clustered({"NOW": -0.1, "-6M": 0.05, "-12M": -0.2},
                                           baseline_mae=BASE_MAE),
       "scalar clusters must raise")
raises(lambda: classify_ablation_clustered({"NOW": [-0.1], "-6M": [0.05], "-12M": [-0.2]}),
       "missing baseline_mae must raise")

print(f"ab/check_verdict.py: {checks} checks passed")
