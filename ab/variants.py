"""
Variant definitions for the A/B backtest flow. See docs/AB_TESTING.md (the
agent playbook covers editing CANDIDATE step by step).

BASELINE mirrors production exactly (model._fit_models / model.TARGETS).
CANDIDATE is the ONLY thing an experimenter edits -- typically a handful of
lines. transform() may add or modify COLUMNS only; it must never add, drop,
or reorder rows (checked in ab/verdict.py), since BASELINE and CANDIDATE must
run on identical row sets for the drift-free same-slice comparison to hold.

To change the feature set (e.g. an ablation), override `targets` with the
reduced/expanded feature lists -- the harness passes `targets` to `fit_fn`, so
the default fit_fn (model._fit_models) fits and predicts on the SAME lists (no
custom fit_fn needed). A custom fit_fn must accept the signature
`fit_fn(train_slice, targets) -> {name: model}`.

Before running anything, decide whether a column the variant adds has to be
FROZEN across the forecast horizon -- see the note on `frozen_features` below.
Getting that wrong is the one mistake here that produces a confident number
production cannot reproduce.

Workflow: edit CANDIDATE, run `python ab_test.py run`, read the verdict
table, then revert CANDIDATE to the no-op below. Variants are disposable;
this scaffolding is permanent.
"""
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import model as model_module


def _identity(data: pd.DataFrame) -> pd.DataFrame:
    return data


@dataclass
class Variant:
    name: str
    transform: Callable[[pd.DataFrame], pd.DataFrame] = _identity
    # fit_fn(train_slice, targets) -> {name: fitted model}. The harness passes the
    # variant's `targets`, so overriding `targets` alone is enough for a feature-set
    # change; the default fits and predicts on the same lists.
    fit_fn: Callable[[pd.DataFrame, dict], dict] = model_module._fit_models
    targets: dict = field(default_factory=lambda: dict(model_module.TARGETS))
    # Horizon honesty. Production runs ONCE and issues d+1..d+7 together, so a
    # lag-type value it cannot know per day is frozen across the whole window
    # (features.FORECAST_FROZEN_FEATURES). Walk-forward test rows instead carry
    # each day's own TRUE lag, so a new lag-like column added here is measured
    # with knowledge production will never have unless it is frozen too: pass
    # FORECAST_FROZEN_FEATURES + [your column]. Leaving these None keeps the
    # production lists, which do NOT mention your new column -- the default is
    # "not frozen", i.e. the leaky direction.
    #
    # apply_forecast_freeze silently skips names that are not columns of the
    # frame, so a typo reads as frozen and is not. Assert both.
    #
    # Leave them None only when the column genuinely IS recomputable per
    # forecast day in production (weather, calendar, HDD, residual load), or
    # when the arm is a deliberate upper-bound "oracle" -- in which case label it
    # as leaky in the ledger entry. See docs/AB_TESTING.md "Horizon honesty".
    frozen_features: list = None
    frozen_rolling: frozenset = None


BASELINE = Variant(name="base")

# --- Edit below for the experiment at hand. Revert to the no-op Variant()
# once the verdict is recorded (see docs/AB_TESTING.md, the playbook). ---
CANDIDATE = Variant(name="candidate")
