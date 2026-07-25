"""
Variant definitions for the A/B backtest flow. See the README "A/B Backtest
Flow" section (the agent playbook covers editing CANDIDATE step by step).

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
    frozen_features: list = None
    frozen_rolling: frozenset = None


BASELINE = Variant(name="base")

# --- Edit below for the experiment at hand. Revert to the no-op Variant()
# once the verdict is recorded (see the README "A/B Backtest Flow" playbook). ---
CANDIDATE = Variant(name="candidate")
