"""
Scoring the cheap2h band (cheap2h_low / cheap2h_high / cheap2h_risk). See
docs/AB_TESTING.md "Scoring the band".

MAE cannot see the band: a change can improve the point estimate and still move
`cheap2h_high` from a useful warning to a useless one. So the band gets its own
walk-forward on the same rolling geometry as the A/B, and its own scorecard. The
point estimate it is scored around is the variant's own min/cheap2h pair after
coherence, exactly as Home Assistant receives it.

`python ab_test.py band` runs BASELINE and CANDIDATE through band_walk_forward on
a cluster grid and prints score_band per cluster plus the most recent weeks;
band_regressions() names every metric where the candidate is worse than the
baseline by more than BAND_TOLERANCE. A breach is not an automatic rejection --
the band is a product decision -- but it must be written down and decided on.
"""
import numpy as np
import pandas as pd

import model as model_module
from features import apply_forecast_freeze
from ab.harness import apply_shift, PRODUCTION_TRAIN_ROWS

TARGET = model_module.INTERVAL_TARGET
POINT_TARGETS = ("min", TARGET)     # coherence needs both; avg/max never touch the band

# How much worse than the baseline a candidate may be, per metric, before it is a
# breach that has to be decided on. Rates are fractions of days, levels EUR/MWh.
# Set 2026-10-16 from the size of the round-24 effects; widen only with a reason.
BAND_TOLERANCE = {
    "above_high": 0.02,        # more outcomes over the warning light
    "below_low": 0.02,         # more outcomes under the floor
    "recall": 0.05,            # fewer expensive days warned
    "precision": 0.05,         # more false alarms per warning
    "calm_high": 5.0,          # a higher (less informative) high on calm days
}
# Metrics where a LOWER value is worse (the rest: higher is worse).
_HIGHER_IS_BETTER = {"recall", "precision", "coverage"}


def band_walk_forward(data: pd.DataFrame, variant, train_window: int = PRODUCTION_TRAIN_ROWS,
                      step: int = 7, iterations: int = 52) -> pd.DataFrame:
    """
    One row per test day: actual cheap2h, the variant's point estimate (after
    coherence), its low / high / risk, and whether the day was coupled. Same
    rolling geometry and horizon-honest freeze as ab.harness.run_walk_forward.
    """
    data = data.reset_index(drop=True)
    min_train = len(data) - iterations * step
    if min_train < train_window:
        raise ValueError(f"need {train_window + iterations * step} rows, got {len(data)}")
    targets = {k: variant.targets[k] for k in POINT_TARGETS}
    rows = []
    for i in range(iterations):
        te = min_train + i * step
        train = data.iloc[te - train_window:te]
        raw = data.iloc[te:te + step]
        frozen = apply_forecast_freeze(raw, train, frozen_features=variant.frozen_features,
                                       frozen_rolling=variant.frozen_rolling)
        models = variant.fit_fn(train, targets=targets)
        point = model_module.enforce_coherence(
            {k: models[k].predict(frozen[cols].values) for k, (_, cols) in targets.items()})[TARGET]
        band = variant.band_fn(train)
        out = band.predict(frozen, point)
        low, high = out[0], out[1]
        risk = out[2] if len(out) > 2 else np.full(len(raw), np.nan)
        # "Warned" is the band's own state: its threshold, not a fixed one, or a
        # variant that moves the threshold is scored against the wrong warning.
        threshold = getattr(band, "warning_threshold", model_module.COUPLING_WARNING_PROBA)
        warned = np.where(np.isnan(risk), np.nan, risk >= threshold)
        coupled = model_module.coupling_label(raw)
        for j in range(len(raw)):
            rows.append(dict(window=i, h=j + 1, date=raw["date"].iloc[j],
                             actual=float(raw[targets[TARGET][0]].iloc[j]), point=float(point[j]),
                             low=float(low[j]), high=float(high[j]), risk=float(risk[j]),
                             warned=float(warned[j]),
                             coupled=int(coupled[j]), workday=int(raw["is_workday"].iloc[j])))
    return pd.DataFrame(rows)


def score_band(df: pd.DataFrame) -> dict:
    """
    The band scorecard. Every metric is a plain fraction or a mean in EUR/MWh:

      coverage     low <= actual <= high
      above_high   actual > high  -- the warning light failed to warn high enough
      below_low    actual < low
      width        mean high - low
      calm_high    mean high on days that did NOT couple -- how informative the
                   light is when nothing happens (lower is better)
      recall       of the coupled (expensive) days, the share the band warned on
                   (its own `warned` state: risk >= the band's warning_threshold)
      precision    of the warned days, the share that coupled
      warn_rate    share of days warned
      missed_cost  mean (actual - high) over coupled days that ended above high
      point_mae    MAE of the point estimate the band is built around

    recall / precision / warn_rate need a warning state; a band without a risk (the
    pre-2026-10-16 symmetric band) scores them as NaN.
    """
    a, lo, hi = df["actual"], df["low"], df["high"]
    coupled = df["coupled"] == 1
    out = {
        "n": int(len(df)),
        "coverage": float(((a >= lo) & (a <= hi)).mean()),
        "above_high": float((a > hi).mean()),
        "below_low": float((a < lo).mean()),
        "width": float((hi - lo).mean()),
        "calm_high": float(hi[~coupled].mean()) if (~coupled).any() else np.nan,
        "expensive_share": float(coupled.mean()),
        "point_mae": float((df["point"] - a).abs().mean()),
    }
    missed = coupled & (a > hi)
    out["missed_cost"] = float((a - hi)[missed].mean()) if missed.any() else 0.0
    if df["warned"].notna().all():
        warned = df["warned"] == 1
        out["warn_rate"] = float(warned.mean())
        out["recall"] = float((warned & coupled).sum() / max(1, coupled.sum()))
        out["precision"] = float((warned & coupled).sum() / max(1, warned.sum()))
    else:
        out.update(warn_rate=np.nan, recall=np.nan, precision=np.nan)
    return out


def band_regressions(base: dict, cand: dict, tolerance: dict = None) -> list:
    """Metrics where `cand` is worse than `base` by more than the tolerance."""
    tolerance = BAND_TOLERANCE if tolerance is None else tolerance
    breaches = []
    for metric, tol in tolerance.items():
        b, c = base.get(metric), cand.get(metric)
        if b is None or c is None or np.isnan(b) or np.isnan(c):
            continue
        worse = (b - c) if metric in _HIGHER_IS_BETTER else (c - b)
        if worse > tol:
            breaches.append(f"{metric}: {b:.3f} -> {c:.3f}")
    return breaches


def run_band_ab(data: pd.DataFrame, clusters: dict, baseline, candidate,
                train_window: int = PRODUCTION_TRAIN_ROWS, recent_days: int = 56,
                workers: int = 1) -> dict:
    """
    BASELINE vs CANDIDATE band scorecards on a cluster grid ({name: [shifts]}).
    Prints one table per cluster, one for the last `recent_days` of the newest
    point, and the breaches. Returns {"base": df, "cand": df} with a `cluster`
    and `shift` column, for any further question.
    """
    jobs = [(name, s, which) for name, shifts in clusters.items() for s in shifts
            for which in ("base", "cand")]
    variants = {"base": baseline, "cand": candidate}
    args = [(data, j, variants, train_window) for j in jobs]
    if workers > 1:
        # Process workers pickle the variants: module-level functions only, no lambdas.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            frames = list(pool.map(_band_job, args))
    else:
        frames = [_band_job(a) for a in args]
    allrows = pd.concat(frames, ignore_index=True)

    newest = min(s for shifts in clusters.values() for s in shifts)
    groups = [(name, allrows[allrows["cluster"] == name]) for name in clusters]
    recent = allrows[allrows["shift"] == newest]
    cutoff = recent["date"].max() - pd.Timedelta(days=recent_days - 1)
    groups.append((f"last {recent_days}d", recent[recent["date"] >= cutoff]))

    any_breach = False
    for name, g in groups:
        sb = score_band(g[g["arm"] == "base"])
        sc = score_band(g[g["arm"] == "cand"])
        table = pd.DataFrame({"base": sb, "cand": sc}).T
        print(f"\n--- {name} (n={sb['n']} days per arm)")
        print(table.round(3).to_string())
        breaches = band_regressions(sb, sc)
        if breaches:
            any_breach = True
            print("  BREACH: " + "; ".join(breaches))
    print("\nBand verdict: " + ("breaches above -- decide and record them before shipping"
                                if any_breach else "no metric worse than BAND_TOLERANCE"))
    return {"base": allrows[allrows["arm"] == "base"], "cand": allrows[allrows["arm"] == "cand"]}


def _band_job(args):
    data, (name, s, which), variants, train_window = args
    df = band_walk_forward(apply_shift(data, s), variants[which], train_window=train_window)
    return df.assign(cluster=name, shift=s, arm=which)
