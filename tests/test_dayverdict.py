"""The day-level interval verdict, the year grid and seeded fits: the gate of docs/AB_TESTING.md
("The adoption bar", 2026-11-15). Synthetic errors only -- no model is fitted."""
import numpy as np
import pandas as pd
import pytest

from se4 import model
from se4.ab import dayverdict as dv
from se4.ab import verdict
from se4.ab.harness import ARCHIVE, LEADS, SeededFit

DATES = pd.date_range("2024-11-09", periods=364 + 371)      # room for shifts 0..370
YEAR_GRID = list(range(7)) + list(range(364, 371))


def _wobble(days, scale=1.0, salt=0.0):
    """A fixed pseudo-random number per calendar day, mean ~0."""
    x = np.sin(days.astype("int64") * 12.9898 + salt) * 43758.5453
    return scale * ((x - np.floor(x)) - 0.5)


def _errors(shifts, cand_minus_base, base_level=10.0, forecast=False):
    """{(shift, arm): harness-shaped errors} where both arms are functions of the calendar
    day alone, so points sharing a day agree exactly -- as identical fits do."""
    out = {}
    for shift in shifts:
        days = dv.point_dates(DATES, shift)
        base = base_level + _wobble(days, 4.0)
        for which, arr in (("base", base), ("cand", base + cand_minus_base(days))):
            out[(shift, which)] = {ARCHIVE: {"t": arr}}
            if forecast:
                for lead in LEADS:
                    out[(shift, which)][lead] = {"t": arr + 100.0 * lead}
    return out


def _const(value):
    return lambda days: np.full(days.shape, value)


def test_points_a_week_apart_count_each_day_once():
    errors = _errors([0, 7], _const(-0.5))
    table = dv.day_table(DATES, errors, [0, 7], "t", ARCHIVE)
    # shift 7 is shift 0 moved one week back: 51 weeks shared, one new -> 53 weeks x 6 sent rows
    assert len(table) == 53 * 6
    assert table["alignment"].nunique() == 1
    assert table.attrs["max_duplicate_gap"] == 0.0


def test_fits_that_disagree_on_a_shared_day_are_reported():
    errors = _errors([0, 7], _const(-0.5))
    errors[(7, "cand")][ARCHIVE]["t"] = errors[(7, "cand")][ARCHIVE]["t"] + 0.25
    v = dv.day_verdict(DATES, errors, [0, 7], "t", ARCHIVE)
    assert v.max_duplicate_gap == pytest.approx(0.25)
    assert any("not identical" in note for note in v.notes)


def test_alignment_is_the_weekday_the_test_week_starts_on():
    for shift in range(7):
        table = dv.day_table(DATES, _errors([shift], _const(0.0)), [shift], "t", ARCHIVE)
        week_start = pd.Timestamp(dv.point_dates(DATES, shift)[0, 0])
        assert table["alignment"].unique().tolist() == [week_start.weekday()]


def test_forecast_reads_sent_row_j_at_lead_j_plus_2_and_archive_drops_row_6():
    errors = _errors([0], _const(0.0), forecast=True)[(0, "base")]
    archive = dv.sent_errors(errors, "t", ARCHIVE)
    forecast = dv.sent_errors(errors, "t", dv.FORECAST)
    assert archive.shape == forecast.shape == (52, 6)
    np.testing.assert_allclose(forecast - archive, np.tile(100.0 * np.arange(2, 8), (52, 1)))


def test_a_steady_gain_on_the_year_grid_is_real_and_a_gate():
    v = dv.day_verdict(DATES, _errors(YEAR_GRID, _const(-0.5)), YEAR_GRID, "t", ARCHIVE)
    assert v.verdict == "REAL" and v.gate
    assert v.delta == pytest.approx(-0.5) and v.high < 0
    # 734 days are evaluated; the newest is only ever row 6 (D+8), which production does not send
    assert v.alignments == 7 and v.days == 733
    assert list(v.years) == ["NOW", "PREV"]
    assert v.years["NOW"]["days"] == 364 and v.years["PREV"]["days"] == 369   # the 5 spare days join PREV


def test_a_steady_loss_is_harmful_and_fails_the_removal_bar():
    v = dv.day_verdict(DATES, _errors(YEAR_GRID, _const(+0.5)), YEAR_GRID, "t", ARCHIVE)
    assert v.verdict == "HARMFUL" and not v.non_inferior


def test_noise_around_zero_is_unresolved():
    v = dv.day_verdict(DATES, _errors(YEAR_GRID, lambda d: _wobble(d, 6.0, salt=1.0)), YEAR_GRID, "t", ARCHIVE)
    assert v.verdict == "UNRESOLVED" and v.low < 0 < v.high
    assert not v.non_inferior            # an interval this wide does not bound the cost either


def test_a_removal_with_a_bounded_cost_passes_its_bar_without_being_real():
    v = dv.day_verdict(DATES, _errors(YEAR_GRID, lambda d: _wobble(d, 0.2, salt=2.0)), YEAR_GRID, "t", ARCHIVE)
    assert v.verdict == "UNRESOLVED"
    assert v.tolerance == pytest.approx(dv.TOLERANCE_FRACTION * v.base_mae)
    assert v.high < v.tolerance and v.non_inferior


def test_a_gain_paid_for_by_the_other_year_is_mixed():
    newest = DATES[-1].to_datetime64().astype("datetime64[D]")

    def one_year_pays(days):
        return np.where((newest - days).astype("int64") < 364, -1.0, +0.4)

    v = dv.day_verdict(DATES, _errors(YEAR_GRID, one_year_pays), YEAR_GRID, "t", ARCHIVE)
    assert v.high < 0 and v.years["PREV"]["delta"] > v.tolerance
    assert v.verdict == "MIXED" and not v.non_inferior


def test_identical_arms_are_no_change():
    v = dv.day_verdict(DATES, _errors(YEAR_GRID, _const(0.0)), YEAR_GRID, "t", ARCHIVE)
    assert v.verdict == "NO_CHANGE" and not v.non_inferior


def test_a_shift_sweep_is_a_look_not_a_gate():
    shifts = [0, 1, 2, 3, 4, 5]
    v = dv.day_verdict(DATES, _errors(shifts, _const(-0.5)), shifts, "t", ARCHIVE)
    assert v.verdict == "REAL" and not v.gate
    assert any("not a gate" in note for note in v.notes)


def test_since_keeps_only_later_test_days():
    since = DATES[-100]
    v = dv.day_verdict(DATES, _errors(YEAR_GRID, _const(-0.5)), YEAR_GRID, "t", ARCHIVE, since=since)
    assert v.days == 99 and not v.gate          # 100 days, less the unsent newest one
    assert v.years["NOW"]["first"] == str(since.date())
    with pytest.raises(ValueError):
        dv.day_verdict(DATES, _errors([0], _const(-0.5)), [0], "t", ARCHIVE, since="2031-01-01")


def test_the_interval_is_reproducible_and_narrows_with_more_days():
    errors = _errors(YEAR_GRID, lambda d: -0.3 + _wobble(d, 6.0, salt=3.0))
    table = dv.day_table(DATES, errors, YEAR_GRID, "t", ARCHIVE)
    assert dv.bootstrap_interval(table) == dv.bootstrap_interval(table)
    _, low, high = dv.bootstrap_interval(table)
    one_year = dv.split_years(table)["NOW"]
    _, low1, high1 = dv.bootstrap_interval(one_year)
    assert (high1 - low1) > (high - low)
    assert np.isnan(dv.bootstrap_interval(table.iloc[:5])[1])      # one block: no interval


def test_year_grid_covers_seven_alignments_in_each_year():
    assert verdict.year_grid(1088 + 364 + 370, 1088) == {"NOW": list(range(7)), "PREV": list(range(364, 371))}
    short = verdict.year_grid(1088 + 364 + 300, 1088)               # PREV overlaps NOW
    assert short["PREV"] == list(range(294, 301))
    assert {s % 7 for s in short["PREV"]} == set(range(7))
    assert verdict.year_grid(1088 + 364 + 8, 1088) == {"NOW": list(range(7))}
    with pytest.raises(ValueError):
        verdict.year_grid(1088 + 364 + 3, 1088)


def test_seeded_fit_reseeds_the_model_and_restores_the_seed():
    seen = []

    def fit(train_slice, targets=None):
        seen.append((model.RANDOM_STATE, model._make_regressor().get_params()["random_state"]))
        return {}

    assert model.RANDOM_STATE == 42
    SeededFit(fit, 7)("rows", targets={})
    assert seen == [(7, 7)] and model.RANDOM_STATE == 42
    with pytest.raises(RuntimeError):
        SeededFit(lambda rows, targets=None: (_ for _ in ()).throw(RuntimeError()), 7)("rows")
    assert model.RANDOM_STATE == 42


def test_one_seed_is_the_untouched_fit_and_more_seeds_average_the_errors():
    assert verdict.seed_list(1) == [None]
    assert verdict.seed_list(3)[0] is None and len(set(verdict.seed_list(3))) == 3
    one = {ARCHIVE: {"t": np.array([[1.0, 3.0]])}}
    two = {ARCHIVE: {"t": np.array([[3.0, 5.0]])}}
    assert verdict._mean_errors([one]) is one
    np.testing.assert_allclose(verdict._mean_errors([one, two])[ARCHIVE]["t"], [[2.0, 4.0]])


def _args(**kwargs):
    import argparse
    return argparse.Namespace(**{"shifts": None, "grid": None, **kwargs})


def test_run_defaults_to_the_year_grid_and_shifts_are_a_look(monkeypatch):
    import ab_test
    monkeypatch.setattr(ab_test, "_grid", lambda inputs, train_window, kind: {"kind": [kind]})
    assert ab_test._resolve_points(_args(), {}, 1088) == ({"kind": ["year"]}, ["year"], "year")
    assert ab_test._resolve_points(_args(grid="clusters"), {}, 1088)[2] == "clusters"
    assert ab_test._resolve_points(_args(shifts="0-2"), {}, 1088) == (None, [0, 1, 2], None)
    # the expanding geometry has no rolling window to place a grid from: the old shift sweep
    assert ab_test._resolve_points(_args(), {}, None) == (None, [0, 1, 2, 3, 4, 5], None)
    with pytest.raises(SystemExit):
        ab_test._resolve_points(_args(shifts="0-2", grid="year"), {}, 1088)
