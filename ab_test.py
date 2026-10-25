"""
CLI for the A/B backtest flow. See docs/AB_TESTING.md for
the full design and the agent playbook.

    python ab_test.py fetch                         # ~5-year snapshot -> ab_cache/long/
    python ab_test.py fetch --days 1095             # production-shaped -> ab_cache/
    python ab_test.py list                          # list long snapshots
    python ab_test.py list --root ab_cache          # list production-shaped snapshots
    python ab_test.py run [--snapshot YYYY-MM-DD] [--shifts 0-5 | --grid] [--expanding]
                          [--no-vintage] [--workers 4]
    python ab_test.py vintages [--snapshot YYYY-MM-DD] [--grid]   # warm the vintage cache
    python ab_test.py band [--snapshot YYYY-MM-DD] [--workers 4]   # cheap2h band scorecard

Needs ENTSO_E_TOKEN (and the other source-API env vars) only for `fetch`;
`run` and `list` are offline once a snapshot exists -- except that `run` also
scores every fit on forecast-vintage weather (Open-Meteo Previous Runs, public,
no token), which fetches ab_cache/vintages/ the first time and extends it when
a newer snapshot reaches past it. `--no-vintage` skips that and reports archive
weather only. The report always labels the two numbers ARCHIVE and FORECAST: the
eval's weather is ERA5 archive, a forecast with no error, and FORECAST is what
production pays (docs/AB_TESTING.md, "Archive weather is a perfect forecast").

LONG SNAPSHOTS ARE THE DEFAULT (changed 2026-10-10, and `run` now trains on a
rolling window). A ~3-year fetch builds to ~1088 rows, which IS production's
training set -- and that makes it exactly one evaluation year too short to
measure production honestly, because the walk-forward has to take its 364 test
days out of that same window: its early test weeks then train on ~731 rows
instead of 1088. Only a longer snapshot can give every test week production's
own training length, so `fetch` defaults to ~5 years (LONG_FETCH_DAYS) under
ab_cache/long/, and `run` loads from there and trains each week on a rolling
ab.harness.PRODUCTION_TRAIN_ROWS-row window.

`--days 1095` still fetches a production-shaped snapshot into ab_cache/ for the
cases that want production's exact frame (the local train -> predict integration
check before shipping). `--expanding` on `run` falls back to the headline eval's
geometry, which is what every grid before 2026-10 used. What the switch changed
when it was measured is in docs/AB_TESTING.md ("Measurement grids") and
docs/DECISIONS.md.

Every fetch rebuilds the merged frame and checks it for date gaps before
finishing: the grid scripts equate rows with consecutive days (shift arithmetic,
end-date dedup), so a single missing source day anywhere in the window would
otherwise skew every downstream measurement silently. The snapshot is still
saved if a gap is found -- the data remains useful -- but the gap is printed
loudly so it gets investigated before being built into a grid. A long fetch also
prints which period clusters a rolling grid can reach on the rows it got.
"""
import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

from ab.snapshot import (save_snapshot, list_snapshots, load_snapshot, read_meta,
                         DEFAULT_CACHE_ROOT, LONG_CACHE_ROOT)
from ab.harness import parse_shift_range, PRODUCTION_TRAIN_ROWS, LAG_WARMUP_ROWS
from ab.verdict import run_ab
from ab.vintage import default_root as default_vintage_root

# Default fetch length. 1825 days (~5 years) is the right number and asking for
# more buys nothing: the frame cannot start before EUA carbon data does
# (2021-10-18), so a longer window only widens the head that gets dropped. See
# docs/AB_TESTING.md "How long a snapshot to fetch".
LONG_FETCH_DAYS = 1825

# One evaluation year, in rows -- the 52 x 7 test days every grid point spends.
EVAL_DAYS = 364

# The canonical period clusters, as shifts off the end of the frame. Printed
# after a long fetch so it is obvious which of them the rows actually reach.
CLUSTERS = {"NOW": (0, 1, 4, 7), "-6M": (180, 183, 186, 189),
            "-12M": (360, 363, 366, 369), "-21M": (637, 640, 643, 646)}


def _check_date_continuity(inputs: dict) -> tuple:
    """
    Rebuild the merged frame and return
    ([(before, after, n_missing_days), ...], n_rows) -- one tuple per
    non-consecutive gap, plus the row count the rolling-grid capacity depends
    on. Cheap (local pandas, no network) relative to the fetch it follows.
    """
    import pandas as pd
    import features as feat

    data = feat.build_training_data(**inputs)
    dates = pd.to_datetime(data["date"]).reset_index(drop=True)
    diffs = dates.diff().dt.days.iloc[1:]
    gaps = [(str(dates.iloc[i - 1].date()), str(dates.iloc[i].date()), int(d) - 1)
            for i, d in zip(diffs.index, diffs) if d != 1]
    print(f"  merged frame: {dates.iloc[0].date()} -> {dates.iloc[-1].date()} "
          f"({len(data)} rows)")
    return gaps, len(data)


def cmd_fetch(args):
    import fetch_data

    # PRODUCTION_TRAIN_ROWS is what every rolling grid trains on, and it is only
    # correct as long as production still fetches TRAINING_DAYS and loses the
    # same lag warm-up. Fail here rather than measure the wrong thing quietly.
    production_days = fetch_data.TRAINING_DAYS
    assert PRODUCTION_TRAIN_ROWS == production_days - LAG_WARMUP_ROWS, (
        f"PRODUCTION_TRAIN_ROWS ({PRODUCTION_TRAIN_ROWS}) no longer equals "
        f"TRAINING_DAYS ({production_days}) - LAG_WARMUP_ROWS ({LAG_WARMUP_ROWS}). "
        f"Re-measure it on a production-shaped snapshot and update ab/harness.py "
        f"before running any rolling grid."
    )

    is_long = args.days > production_days
    fetch_data.TRAINING_DAYS = args.days

    if args.root:
        cache_root = Path(args.root)
    elif is_long:
        cache_root = LONG_CACHE_ROOT
    else:
        cache_root = DEFAULT_CACHE_ROOT

    today = datetime.now(UTC).date()
    print(f"Fetching training inputs for {today} "
          f"({fetch_data.TRAINING_DAYS} days) -> {cache_root}/ ...")
    inputs = fetch_data.fetch_training_inputs(today)
    snapshot_dir = save_snapshot(inputs, today, cache_root=cache_root)
    print(f"\nSaved snapshot: {snapshot_dir}")

    print("\nVerifying date continuity...")
    gaps, rows = _check_date_continuity(inputs)
    if gaps:
        print(f"\nWARNING -- {len(gaps)} gap(s) in the merged frame:")
        for before, after, missing in gaps:
            print(f"  {before} -> {after}  ({missing} missing day(s))")
        print("\nSnapshot is saved, but grid scripts equate rows with consecutive")
        print("days -- do not build a shift grid on this snapshot until the gap(s)")
        print("are understood.")
    else:
        print("  OK -- no date gaps.")

    if is_long:
        max_shift = rows - EVAL_DAYS - PRODUCTION_TRAIN_ROWS
        print(f"\nRolling-grid capacity: {rows} rows, so a production-faithful point "
              f"(train {PRODUCTION_TRAIN_ROWS} + evaluate {EVAL_DAYS}) fits at "
              f"shifts 0..{max_shift}.")
        for name, shifts in CLUSTERS.items():
            fits = max(shifts) <= max_shift
            print(f"  {name:<5} shifts {shifts[0]}-{shifts[-1]}: "
                  f"{'fits' if fits else f'does NOT fit (needs {max(shifts)})'}")
        if max_shift < max(CLUSTERS['-12M']):
            print("  A cluster that does not fit needs a later fetch -- the frame grows "
                  "one row per day, and the head cannot move before 2021-10-21.")
    else:
        print(f"\nThis is a production-shaped snapshot ({args.days} days) in "
              f"{cache_root}/. It builds to ~{PRODUCTION_TRAIN_ROWS} rows, which is "
              f"production's own training set -- use it for the local "
              f"train -> predict integration check, not for a rolling A/B grid "
              f"(it is one evaluation year too short; see the module docstring).")


def cmd_list(args):
    cache_root = Path(args.root) if args.root else LONG_CACHE_ROOT
    snapshots = list_snapshots(cache_root=cache_root)
    if not snapshots:
        print(f"No snapshots found under {cache_root}/. Run 'python ab_test.py fetch' first.")
        if cache_root == LONG_CACHE_ROOT and list_snapshots(cache_root=DEFAULT_CACHE_ROOT):
            print(f"(Production-shaped snapshots exist under {DEFAULT_CACHE_ROOT}/ -- "
                  f"list them with: python ab_test.py list --root {DEFAULT_CACHE_ROOT}. "
                  f"They are one evaluation year too short for a rolling grid.)")
        return
    print(f"Available snapshots under {cache_root}/:")
    print(f"  {'snapshot':<12} {'weather tail':>12}")
    for name in snapshots:
        meta = read_meta(name, cache_root=cache_root)
        tail = meta.get("weather_tail_days")
        # "*" = the snapshot predates the field and it was derived from the
        # data on read, not recorded at save time. Same number either way.
        mark = "*" if meta.get("weather_tail_derived") else ""
        print(f"  {name:<12} {(f'{tail}d{mark}') if tail is not None else '?':>12}")
    print("\n'weather tail' is how many days behind the snapshot's own `today` its")
    print("weather stops -- 1d = topped up to yesterday, ~5d = archive-only (fetched")
    print("before 2026-08-23). Snapshots with DIFFERENT tails must not share one")
    print("measurement grid: the merged frame is ~4 rows longer on a 1d snapshot, so")
    print("a shift grid built across both would silently compare different windows.")
    print("'*' means the snapshot predates the field and it was derived on read.")


def _grid(inputs, train_window):
    """The per-snapshot NOW / -6M / FAR cluster grid (ab.verdict.default_clusters)."""
    import features as feat
    from ab.verdict import default_clusters
    if train_window is None:
        sys.exit("--grid needs the rolling geometry (it places FAR from the training window); "
                 "drop --expanding or pass --shifts.")
    return default_clusters(len(feat.build_training_data(**inputs)), train_window)


def cmd_run(args):
    cache_root = Path(args.root) if args.root else LONG_CACHE_ROOT
    inputs, meta = load_snapshot(args.snapshot, cache_root=cache_root)
    print(f"Using snapshot: {meta['today']} (fetched {meta['fetched_at']}, root={cache_root})")
    train_window = None if args.expanding else args.train_window
    if args.grid:
        clusters = _grid(inputs, train_window)
        shifts = [s for ss in clusters.values() for s in ss]
        print(f"  cluster grid: {clusters}")
    else:
        clusters, shifts = None, parse_shift_range(args.shifts)
    # Forecast-vintage scoring is the default: every weather input in the eval is a perfect
    # forecast, and a run that reports only that number invites reading it as production's.
    # Opting out is explicit, and the report says so.
    vintage_root = None if args.no_vintage else (
        Path(args.vintage_root) if args.vintage_root else default_vintage_root(cache_root))
    run_ab(inputs, shifts, train_window=train_window, clusters=clusters,
           vintage_root=vintage_root, workers=args.workers, save=args.save_errors)


def cmd_vintages(args):
    """Fetch/extend the vintage cache for a snapshot's evaluated days and check coverage."""
    import features as feat
    from ab.vintage import ensure_vintages, check_coverage, evaluated_dates
    cache_root = Path(args.root) if args.root else LONG_CACHE_ROOT
    inputs, meta = load_snapshot(args.snapshot, cache_root=cache_root)
    print(f"Using snapshot: {meta['today']} (fetched {meta['fetched_at']}, root={cache_root})")
    data = feat.build_training_data(**inputs)
    if args.grid:
        shifts = [s for ss in _grid(inputs, args.train_window).values() for s in ss]
    else:
        shifts = parse_shift_range(args.shifts)
    days = evaluated_dates(data["date"], shifts)
    vintage_root = Path(args.vintage_root) if args.vintage_root else default_vintage_root(cache_root)
    print(f"Vintage cache: {vintage_root}  (evaluated days {days[0]} -> {days[-1]}, {len(days)})")
    vintages = ensure_vintages(max(days), root=vintage_root)
    check_coverage(vintages, days)
    print(f"OK -- {len(vintages)} points, every evaluated hour has a finite vintage at leads 1-7.")


def cmd_band(args):
    from ab.band import run_band_ab
    from ab.variants import BASELINE, CANDIDATE
    from features import build_training_data
    cache_root = Path(args.root) if args.root else LONG_CACHE_ROOT
    inputs, meta = load_snapshot(args.snapshot, cache_root=cache_root)
    print(f"Using snapshot: {meta['today']} (fetched {meta['fetched_at']}, root={cache_root})")
    data = build_training_data(**inputs)
    # One point per cluster by default (FAR = the oldest the rolling geometry can still
    # fit); the band is scored per cluster, never pooled -- docs/AB_TESTING.md.
    far = len(data) - PRODUCTION_TRAIN_ROWS - EVAL_DAYS
    clusters = ({"NOW": [0], "-6M": [180], "FAR": [far]} if args.shifts is None else
                {"grid": parse_shift_range(args.shifts)})
    print(f"  {len(data)} rows; clusters {clusters}; BASELINE={BASELINE.name} CANDIDATE={CANDIDATE.name}")
    run_band_ab(data, clusters, BASELINE, CANDIDATE, recent_days=args.recent_days,
                workers=args.workers)


def main():
    parser = argparse.ArgumentParser(description="A/B backtest flow for SE4 price prediction models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and cache today's training inputs.")
    fetch_parser.add_argument("--days", type=int, default=LONG_FETCH_DAYS,
                               help=f"Training window length in days (default: "
                                    f"{LONG_FETCH_DAYS}, ~5 years, saved under "
                                    f"ab_cache/long/). Asking for more buys no extra "
                                    f"rows: the frame cannot start before EUA carbon "
                                    f"data does (2021-10-18) and the head is dropped. "
                                    f"Pass 1095 for a production-shaped snapshot in "
                                    f"ab_cache/ -- useful for the integration check, "
                                    f"too short for a rolling grid. See "
                                    f"docs/AB_TESTING.md 'How long a snapshot to "
                                    f"fetch'.")
    fetch_parser.add_argument("--root", default=None,
                               help="Cache root to save under (default: ab_cache/long/ "
                                    "for the default length, ab_cache/ for --days 1095).")

    list_parser = subparsers.add_parser("list", help="List cached snapshots.")
    list_parser.add_argument("--root", default=None,
                              help="Cache root to list (default: ab_cache/long/; pass "
                                   "ab_cache for the production-shaped ones).")

    run_parser = subparsers.add_parser("run", help="Run the BASELINE vs CANDIDATE A/B across simulated shifts.")
    run_parser.add_argument("--snapshot", default=None, help="Snapshot date (YYYY-MM-DD). Default: newest.")
    run_parser.add_argument("--shifts", default="0-5", help="Shift range, e.g. '0-5' or '0,2,4'. Default: 0-5.")
    run_parser.add_argument("--root", default=None,
                             help="Cache root to load from (default: ab_cache/long/).")
    run_parser.add_argument("--train-window", type=int, default=PRODUCTION_TRAIN_ROWS,
                             help=f"Rows each test week trains on (default: "
                                  f"{PRODUCTION_TRAIN_ROWS}, what production fits on). "
                                  f"Needs a long snapshot: a point costs "
                                  f"train_window + {EVAL_DAYS} rows plus its shift.")
    run_parser.add_argument("--expanding", action="store_true",
                             help="Train on everything before each test week instead "
                                  "(the headline eval's geometry, and what every grid "
                                  "before 2026-10 used). Only way to run on a "
                                  "production-shaped snapshot.")
    run_parser.add_argument("--grid", action="store_true",
                             help="Run the per-snapshot cluster grid (NOW 0/1/4/7, -6M "
                                  "180/183/186/189, FAR = the oldest four shifts the "
                                  "rolling geometry fits) instead of --shifts, and report "
                                  "per cluster.")
    run_parser.add_argument("--no-vintage", action="store_true",
                             help="Archive weather only. By default every fit is ALSO "
                                  "scored on the forecast Open-Meteo issued L days ahead "
                                  "(Previous Runs, no token) and the report carries both "
                                  "numbers; the first run fetches the vintage cache "
                                  "(~3 min), later runs are offline.")
    run_parser.add_argument("--vintage-root", default=None,
                             help="Vintage cache directory (default: ab_cache/vintages, "
                                  "next to the snapshot root).")
    run_parser.add_argument("--workers", type=int, default=1,
                             help="Processes for the (variant, shift) jobs. Pin "
                                  "OMP_NUM_THREADS yourself (4 for the ledger): XGBoost is "
                                  "not bit-identical across thread counts.")
    run_parser.add_argument("--save-errors", default=None, metavar="PATH",
                             help="Pickle every point's per-row error arrays (all "
                                  "conditions, all targets) for later analysis.")

    vintage_parser = subparsers.add_parser(
        "vintages", help="Fetch/extend the forecast-vintage cache for a snapshot and check its coverage.")
    vintage_parser.add_argument("--snapshot", default=None, help="Snapshot date (YYYY-MM-DD). Default: newest.")
    vintage_parser.add_argument("--shifts", default="0-5", help="Shift range to cover. Default: 0-5.")
    vintage_parser.add_argument("--grid", action="store_true", help="Cover the cluster grid, as `run --grid`.")
    vintage_parser.add_argument("--train-window", type=int, default=PRODUCTION_TRAIN_ROWS,
                                help="Training rows (only places FAR for --grid).")
    vintage_parser.add_argument("--root", default=None, help="Cache root of the snapshot (default: ab_cache/long/).")
    vintage_parser.add_argument("--vintage-root", default=None, help="Vintage cache directory.")

    band_parser = subparsers.add_parser(
        "band", help="Score the cheap2h band (low/high/risk) for BASELINE vs CANDIDATE.")
    band_parser.add_argument("--snapshot", default=None, help="Snapshot date (YYYY-MM-DD). Default: newest.")
    band_parser.add_argument("--shifts", default=None,
                             help="Shifts as one cluster, e.g. '0,4'. Default: NOW / -6M / FAR, one point each.")
    band_parser.add_argument("--root", default=None, help="Cache root (default: ab_cache/long/).")
    band_parser.add_argument("--recent-days", type=int, default=56,
                             help="Also score the last N days of the newest point (default 56).")
    band_parser.add_argument("--workers", type=int, default=1,
                             help="Parallel processes (variants must use module-level functions).")

    args = parser.parse_args()

    {"fetch": cmd_fetch, "list": cmd_list, "run": cmd_run, "band": cmd_band,
     "vintages": cmd_vintages}[args.command](args)


if __name__ == "__main__":
    main()
