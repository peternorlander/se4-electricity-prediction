"""
CLI for the A/B backtest flow. See the README "A/B Backtest Flow" section for
the full design and the agent playbook.

    python ab_test.py fetch                        # fetch + cache today's training inputs
    python ab_test.py fetch --days 1825             # long-window fetch (~5y), auto-routed
                                                     # to a separate cache root -- see below
    python ab_test.py list                          # list cached snapshots
    python ab_test.py list --root ab_cache/long      # list long-window snapshots
    python ab_test.py run [--snapshot YYYY-MM-DD] [--shifts 0-5] [--root ab_cache/long]

Needs ENTSO_E_TOKEN (and the other source-API env vars) only for `fetch`;
`run` and `list` are offline once a snapshot exists.

LONG-WINDOW SNAPSHOTS (added 2026-08-06 for Round 15, IMPROVEMENT_PLAN.md).
`--days` overrides fetch_data.TRAINING_DAYS (normally 1095, ~3 years) for one
fetch -- same fetch_training_inputs() call, same single contiguous range up to
today, just with the start pushed further back. A run with --days longer than
the normal window is auto-routed to ab.snapshot.LONG_CACHE_ROOT
(ab_cache/long/) unless --root is given explicitly, so it can never collide
with -- or be silently picked up as "newest" by -- a routine `ab_test.py run`
against the normal 3-year snapshots that every grid script assumes. Every
fetch (long or normal) rebuilds the merged frame and checks it for date gaps
before finishing: the grid scripts equate rows with consecutive days (shift
arithmetic, end-date dedup), so a single missing source day anywhere in the
window would otherwise skew every downstream measurement silently. The
snapshot is still saved if a gap is found -- the data remains useful -- but the
gap is printed loudly so it gets investigated before being built into a grid.
"""
import argparse
import sys
from datetime import datetime, UTC
from pathlib import Path

from ab.snapshot import save_snapshot, list_snapshots, load_snapshot, DEFAULT_CACHE_ROOT, LONG_CACHE_ROOT
from ab.harness import parse_shift_range
from ab.verdict import run_ab


def _check_date_continuity(inputs: dict) -> list:
    """
    Rebuild the merged frame and return [(before, after, n_missing_days), ...]
    for every non-consecutive gap. Cheap (local pandas, no network) relative to
    the fetch it follows.
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
    return gaps


def cmd_fetch(args):
    import fetch_data

    default_days = fetch_data.TRAINING_DAYS
    is_long = args.days is not None and args.days > default_days
    if args.days is not None:
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
    gaps = _check_date_continuity(inputs)
    if gaps:
        print(f"\nWARNING -- {len(gaps)} gap(s) in the merged frame:")
        for before, after, missing in gaps:
            print(f"  {before} -> {after}  ({missing} missing day(s))")
        print("\nSnapshot is saved, but grid scripts equate rows with consecutive")
        print("days -- do not build a shift grid on this snapshot until the gap(s)")
        print("are understood.")
    else:
        print("  OK -- no date gaps.")

    if is_long and not args.root:
        print(f"\nThis is a long-window snapshot ({args.days} days > default {default_days}) "
              f"-- auto-routed to {LONG_CACHE_ROOT}/, kept separate from the normal "
              f"snapshots so it can't be picked up as \"newest\" by a routine run.")
        print(f"Load it with: python ab_test.py run --snapshot {today} --root {LONG_CACHE_ROOT}")


def cmd_list(args):
    cache_root = Path(args.root) if args.root else DEFAULT_CACHE_ROOT
    snapshots = list_snapshots(cache_root=cache_root)
    if not snapshots:
        print(f"No snapshots found under {cache_root}/. Run 'python ab_test.py fetch' first.")
        if cache_root == DEFAULT_CACHE_ROOT and LONG_CACHE_ROOT.exists():
            print(f"(Long-window snapshots exist under {LONG_CACHE_ROOT}/ -- "
                  f"list them with: python ab_test.py list --root {LONG_CACHE_ROOT})")
        return
    print(f"Available snapshots under {cache_root}/:")
    for name in snapshots:
        print(f"  {name}")


def cmd_run(args):
    cache_root = Path(args.root) if args.root else DEFAULT_CACHE_ROOT
    inputs, meta = load_snapshot(args.snapshot, cache_root=cache_root)
    print(f"Using snapshot: {meta['today']} (fetched {meta['fetched_at']}, root={cache_root})")
    shifts = parse_shift_range(args.shifts)
    run_ab(inputs, shifts)


def main():
    parser = argparse.ArgumentParser(description="A/B backtest flow for SE4 price prediction models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and cache today's training inputs.")
    fetch_parser.add_argument("--days", type=int, default=None,
                               help="Override the training window length in days "
                                    "(default: fetch_data.TRAINING_DAYS, ~3 years). "
                                    "A value longer than the default auto-routes to "
                                    "ab_cache/long/ unless --root is given.")
    fetch_parser.add_argument("--root", default=None,
                               help="Cache root to save under (default: ab_cache/, "
                                    "or ab_cache/long/ for a long --days fetch).")

    list_parser = subparsers.add_parser("list", help="List cached snapshots.")
    list_parser.add_argument("--root", default=None,
                              help="Cache root to list (default: ab_cache/).")

    run_parser = subparsers.add_parser("run", help="Run the BASELINE vs CANDIDATE A/B across simulated shifts.")
    run_parser.add_argument("--snapshot", default=None, help="Snapshot date (YYYY-MM-DD). Default: newest.")
    run_parser.add_argument("--shifts", default="0-5", help="Shift range, e.g. '0-5' or '0,2,4'. Default: 0-5.")
    run_parser.add_argument("--root", default=None,
                             help="Cache root to load from (default: ab_cache/).")

    args = parser.parse_args()

    {"fetch": cmd_fetch, "list": cmd_list, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
