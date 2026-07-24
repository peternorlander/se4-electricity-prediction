"""
CLI for the A/B backtest flow. See the README "A/B Backtest Flow" section for
the full design and the agent playbook.

    python ab_test.py fetch                        # fetch + cache today's training inputs
    python ab_test.py list                          # list cached snapshots
    python ab_test.py run [--snapshot YYYY-MM-DD] [--shifts 0-5]

Needs ENTSO_E_TOKEN (and the other source-API env vars) only for `fetch`;
`run` and `list` are offline once a snapshot exists.
"""
import argparse
import sys
from datetime import datetime, UTC

from ab.snapshot import save_snapshot, list_snapshots, load_snapshot
from ab.harness import parse_shift_range
from ab.verdict import run_ab


def cmd_fetch(args):
    from fetch_data import fetch_training_inputs

    today = datetime.now(UTC).date()
    print(f"Fetching training inputs for {today}...")
    inputs = fetch_training_inputs(today)
    snapshot_dir = save_snapshot(inputs, today)
    print(f"\nSaved snapshot: {snapshot_dir}")


def cmd_list(args):
    snapshots = list_snapshots()
    if not snapshots:
        print("No snapshots found. Run 'python ab_test.py fetch' first.")
        return
    print("Available snapshots:")
    for name in snapshots:
        print(f"  {name}")


def cmd_run(args):
    inputs, meta = load_snapshot(args.snapshot)
    print(f"Using snapshot: {meta['today']} (fetched {meta['fetched_at']})")
    shifts = parse_shift_range(args.shifts)
    run_ab(inputs, shifts)


def main():
    parser = argparse.ArgumentParser(description="A/B backtest flow for SE4 price prediction models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch", help="Fetch and cache today's training inputs.")
    subparsers.add_parser("list", help="List cached snapshots.")

    run_parser = subparsers.add_parser("run", help="Run the BASELINE vs CANDIDATE A/B across simulated shifts.")
    run_parser.add_argument("--snapshot", default=None, help="Snapshot date (YYYY-MM-DD). Default: newest.")
    run_parser.add_argument("--shifts", default="0-5", help="Shift range, e.g. '0-5' or '0,2,4'. Default: 0-5.")

    args = parser.parse_args()

    {"fetch": cmd_fetch, "list": cmd_list, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
