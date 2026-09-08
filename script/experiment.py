#!/usr/bin/env python3
##
# @file script/experiment.py
# @brief Run the controlled comparison experiment and write the report.
#
# This is the script that produces the numbers the dissertation reports. It
# fits every detector in the chosen preset on the dataset's frozen training
# split, scores the full test split with each, and writes a comparison table,
# an alert-budget sweep, a per-role false-positive breakdown and a run
# manifest recording exactly what was run.
#
# Usage:
#   python script/experiment.py                       # core comparison
#   python script/experiment.py --preset full         # every detector
#   python script/experiment.py --preset ablation     # scope x statistic x algorithm
#   python script/experiment.py --temporal            # add rolling-window features
#   python script/experiment.py --dataset data/artifacts/dataset_cohort400_seed42.parquet
##

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.colors import amber, blue, dim, green
from src.detectors import DEFAULT_FIT_SUBSAMPLE, PRESETS
from src.evaluate import BUDGET_SWEEP, DEFAULT_ALERT_BUDGET
from src.experiment import headline_table, run_comparison, write_report, _md_table
from src.progress import divider, elapsed_str, header, kv, ok, step

DEFAULT_DATASET = os.path.join(ROOT, "data", "artifacts", "dataset.parquet")
DEFAULT_OUT_DIR = os.path.join(ROOT, "results")


def main() -> int:
    """@brief Parse arguments, run the comparison, write and print the report."""
    parser = argparse.ArgumentParser(
        description="Run the population-vs-per-user comparison experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, metavar="PATH",
                        help="Prepared dataset to evaluate on.")
    parser.add_argument("--preset", choices=PRESETS, default="core",
                        help="Which detectors to compare (default: core).")
    parser.add_argument("--temporal", action="store_true",
                        help="Add causal rolling-window features for every detector.")
    parser.add_argument("--budget", type=float, default=DEFAULT_ALERT_BUDGET,
                        metavar="F",
                        help="Alert budget: fraction of test rows each detector "
                             "may flag (default: %(default)s).")
    parser.add_argument("--contamination", type=float, default=0.02,
                        help="Expected anomalous proportion for the anomaly models.")
    parser.add_argument("--min-history-days", type=int, default=14,
                        help="Cold-start threshold for the per-user framework.")
    parser.add_argument("--fit-subsample", type=int, default=DEFAULT_FIT_SUBSAMPLE,
                        metavar="N",
                        help="Training-row cap for LOF / One-Class SVM "
                             "(0 disables; default: %(default)s).")
    parser.add_argument("--max-users", type=int, default=None, metavar="N",
                        help="Limit to the first N users, for smoke runs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, metavar="DIR",
                        help="Directory for the report (default: results/).")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"\n  No prepared dataset at {os.path.relpath(args.dataset, ROOT)}.")
        print(f"  Build it with:  {amber('python script/prepare.py')}\n")
        return 1

    t0 = time.perf_counter()
    header("COS700 — Detector Comparison Experiment")

    step(1, "Configuration")
    kv("Dataset",        os.path.relpath(args.dataset, ROOT))
    kv("Preset",         args.preset)
    kv("Features",       "base + rolling windows" if args.temporal else "base")
    kv("Alert budget",   f"{args.budget:.5f}  ({args.budget * 100:.3f}% of test rows)")
    kv("Fit subsample",  args.fit_subsample or "none")
    kv("Seed",           args.seed)

    step(2, "Running detectors")

    def on_detector(i, total, detector):
        print(f"  {dim(f'[{i}/{total}]')} {blue(detector.name)}  "
              f"{dim('(' + detector.family + ')')}")

    outcome = run_comparison(
        dataset_path=args.dataset,
        preset=args.preset,
        temporal=args.temporal,
        budget=args.budget,
        budgets=tuple(sorted(set(BUDGET_SWEEP) | {args.budget})),
        contamination=args.contamination,
        min_history_days=args.min_history_days,
        fit_subsample=args.fit_subsample or None,
        max_users=args.max_users,
        random_state=args.seed,
        on_detector=on_detector,
    )

    step(3, "Writing the report")
    out_dir = os.path.join(args.out, f"{args.preset}"
                                     + ("-temporal" if args.temporal else ""))
    paths = write_report(outcome, out_dir)
    for label, path in paths.items():
        ok(f"{label:12s} -> {os.path.relpath(path, ROOT)}")

    m = outcome["manifest"]
    print()
    divider("═", 70)
    print(green("  Comparison complete.") + f"  Total elapsed: {elapsed_str(t0)}")
    divider("─", 70)
    print(f"  Test rows        : {m['n_test_rows']:,}")
    print(f"  Malicious in test: {m['n_malicious_test']}")
    print(f"  Alert budget     : top {m['alert_budget'] * 100:.3f}% "
          f"(~{int(m['n_test_rows'] * m['alert_budget']):,} alerts)")
    print()
    print(_md_table(headline_table(outcome["results"])))
    print()
    print(f"  {dim('Full report:')} {blue(os.path.relpath(paths['report_md'], ROOT))}")
    divider("═", 70)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
