#!/usr/bin/env python3
##
# @file script/prepare.py
# @brief Freeze the aggregated counts into one self-contained training dataset.
#
# This is the last step that touches data/raw/. It joins the aggregated
# per-(user, day) counts to the LDAP org structure and the ground-truth
# labels, applies the train/test split, and writes a single Parquet file plus
# a provenance manifest. From then on train.py and app.py read only that
# dataset -- data/raw/ can be deleted entirely.
#
# Usage:
#   python script/prepare.py                       # full population
#   python script/prepare.py --cohort 400          # reproducible subset
#   python script/prepare.py --train-frac 0.8      # different split
#   python script/prepare.py --from <table>.parquet
##

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR      = os.path.join(ROOT, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
LDAP_DIR      = os.path.join(RAW_DIR, "ldap")
ANS_DIR       = os.path.join(RAW_DIR, "answers")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

USERDAY_TABLE   = os.path.join(ARTIFACTS_DIR, "userday.parquet")
DEFAULT_DATASET = os.path.join(ARTIFACTS_DIR, "dataset.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the aggregated counts into a self-contained dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from", dest="source", default=None, metavar="PATH",
                        help="Aggregated user-day table to build from "
                             "(default: data/artifacts/userday.parquet).")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output .parquet path. Defaults to dataset.parquet, "
                             "or dataset_cohortN_seedS.parquet with --cohort.")
    parser.add_argument("--cohort", type=int, default=None, metavar="N",
                        help="Restrict to a reproducible N-user subset "
                             "(all insiders + role-stratified benign sample).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for cohort sampling (default: 42).")
    parser.add_argument("--train-frac", type=float, default=0.7, metavar="F",
                        help="Fraction of the date range used for training "
                             "(default: 0.7).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the output already exists.")
    args = parser.parse_args()

    import pandas as pd

    from src.colors import blue, green, amber, dim
    from src.progress import header, step, ok, warn, kv, divider, elapsed_str, human_size
    from src.preprocess import load_org_structure, load_ground_truth, load_malicious_days
    from src.prepare import build_prepared, save_prepared, make_manifest
    from src.cohort import build_cohort

    source = args.source or USERDAY_TABLE
    # The laptop path (aggregate.py --cohort N) never builds the full table.
    # Cohort selection is deterministic, so preparing from the cohort's own
    # table gives the same dataset as slicing the full one.
    if args.cohort and args.source is None and not os.path.exists(USERDAY_TABLE):
        cohort_table = os.path.join(ARTIFACTS_DIR,
                                    f"userday_cohort{args.cohort}_seed{args.seed}.parquet")
        if os.path.exists(cohort_table):
            source = cohort_table
    if args.out is None:
        args.out = (os.path.join(ARTIFACTS_DIR,
                                 f"dataset_cohort{args.cohort}_seed{args.seed}.parquet")
                    if args.cohort else DEFAULT_DATASET)

    t0 = time.perf_counter()
    header("COS700 — Dataset Preparation")

    if os.path.exists(args.out) and not args.force:
        warn(f"{os.path.relpath(args.out)} already exists — use --force to rebuild.")
        return 0

    if not os.path.exists(source):
        warn(f"No aggregated table at {os.path.relpath(source)}.")
        print()
        print("  Reduce the raw logs once first:")
        print(f"      {amber('python script/aggregate.py' + (f' --cohort {args.cohort}' if args.cohort else ''))}")
        print()
        return 1

    step(1, "Loading inputs")
    counts = pd.read_parquet(source)
    kv("Aggregated table", os.path.relpath(source))
    kv("User-days", f"{len(counts):,}")

    org = load_org_structure()
    gt  = load_ground_truth()
    days = load_malicious_days()
    kv("LDAP users", f"{len(org):,}")
    kv("Insiders", f"{gt['user'].nunique()}")

    cohort_manifest = None
    if args.cohort:
        step(2, "Selecting user cohort")
        insiders = sorted(gt["user"].astype(str).unique())
        cohort_manifest = build_cohort(org, insiders, n_users=args.cohort,
                                       seed=args.seed)
        keep = set(cohort_manifest["users"])
        counts = counts[counts["user"].isin(keep)].reset_index(drop=True)
        kv("Cohort size",   cohort_manifest["n_users"])
        kv("Insiders kept", f"{cohort_manifest['n_insiders']} (all)")
        kv("Roles covered", f"{cohort_manifest['n_roles_covered']}"
                            f" / {cohort_manifest['n_roles_total']}")
        kv("Seed",          cohort_manifest["seed"])
        kv("User-days kept", f"{len(counts):,}")
    else:
        step(2, "Selecting user cohort")
        kv("Cohort", "full population (no subsetting)")

    step(3, "Attaching context, labels and the train/test split")
    df, stats = build_prepared(counts, org, gt, days, train_frac=args.train_frac)

    for label, value in [
        ("Rows",              f"{stats['n_rows']:,}"),
        ("Users",             f"{stats['n_users']:,}"),
        ("Roles",             stats["n_roles"]),
        ("Date range",        f"{stats['date_min']} .. {stats['date_max']}"),
        ("Split cutoff",      stats["split_cutoff"]),
        ("Train / Test rows", f"{stats['n_train']:,} / {stats['n_test']:,}"),
        ("Malicious rows",    stats["n_malicious"]),
        ("  in train / test", f"{stats['n_malicious_train']} / {stats['n_malicious_test']}"),
        ("Insiders present",  stats["n_insiders"]),
        ("Label source",      stats["label_source"]),
    ]:
        kv(label, value)

    if stats["n_malicious_train"] == 0:
        warn("No malicious rows fall in the training split — the supervised "
             "baseline will have nothing positive to learn from.")

    step(4, "Writing the dataset")
    manifest = make_manifest(
        stats,
        sources={
            "userday_table": source,
            "insiders":      os.path.join(ANS_DIR, "insiders.csv"),
            "malicious_days": os.path.join(ANS_DIR, "malicious_days.csv"),
            "ldap":          os.path.join(LDAP_DIR, sorted(os.listdir(LDAP_DIR))[-1])
                             if os.path.isdir(LDAP_DIR) and os.listdir(LDAP_DIR) else "",
        },
        cohort=cohort_manifest,
    )
    manifest["insider_windows"] = [
        {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}
        for row in gt.to_dict("records")
    ]
    manifest_path = save_prepared(df, manifest, args.out)

    ok(f"dataset  -> {os.path.relpath(args.out)}  "
       f"({human_size(os.path.getsize(args.out))})")
    ok(f"manifest -> {os.path.relpath(manifest_path)}")

    print()
    divider("═", 70)
    print(green("  Dataset ready.") + f"  Total elapsed: {elapsed_str(t0)}")
    divider("─", 70)
    print(f"  Dataset : {blue(os.path.relpath(args.out))}")
    print()
    print(f"  {dim('This file is self-contained — it carries the features, org')}")
    print(f"  {dim('context, labels and the train/test split. Nothing downstream')}")
    print(f"  {dim('reads data/raw/ any more.')}")
    print()
    print(f"  {dim('Next:')}")
    train_cmd = "python train.py" + (
        "" if args.out == DEFAULT_DATASET
        else f" --dataset {os.path.relpath(args.out)}")
    print(f"    {amber(train_cmd)}")
    print()
    divider("═", 70)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
