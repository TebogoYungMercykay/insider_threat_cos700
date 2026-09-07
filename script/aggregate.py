#!/usr/bin/env python3
##
# @file script/aggregate.py
# @brief One-time reduction of the raw CERT logs into a compact per-(user, day)
#        feature table that training can use forever after.
#
# The r6.2 logs are ~100 GB on disk and cannot be loaded into a 32 GB machine.
# Every feature this framework uses is a per-(user, day) aggregate, so the raw
# events only ever need to be read once, sequentially, and reduced. Afterwards
# the raw CSVs are redundant and can be deleted — the ~24 GB archive is enough
# to rebuild the table if the feature definitions ever change.
#
# Usage:
#   python script/aggregate.py                      # full population
#   python script/aggregate.py --cohort 400         # 400-user stratified subset
#   python script/aggregate.py --cohort 400 --seed 7
#   python script/aggregate.py --from-archive       # stream from r6.2.tar.bz2
#   python script/aggregate.py --workers 8
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
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
ARCHIVE       = os.path.join(DATA_DIR, "r6.2.tar.bz2")

DEFAULT_TABLE  = os.path.join(ARTIFACTS_DIR, "userday.parquet")
COHORT_MANIFEST = os.path.join(ARTIFACTS_DIR, "cohort.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reduce the raw CERT logs to a per-(user, day) feature table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cohort", type=int, default=None, metavar="N",
                        help="Restrict to a reproducible N-user subset "
                             "(all insiders + role-stratified benign sample). "
                             "Default: every user.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for cohort sampling (default: 42).")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output .parquet path. Defaults to "
                             "userday.parquet, or userday_cohortN_seedS.parquet "
                             "when --cohort is given, so a cohort run never "
                             "overwrites the full-population table.")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="Worker processes (default: CPU count, max 12).")
    parser.add_argument("--rescan", action="store_true",
                        help="Force a fresh pass over the raw logs even when a "
                             "full-population table already exists to slice a "
                             "cohort out of.")
    parser.add_argument("--from-archive", action="store_true",
                        help="Stream from r6.2.tar.bz2 instead of the extracted "
                             "CSVs (slower, but needs no extracted copy on disk).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the output already exists.")
    args = parser.parse_args()

    if args.out is None:
        args.out = (os.path.join(ARTIFACTS_DIR,
                                 f"userday_cohort{args.cohort}_seed{args.seed}.parquet")
                    if args.cohort else DEFAULT_TABLE)
    cohort_manifest_path = (
        os.path.splitext(args.out)[0] + ".cohort.json" if args.cohort else COHORT_MANIFEST)

    from src.colors import blue, green, amber, dim
    from src.progress import header, step, ok, warn, kv, divider, elapsed_str
    from src.aggregate import build_userday_table, save_userday_table
    from src.cohort import build_cohort, save_cohort

    t0 = time.perf_counter()
    header("COS700 — Raw Log Aggregation")

    if os.path.exists(args.out) and not args.force:
        warn(f"{os.path.relpath(args.out)} already exists — use --force to rebuild.")
        return 0

    cohort_users = None
    if args.cohort:
        step(1, "Selecting user cohort")
        import pandas as pd
        from src.preprocess import load_org_structure

        org = load_org_structure()
        insiders = sorted(pd.read_csv(
            os.path.join(RAW_DIR, "answers", "insiders.csv"))["user"].astype(str).unique())
        manifest = build_cohort(org, insiders, n_users=args.cohort, seed=args.seed)
        save_cohort(manifest, cohort_manifest_path)
        cohort_users = manifest["users"]

        kv("Cohort size",   manifest["n_users"])
        kv("Insiders kept", f"{manifest['n_insiders']} / {manifest['n_insiders']} (all)")
        kv("Benign sample", manifest["n_benign"])
        kv("Roles covered", f"{manifest['n_roles_covered']} / {manifest['n_roles_total']}")
        kv("Seed",          manifest["seed"])
        ok(f"cohort manifest -> {os.path.relpath(cohort_manifest_path)}")
    else:
        step(1, "Selecting user cohort")
        kv("Cohort", "full population (no subsetting)")

    # A cohort is a subset of rows, so if the full-population table has
    # already been built there is nothing to learn from re-reading 93 GB of
    # raw logs -- slicing the existing table gives a bit-identical result in
    # about a second. Verified in tests/test_pipeline.py.
    can_slice = (cohort_users is not None
                 and not args.rescan
                 and not args.from_archive
                 and os.path.exists(DEFAULT_TABLE)
                 and os.path.abspath(args.out) != os.path.abspath(DEFAULT_TABLE))

    if can_slice:
        step(2, "Slicing the cohort out of the full-population table")
        import pandas as pd

        full = pd.read_parquet(DEFAULT_TABLE)
        df = full[full["user"].isin(set(cohort_users))].reset_index(drop=True)
        kv("Source", os.path.relpath(DEFAULT_TABLE))
        kv("Rows", f"{len(df):,} of {len(full):,}")
        found = df["user"].nunique()
        if found < len(cohort_users):
            warn(f"{len(cohort_users) - found} cohort user(s) have no activity "
                 f"in the aggregated table.")
        ok("cohort sliced without re-reading the raw logs")
    else:
        step(2, "Streaming raw logs into per-(user, day) aggregates")
        df = build_userday_table(
            raw_dir=RAW_DIR,
            archive=ARCHIVE,
            cohort=cohort_users,
            workers=args.workers,
            prefer_archive=args.from_archive,
        )

    step(3, "Writing the aggregated table")
    save_userday_table(df, args.out)

    kv("Users",     f"{df['user'].nunique():,}")
    kv("User-days", f"{len(df):,}")
    kv("Date range", f"{df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")

    print()
    divider("═", 70)
    print(green("  Aggregation complete.") + f"  Total elapsed: {elapsed_str(t0)}")
    divider("─", 70)
    print(f"  Table : {blue(os.path.relpath(args.out))}")
    print()
    print(f"  {dim('Next — attach context, labels and the train/test split:')}")
    prepare_cmd = "python script/prepare.py" + (
        "" if args.out == DEFAULT_TABLE else f" --from {os.path.relpath(args.out)}")
    print(f"    {amber(prepare_cmd)}")
    print()
    print(f"  {dim('The five raw CSVs are now redundant. To reclaim ~93 GB:')}")
    print(f"    {dim('rm data/raw/{http,email,file,device,logon}.csv')}")
    print(f"  {dim('Keep data/raw/ldap/ and data/raw/answers/ — prepare.py reads them.')}")
    print(f"  {dim('Rebuild later from the archive with:')}")
    print(f"    {dim('python script/aggregate.py --from-archive --force')}")
    divider("═", 70)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
