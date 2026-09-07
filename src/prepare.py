##
# @file src/prepare.py
# @brief Freeze the aggregated user-day counts into a single self-contained
#        training dataset.
#
# WHY THIS EXISTS
# ---------------
# src/aggregate.py answers "what did each user do each day?" by reducing the
# ~93 GB of raw logs to a compact count table. That table is still only half a
# dataset: it carries no org context, no labels, and no train/test split, so
# every training run had to reach back into data/raw/ldap/ and
# data/raw/answers/ and re-derive them.
#
# This module performs that derivation once and writes the result as one
# Parquet file plus one JSON manifest. Afterwards nothing in the training or
# evaluation path reads data/raw/ at all -- the dataset is a standalone
# artifact that can be copied, archived alongside a dissertation, or handed to
# someone else to reproduce a result exactly.
#
# The manifest records where every part came from (source table, LDAP
# snapshots, ground-truth files, cohort, seed, split cutoff) together with the
# content hashes of those inputs, so a frozen dataset can always be traced
# back to the release it was built from.
##

import hashlib
import json
import os

import numpy as np
import pandas as pd

from src.schema import FEATURE_COLS
from src.progress import substep

## Columns carried by a prepared dataset, beyond the features themselves.
CONTEXT_COLS = ["role", "functional_unit", "department", "supervisor",
                "is_high_privilege"]
LABEL_COLS   = ["is_malicious"]
SPLIT_COLS   = ["split"]

DATASET_COLUMNS = ["user", "date"] + FEATURE_COLS + CONTEXT_COLS + LABEL_COLS + SPLIT_COLS

## Bumped whenever the on-disk layout changes in a way that older readers
#  cannot interpret.
DATASET_FORMAT_VERSION = 1


def _digest(path: str, limit: int = 64 * 1024 * 1024) -> str:
    """
    @brief Short content hash of a file, for provenance.

    Hashes at most `limit` bytes: enough to detect an input being swapped or
    regenerated, without re-reading a large table in full.

    @param path   File to hash.
    @param limit  Maximum bytes to read.
    @return       Truncated hex digest, or "missing".
    """
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def time_split_cutoff(dates: pd.Series, train_frac: float) -> pd.Timestamp:
    """
    @brief Date at which the training window ends and the test window begins.

    The cutoff is taken over the *distinct* days present, not over rows, so
    that a handful of very active users cannot drag the boundary.

    @param dates       All observation dates.
    @param train_frac  Fraction of the date range assigned to training.
    @return            First date belonging to the test split.
    """
    unique = np.sort(pd.to_datetime(dates).unique())
    index = min(int(len(unique) * train_frac), len(unique) - 1)
    return pd.Timestamp(unique[index])


## Role names that denote elevated privilege. Matched by substring rather
#  than an exact list: r6.2's LDAP writes "ITAdmin", while the synthetic
#  generator writes "SysAdmin"/"DBAdmin", and other CERT releases use
#  spaced forms like "Sys Admin".
HIGH_PRIVILEGE_PATTERN = r"sys.?admin|db.?admin|database.?admin|it.?admin|administrator"


def attach_org_context(df: pd.DataFrame, org_df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Join role and org-structure columns onto a per-(user, day) table.

    @param df      Per-(user, day) rows with a 'user' column.
    @param org_df  Collapsed LDAP org structure.
    @return        df with role, functional_unit, department, supervisor and
                   is_high_privilege appended.
    """
    out = df.merge(org_df, on="user", how="left")
    out["role"] = out["role"].fillna("Employee")
    out["functional_unit"] = out["functional_unit"].fillna("Operations")
    for col in ("department", "supervisor"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    out["is_high_privilege"] = out["role"].str.contains(
        HIGH_PRIVILEGE_PATTERN, case=False, regex=True, na=False)
    return out


def attach_labels(df: pd.DataFrame, gt_df: pd.DataFrame,
                  malicious_days: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """
    @brief Add the is_malicious label to a per-(user, day) table.

    Prefers the exact event-derived day labels. Filling in each insider's
    [start_date, end_date] window instead marks 131 user-days malicious where
    only 44 carry malicious activity, because scenarios 2 and 4 unfold in
    bursts over one to two months.

    @param df              Per-(user, day) rows.
    @param gt_df           Insider window index, used only as a fallback.
    @param malicious_days  Exact malicious (user, day) labels, or None.
    @return                (df with is_malicious, description of the source).
    """
    out = df.copy()
    day = out["date"].dt.floor("D")

    if malicious_days is not None and len(malicious_days):
        labels = set(zip(malicious_days["user"].astype(str),
                         pd.to_datetime(malicious_days["date"]).dt.floor("D")))
        out["is_malicious"] = [(u, d) in labels
                               for u, d in zip(out["user"].astype(str), day)]
        return out, "exact malicious-day labels"

    out["is_malicious"] = False
    for _, row in gt_df.iterrows():
        out.loc[(out["user"] == row["user"])
                & (out["date"] >= row["start_date"])
                & (out["date"] <= row["end_date"]), "is_malicious"] = True
    return out, "insider windows (approximate)"


def build_prepared(userday_df: pd.DataFrame, org_df: pd.DataFrame,
                   gt_df: pd.DataFrame, malicious_days: pd.DataFrame | None,
                   train_frac: float = 0.7) -> tuple[pd.DataFrame, dict]:
    """
    @brief Attach org context, labels and a time-based split to the counts.

    @param userday_df      Aggregated per-(user, day) counts.
    @param org_df          Collapsed LDAP org structure.
    @param gt_df           Insider window index (one row per insider).
    @param malicious_days  Exact malicious (user, day) labels, or None to fall
                           back to filling in each insider's window.
    @param train_frac      Fraction of the date range assigned to training.
    @return                (dataset, stats) ready to be written.
    """
    df = attach_org_context(userday_df, org_df)
    df, label_source = attach_labels(df, gt_df, malicious_days)

    cutoff = time_split_cutoff(df["date"], train_frac)
    df["split"] = np.where(df["date"] < cutoff, "train", "test")

    df = df[DATASET_COLUMNS].sort_values(["user", "date"], ignore_index=True)

    train_mask = df["split"] == "train"
    stats = {
        "n_rows":            int(len(df)),
        "n_users":           int(df["user"].nunique()),
        "n_roles":           int(df["role"].nunique()),
        "date_min":          str(df["date"].min().date()),
        "date_max":          str(df["date"].max().date()),
        "split_cutoff":      str(cutoff.date()),
        "train_frac":        train_frac,
        "n_train":           int(train_mask.sum()),
        "n_test":            int((~train_mask).sum()),
        "n_malicious":       int(df["is_malicious"].sum()),
        "n_malicious_train": int((df["is_malicious"] & train_mask).sum()),
        "n_malicious_test":  int((df["is_malicious"] & ~train_mask).sum()),
        "n_insiders":        int(df.loc[df["is_malicious"], "user"].nunique()),
        "label_source":      label_source,
    }
    return df, stats


def save_prepared(df: pd.DataFrame, manifest: dict, path: str) -> str:
    """
    @brief Write a prepared dataset and its manifest.

    @param df        Prepared dataset.
    @param manifest  Provenance and summary statistics.
    @param path      Destination .parquet path; the manifest goes alongside it.
    @return          Path to the manifest.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = df.copy()
    for col in ["user", "role", "functional_unit", "department", "supervisor", "split"]:
        out[col] = out[col].astype("category")     # dictionary-encoded in Parquet
    out.to_parquet(path, index=False, compression="zstd")

    manifest_path = manifest_path_for(path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def manifest_path_for(dataset_path: str) -> str:
    """@brief Manifest path that pairs with a dataset path."""
    return os.path.splitext(dataset_path)[0] + ".meta.json"


def load_prepared(path: str) -> tuple[pd.DataFrame, dict]:
    """
    @brief Load a prepared dataset and its manifest.

    This is the only load the training pipeline needs: the returned frame
    already carries context, labels and the split, so nothing reads data/raw/.

    @param path  Path to the dataset .parquet.
    @return      (dataset, manifest). The manifest is {} if absent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No prepared dataset at {path}.\n"
            f"Build it with:  python script/prepare.py"
        )
    with substep("loading prepared dataset") as s:
        df = pd.read_parquet(path)
        for col in ["user", "role", "functional_unit", "department",
                    "supervisor", "split"]:
            if col in df.columns and str(df[col].dtype) == "category":
                df[col] = df[col].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        s.set_detail(f"{len(df):,} rows, {df['user'].nunique():,} users")

    manifest = {}
    mpath = manifest_path_for(path)
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    return df, manifest


def ground_truth_from_manifest(manifest: dict) -> pd.DataFrame:
    """
    @brief Rebuild the insider window table from a dataset manifest.

    The five insider windows are embedded in the manifest so that evaluation
    keeps working without data/raw/answers/insiders.csv.

    @param manifest  Manifest from load_prepared().
    @return          DataFrame of insider windows (possibly empty).
    """
    windows = manifest.get("insider_windows", [])
    gt = pd.DataFrame(windows)
    for col in ("start_date", "end_date"):
        if col in gt.columns:
            gt[col] = pd.to_datetime(gt[col])
    return gt


def make_manifest(stats: dict, sources: dict, cohort: dict | None) -> dict:
    """
    @brief Assemble the provenance manifest written next to a dataset.

    @param stats    Summary statistics from build_prepared().
    @param sources  {label: path} of every input that shaped the dataset.
    @param cohort   Cohort manifest, or None for the full population.
    @return         Manifest dict.
    """
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "created_at":     pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_cols":   list(FEATURE_COLS),
        "columns":        list(DATASET_COLUMNS),
        **stats,
        "cohort": None if cohort is None else {
            k: cohort[k] for k in
            ("n_users", "n_insiders", "n_benign", "seed", "strategy",
             "n_roles_covered", "n_roles_total")
            if k in cohort
        },
        "cohort_users": None if cohort is None else cohort.get("users"),
        "sources": {
            label: {"path": os.path.basename(path), "sha256_16": _digest(path)}
            for label, path in sources.items()
        },
    }
