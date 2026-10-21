##
# @file src/preprocess.py
# @brief Log ingestion and behavioural feature engineering for insider threat detection.
#
# Loads the CERT-schema raw logs from data/raw/ (produced either by the real
# CERT r6.2 dataset or by script/dataset.py's synthetic generator),
# aggregates them into one row per (user, day) with behavioural count
# features, joins role/org-structure context from the LDAP snapshots, and
# labels rows against the ground-truth insider window index.
#
# Run script/dataset.py once before using this module.
##

import os

import numpy as np
import pandas as pd

from src.progress import substep, Progress, warn
from src.schema import FEATURE_COLS as _FEATURE_COLS
from src.schema import netloc_of
from src.aggregate import _has_external_recipient, _count_attachments, detect_org_domain


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
LDAP_DIR = os.path.join(RAW_DIR, "ldap")
ANS_DIR  = os.path.join(RAW_DIR, "answers")

LOG_FILES = ["logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"]

ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

## Default location of the aggregated per-(user, day) table produced by
#  script/aggregate.py. This, not the raw CSVs, is what training reads.
USERDAY_TABLE = os.path.join(ARTIFACTS_DIR, "userday.parquet")

# The feature set is declared in src/schema.py so that the streaming
# aggregator and this module cannot drift apart. Re-exported here because
# baseline.py, evaluate.py and app.py already import it from this module.
FEATURE_COLS = list(_FEATURE_COLS)


def _org_domain_from_frame(email_df: pd.DataFrame) -> str:
    """
    @brief Infer the organisation's email domain from a log's own senders.

    The modal sender domain is the organisation's: employees send most of
    the mail. Falls back to the LDAP-based detector when there is no usable
    `from` column.

    @param email_df  Email log rows.
    @return          The organisation's email domain, lower-cased.
    """
    if "from" in email_df.columns:
        domains = (email_df["from"].dropna().astype(str)
                   .str.rsplit("@", n=1).str[-1].str.strip().str.lower())
        domains = domains[domains != ""]
        if len(domains):
            return domains.mode().iloc[0]
    return detect_org_domain(RAW_DIR)


def _as_bool(series: pd.Series) -> pd.Series:
    """
    @brief Coerce a truthy column to booleans across both dataset flavours.

    r6.2 stores "True"/"False" strings; the synthetic generator stores 1/0.

    @param series  Raw column.
    @return        Boolean Series.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _require_raw_data() -> None:
    """@brief Raise FileNotFoundError with setup instructions if raw logs are missing."""
    missing = [f for f in LOG_FILES if not os.path.exists(os.path.join(RAW_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing raw log files: {missing}\n"
            f"Run:  python script/dataset.py\n"
            f"Expected directory: {RAW_DIR}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# Header-name aliases tolerated when reading LDAP snapshots. The synthetic
# generator writes the canonical names on the left; the official CERT
# release (and some third-party re-packagings) use variants of these, and
# also carries extra columns (project, business_unit, team, email) that
# this pipeline doesn't use and simply ignores.
_LDAP_COLUMN_ALIASES = {
    "user":            ["user", "user_id", "userid", "id"],
    "role":            ["role", "position", "title", "job_title", "designation"],
    "functional_unit": ["functional_unit", "functional unit", "unit"],
    "department":      ["department", "dept", "business_unit", "business unit"],
    "supervisor":      ["supervisor", "manager", "supervisor_id"],
}


def _resolve_ldap_columns(columns: list) -> dict:
    """
    @brief Map this snapshot's actual (possibly differently-named/cased)
           columns onto the canonical names this pipeline expects.

    @param columns  The DataFrame's actual column names.
    @return         Dict {canonical_name: actual_column_name} for every
                     canonical field that could be resolved.
    """
    lower_map = {c.lower().strip(): c for c in columns}
    resolved = {}
    for canonical, aliases in _LDAP_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                resolved[canonical] = lower_map[alias]
                break
    return resolved


def load_org_structure() -> pd.DataFrame:
    """
    @brief Load and collapse monthly LDAP snapshots into one row per user.

    Users occasionally change role across snapshots (promotions, transfers);
    the most recent snapshot for each user is kept. Column names are
    resolved via a small alias table (see _LDAP_COLUMN_ALIASES) so this
    works against both the synthetic generator's output and the official
    CERT release's LDAP snapshot schema without modification.

    @return  DataFrame: user, role, functional_unit, department, supervisor.
    """
    files = sorted(os.path.join(LDAP_DIR, f) for f in os.listdir(LDAP_DIR) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No LDAP snapshots found in {LDAP_DIR}")

    frames = []
    with substep(f"loading {len(files)} LDAP snapshot(s)") as s:
        prog = Progress("ldap snapshots", len(files), unit="items")
        for path in files:
            month = os.path.basename(path).replace(".csv", "")
            df = pd.read_csv(path)
            colmap = _resolve_ldap_columns(list(df.columns))
            if "user" not in colmap:
                raise ValueError(
                    f"Could not identify a user-id column in {path} "
                    f"(columns found: {list(df.columns)})"
                )
            out = pd.DataFrame({"user": df[colmap["user"]].astype(str)})
            for field in ["role", "functional_unit", "department", "supervisor"]:
                out[field] = df[colmap[field]] if field in colmap else np.nan
            out["snapshot_month"] = month
            frames.append(out)
            prog.update(1)
        prog.done()
        s.set_detail(f"{len(files)} files")

    ldap = pd.concat(frames, ignore_index=True)
    ldap = ldap.sort_values("snapshot_month").drop_duplicates("user", keep="last")
    return ldap[["user", "role", "functional_unit", "department", "supervisor"]]


def load_ground_truth() -> pd.DataFrame:
    """
    @brief Load the simplified ground-truth insider-window index.

    Ground truth is expressed as one row per labelled malicious window:
    a user together with the [start_date, end_date] range during which
    their activity is scenario-malicious. This is a simplification of the
    official CERT per-event answer-key format (which marks individual row
    ids across the five log files); it is sufficient for day-granularity
    evaluation and is easy to hand-author against real CERT answer files
    if finer-grained ground truth is later required.

    @return  DataFrame: user, scenario, threat_type, start_date, end_date.
    """
    path = os.path.join(ANS_DIR, "insiders.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    with substep("loading ground-truth insider windows") as s:
        gt = pd.read_csv(path, parse_dates=["start_date", "end_date"])
        s.set_detail(f"{len(gt):,} windows, {gt['user'].nunique()} insider(s)")
    return gt


def load_malicious_days() -> pd.DataFrame | None:
    """
    @brief Load the exact set of malicious (user, day) labels, if present.

    This is the precise label set, derived from the per-event answer key by
    script/dataset.py: a user-day is malicious only if a labelled malicious
    event actually occurred on it. It is strictly better than filling in
    each insider's [start_date, end_date] window, because scenarios 2 and 4
    unfold in short bursts spread over one to two months, window-filling
    marks 131 user-days malicious where only 44 carry malicious activity,
    so two thirds of the "positives" would be ordinary behaviour.

    @return  DataFrame of user, date (and scenario), or None if the file
             predates this labelling and only windows are available.
    """
    path = os.path.join(ANS_DIR, "malicious_days.csv")
    if not os.path.exists(path):
        return None
    with substep("loading exact malicious-day labels") as s:
        days = pd.read_csv(path, parse_dates=["date"])
        s.set_detail(f"{len(days):,} malicious user-days")
    return days


## Refuse to read a raw log larger than this into memory. The real r6.2
#  http.csv is ~90 GB; pandas would need many times that in RAM, which is
#  exactly the failure this pipeline is designed to avoid.
MAX_INMEMORY_LOG_BYTES = 2 * 1024 ** 3


def load_daily_features(path: str | None = None) -> pd.DataFrame:
    """
    @brief Load the pre-aggregated per-(user, day) feature table.

    This is the normal entry point for real data. The table is produced once
    by script/aggregate.py, which streams the raw logs and reduces them
    ~1000:1; everything downstream works from the compact result.

    @param path  Parquet path (default: data/artifacts/userday.parquet).
    @return      DataFrame of user, date, and FEATURE_COLS.
    """
    path = path or USERDAY_TABLE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No aggregated user-day table at {path}.\n"
            f"Build it once (reads the raw logs in a single streaming pass):\n"
            f"    python script/aggregate.py"
        )
    with substep("loading aggregated user-day table") as s:
        df = pd.read_parquet(path)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df["date"] = pd.to_datetime(df["date"])
        s.set_detail(f"{len(df):,} user-days, {df['user'].nunique():,} users")
    return df[["user", "date"] + FEATURE_COLS]


def load_raw_logs() -> dict:
    """
    @brief Load the five raw CERT-schema event logs into memory.

    Only viable for the synthetic generator's output or a small hand-made
    sample. For the real r6.2 release use script/aggregate.py and
    load_daily_features() instead. See docs/data-pipeline.md.

    @return  Dict of DataFrames keyed by "logon", "device", "file", "http", "email".
    """
    _require_raw_data()

    oversized = [
        (f, os.path.getsize(os.path.join(RAW_DIR, f)))
        for f in LOG_FILES
        if os.path.getsize(os.path.join(RAW_DIR, f)) > MAX_INMEMORY_LOG_BYTES
    ]
    if oversized:
        detail = ", ".join(f"{f} ({size / 1024**3:.1f} GB)" for f, size in oversized)
        raise MemoryError(
            f"Refusing to load these raw logs into memory: {detail}.\n"
            f"They exceed the {MAX_INMEMORY_LOG_BYTES / 1024**3:.0f} GB in-memory "
            f"limit and would exhaust RAM.\n"
            f"Use the streaming aggregator instead:\n"
            f"    python script/aggregate.py\n"
            f"then train as usual, train.py reads the aggregated table."
        )

    logs = {}
    for name in ["logon", "device", "file", "http", "email"]:
        with substep(f"loading {name}.csv") as s:
            df = pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"), parse_dates=["date"])
            logs[name] = df
            s.set_detail(f"{len(df):,} rows")
    return logs


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _daily_counts(df: pd.DataFrame, group_extra: dict | None = None) -> pd.DataFrame:
    """
    @brief Collapse an event log to per-(user, day) counts, optionally with
           extra conditional counts.

    @param df           Event DataFrame with 'user' and 'date' columns.
    @param group_extra  Dict of {output_col: boolean_mask_series} for
                        additional conditional counts (e.g. after-hours only).
    @return             DataFrame indexed by (user, day) with a 'n_total' column
                        plus one column per entry in group_extra.
    """
    day = df["date"].dt.floor("D")
    out = df.groupby(["user", day]).size().rename("n_total").reset_index()
    out = out.rename(columns={"date": "day"})

    if group_extra:
        for col, mask in group_extra.items():
            sub = df.loc[mask].groupby(["user", day]).size().rename(col).reset_index()
            sub = sub.rename(columns={"date": "day"})
            out = out.merge(sub, on=["user", "day"], how="left")
            out[col] = out[col].fillna(0)

    return out


def build_daily_features(logs: dict) -> pd.DataFrame:
    """
    @brief Build one row per (user, day) with behavioural count features.

    This is the direct analogue of the Reuters "document" in the COS783
    keyword-search system: here, each user-day is the unit that gets
    profiled, normalised, and scored, in place of each news article being
    indexed and searched.

    @param logs  Dict of raw log DataFrames, as returned by load_raw_logs().
    @return      DataFrame with one row per (user, day) and behavioural
                 count columns as declared in FEATURE_COLS.
    """
    logon  = logs["logon"]
    device = logs["device"]
    file_  = logs["file"]
    http   = logs["http"]
    email  = logs["email"]

    is_afterhours = lambda d: (d["date"].dt.hour < 7) | (d["date"].dt.hour >= 19)

    with substep(f"aggregating logon events ({len(logon):,} rows)") as s:
        logon_d = _daily_counts(logon, {
            "n_logon_only":       logon["activity"] == "Logon",
            "n_logoff_only":      logon["activity"] == "Logoff",
            "n_afterhours_logon": is_afterhours(logon) & (logon["activity"] == "Logon"),
        }).rename(columns={"n_total": "n_logon_total"})
        s.set_detail(f"{len(logon_d):,} user-days")

    with substep(f"aggregating device events ({len(device):,} rows)") as s:
        device_d = _daily_counts(device, {
            "n_device_connect": device["activity"] == "Connect",
        }).rename(columns={"n_total": "n_device_total"})
        s.set_detail(f"{len(device_d):,} user-days")

    with substep(f"aggregating file events ({len(file_):,} rows)") as s:
        # r6.2 writes the literal strings "True"/"False"; the synthetic
        # generator writes 1/0. Normalise both to a boolean mask -- comparing
        # against 1 alone silently yields all-False on real data, which held
        # n_file_to_removable at zero and disabled the single most
        # discriminative exfiltration feature.
        file_d = _daily_counts(file_, {
            "n_file_to_removable": _as_bool(file_["to_removable_media"]),
        }).rename(columns={"n_total": "n_file_events"})
        s.set_detail(f"{len(file_d):,} user-days")

    with substep(f"aggregating http events ({len(http):,} rows)") as s:
        http_d = _daily_counts(http, {
            "n_http_upload": http["activity"] == "WWW Upload",
        }).rename(columns={"n_total": "n_http_events"})
        # Count distinct hosts, not distinct URLs: r6.2 URLs carry unique
        # per-visit path components, so a URL nunique is very nearly a
        # duplicate of n_http_events rather than a breadth-of-browsing signal.
        domains_d = (
            http.assign(day=http["date"].dt.floor("D"),
                        _host=http["url"].map(netloc_of))
            .groupby(["user", "day"])["_host"].nunique()
            .rename("n_distinct_domains").reset_index()
        )
        http_d = http_d.merge(domains_d, on=["user", "day"], how="left")
        s.set_detail(f"{len(http_d):,} user-days")

    with substep(f"aggregating email events ({len(email):,} rows)") as s:
        # Only "Send" rows are user-initiated egress; "View" rows are mail
        # received, and counting them inflated n_email_sent by ~60%.
        sent = email[email["activity"].astype(str) == "Send"] \
            if "activity" in email.columns else email
        sent = sent.reset_index(drop=True)

        # A recipient is external if it is outside the organisation's own
        # domain, checked across to/cc/bcc. The previous test looked for
        # "@corp.example.com", which appears nowhere in r6.2 (the fictional
        # company is DTAA), so every message counted as external.
        fields = [c for c in ("to", "cc", "bcc") if c in sent.columns]
        if len(sent):
            # Infer the organisation's domain from the frame in hand rather
            # than from RAW_DIR, so this stays correct for a synthetic or
            # hand-made log that is not the one on disk.
            org_domain = _org_domain_from_frame(sent)
            recipients = sent[fields].fillna("").agg(";".join, axis=1)
            is_external = pd.Series(
                [_has_external_recipient(r, "", "", org_domain) for r in recipients],
                index=sent.index,
            )
        else:
            is_external = pd.Series(dtype=bool)

        email_d = _daily_counts(sent, {
            "n_email_external": is_external,
        }).rename(columns={"n_total": "n_email_sent"})

        # r6.2's `attachments` column is a ";"-separated list of
        # "path(size)" entries, not a number -- summing it directly is
        # meaningless. Count the entries instead.
        attach_counts = sent["attachments"].fillna("").map(_count_attachments) \
            if "attachments" in sent.columns else 0
        attach_d = (
            sent.assign(day=sent["date"].dt.floor("D"),
                        _n_attach=attach_counts,
                        _size=pd.to_numeric(sent.get("size"), errors="coerce").fillna(0.0))
            .groupby(["user", "day"])[["_n_attach", "_size"]].sum()
            .rename(columns={"_n_attach": "n_email_attachments", "_size": "total_email_size"})
            .reset_index()
        )
        email_d = email_d.merge(attach_d, on=["user", "day"], how="left")
        s.set_detail(f"{len(email_d):,} user-days")

    with substep("merging per-log-type tables into one (user, day) feature table") as s:
        merged = logon_d[["user", "day", "n_logon_only", "n_logoff_only", "n_afterhours_logon"]]
        merged = merged.rename(columns={"n_logon_only": "n_logon", "n_logoff_only": "n_logoff"})

        for piece, cols in [
            (device_d,  ["n_device_connect"]),
            (file_d,    ["n_file_events", "n_file_to_removable"]),
            (http_d,    ["n_http_events", "n_http_upload", "n_distinct_domains"]),
            (email_d,   ["n_email_sent", "n_email_external", "n_email_attachments", "total_email_size"]),
        ]:
            merged = merged.merge(piece[["user", "day"] + cols], on=["user", "day"], how="outer")

        merged = merged.fillna(0)
        for col in FEATURE_COLS:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].astype(float)

        merged = merged.rename(columns={"day": "date"}).sort_values(["user", "date"]).reset_index(drop=True)
        s.set_detail(f"{len(merged):,} user-days total")

    return merged[["user", "date"] + FEATURE_COLS]


def attach_context(daily_df: pd.DataFrame, org_df: pd.DataFrame,
                   gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Join role/org context and ground-truth malicious-day labels onto
           the per-(user, day) feature table.

    Delegates to src.prepare so that this path and the frozen-dataset path
    (script/prepare.py) can never disagree about what a label or a
    high-privilege role means.

    @param daily_df  Output of build_daily_features().
    @param org_df    Output of load_org_structure().
    @param gt_df     Output of load_ground_truth().
    @return          daily_df with 'role', 'functional_unit', 'department',
                     'supervisor', 'is_high_privilege' and 'is_malicious'
                     columns appended.
    """
    from src.prepare import attach_org_context, attach_labels

    df = attach_org_context(daily_df, org_df)
    days_df = load_malicious_days()
    if days_df is None or not len(days_df):
        warn("No malicious_days.csv, falling back to window filling, which "
             "over-labels benign days inside an insider's active period.")

    n_labels = 0 if days_df is None else len(days_df)
    with substep(f"labelling rows against "
                 f"{n_labels or len(gt_df):,} ground-truth label(s)") as s:
        df, source = attach_labels(df, gt_df, days_df)
        s.set_detail(f"{int(df['is_malicious'].sum()):,} malicious rows "
                     f"labelled from {source}")
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.7) -> pd.DataFrame:
    """
    @brief Add a 'split' column ("train"/"test") using a global time cutoff.

    A time-based split (rather than a random split) prevents future
    information from leaking into the training window and matches how a
    deployed system would actually be trained and evaluated: on data
    collected before a cutoff date, evaluated on data collected after it.

    @param df           Context-attached daily feature DataFrame.
    @param train_frac   Fraction of the date range assigned to "train".
    @return             df with an added 'split' column.
    """
    from src.prepare import time_split_cutoff

    df = df.copy()
    cutoff = time_split_cutoff(df["date"], train_frac)
    df["split"] = np.where(df["date"] < cutoff, "train", "test")
    return df


def load_dataset(max_users: int | None = None,
                 userday_table: str | None = None) -> pd.DataFrame:
    """
    @brief End-to-end loader: daily features -> context -> time split.

    Prefers the pre-aggregated per-(user, day) table when one exists, and
    only falls back to reading the raw logs in memory when it does not
    (the synthetic-data path). The fallback is guarded by
    MAX_INMEMORY_LOG_BYTES.

    @param max_users      Optional cap on distinct users (quick smoke-tests).
    @param userday_table  Explicit aggregated-table path; defaults to
                          data/artifacts/userday.parquet.
    @return               Fully assembled per-(user, day) DataFrame.
    """
    path = userday_table or USERDAY_TABLE
    if os.path.exists(path):
        daily = load_daily_features(path)
        if max_users:
            keep = set(pd.Index(daily["user"].unique()).sort_values()[:max_users])
            daily = daily[daily["user"].isin(keep)].reset_index(drop=True)
    else:
        logs = load_raw_logs()
        if max_users:
            keep_users = set(logs["logon"]["user"].drop_duplicates().head(max_users))
            logs = {k: v[v["user"].isin(keep_users)].reset_index(drop=True)
                    for k, v in logs.items()}
        daily = build_daily_features(logs)

    org = load_org_structure()
    gt  = load_ground_truth()
    df  = attach_context(daily, org, gt)
    df  = time_split(df)
    return df
