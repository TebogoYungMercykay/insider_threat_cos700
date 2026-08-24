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

from src.progress import substep, Progress

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
LDAP_DIR = os.path.join(RAW_DIR, "ldap")
ANS_DIR  = os.path.join(RAW_DIR, "answers")

LOG_FILES = ["logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"]

# Behavioural feature columns produced by build_daily_features(). Declared
# once here so downstream modules (baseline, evaluate, app) share a single
# source of truth for the feature set.
FEATURE_COLS = [
    "n_logon", "n_logoff", "n_afterhours_logon",
    "n_device_connect",
    "n_file_events", "n_file_to_removable",
    "n_http_events", "n_http_upload", "n_distinct_domains",
    "n_email_sent", "n_email_external", "n_email_attachments", "total_email_size",
]


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
        s.set_detail(f"{len(gt):,} windows")
    return gt


def load_raw_logs() -> dict:
    """
    @brief Load the five raw CERT-schema event logs from data/raw/.

    @return  Dict of DataFrames keyed by "logon", "device", "file", "http", "email".
    """
    _require_raw_data()
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
        file_d = _daily_counts(file_, {
            "n_file_to_removable": file_["to_removable_media"] == 1,
        }).rename(columns={"n_total": "n_file_events"})
        s.set_detail(f"{len(file_d):,} user-days")

    with substep(f"aggregating http events ({len(http):,} rows)") as s:
        http_d = _daily_counts(http, {
            "n_http_upload": http["activity"] == "WWW Upload",
        }).rename(columns={"n_total": "n_http_events"})
        domains_d = (
            http.assign(day=http["date"].dt.floor("D"))
            .groupby(["user", "day"])["url"].nunique()
            .rename("n_distinct_domains").reset_index()
        )
        http_d = http_d.merge(domains_d, on=["user", "day"], how="left")
        s.set_detail(f"{len(http_d):,} user-days")

    with substep(f"aggregating email events ({len(email):,} rows)") as s:
        is_external = ~email["to"].str.contains("@corp.example.com", na=False)
        email_d = _daily_counts(email, {
            "n_email_external": is_external,
        }).rename(columns={"n_total": "n_email_sent"})
        attach_d = (
            email.assign(day=email["date"].dt.floor("D"))
            .groupby(["user", "day"])[["attachments", "size"]].sum()
            .rename(columns={"attachments": "n_email_attachments", "size": "total_email_size"})
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


def attach_context(daily_df: pd.DataFrame, org_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Join role/org context and ground-truth malicious-day labels onto
           the per-(user, day) feature table.

    @param daily_df  Output of build_daily_features().
    @param org_df    Output of load_org_structure().
    @param gt_df     Output of load_ground_truth().
    @return          daily_df with 'role', 'functional_unit', 'is_high_privilege',
                     and 'is_malicious' columns appended.
    """
    df = daily_df.merge(org_df, on="user", how="left")
    df["role"] = df["role"].fillna("Employee")
    df["functional_unit"] = df["functional_unit"].fillna("Operations")
    # Match by substring (case-insensitive) rather than an exact role list:
    # the synthetic generator's roles are "SysAdmin"/"DBAdmin", but the real
    # CERT LDAP data uses its own role-name strings (e.g. "Sys Admin",
    # "Database Administrator", "IT Admin"). This keeps the role-aware
    # false-positive-reduction comparison meaningful against real data too.
    _HP_PATTERN = r"sys.?admin|db.?admin|database.?admin|it.?admin|administrator"
    df["is_high_privilege"] = df["role"].str.contains(_HP_PATTERN, case=False, regex=True, na=False)

    df["is_malicious"] = False
    with substep(f"labelling malicious (user, day) rows against {len(gt_df):,} ground-truth window(s)") as s:
        prog = Progress("ground-truth windows", len(gt_df), unit="items")
        for _, row in gt_df.iterrows():
            mask = (
                (df["user"] == row["user"])
                & (df["date"] >= row["start_date"])
                & (df["date"] <= row["end_date"])
            )
            df.loc[mask, "is_malicious"] = True
            prog.update(1)
        prog.done()
        s.set_detail(f"{int(df['is_malicious'].sum()):,} malicious rows labelled")

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
    df = df.copy()
    dates = df["date"].sort_values().unique()
    cutoff = dates[int(len(dates) * train_frac)]
    df["split"] = np.where(df["date"] < cutoff, "train", "test")
    return df


def load_dataset(max_users: int | None = None) -> pd.DataFrame:
    """
    @brief End-to-end convenience loader: raw logs -> daily features ->
           context -> time split.

    @param max_users  Optional cap on distinct users (for quick smoke-tests).
    @return           Fully assembled per-(user, day) DataFrame.
    """
    logs = load_raw_logs()
    if max_users:
        keep_users = set(logs["logon"]["user"].drop_duplicates().head(max_users))
        logs = {k: v[v["user"].isin(keep_users)].reset_index(drop=True) for k, v in logs.items()}

    daily = build_daily_features(logs)
    org   = load_org_structure()
    gt    = load_ground_truth()
    df    = attach_context(daily, org, gt)
    df    = time_split(df)
    return df
