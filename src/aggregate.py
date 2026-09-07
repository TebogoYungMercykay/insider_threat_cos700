##
# @file src/aggregate.py
# @brief Bounded-memory streaming reduction of the raw CERT logs into a
#        compact per-(user, day) behavioural feature table.
#
# WHY THIS EXISTS
# ---------------
# The r6.2 release is ~100 GB of raw event CSV, dominated by http.csv
# (~90 GB / ~117 M rows), and most of that volume is the free-text `content`
# column that this framework never uses. The original loader called
# pandas.read_csv() on each file in full, which cannot complete on a 32 GB
# machine.
#
# The key observation is that every engineered feature is a *per-(user, day)
# aggregate*: a count, a sum, or a distinct-count. Aggregates are computable
# in a single sequential pass with memory proportional to the number of
# (user, day) pairs -- not to the number of events. For r6.2 that is
# ~4 000 users x ~516 days ~= 2.1 M rows, roughly 200 MB in memory and well
# under 100 MB on disk as Parquet.
#
# So the 100 GB of raw logs is read exactly once, reduced ~1000:1, and every
# subsequent experiment trains off the compact table. The raw CSVs can then
# be deleted; see docs/data-pipeline.md.
#
# HOW IT IS MADE FAST
# -------------------
#   * Byte-range sharding. Every r6.2 record occupies exactly one physical
#     line (verified: newline count == CSV record count), so a file can be
#     split at arbitrary byte offsets snapped to line boundaries and the
#     shards parsed concurrently with no record ever split or double-counted.
#   * Fixed-width timestamp slicing instead of datetime parsing
#     (see src/schema.day_ordinal).
#   * Per-shard accumulation into small integer-slot vectors, merged in the
#     parent as each shard completes, so peak memory stays bounded.
#
# Distinct-domain counts are the one feature that cannot simply be summed
# across shards, so shards return per-(user, day) sets of 32-bit domain ids
# and the parent unions them before taking cardinalities. Ids come from a
# CRC-32 of the host, which needs no shared vocabulary between processes.
##

import csv
import io
import multiprocessing as mp
import os
import tarfile
import time
import zlib

import numpy as np
import pandas as pd

from src.schema import (
    FEATURE_COLS, N_FEATURES, SLOT, LOG_COLUMNS, LOG_KINDS,
    INTERNAL_EMAIL_DOMAIN, day_ordinal, is_afterhours, netloc_of,
)
from src.progress import Progress, substep, kv, ok, warn, human_size

csv.field_size_limit(1 << 30)

## Target size of one parallel shard. Small enough to keep the progress
#  display responsive and to balance load across workers, large enough that
#  per-task overhead is negligible.
SHARD_BYTES = 512 * 1024 * 1024

## Read granularity inside a shard.
READ_BYTES = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Per-log-kind record handlers
# ---------------------------------------------------------------------------
#
# Each handler consumes one parsed CSV row and updates the accumulators.
# They are deliberately written as flat index arithmetic on a plain list --
# this loop runs ~120 million times, so attribute lookups and DataFrame
# machinery are avoided entirely.

_S_LOGON      = SLOT["n_logon"]
_S_LOGOFF     = SLOT["n_logoff"]
_S_AFTERHOURS = SLOT["n_afterhours_logon"]
_S_DEVICE     = SLOT["n_device_connect"]
_S_FILE       = SLOT["n_file_events"]
_S_FILE_REM   = SLOT["n_file_to_removable"]
_S_HTTP       = SLOT["n_http_events"]
_S_UPLOAD     = SLOT["n_http_upload"]
_S_EMAIL      = SLOT["n_email_sent"]
_S_EMAIL_EXT  = SLOT["n_email_external"]
_S_ATTACH     = SLOT["n_email_attachments"]
_S_SIZE       = SLOT["total_email_size"]

## Values that mean "yes" in a raw log's boolean column. r6.2 writes the
## literal strings "True"/"False"; the synthetic generator writes 1/0.
## Accepting only one dialect silently zeroes the column for the other.
_TRUTHY = frozenset({"True", "true", "TRUE", "1", "yes", "Yes", "YES"})


def _domain_of(address: str) -> str:
    """@brief Lower-cased domain part of an email address, or "" if malformed."""
    at = address.rfind("@")
    return address[at + 1:].strip().lower() if at >= 0 else ""


def _has_external_recipient(to: str, cc: str, bcc: str,
                            org_domain: str = INTERNAL_EMAIL_DOMAIN) -> bool:
    """
    @brief True if any recipient address lies outside the organisation.

    The organisation's domain is supplied rather than hard-coded: r6.2's
    fictional company is DTAA (@dtaa.com) while the synthetic generator uses
    @corp.example.com, and a hard-coded value silently marks every message in
    the other dataset as external -- which is how this feature became a
    constant. See detect_org_domain().

    Note the domain must come from the organisation, not from each message's
    sender: about a quarter of r6.2's "Send" rows originate from an employee's
    *personal* webmail account (juno.com, msn.com, ...) on a company machine.
    Judging those against their own sender domain would classify
    personal-to-personal mail as internal, when it is the most external
    traffic in the dataset.

    r6.2 packs multiple recipients into one ";"-separated field, and scenarios
    2, 4 and 5 all hinge on mail leaving the company, so to/cc/bcc are all
    considered.

    @param to,cc,bcc  Raw recipient fields from email.csv.
    @param org_domain The organisation's own email domain.
    @return           True if at least one recipient is external.
    """
    for field in (to, cc, bcc):
        if not field:
            continue
        for addr in field.split(";"):
            addr = addr.strip().lower()
            if addr and _domain_of(addr) != org_domain:
                return True
    return False


def detect_org_domain(raw_dir: str, ldap_dir: str | None = None,
                      sample_bytes: int = 64 * 1024 * 1024) -> str:
    """
    @brief Work out which email domain belongs to the organisation.

    Prefers the LDAP directory, which lists every employee's corporate
    address and is authoritative and tiny. Falls back to the most common
    sender domain in a sample of email.csv, which works for any dataset whose
    LDAP snapshots carry no email column (the synthetic generator's do not).

    @param raw_dir       Directory holding the extracted CSVs.
    @param ldap_dir      Directory of LDAP snapshots (default: raw_dir/ldap).
    @param sample_bytes  How much of email.csv to sample for the fallback.
    @return              The organisation's email domain, lower-cased.
    """
    from collections import Counter

    ldap_dir = ldap_dir or os.path.join(raw_dir, "ldap")
    if os.path.isdir(ldap_dir):
        snapshots = sorted(f for f in os.listdir(ldap_dir) if f.endswith(".csv"))
        if snapshots:
            with open(os.path.join(ldap_dir, snapshots[0]), newline="",
                      encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = [c.strip().lower() for c in next(reader, [])]
                if "email" in header:
                    col = header.index("email")
                    domains = Counter(
                        _domain_of(row[col]) for _, row in zip(range(5000), reader)
                        if len(row) > col and _domain_of(row[col])
                    )
                    if domains:
                        return domains.most_common(1)[0][0]

    email_path = os.path.join(raw_dir, "email.csv")
    if os.path.exists(email_path):
        with open(email_path, "rb") as f:
            blob = f.read(sample_bytes)
        text = blob.decode("utf-8", "replace")
        text = text[:text.rfind("\n")] if "\n" in text else text
        reader = csv.reader(io.StringIO(text))
        header = next(reader, [])
        if "from" in header:
            col = header.index("from")
            domains = Counter(
                _domain_of(row[col]) for row in reader
                if len(row) > col and _domain_of(row[col])
            )
            if domains:
                return domains.most_common(1)[0][0]

    return INTERNAL_EMAIL_DOMAIN


def _count_attachments(attachments: str) -> int:
    """
    @brief Number of attachments on an email.

    The r6.2 `attachments` column is not a count -- it is a ";"-separated
    list of "path(size)" entries, empty when there are none.

    @param attachments  Raw attachments field.
    @return             Number of attached files.
    """
    if not attachments:
        return 0
    return sum(1 for part in attachments.split(";") if part.strip())


# ---------------------------------------------------------------------------
# Shard reading
# ---------------------------------------------------------------------------

def _line_ranges(path: str, shard_bytes: int = SHARD_BYTES) -> list[tuple[int, int]]:
    """
    @brief Split a CSV into byte ranges snapped to record boundaries.

    The header line is excluded from every range. Each returned range starts
    at the first byte of a record and ends immediately after the newline that
    terminates a record, so shards partition the file's records exactly.

    @param path         Path to the raw CSV.
    @param shard_bytes  Approximate target size of one shard.
    @return             List of (start, end) byte offsets.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.readline()                       # skip header
        first = f.tell()

        bounds = [first]
        pos = first
        while True:
            pos += shard_bytes
            if pos >= size:
                break
            f.seek(pos)
            f.readline()                   # discard the straddled record
            snapped = f.tell()
            if snapped >= size:
                break
            if snapped > bounds[-1]:
                bounds.append(snapped)
        bounds.append(size)

    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _iter_lines(path: str, start: int, end: int):
    """
    @brief Yield decoded records in [start, end), completing a straddling
           record if one exists.

    @param path   Path to the raw CSV.
    @param start  First byte of the shard (a record boundary).
    @param end    One past the last byte of the shard.
    @return       Generator of record strings, without trailing newlines.
    """
    with open(path, "rb") as f:
        f.seek(start)
        pos = start
        tail = b""
        while pos < end:
            chunk = f.read(min(READ_BYTES, end - pos))
            if not chunk:
                break
            pos += len(chunk)
            parts = (tail + chunk).split(b"\n")
            tail = parts.pop()
            for line in parts:
                yield line.decode("utf-8", "replace")
        if tail:
            # Range boundaries are snapped to record starts, so this should
            # not normally fire; completing the record keeps the reader
            # correct even if a file is re-sharded with unsnapped offsets.
            yield (tail + f.readline()).decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------

def _consume(kind: str, rows, cohort: frozenset | None,
             counts: dict, domains: dict,
             org_domain: str = INTERNAL_EMAIL_DOMAIN) -> int:
    """
    @brief Fold a stream of parsed CSV rows into the shard accumulators.

    @param kind     Log kind: one of src.schema.LOG_KINDS.
    @param rows     Iterable of parsed CSV rows (lists of str).
    @param cohort   If given, only these users are accumulated.
    @param counts   dict {(user, day_ordinal): [counter slots]} -- updated.
    @param domains  dict {(user, day_ordinal): set[int]} -- updated (http only).
    @param org_domain  The organisation's email domain (see detect_org_domain).
    @return         Number of records consumed.
    """
    cols = LOG_COLUMNS[kind]
    i_date, i_user = cols.index("date"), cols.index("user")
    n_cols = len(cols)
    n_rows = 0

    # Bind hot lookups to locals; this loop runs up to ~117 M times per file.
    _ordinal, _afterhours, _netloc, _crc = day_ordinal, is_afterhours, netloc_of, zlib.crc32

    for row in rows:
        if len(row) != n_cols:
            continue                       # malformed / truncated record
        user = row[i_user]
        if cohort is not None and user not in cohort:
            continue
        n_rows += 1

        stamp = row[i_date]
        key = (user, _ordinal(stamp))
        vec = counts.get(key)
        if vec is None:
            vec = counts[key] = [0.0] * N_FEATURES

        if kind == "logon":
            activity = row[4]
            if activity == "Logon":
                vec[_S_LOGON] += 1.0
                if _afterhours(stamp):
                    vec[_S_AFTERHOURS] += 1.0
            elif activity == "Logoff":
                vec[_S_LOGOFF] += 1.0

        elif kind == "device":
            if row[5] == "Connect":
                vec[_S_DEVICE] += 1.0

        elif kind == "file":
            vec[_S_FILE] += 1.0
            if row[6] in _TRUTHY:
                vec[_S_FILE_REM] += 1.0

        elif kind == "http":
            vec[_S_HTTP] += 1.0
            if row[5] == "WWW Upload":
                vec[_S_UPLOAD] += 1.0
            host = _netloc(row[4])
            if host:
                seen = domains.get(key)
                if seen is None:
                    seen = domains[key] = set()
                seen.add(_crc(host.encode("utf-8", "replace")))

        elif kind == "email":
            # "View" rows are mail received; only "Send" is user-initiated
            # egress, which is what the exfiltration scenarios turn on.
            if row[8] != "Send":
                continue
            vec[_S_EMAIL] += 1.0
            if _has_external_recipient(row[4], row[5], row[6], org_domain):
                vec[_S_EMAIL_EXT] += 1.0
            vec[_S_ATTACH] += _count_attachments(row[10])
            size = row[9]
            if size:
                try:
                    vec[_S_SIZE] += float(size)
                except ValueError:
                    pass

    return n_rows


def _shard_worker(task):
    """
    @brief Process one byte-range shard of one raw log file.

    @param task  (kind, path, start, end, cohort, org_domain) tuple.
    @return      (counts, domains, n_rows, n_bytes) for the shard.
    """
    kind, path, start, end, cohort, org_domain = task
    counts: dict = {}
    domains: dict = {}
    reader = csv.reader(_iter_lines(path, start, end))
    n_rows = _consume(kind, reader, cohort, counts, domains, org_domain)
    return counts, domains, n_rows, end - start


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _merge(counts: dict, domains: dict, part_counts: dict, part_domains: dict) -> None:
    """
    @brief Fold one shard's accumulators into the running totals, in place.

    @param counts,domains            Running totals -- updated.
    @param part_counts,part_domains  One shard's result.
    """
    for key, vec in part_counts.items():
        acc = counts.get(key)
        if acc is None:
            counts[key] = vec
        else:
            for j in range(N_FEATURES):
                acc[j] += vec[j]

    for key, seen in part_domains.items():
        acc = domains.get(key)
        if acc is None:
            domains[key] = seen
        else:
            acc |= seen


def _to_frame(counts: dict, domains: dict) -> pd.DataFrame:
    """
    @brief Materialise the merged accumulators as a per-(user, day) DataFrame.

    @param counts   {(user, day_ordinal): [counter slots]}.
    @param domains  {(user, day_ordinal): set[domain id]}.
    @return         DataFrame of user, date, and FEATURE_COLS.
    """
    if not counts:
        return pd.DataFrame(columns=["user", "date"] + FEATURE_COLS)

    keys = list(counts)
    users = np.array([k[0] for k in keys], dtype=object)
    ordinals = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
    matrix = np.array([counts[k] for k in keys], dtype=np.float32)

    slot = SLOT["n_distinct_domains"]
    matrix[:, slot] = np.fromiter(
        (len(domains[k]) if k in domains else 0 for k in keys),
        dtype=np.float32, count=len(keys),
    )

    df = pd.DataFrame(matrix, columns=FEATURE_COLS)
    df.insert(0, "date", pd.to_datetime(
        pd.Series(ordinals).map(lambda o: pd.Timestamp.fromordinal(int(o)))
    ))
    df.insert(0, "user", users)
    return df.sort_values(["user", "date"], ignore_index=True)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _pool_context():
    """
    @brief Choose a multiprocessing start method that is safe to call from a
           process that may already have threads running.

    "fork" is cheapest but is unsafe (and deprecated) in a multi-threaded
    parent, which is exactly the case when this is driven from a test runner
    or a notebook. "forkserver" forks from a clean single-threaded helper, so
    it keeps most of fork's speed without the deadlock hazard.

    @return  A multiprocessing context.
    """
    available = mp.get_all_start_methods()
    for method in ("forkserver", "fork", "spawn"):
        if method in available:
            return mp.get_context(method)
    return mp.get_context()


def discover_sources(raw_dir: str, archive: str | None = None,
                     prefer_archive: bool = False) -> tuple[dict, str]:
    """
    @brief Decide where the raw events will be read from.

    Extracted CSVs are preferred because they can be sharded and read in
    parallel. The bzip2 archive is the fallback, so that the ~100 GB of
    extracted CSVs can be deleted and the aggregation still re-run later
    from the ~24 GB archive alone, without ever unpacking it to disk.

    @param raw_dir         Directory holding the extracted CSVs.
    @param archive         Path to r6.2.tar.bz2, or None.
    @param prefer_archive  Read from the archive even if the CSVs are present.
    @return                ({kind: path}, mode) where mode is "csv" or "archive".
    """
    if prefer_archive and archive and os.path.exists(archive):
        return {"archive": archive}, "archive"

    csvs = {k: os.path.join(raw_dir, f"{k}.csv") for k in LOG_KINDS}
    present = {k: p for k, p in csvs.items() if os.path.exists(p)}
    if len(present) == len(LOG_KINDS):
        return present, "csv"

    if archive and os.path.exists(archive):
        return {"archive": archive}, "archive"

    missing = [k for k in LOG_KINDS if k not in present]
    raise FileNotFoundError(
        f"Cannot find raw events. Missing CSVs: {missing} in {raw_dir}, "
        f"and no archive at {archive}.\n"
        f"Run:  python script/dataset.py"
    )


def _aggregate_from_archive(archive: str, cohort: frozenset | None,
                            org_domain: str) -> tuple[dict, dict]:
    """
    @brief Single-pass aggregation streamed straight out of r6.2.tar.bz2.

    Decompresses on the fly and never materialises a member on disk, so this
    runs with the ~24 GB archive as the only copy of the data. It is
    single-threaded because bzip2 streams are not seekable -- expect this to
    be substantially slower than the CSV path, and to be used only when the
    extracted CSVs have been deleted.

    @param archive  Path to r6.2.tar.bz2.
    @param cohort   If given, only these users are accumulated.
    @return         (counts, domains) accumulators.
    """
    counts: dict = {}
    domains: dict = {}
    wanted = {f"{k}.csv": k for k in LOG_KINDS}

    warn("Reading from the compressed archive (single-threaded, slow path).")
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tar:
            kind = wanted.get(os.path.basename(member.name))
            if kind is None or not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with substep(f"streaming {kind}.csv from archive "
                         f"({human_size(member.size)})") as s:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace",
                                        newline="")
                reader = csv.reader(text)
                next(reader, None)                      # header
                n = _consume(kind, reader, cohort, counts, domains, org_domain)
                s.set_detail(f"{n:,} rows")
    return counts, domains


def _aggregate_from_csvs(sources: dict, cohort: frozenset | None,
                         workers: int, org_domain: str) -> tuple[dict, dict]:
    """
    @brief Parallel byte-range aggregation over the extracted CSVs.

    @param sources  {kind: path} for the five raw logs.
    @param cohort   If given, only these users are accumulated.
    @param workers  Number of worker processes.
    @return         (counts, domains) accumulators.
    """
    tasks = []
    for kind in LOG_KINDS:
        path = sources[kind]
        for start, end in _line_ranges(path):
            tasks.append((kind, path, start, end, cohort, org_domain))

    total_bytes = sum(os.path.getsize(sources[k]) for k in LOG_KINDS)
    kv("Raw volume", human_size(total_bytes))
    kv("Shards", f"{len(tasks):,} x ~{human_size(SHARD_BYTES)}")
    kv("Workers", workers)

    counts: dict = {}
    domains: dict = {}
    n_rows = 0

    progress = Progress("scanning raw logs", total_bytes, unit="bytes")
    with _pool_context().Pool(processes=workers) as pool:
        for part_counts, part_domains, rows, nbytes in pool.imap_unordered(
                _shard_worker, tasks, chunksize=1):
            _merge(counts, domains, part_counts, part_domains)
            n_rows += rows
            progress.update(nbytes)
    progress.done()

    kv("Events aggregated", f"{n_rows:,}")
    return counts, domains


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_userday_table(raw_dir: str, archive: str | None = None,
                        cohort=None, workers: int | None = None,
                        prefer_archive: bool = False) -> pd.DataFrame:
    """
    @brief Reduce the raw CERT logs to one row per (user, day).

    Memory use is proportional to the number of distinct (user, day) pairs
    retained, never to the number of raw events, so this completes on a
    consumer laptop regardless of how large the raw logs are.

    @param raw_dir  Directory holding the extracted CSVs.
    @param archive  Path to r6.2.tar.bz2, used if the CSVs are absent.
    @param cohort          Optional iterable of user ids to restrict to.
    @param workers         Worker process count (default: CPU count, max 12).
    @param prefer_archive  Read from the archive even if the CSVs are present.
    @return                DataFrame of user, date, and FEATURE_COLS.
    """
    sources, mode = discover_sources(raw_dir, archive, prefer_archive)
    cohort = frozenset(cohort) if cohort is not None else None
    if cohort is not None:
        kv("Cohort", f"{len(cohort):,} users")
    else:
        kv("Cohort", "all users")

    org_domain = detect_org_domain(raw_dir)
    kv("Org email domain", org_domain)

    t0 = time.perf_counter()
    if mode == "archive":
        counts, domains = _aggregate_from_archive(sources["archive"], cohort, org_domain)
    else:
        if workers is None:
            workers = min(12, max(1, (os.cpu_count() or 4)))
        counts, domains = _aggregate_from_csvs(sources, cohort, workers, org_domain)

    with substep("materialising per-(user, day) table") as s:
        df = _to_frame(counts, domains)
        s.set_detail(f"{len(df):,} user-days")

    kv("Reduction", f"1 row per user-day  ({len(df):,} rows)")
    kv("Wall time", f"{time.perf_counter() - t0:,.1f}s")
    return df


def save_userday_table(df: pd.DataFrame, path: str) -> None:
    """
    @brief Write the per-(user, day) table to Parquet.

    @param df    Table from build_userday_table().
    @param path  Destination .parquet path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    ok(f"user-day table -> {os.path.relpath(path)}  ({human_size(os.path.getsize(path))})")


def load_userday_table(path: str) -> pd.DataFrame:
    """
    @brief Read a previously built per-(user, day) table.

    @param path  Path to the .parquet file.
    @return      DataFrame of user, date, and FEATURE_COLS.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No aggregated user-day table at {path}.\n"
            f"Build it once with:  python script/aggregate.py"
        )
    return pd.read_parquet(path)
