##
# @file script/dataset.py
# @brief Populate data/raw/ with CERT r6.2-schema logs, either from the
#        official CERT Insider Threat Test Dataset archives, or — if those
#        aren't present — a schema-identical synthetic generator, so the
#        rest of the pipeline (src/preprocess.py, train.py, app.py) is
#        completely agnostic to which one produced the data.
#
# REAL DATA PATH
# ---------------
# Download the official dataset yourself from
#     https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
# (Carnegie Mellon University / CERT Division, Lindauer 2020) and place the
# following in data/ (next to this script's parent directory):
#     r6.2.tar.bz2          the log files (logon/device/file/http/email) + LDAP
#     answers.tar.bz2       the ground-truth answer key
#     CHECKSUMS-sha256.txt  (optional but recommended) published checksums
#     CHECKSUMS-sha1.txt    (optional but recommended) published checksums
#
# SYNTHETIC PATH
# ---------------
# If the official archives aren't found (or --synthetic is passed), a
# schema-identical synthetic dataset is generated instead: normal
# role-conditioned daily activity for every user, plus two injected
# attacker archetypes (Section 4.3/4.5 of the research proposal):
#   - burst      : large, short-window activity spikes
#   - low_slow   : small, sustained per-user drift, the case a population
#                  baseline is expected to miss
#
# Either path leaves data/raw/ in the same shape, at every step logging what
# is currently happening — this script never runs a multi-second operation
# without first saying so, and never leaves you watching a blank terminal.
#
# Usage:
#   python script/dataset.py                          # real if present, else synthetic
#   python script/dataset.py --synthetic               # force synthetic
#   python script/dataset.py --n-users 800 --n-days 270  # larger synthetic run
#   python script/dataset.py --r6-archive data/r6.2.tar.bz2 --answers-archive data/answers.tar.bz2
#   python script/dataset.py --no-verify-checksums
#   python script/dataset.py --force
##

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import tarfile
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
LDAP_DIR   = os.path.join(RAW_DIR, "ldap")
ANS_DIR    = os.path.join(RAW_DIR, "answers")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.colors import blue, green, red, amber, dim
from src.progress import (
    step, ok, warn, err, kv, divider, header, elapsed_str,
    Progress, substep, Heartbeat, human_size,
)

REQUIRED_REAL_FILES = ["logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"]
CHECKSUM_FILES = ["CHECKSUMS-sha256.txt", "CHECKSUMS-sha1.txt"]
_BSD_CHECKSUM_RE = re.compile(r"^\w+\(([^)]+)\)\s*=\s*([0-9a-fA-F]+)$")
_VALID_LOG_TYPES = {"logon", "device", "file", "http", "email"}

_CORP_DOMAIN = "corp.example.com"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _raw_data_present() -> bool:
    """
    @brief True if data/raw/ looks fully populated: the five required log
           CSVs, at least one LDAP snapshot, and the derived
           answers/insiders.csv all exist — regardless of whether they came
           from the real archives or the synthetic generator.
    """
    if not all(os.path.exists(os.path.join(RAW_DIR, f)) for f in REQUIRED_REAL_FILES):
        return False
    if not (os.path.isdir(LDAP_DIR) and any(f.endswith(".csv") for f in os.listdir(LDAP_DIR))):
        return False
    if not os.path.exists(os.path.join(ANS_DIR, "insiders.csv")):
        return False
    return True


def _already_extracted(dest_path: str) -> bool:
    """@brief True if dest_path exists as a complete file (not mid-write)."""
    return os.path.exists(dest_path) and not os.path.exists(dest_path + ".part")


# ===========================================================================
# REAL DATA PATH — ingest the official r6.2 / answers archives
# ===========================================================================

# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

def _hash_file(path: str, algo: str) -> str:
    h = hashlib.sha256() if algo == "sha256" else hashlib.sha1()
    total = os.path.getsize(path)
    prog = Progress(f"hashing {os.path.basename(path)}", total, unit="bytes")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            prog.update(len(chunk))
    prog.done()
    return h.hexdigest()


def verify_checksums(archive_paths: list) -> bool:
    """
    @brief Best-effort verification of downloaded archives against
           data/CHECKSUMS-sha256.txt / CHECKSUMS-sha1.txt, if present.

    Accepts both "<hash>  <filename>" (coreutils) and "SHA256(<filename>)="
    (BSD) formats. A checksum file with no entry for a given archive is
    logged explicitly (not silently treated the same as "no checksum file
    at all") so a mismatch between the archive's actual filename and what
    the checksum file lists doesn't masquerade as a passed verification.

    @return  True if every archive with a matching entry verified OK; False
             on any mismatch. Always prints what it did and did not check.
    """
    entries = {}
    for cf in CHECKSUM_FILES:
        path = os.path.join(DATA_DIR, cf)
        if not os.path.exists(path):
            continue
        algo = "sha256" if "sha256" in cf else "sha1"
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = _BSD_CHECKSUM_RE.match(line)
                if m:
                    entries.setdefault(os.path.basename(m.group(1)), (algo, m.group(2).lower()))
                    continue
                parts = line.split()
                if len(parts) == 2:
                    hashval, fname = parts
                    entries.setdefault(os.path.basename(fname.lstrip("*")), (algo, hashval.lower()))

    if not entries:
        _log(f"  {dim('(no CHECKSUMS-sha*.txt found under data/ — skipping verification)')}")
        return True

    all_ok = True
    for path in archive_paths:
        fname = os.path.basename(path)
        if not os.path.exists(path):
            continue
        if fname not in entries:
            warn(f"no checksum entry found for {fname} in CHECKSUMS-sha*.txt — skipping verification for this file")
            continue
        algo, expected = entries[fname]
        actual = _hash_file(path, algo)
        if actual.lower() == expected.lower():
            ok(f"checksum OK : {fname} ({algo})")
        else:
            err(f"CHECKSUM MISMATCH : {fname} ({algo})")
            _log(f"      expected: {expected}")
            _log(f"      actual:   {actual}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Archive extraction — stream each wanted member straight to its final
# destination (never staged), atomically (write to "<name>.part", rename on
# success) so an interrupted run leaves either a complete file or nothing.
# ---------------------------------------------------------------------------

def _stream_member_to(tar: tarfile.TarFile, member: tarfile.TarInfo, dest_path: str) -> None:
    src = tar.extractfile(member)
    if src is None:
        return
    name = os.path.basename(dest_path)
    _log(f"  {blue(chr(0x2192))} extracting {name} ({human_size(member.size)}) ...")
    tmp_path = dest_path + ".part"
    prog = Progress(name, member.size, unit="bytes")
    try:
        with open(tmp_path, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
                prog.update(len(chunk))
        prog.done()
        os.replace(tmp_path, dest_path)
    except OSError as e:
        prog.done()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        if e.errno == 122:  # EDQUOT
            raise OSError(
                f"{e}\n"
                f"  Disk quota exceeded writing {dest_path}. This script streams files "
                f"directly to their destination (no staging copy), so this means the "
                f"destination filesystem itself is out of space — free up space under "
                f"{RAW_DIR}. Already-completed files were left untouched; re-run once "
                f"space is available to resume from here."
            ) from e
        raise


def ingest_r6_archive(archive_path: str, force: bool = False) -> None:
    """@brief Stream logon/device/file/http/email.csv + LDAP/*.csv out of r6.2.tar.bz2 into data/raw/."""
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"r6.2 archive not found: {archive_path}")

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LDAP_DIR, exist_ok=True)

    _log(f"  Extracting {os.path.basename(archive_path)} ...")
    found_files, found_ldap = {}, []
    skipped_files, skipped_ldap = 0, 0

    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)

            if base in REQUIRED_REAL_FILES and base not in found_files:
                dest = os.path.join(RAW_DIR, base)
                if not force and _already_extracted(dest):
                    found_files[base] = "(already present, skipped)"
                    skipped_files += 1
                    warn(f"{base} already present — skipped")
                    continue
                _stream_member_to(tar, member, dest)
                found_files[base] = member.name
                ok(f"{base:<14s} <- {member.name}")
                continue

            parts = member.name.replace("\\", "/").split("/")[:-1]
            if base.lower().endswith(".csv") and any(p.lower() == "ldap" for p in parts):
                dest = os.path.join(LDAP_DIR, base)
                if not force and _already_extracted(dest):
                    found_ldap.append(base)
                    skipped_ldap += 1
                    continue
                _stream_member_to(tar, member, dest)
                found_ldap.append(base)

    missing = [f for f in REQUIRED_REAL_FILES if f not in found_files]
    if missing:
        raise FileNotFoundError(
            f"Could not locate {missing} inside {archive_path}. "
            f"Check that this is the official r6.2 release archive."
        )
    if not found_ldap:
        raise FileNotFoundError(f"Could not locate any LDAP/*.csv snapshots inside {archive_path}")

    ok(f"ldap/         <- {len(found_ldap)} monthly snapshot(s)"
       + (f" ({skipped_ldap} already present, skipped)" if skipped_ldap else ""))


def _scenario_id_from_filename(path: str, fallback: int) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else str(fallback)


class _CountingStream(io.RawIOBase):
    """@brief Wraps a tarfile member's binary stream, reporting bytes read as they're consumed."""

    def __init__(self, raw, on_read):
        self._raw = raw
        self._on_read = on_read

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self._raw.read(len(b))
        n = len(data)
        b[:n] = data
        if n:
            self._on_read(n)
        return n


def build_ground_truth_from_answers(archive_path: str) -> pd.DataFrame:
    """
    @brief Convert the official per-event answer key into the simplified
           per-user malicious-window ground truth this pipeline expects.

    Streams every answers/*.csv member straight out of the archive (nothing
    is written to disk) via a byte-counting wrapper so parsing progress —
    normally the least visible step, since it's CPU-bound row parsing
    rather than a simple byte copy — is reported the same way extraction is.
    """
    rows = []
    n_files = 0
    with tarfile.open(archive_path, "r:bz2") as tar:
        members = sorted(
            (m for m in tar if m.isfile() and m.name.lower().endswith(".csv")),
            key=lambda m: m.name,
        )
        if not members:
            raise FileNotFoundError(f"No answer-key CSV files found inside {archive_path}")

        for i, member in enumerate(members, start=1):
            n_files += 1
            scenario = _scenario_id_from_filename(member.name, i)
            _log(f"  {blue(chr(0x2192))} reading {os.path.basename(member.name)} ({human_size(member.size)}) ...")
            src = tar.extractfile(member)
            if src is None:
                continue

            prog = Progress(os.path.basename(member.name), member.size, unit="bytes")
            counted = io.BufferedReader(_CountingStream(src, prog.update))
            text = io.TextIOWrapper(counted, encoding="utf-8", errors="replace", newline="")

            n_parsed, n_skipped = 0, 0
            for rec in csv.reader(text):
                if len(rec) < 4:
                    n_skipped += 1
                    continue
                log_type, event_id, date_str, user = rec[0], rec[1], rec[2], rec[3]
                if log_type not in _VALID_LOG_TYPES:
                    n_skipped += 1
                    continue
                try:
                    ts = pd.to_datetime(date_str)
                except (ValueError, TypeError):
                    n_skipped += 1
                    continue
                rows.append({"scenario": scenario, "user": user, "date": ts})
                n_parsed += 1
            prog.done()
            ok(f"{os.path.basename(member.name):<20s} scenario {scenario:<4s} "
               f"{n_parsed:>7,} malicious events parsed  ({n_skipped} skipped)")

    if not rows:
        raise ValueError(f"Parsed 0 malicious events from {n_files} answer-key file(s).")

    events = pd.DataFrame(rows)
    grouped = events.groupby(["scenario", "user"])["date"].agg(["min", "max", "count"]).reset_index()
    grouped = grouped.rename(columns={"min": "start_date", "max": "end_date", "count": "n_events"})
    grouped["threat_type"] = "scenario_" + grouped["scenario"].astype(str)
    return grouped[["user", "scenario", "threat_type", "start_date", "end_date", "n_events"]]


def ingest_answers_archive(archive_path: str, force: bool = False) -> None:
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"answers archive not found: {archive_path}")

    os.makedirs(ANS_DIR, exist_ok=True)
    out_path = os.path.join(ANS_DIR, "insiders.csv")

    if not force and _already_extracted(out_path):
        warn("answers/insiders.csv already present — skipped")
        return

    _log(f"  Reading {os.path.basename(archive_path)} ...")
    gt_df = build_ground_truth_from_answers(archive_path)

    tmp_path = out_path + ".part"
    gt_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    ok(f"answers/insiders.csv <- {len(gt_df)} labelled (scenario, user) windows across "
       f"{gt_df['scenario'].nunique()} scenario(s), {gt_df['n_events'].sum():,} total events")


def ingest_real_data(r6_archive: str, answers_archive: str, verify: bool = True, force: bool = False) -> None:
    header("Ingesting official CERT r6.2 dataset from local archives")

    if verify:
        step(1, "Verifying checksums")
        if not verify_checksums([r6_archive, answers_archive]):
            err("Checksum mismatch detected — the archive may be corrupt or truncated.")
            _log("  Re-download it, or re-run with --no-verify-checksums to proceed anyway.")
            sys.exit(1)

    step(2, "Extracting log files + LDAP snapshots")
    ingest_r6_archive(r6_archive, force=force)

    step(3, "Extracting and converting the answer key")
    ingest_answers_archive(answers_archive, force=force)

    print()
    divider("═", 70, green)
    print(green(f"  Done — real data ready at {RAW_DIR}", bold=True))
    divider("═", 70, green)


# ===========================================================================
# SYNTHETIC PATH — schema-identical generator with two attacker archetypes
# ===========================================================================

ROLES = ["Employee", "Employee", "Employee", "Employee", "SysAdmin", "DBAdmin"]
FUNCTIONAL_UNITS = ["Engineering", "Finance", "Sales", "HR", "IT", "Legal"]
DEPARTMENTS = ["North America", "EMEA", "APAC"]


def _write_csv_progress(path: str, df: pd.DataFrame, label: str) -> None:
    """@brief Write df to path with a byte-based progress bar, atomically."""
    tmp_path = path + ".part"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    prog = Progress(label, len(csv_bytes), unit="bytes")
    with open(tmp_path, "wb") as f:
        chunk_size = 1 << 20
        for i in range(0, len(csv_bytes), chunk_size):
            chunk = csv_bytes[i:i + chunk_size]
            f.write(chunk)
            prog.update(len(chunk))
    prog.done()
    os.replace(tmp_path, path)


def generate_synthetic(n_users: int = 300, n_days: int = 180, seed: int = 42) -> None:
    """
    @brief Generate a schema-identical synthetic dataset: normal
           role-conditioned daily activity for every user, an injected
           `burst` insider cohort, an injected `low_slow` insider cohort,
           and a legitimate high-privilege cohort with naturally elevated
           (but non-malicious) activity, to exercise the false-positive
           subgroup metric honestly.

    @param n_users  Number of distinct users to simulate.
    @param n_days   Number of consecutive days of activity to simulate.
    @param seed     RNG seed, for reproducible smoke-test runs.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LDAP_DIR, exist_ok=True)
    os.makedirs(ANS_DIR, exist_ok=True)

    header(f"Generating synthetic CERT-schema dataset  ({n_users} users x {n_days} days)")

    # --- users / roles -----------------------------------------------------
    with substep("assigning users to roles and functional units") as s:
        users = [f"USER{u:04d}" for u in range(n_users)]
        roles = rng.choice(ROLES, size=n_users)
        units = rng.choice(FUNCTIONAL_UNITS, size=n_users)
        depts = rng.choice(DEPARTMENTS, size=n_users)
        supervisors = rng.choice(users, size=n_users)
        is_high_priv = np.isin(roles, ["SysAdmin", "DBAdmin"])
        s.set_detail(f"{int(is_high_priv.sum())} high-privilege / {n_users - int(is_high_priv.sum())} standard")

    start_date = pd.Timestamp("2010-01-04")  # matches the real r6.2 release's window start
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    # --- pick insider scenarios ---------------------------------------------
    n_burst    = max(1, int(n_users * 0.03))
    n_low_slow = max(1, int(n_users * 0.03))
    insider_idx = rng.choice(n_users, size=min(n_burst + n_low_slow, n_users), replace=False)
    burst_idx, low_slow_idx = insider_idx[:n_burst], insider_idx[n_burst:n_burst + n_low_slow]

    insider_windows = []
    for scen_n, idx in enumerate(burst_idx, start=1):
        w_start = rng.integers(int(n_days * 0.2), max(int(n_days * 0.2) + 1, int(n_days * 0.8)))
        w_len = rng.integers(2, 6)
        insider_windows.append({
            "user": users[idx], "scenario": f"synth_burst_{scen_n}", "threat_type": "burst",
            "archetype": "burst",
            "start_date": dates[w_start], "end_date": dates[min(w_start + w_len, n_days - 1)],
        })
    for scen_n, idx in enumerate(low_slow_idx, start=1):
        w_start = rng.integers(int(n_days * 0.1), max(int(n_days * 0.1) + 1, int(n_days * 0.6)))
        w_len = rng.integers(20, 45)
        insider_windows.append({
            "user": users[idx], "scenario": f"synth_low_slow_{scen_n}", "threat_type": "low_slow",
            "archetype": "low_slow",
            "start_date": dates[w_start], "end_date": dates[min(w_start + w_len, n_days - 1)],
        })

    # --- per-log-type generation --------------------------------------------
    logon_rows, device_rows, file_rows, http_rows, email_rows = [], [], [], [], []
    n_events = {w["scenario"]: 0 for w in insider_windows}
    windows_by_user = {}
    for w in insider_windows:
        windows_by_user.setdefault(w["user"], []).append(w)

    prog = Progress("simulating daily activity", n_users, unit="items")
    t0 = time.perf_counter()
    for i, user in enumerate(users):
        role = roles[i]
        hp = is_high_priv[i]
        base_logon      = rng.normal(3.0 + (1.5 if hp else 0.0), 0.6, n_days).clip(min=0)
        base_device     = rng.poisson(0.4 + (0.6 if hp else 0.0), n_days)
        base_file       = rng.poisson(4.0 + (3.0 if hp else 0.0), n_days)
        base_http       = rng.poisson(8.0, n_days)
        base_email      = rng.poisson(3.0, n_days)
        user_windows = windows_by_user.get(user, [])

        for d, day in enumerate(dates):
            n_logon = int(base_logon[d])
            for _ in range(max(n_logon, 1)):
                hour = int(rng.normal(9, 1.5)) % 24
                logon_rows.append((f"L{{{user}-{d}}}", day + pd.Timedelta(hours=hour), user, f"PC-{i%50:04d}", "Logon"))
                logon_rows.append((f"L{{{user}-{d}o}}", day + pd.Timedelta(hours=(hour + 8) % 24), user, f"PC-{i%50:04d}", "Logoff"))

            for _ in range(base_device[d]):
                device_rows.append((f"D{{{user}-{d}}}", day + pd.Timedelta(hours=11), user, f"PC-{i%50:04d}", "Removable Media", "Connect"))

            for _ in range(base_file[d]):
                file_rows.append((f"F{{{user}-{d}}}", day + pd.Timedelta(hours=13), user, f"PC-{i%50:04d}",
                                   f"C:\\docs\\file_{rng.integers(0,9999)}.doc", "File Open", 0, 0, ""))

            for _ in range(base_http[d]):
                http_rows.append((f"H{{{user}-{d}}}", day + pd.Timedelta(hours=14), user, f"PC-{i%50:04d}",
                                   f"http://intranet.{_CORP_DOMAIN}/page{rng.integers(0,500)}", "WWW Visit", ""))

            for _ in range(base_email[d]):
                email_rows.append((f"E{{{user}-{d}}}", day + pd.Timedelta(hours=10), user, f"PC-{i%50:04d}",
                                    f"colleague{rng.integers(0,n_users)}@{_CORP_DOMAIN}", "", "", f"{user}@{_CORP_DOMAIN}",
                                    "Send", int(rng.integers(500, 5000)), 0, ""))

            # inject archetype-specific malicious activity for active windows
            for w in user_windows:
                if not (w["start_date"] <= day <= w["end_date"]):
                    continue
                is_burst = w["threat_type"] == "burst"
                n_mal = rng.integers(8, 20) if is_burst else rng.integers(1, 3)
                for _ in range(n_mal):
                    hour = rng.integers(20, 24) if is_burst else rng.integers(0, 24)
                    device_rows.append((f"MD{{{user}-{d}-{_}}}", day + pd.Timedelta(hours=int(hour)), user,
                                         f"PC-{i%50:04d}", "Removable Media", "Connect"))
                    file_rows.append((f"MF{{{user}-{d}-{_}}}", day + pd.Timedelta(hours=int(hour)), user,
                                       f"PC-{i%50:04d}", f"R:\\export_{rng.integers(0,9999)}.zip", "File Copy", 1, 0, ""))
                    http_rows.append((f"MH{{{user}-{d}-{_}}}", day + pd.Timedelta(hours=int(hour)), user,
                                       f"PC-{i%50:04d}", "http://external-file-share.example.net/upload", "WWW Upload", ""))
                    email_rows.append((f"ME{{{user}-{d}-{_}}}", day + pd.Timedelta(hours=int(hour)), user,
                                        f"PC-{i%50:04d}", f"personal{rng.integers(0,999)}@webmail.example.net", "", "",
                                        f"{user}@{_CORP_DOMAIN}", "Send", int(rng.integers(2000, 20000)),
                                        int(rng.integers(1, 3)), ""))
                    n_events[w["scenario"]] += 4

        prog.update(1)
    prog.done()
    ok(f"activity simulated for {n_users} users over {n_days} days  ({elapsed_str(t0)})")

    # --- write log CSVs ------------------------------------------------------
    step(1, "Writing simulated event logs")
    log_specs = [
        ("logon.csv", logon_rows, ["id", "date", "user", "pc", "activity"]),
        ("device.csv", device_rows, ["id", "date", "user", "pc", "file_tree", "activity"]),
        ("file.csv", file_rows, ["id", "date", "user", "pc", "filename", "activity", "to_removable_media", "from_removable_media", "content"]),
        ("http.csv", http_rows, ["id", "date", "user", "pc", "url", "activity", "content"]),
        ("email.csv", email_rows, ["id", "date", "user", "pc", "to", "cc", "bcc", "from", "activity", "size", "attachments", "content"]),
    ]
    for fname, rows, cols in log_specs:
        dest = os.path.join(RAW_DIR, fname)
        df = pd.DataFrame(rows, columns=cols)
        _write_csv_progress(dest, df, f"writing {fname}")
        ok(f"{fname:<14s} <- {len(df):,} events")

    # --- LDAP monthly snapshots -----------------------------------------------
    step(2, "Writing LDAP org-structure snapshots")
    months = pd.period_range(start_date, periods=max(1, n_days // 30 + 1), freq="M")
    with substep(f"writing {len(months)} monthly snapshot(s)") as s:
        for month in months:
            ldap_df = pd.DataFrame({
                "user": users, "role": roles, "functional_unit": units,
                "department": depts, "supervisor": supervisors,
            })
            ldap_df.to_csv(os.path.join(LDAP_DIR, f"{month}.csv"), index=False)
        s.set_detail(f"{len(months)} files, {n_users} users each")
    ok(f"ldap/ <- {len(months)} monthly snapshot(s)")

    # --- ground truth ----------------------------------------------------------
    step(3, "Writing ground-truth insider windows")
    gt_rows = []
    for w in insider_windows:
        gt_rows.append({
            "user": w["user"], "scenario": w["scenario"], "threat_type": w["threat_type"],
            "archetype": w["archetype"], "start_date": w["start_date"], "end_date": w["end_date"],
            "n_events": n_events[w["scenario"]],
        })
    gt_df = pd.DataFrame(gt_rows)
    out_path = os.path.join(ANS_DIR, "insiders.csv")
    tmp_path = out_path + ".part"
    gt_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    ok(f"answers/insiders.csv <- {len(gt_df)} labelled windows "
       f"({len(burst_idx)} burst, {len(low_slow_idx)} low_slow)")

    print()
    divider("═", 70, green)
    print(green(f"  Done — synthetic data ready at {RAW_DIR}", bold=True))
    divider("═", 70, green)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate data/raw/ from the official CERT r6.2 archives if present, "
                     "else from a schema-identical synthetic generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--r6-archive", default=os.path.join(DATA_DIR, "r6.2.tar.bz2"),
                         help="Path to the official r6.2.tar.bz2 archive.")
    parser.add_argument("--answers-archive", default=os.path.join(DATA_DIR, "answers.tar.bz2"),
                         help="Path to the official answers.tar.bz2 archive.")
    parser.add_argument("--no-verify-checksums", action="store_true",
                         help="Skip checksum verification against data/CHECKSUMS-sha*.txt.")
    parser.add_argument("--synthetic", action="store_true",
                         help="Force the synthetic generator, even if real archives are present.")
    parser.add_argument("--n-users", type=int, default=300, metavar="N",
                         help="Synthetic generator: number of users to simulate (default: 300).")
    parser.add_argument("--n-days", type=int, default=180, metavar="N",
                         help="Synthetic generator: number of days to simulate (default: 180).")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic generator RNG seed.")
    parser.add_argument("--force", action="store_true",
                         help="Redo ingestion/generation from scratch even if data/raw/ is already populated.")
    args = parser.parse_args()

    if _raw_data_present() and not args.force:
        ok("data/raw/ is already populated — leaving it untouched.")
        _log(f"  {dim('(pass --force to redo from scratch)')}")
        sys.exit(0)

    if args.synthetic:
        generate_synthetic(n_users=args.n_users, n_days=args.n_days, seed=args.seed)
        sys.exit(0)

    have_real = os.path.exists(args.r6_archive) and os.path.exists(args.answers_archive)
    if have_real:
        ingest_real_data(args.r6_archive, args.answers_archive,
                          verify=not args.no_verify_checksums, force=args.force)
    else:
        warn("Official r6.2 archives not found under data/ — falling back to the synthetic generator.")
        _log(f"    expected r6 archive      : {args.r6_archive}")
        _log(f"    expected answers archive : {args.answers_archive}")
        _log(f"  {dim('(pass --r6-archive/--answers-archive to point at a different location,')}")
        _log(f"  {dim(' or download the real dataset — see data/README.md)')}")
        print()
        generate_synthetic(n_users=args.n_users, n_days=args.n_days, seed=args.seed)
