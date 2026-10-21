##
# @file tests/test_pipeline.py
# @brief Regression tests for the streaming aggregation pipeline.
#
# The bugs these cover were all silent: they produced a table of the right
# shape whose columns were constant or wrong, so training "worked" and simply
# never learned anything. Each test below pins one of them.
#
# Run:  pytest tests/ -q
##

import csv
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import aggregate as A
from src import schema as S
from src.cohort import build_cohort

RAW_DIR   = os.path.join(ROOT, "data", "raw")
ANS_DIR   = os.path.join(RAW_DIR, "answers")
ARTIFACTS = os.path.join(ROOT, "data", "artifacts")

# Tests carrying this marker stream the raw logs. They are the reason the
# aggregation is trustworthy, and they are also why the full suite takes
# minutes -- hence the `slow` marker, so `pytest -m "not slow"` stays fast.
requires_raw = pytest.mark.slow(pytest.mark.skipif(
    not os.path.exists(os.path.join(RAW_DIR, "logon.csv")),
    reason="raw CERT logs not present",
))
requires_table = pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACTS, "userday.parquet")),
    reason="aggregated user-day table not built",
)


# ---------------------------------------------------------------------------
# Timestamp and URL parsing
# ---------------------------------------------------------------------------

def test_day_ordinal_matches_datetime():
    """The fast fixed-width date slice must agree with a real parser."""
    import datetime as dt
    for stamp in ["01/02/2010 02:19:18", "12/31/2010 23:59:59", "06/01/2011 07:26:02"]:
        expected = dt.datetime.strptime(stamp, "%m/%d/%Y %H:%M:%S").date().toordinal()
        assert S.day_ordinal(stamp) == expected


def test_hour_and_afterhours():
    assert S.hour_of("01/02/2010 00:00:00") == 0
    assert S.hour_of("01/02/2010 23:59:59") == 23
    assert S.is_afterhours("01/02/2010 06:59:59")     # before 07:00
    assert not S.is_afterhours("01/02/2010 07:00:00")
    assert not S.is_afterhours("01/02/2010 18:59:59")
    assert S.is_afterhours("01/02/2010 19:00:00")     # from 19:00


def test_netloc_collapses_urls_to_hosts():
    """n_distinct_domains counts hosts; r6.2 URLs are unique per visit."""
    assert S.netloc_of("http://www.dropbox.com/home/x.aspx") == "dropbox.com"
    assert S.netloc_of("https://dropbox.com/y") == "dropbox.com"
    assert S.netloc_of("http://wikileaks.org") == "wikileaks.org"
    assert S.netloc_of("") == ""


# ---------------------------------------------------------------------------
# Feature semantics that were silently broken against real r6.2 data
# ---------------------------------------------------------------------------

def test_external_recipient_detection():
    """Recipients are judged against the org domain; cc/bcc count too."""
    org      = "dtaa.com"
    internal = "Alice.B.Smith@dtaa.com"
    external = "someone@gmail.com"

    assert not A._has_external_recipient(internal, "", "", org)
    assert A._has_external_recipient(external, "", "", org)
    assert A._has_external_recipient(internal, external, "", org)   # via cc
    assert A._has_external_recipient(internal, "", external, org)   # via bcc
    assert A._has_external_recipient(f"{internal};{external}", "", "", org)
    assert not A._has_external_recipient("", "", "", org)


def test_personal_to_personal_mail_is_external():
    """
    About a quarter of r6.2's Send rows come from an employee's own webmail
    account. Judging recipients against the *sender's* domain would call
    juno.com -> juno.com internal; it is the most external traffic there is.
    """
    assert A._has_external_recipient(
        "Luke.D.Norman@juno.com;Sloane_Campbell@juno.com", "", "", "dtaa.com")


@requires_raw
def test_org_domain_is_detected_not_hardcoded():
    """
    r6.2 uses @dtaa.com; the synthetic generator uses @corp.example.com.
    Hard-coding either marks every message in the other as external, which is
    how n_email_external became a constant.
    """
    assert A.detect_org_domain(RAW_DIR) == "dtaa.com"


def test_org_domain_falls_back_to_sender_mode(tmp_path):
    """With no LDAP email column, the modal sender domain is used."""
    (tmp_path / "email.csv").write_text(
        "id,date,user,pc,to,cc,bcc,from,activity,size,attachments,content\n"
        + "".join(f"i,01/02/2010 01:00:00,U,PC,a@x.net,,,u{i}@corp.example.com,Send,1,,c\n"
                 for i in range(10))
        + "i,01/02/2010 01:00:00,U,PC,a@x.net,,,one@other.org,Send,1,,c\n"
    )
    assert A.detect_org_domain(str(tmp_path)) == "corp.example.com"


def test_attachment_counting():
    """`attachments` is a ;-separated list of path(size) entries, not a number."""
    assert A._count_attachments("") == 0
    assert A._count_attachments(r"C:\a\B.doc(1119253)") == 1
    assert A._count_attachments(r"C:\a\B.doc(1119253);C:\c\D.doc(155895)") == 2


def test_removable_media_flag_is_a_string_in_r6_2():
    """to_removable_media is "True"/"False"; comparing to 1 yields all-False."""
    from src.preprocess import _as_bool
    got = _as_bool(pd.Series(["True", "False", "true", "1", "0", ""]))
    assert got.tolist() == [True, False, True, True, False, False]


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

@requires_raw
@pytest.mark.parametrize("shard_bytes", [4 << 20, 17 << 20, 100 << 20])
def test_shard_invariance(shard_bytes):
    """
    Parallel byte-range sharding must produce exactly the same aggregates as
    one sequential pass -- no record split, dropped, or double-counted.
    """
    path, kind = os.path.join(RAW_DIR, "device.csv"), "device"

    ref_counts, ref_domains = {}, {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)
        A._consume(kind, reader, None, ref_counts, ref_domains)

    counts, domains = {}, {}
    ranges = A._line_ranges(path, shard_bytes)
    assert len(ranges) > 1
    for start, end in ranges:
        part_c, part_d, _, _ = A._shard_worker(
            (kind, path, start, end, None, S.INTERNAL_EMAIL_DOMAIN))
        A._merge(counts, domains, part_c, part_d)

    assert counts == ref_counts


@requires_raw
def test_line_ranges_partition_the_file_exactly():
    path = os.path.join(RAW_DIR, "device.csv")
    ranges = A._line_ranges(path, 8 << 20)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))     # contiguous
    assert ranges[-1][1] == os.path.getsize(path)                    # reaches EOF
    with open(path, "rb") as f:                                      # header excluded
        assert ranges[0][0] == len(f.readline())


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(ANS_DIR, "insiders.csv")),
                    reason="ground truth not built")
def test_ground_truth_is_r6_2_only():
    """
    The answers archive ships every CERT release at once. r6.2 has exactly
    five insiders; ingesting the archive wholesale previously produced 239.
    """
    gt = pd.read_csv(os.path.join(ANS_DIR, "insiders.csv"))
    assert gt["user"].nunique() == 5
    assert set(gt["user"]) == {"ACM2278", "CMP2946", "PLJ1771", "CDE1846", "MBG3183"}


@pytest.mark.skipif(not os.path.exists(os.path.join(ANS_DIR, "malicious_days.csv")),
                    reason="ground truth not built")
def test_malicious_days_exclude_collateral_users():
    """
    Scenario 3's answer file also records the 24 recipients of the insider's
    mass email. They are victims; labelling them malicious injects 24 false
    positive users into the ground truth.
    """
    days = pd.read_csv(os.path.join(ANS_DIR, "malicious_days.csv"))
    gt = pd.read_csv(os.path.join(ANS_DIR, "insiders.csv"))
    assert set(days["user"]) <= set(gt["user"])
    assert len(days) == 44


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def org():
    from src.preprocess import load_org_structure
    return load_org_structure()


@requires_raw
def test_cohort_is_deterministic_and_keeps_every_insider(org):
    insiders = sorted(pd.read_csv(os.path.join(ANS_DIR, "insiders.csv"))["user"].unique())
    a = build_cohort(org, insiders, n_users=400, seed=42)
    b = build_cohort(org, insiders, n_users=400, seed=42)
    c = build_cohort(org, insiders, n_users=400, seed=7)

    assert a["users"] == b["users"]                  # reproducible
    assert a["users"] != c["users"]                  # seed actually matters
    assert set(insiders) <= set(a["users"])          # no positive discarded
    assert a["n_users"] == 400
    assert a["n_roles_covered"] == a["n_roles_total"]   # every role represented


@requires_raw
def test_cohort_size_is_respected(org):
    insiders = sorted(pd.read_csv(os.path.join(ANS_DIR, "insiders.csv"))["user"].unique())
    for n in (50, 100, 1000):
        assert build_cohort(org, insiders, n_users=n, seed=1)["n_users"] == n


# ---------------------------------------------------------------------------
# End-to-end table properties
# ---------------------------------------------------------------------------

@requires_table
def test_aggregated_table_has_no_dead_features():
    """
    Every feature must vary. n_file_to_removable and n_email_external were
    both structurally zero against real data before the semantics were fixed.
    """
    df = pd.read_parquet(os.path.join(ARTIFACTS, "userday.parquet"))
    dead = [c for c in S.FEATURE_COLS if df[c].max() == 0]
    assert dead == []


@requires_table
def test_distinct_domains_is_not_a_proxy_for_event_count():
    """A URL nunique would make this column nearly equal to n_http_events."""
    df = pd.read_parquet(os.path.join(ARTIFACTS, "userday.parquet"))
    assert df["n_distinct_domains"].mean() < 0.5 * df["n_http_events"].mean()


@requires_table
def test_every_malicious_day_has_a_matching_row():
    df = pd.read_parquet(os.path.join(ARTIFACTS, "userday.parquet"))
    days = pd.read_csv(os.path.join(ANS_DIR, "malicious_days.csv"), parse_dates=["date"])
    rows = set(zip(df["user"], df["date"]))
    assert all((u, d) in rows for u, d in zip(days["user"], days["date"]))


@requires_table
@requires_raw
def test_cohort_slice_equals_direct_aggregation(org):
    """
    Slicing a cohort out of the full table must equal aggregating that cohort
    from the raw logs -- this is what makes the fast path safe.
    """
    insiders = sorted(pd.read_csv(os.path.join(ANS_DIR, "insiders.csv"))["user"].unique())
    users = build_cohort(org, insiders, n_users=25, seed=3)["users"]

    full = pd.read_parquet(os.path.join(ARTIFACTS, "userday.parquet"))
    sliced = full[full["user"].isin(set(users))].reset_index(drop=True)

    direct = A.build_userday_table(RAW_DIR, cohort=users, workers=4)
    pd.testing.assert_frame_equal(sliced, direct, check_dtype=False)


# ---------------------------------------------------------------------------
# The in-memory path (synthetic data) must agree with the streaming path
# ---------------------------------------------------------------------------

def _synthetic_logs():
    """A tiny schema-identical log set in the synthetic generator's dialect."""
    def frame(rows, cols):
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M:%S")
        return df

    day = "01/04/2010"
    return {
        "logon": frame([
            ["i1", f"{day} 02:10:00", "U1", "PC1", "Logon"],     # after hours
            ["i2", f"{day} 09:00:00", "U1", "PC1", "Logon"],
            ["i3", f"{day} 17:00:00", "U1", "PC1", "Logoff"],
        ], ["id", "date", "user", "pc", "activity"]),
        "device": frame([
            ["i4", f"{day} 09:05:00", "U1", "PC1", "R:\\", "Connect"],
            ["i5", f"{day} 09:30:00", "U1", "PC1", "", "Disconnect"],
        ], ["id", "date", "user", "pc", "file_tree", "activity"]),
        "file": frame([
            ["i6", f"{day} 09:10:00", "U1", "PC1", "a.doc", "File Copy", 1, 0, "x"],
            ["i7", f"{day} 09:11:00", "U1", "PC1", "b.doc", "File Open", 0, 0, "x"],
        ], ["id", "date", "user", "pc", "filename", "activity",
            "to_removable_media", "from_removable_media", "content"]),
        "http": frame([
            ["i8",  f"{day} 10:00:00", "U1", "PC1", "http://www.a.com/p/1", "WWW Visit", "x"],
            ["i9",  f"{day} 10:01:00", "U1", "PC1", "http://a.com/p/2", "WWW Visit", "x"],
            ["i10", f"{day} 10:02:00", "U1", "PC1", "http://b.com/p/3", "WWW Upload", "x"],
        ], ["id", "date", "user", "pc", "url", "activity", "content"]),
        "email": frame([
            ["i11", f"{day} 11:00:00", "U1", "PC1", "c1@corp.example.com", "", "",
             "U1@corp.example.com", "Send", 1000, "", "x"],
            ["i12", f"{day} 11:05:00", "U1", "PC1", "p1@webmail.example.net", "", "",
             "U1@corp.example.com", "Send", 2000, r"C:\a.doc(1);C:\b.doc(2)", "x"],
            ["i13", f"{day} 11:09:00", "U1", "PC1", "c2@corp.example.com", "", "",
             "someone@corp.example.com", "View", 3000, "", "x"],
        ], ["id", "date", "user", "pc", "to", "cc", "bcc", "from",
            "activity", "size", "attachments", "content"]),
    }


def test_inmemory_features_are_correct_for_synthetic_dialect():
    """
    The synthetic generator writes 1/0 for to_removable_media and uses
    @corp.example.com, where r6.2 writes "True"/"False" and @dtaa.com. Both
    dialects must produce the same, correct features.
    """
    from src.preprocess import build_daily_features

    row = build_daily_features(_synthetic_logs()).iloc[0]
    assert row["n_logon"] == 2
    assert row["n_logoff"] == 1
    assert row["n_afterhours_logon"] == 1
    assert row["n_device_connect"] == 1              # Disconnect excluded
    assert row["n_file_events"] == 2
    assert row["n_file_to_removable"] == 1           # 1/0 dialect honoured
    assert row["n_http_events"] == 3
    assert row["n_http_upload"] == 1
    assert row["n_distinct_domains"] == 2            # www.a.com and a.com collapse
    assert row["n_email_sent"] == 2                  # the View row excluded
    assert row["n_email_external"] == 1              # only webmail.example.net
    assert row["n_email_attachments"] == 2           # ;-separated entries counted
    assert row["total_email_size"] == 3000           # Send rows only


def test_inmemory_path_matches_streaming_path():
    """
    build_daily_features() (used for synthetic data) and the streaming
    aggregator must produce identical features for identical events.
    """
    import tempfile
    from src.preprocess import build_daily_features

    logs = _synthetic_logs()
    expected = build_daily_features(logs).iloc[0]

    with tempfile.TemporaryDirectory() as tmp:
        for kind, df in logs.items():
            out = df.copy()
            out["date"] = out["date"].dt.strftime("%m/%d/%Y %H:%M:%S")
            out.to_csv(os.path.join(tmp, f"{kind}.csv"), index=False)
        got = A.build_userday_table(tmp, workers=2).iloc[0]

    for col in S.FEATURE_COLS:
        assert got[col] == expected[col], col


def test_streaming_removable_flag_accepts_both_dialects():
    """r6.2 writes "True"/"False"; the synthetic generator writes 1/0."""
    for value, expected in (("True", 1.0), ("true", 1.0), ("1", 1.0),
                            ("False", 0.0), ("0", 0.0), ("", 0.0)):
        counts, domains = {}, {}
        row = ["id", "01/04/2010 09:00:00", "U1", "PC1", "a.doc",
               "File Copy", value, "False", "content"]
        A._consume("file", [row], None, counts, domains)
        assert list(counts.values())[0][S.SLOT["n_file_to_removable"]] == expected, value


# ---------------------------------------------------------------------------
# The prepared dataset must stand alone
# ---------------------------------------------------------------------------

DATASET = os.path.join(ARTIFACTS, "dataset.parquet")

requires_dataset = pytest.mark.skipif(
    not os.path.exists(DATASET), reason="prepared dataset not built")


@requires_dataset
def test_prepared_dataset_is_self_contained():
    """
    The whole point of the prepare step: the dataset carries features, org
    context, labels and the split, so training reads nothing from data/raw/.
    """
    from src.prepare import load_prepared, DATASET_COLUMNS

    df, manifest = load_prepared(DATASET)
    assert list(df.columns) == DATASET_COLUMNS
    assert set(df["split"]) == {"train", "test"}
    assert df["role"].notna().all()
    assert df["is_malicious"].dtype == bool
    assert manifest["n_rows"] == len(df)
    assert manifest["feature_cols"] == list(S.FEATURE_COLS)


@requires_dataset
def test_manifest_carries_ground_truth_for_evaluation():
    """gt_df must survive without data/raw/answers/insiders.csv."""
    from src.prepare import load_prepared, ground_truth_from_manifest

    _, manifest = load_prepared(DATASET)
    gt = ground_truth_from_manifest(manifest)
    assert len(gt) == 5
    assert set(gt["user"]) == {"ACM2278", "CMP2946", "PLJ1771", "CDE1846", "MBG3183"}
    assert pd.api.types.is_datetime64_any_dtype(gt["start_date"])


@requires_dataset
def test_split_is_time_based_with_no_leakage():
    """Every training day must precede every test day."""
    df, manifest = _load_dataset()
    cutoff = pd.Timestamp(manifest["split_cutoff"])
    assert df.loc[df["split"] == "train", "date"].max() < cutoff
    assert df.loc[df["split"] == "test", "date"].min() >= cutoff


def _load_dataset():
    from src.prepare import load_prepared
    return load_prepared(DATASET)


@requires_dataset
def test_labels_match_the_answer_key():
    df, _ = _load_dataset()
    assert int(df["is_malicious"].sum()) == 44
    assert df.loc[df["is_malicious"], "user"].nunique() == 5


@requires_table
@requires_raw
def test_prepare_is_reproducible_and_cohort_is_a_subset():
    """
    Preparing twice gives the same dataset, and a cohort dataset is exactly
    the cohort's rows out of the full one.
    """
    from src.preprocess import load_org_structure, load_ground_truth, load_malicious_days
    from src.prepare import build_prepared
    from src.cohort import build_cohort

    counts = pd.read_parquet(os.path.join(ARTIFACTS, "userday.parquet"))
    org, gt, days = load_org_structure(), load_ground_truth(), load_malicious_days()

    full_a, stats_a = build_prepared(counts, org, gt, days, train_frac=0.7)
    full_b, stats_b = build_prepared(counts, org, gt, days, train_frac=0.7)
    pd.testing.assert_frame_equal(full_a, full_b)
    assert stats_a == stats_b

    users = build_cohort(org, sorted(gt["user"].unique()), n_users=30, seed=5)["users"]
    sub, _ = build_prepared(counts[counts["user"].isin(set(users))].reset_index(drop=True),
                            org, gt, days, train_frac=0.7)
    expected = full_a[full_a["user"].isin(set(users))].reset_index(drop=True)

    # The split cutoff is derived from the dates present, which a subset
    # preserves here, so the rows must match the full run exactly.
    pd.testing.assert_frame_equal(sub, expected)


def test_time_split_cutoff_respects_the_fraction():
    from src.prepare import time_split_cutoff

    dates = pd.Series(pd.date_range("2010-01-01", periods=100, freq="D"))
    assert time_split_cutoff(dates, 0.7) == pd.Timestamp("2010-03-12")   # day 71
    assert time_split_cutoff(dates, 1.0) == pd.Timestamp("2010-04-10")   # last day


@requires_table
@requires_raw
def test_preprocess_and_prepare_agree():
    """
    preprocess.attach_context/time_split and prepare.build_prepared must
    produce identical context, labels and split -- they are two entry points
    onto one implementation, and silent drift between them is what made
    features constant in the first place.
    """
    from src.preprocess import (load_org_structure, load_ground_truth,
                                load_malicious_days, attach_context, time_split)
    from src.prepare import build_prepared, DATASET_COLUMNS

    counts = pd.read_parquet(os.path.join(ARTIFACTS, "userday.parquet")).head(50_000)
    org, gt, days = load_org_structure(), load_ground_truth(), load_malicious_days()

    via_preprocess = time_split(attach_context(counts, org, gt), train_frac=0.7)
    via_preprocess = via_preprocess[DATASET_COLUMNS].sort_values(
        ["user", "date"], ignore_index=True)

    via_prepare, _ = build_prepared(counts, org, gt, days, train_frac=0.7)

    pd.testing.assert_frame_equal(via_preprocess, via_prepare)


# ---------------------------------------------------------------------------
# Low-disk ingestion (dataset.py --logs-from-archive)
# ---------------------------------------------------------------------------

def _load_dataset_script():
    """@brief Import script/dataset.py as a module (scripts are not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dataset_script", os.path.join(ROOT, "script", "dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mini_r6_archive(tmp_path) -> str:
    """@brief A tiny archive laid out like r6.2.tar.bz2: five logs + LDAP snapshots."""
    import io
    import tarfile
    path = tmp_path / "r6.2.tar.bz2"
    members = {f"r6.2/{name}": "id,date,user\n" for name in
               ("logon.csv", "device.csv", "file.csv", "http.csv", "email.csv")}
    members["r6.2/LDAP/2010-01.csv"] = "employee_name,user_id,role\nA B,ABC0001,Salesman\n"
    members["r6.2/LDAP/2010-02.csv"] = "employee_name,user_id,role\nA B,ABC0001,Salesman\n"
    with tarfile.open(path, "w:bz2") as tar:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


@pytest.mark.parametrize("include_logs", [True, False])
def test_low_disk_ingest_leaves_the_logs_in_the_archive(tmp_path, monkeypatch, include_logs):
    ds = _load_dataset_script()
    raw = tmp_path / "raw"
    monkeypatch.setattr(ds, "RAW_DIR", str(raw))
    monkeypatch.setattr(ds, "LDAP_DIR", str(raw / "ldap"))

    ds.ingest_r6_archive(_mini_r6_archive(tmp_path), include_logs=include_logs)

    assert sorted(os.listdir(raw / "ldap")) == ["2010-01.csv", "2010-02.csv"]
    logs = [f for f in ("logon.csv", "device.csv", "file.csv", "http.csv", "email.csv")
            if (raw / f).exists()]
    assert len(logs) == (5 if include_logs else 0)


# ---------------------------------------------------------------------------
# Compact host sets (keeps full aggregation inside an 8 GB laptop)
# ---------------------------------------------------------------------------

def _http_rows(n_users=3, n_days=3, per_day=40):
    """@brief Web events with repeated hosts, so (user, day) keys straddle shards."""
    rows = []
    for d in range(n_days):
        for u in range(n_users):
            for i in range(per_day):
                rows.append([f"i{d}{u}{i}", f"01/{4 + d:02d}/2010 {9 + i % 8:02d}:00:00",
                             f"U{u}", "PC1", f"http://h{(i * (u + 1)) % 17}.com/p/{i}",
                             "WWW Upload" if i % 9 == 0 else "WWW Visit", "x"])
    return rows


def test_hosts_are_packed_and_unioned_exactly_across_shards(tmp_path):
    path = tmp_path / "http.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(S.LOG_COLUMNS["http"])
        w.writerows(_http_rows())

    ref_c, ref_d = {}, {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        A._consume("http", reader, None, ref_c, ref_d)

    counts, domains = {}, {}
    ranges = A._line_ranges(str(path), 300)
    assert len(ranges) > 10
    for start, end in ranges:
        part_c, part_d, _, _ = A._shard_worker(
            ("http", str(path), start, end, None, S.INTERNAL_EMAIL_DOMAIN))
        A._merge(counts, domains, part_c, part_d)

    assert all(isinstance(v, bytes) for v in domains.values())
    pd.testing.assert_frame_equal(A._to_frame(counts, domains), A._to_frame(ref_c, ref_d))


def test_archive_batches_match_the_csv_path(tmp_path, monkeypatch):
    import io
    import tarfile

    logs = _synthetic_logs()
    logs["http"] = pd.DataFrame(_http_rows(), columns=S.LOG_COLUMNS["http"])
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    archive = tmp_path / "r6.2.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        for kind, df in logs.items():
            out = df.copy()
            if not isinstance(out["date"].iloc[0], str):
                out["date"] = out["date"].dt.strftime("%m/%d/%Y %H:%M:%S")
            out.to_csv(csv_dir / f"{kind}.csv", index=False)
            data = (csv_dir / f"{kind}.csv").read_bytes()
            info = tarfile.TarInfo(f"r6.2/{kind}.csv")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    expected = A.build_userday_table(str(csv_dir), workers=2)

    monkeypatch.setattr(A, "ARCHIVE_BATCH_ROWS", 2)     # many batches, many merges
    counts, domains = A._aggregate_from_archive(str(archive), None,
                                                A.detect_org_domain(str(csv_dir)))
    got = A._to_frame(counts, domains)

    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  expected[got.columns].reset_index(drop=True),
                                  check_dtype=False)
