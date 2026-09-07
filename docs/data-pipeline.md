# Data Pipeline — Training on 93 GB with 32 GB of RAM

This document explains how the framework ingests the CERT r6.2 insider threat
dataset on a consumer laptop, and records the correctness problems that were
found and fixed in the process.

---

## The problem

The r6.2 release is ~93 GB of raw event CSV once extracted:

| File | Size | Rows |
| --- | ---: | ---: |
| `http.csv` | 84.0 GB | ~117 M |
| `email.csv` | 7.5 GB | ~10 M |
| `file.csv` | 1.2 GB | ~2 M |
| `logon.csv` | 230 MB | 3.5 M |
| `device.csv` | 133 MB | 1.6 M |
| **Total** | **93.2 GB** | **135 M** |

The original loader called `pandas.read_csv()` on each file in full. That
cannot complete on a 32 GB machine — `http.csv` alone would need several
hundred GB of RAM, because most of its volume is a free-text `content` column
that this framework never uses.

Extracting the archive also consumed most of the free disk: 93 GB of CSVs
plus the 23.8 GB source archive is ~117 GB, against ~96 GB free.

## The key observation

Every feature in `FEATURE_COLS` is a **per-`(user, day)` aggregate** — a
count, a sum, or a distinct-count. Aggregates are computable in a single
sequential pass with memory proportional to the number of `(user, day)` pairs,
**not** to the number of events.

For r6.2 that is 4 000 users × ~350 active days ≈ **1.39 M rows**:

```
93.2 GB / 135,117,169 events   ->   8.6 MB / 1,394,010 user-days
                                    a 10,800:1 reduction
```

So the raw logs are read **once**, reduced, and never touched again. Every
experiment afterwards trains off an 8.6 MB Parquet file that loads in 0.1 s.

## The pipeline

```
data/r6.2.tar.bz2 ──┐
                    ├─► aggregate.py ─► userday.parquet ─┐
data/raw/*.csv ─────┘    (3.3 min)         (8.6 MB)      │
                                                         ├─► prepare.py ─► dataset.parquet ─► train.py ─► app.py
data/raw/ldap/ ─────────────────────────────────────────┤     (3 s)         (8.7 MB)         (7 min)
data/raw/answers/ ──────────────────────────────────────┘
```

Two stages, because they read different things and cost different amounts:

**`aggregate.py`** is the only step that reads the five big CSVs. It answers
*"what did each user do each day?"* and takes 3.3 minutes.

**`prepare.py`** is the only step that reads `ldap/` and `answers/`. It attaches
org context, labels and the train/test split, and takes 3 seconds.

Splitting them means you can re-label, re-split, or carve out a cohort in
seconds without ever re-reading 93 GB. And once `dataset.parquet` exists,
**nothing downstream reads `data/raw/` at all** — verified by deleting the
directory outright and training successfully
(`test_prepared_dataset_is_self_contained`).

### 1. Build the dataset

```bash
python script/aggregate.py                 # raw logs -> counts   — 3.3 min
python script/prepare.py                   # counts -> dataset    — 3 s
```

### The dataset artifact

`dataset.parquet` is 8.7 MB and carries 22 columns:

| Group | Columns |
| --- | --- |
| Identity | `user`, `date` |
| Features | the 13 in `FEATURE_COLS` |
| Org context | `role`, `functional_unit`, `department`, `supervisor`, `is_high_privilege` |
| Label | `is_malicious` |
| Split | `split` (`train`/`test`) |

Alongside it, `dataset.meta.json` records the provenance: which aggregated
table, LDAP snapshot and ground-truth files it was built from (with content
hashes), the cohort and seed if any, the split cutoff, the class counts, and
the five insider windows — so evaluation keeps working without
`answers/insiders.csv`, and a frozen dataset can always be traced back to the
release it came from.

This is the file to archive alongside a dissertation. It is small enough to
keep several of (one per experiment), and reproduces a result exactly.

### 2. Reclaim the disk

Once the table exists the extracted CSVs are redundant:

```bash
rm data/raw/{http,email,file,device,logon}.csv     # frees ~93 GB
```

Keep `data/r6.2.tar.bz2`. If the feature definitions ever change, rebuild
straight from the archive — it decompresses on the fly and never unpacks to
disk:

```bash
python script/aggregate.py --from-archive --force
```

This path is single-threaded (bzip2 streams are not seekable), so it is
considerably slower than reading the extracted CSVs. It exists so that having
deleted the CSVs is never a dead end.

### 3. Train

```bash
python train.py                            # full population — 7 min
```

## How the aggregation is made fast

**Byte-range sharding.** Every r6.2 record occupies exactly one physical line
(verified: newline count == CSV record count). A file can therefore be split at
arbitrary byte offsets snapped to line boundaries and the shards parsed
concurrently, with no record split or double-counted. `tests/test_pipeline.py`
pins this: sharded aggregation must equal a single sequential pass, at three
different shard sizes.

**No datetime parsing.** Timestamps are fixed-width `MM/DD/YYYY HH:MM:SS`, so
the calendar day and hour are extracted by slicing, memoised on the 10-character
date prefix. Running a general-purpose parser 135 million times was the single
largest avoidable cost.

**Flat counter vectors.** Each shard accumulates into plain lists indexed by
slot number, merged in the parent as shards complete, so peak memory stays
bounded regardless of file size.

**Distinct domains without a shared vocabulary.** `n_distinct_domains` is the
one feature that cannot be summed across shards. Shards return per-`(user, day)`
sets of CRC-32 host ids, which the parent unions before taking cardinalities —
no coordination between processes required.

Measured on 14 cores / 12 workers: **93.2 GB in 3.3 minutes** (~470 MB/s).

## Subsets (cohorts)

Aggregation removes the compute barrier, so a subset is no longer needed to
make training *possible*. It is still useful to make iteration *fast*:

| | Full population | 400-user cohort |
| --- | ---: | ---: |
| User-days | 1,394,010 | 139,590 |
| `train.py` wall time | 7.2 min | **20 s** |
| Malicious user-days | 44 | 44 (all retained) |

```bash
python script/prepare.py --cohort 400 --seed 42
python train.py --dataset data/artifacts/dataset_cohort400_seed42.parquet
```

Because a cohort is carved at the prepare stage, it costs 3 seconds and never
re-reads the raw logs. `script/aggregate.py --cohort` also exists, for when you
want the *counts* table itself restricted.

A cohort is built to be defensible in a write-up:

1. **No positive is ever discarded.** All five designated insiders are included
   unconditionally. With only five in the whole release, dropping one would
   materially change the evaluation.
2. **The negative class is downsampled by role-stratified sampling.** The
   per-user framework falls back to a role-group baseline for users with
   insufficient history, so a cohort missing whole roles would degrade that
   fallback and understate the framework's accuracy. At `--cohort 400` all 46
   roles remain represented.
3. **It is reproducible.** The selected user list, the seed, and the role
   breakdown are written to `userday_cohort400_seed42.cohort.json`.

Cohorts are verified to be exact subsets: a cohort dataset equals the cohort's
rows out of the full one (`test_prepare_is_reproducible_and_cohort_is_a_subset`),
and at the aggregation stage a sliced cohort is bit-identical to a fresh scan
(`test_cohort_slice_equals_direct_aggregation`).

---

## Correctness problems found and fixed

These mattered more than the performance work. Each produced a table of the
right *shape* whose columns were constant or wrong, so training appeared to
work and simply never learned anything.

### 1. Ground truth mixed every CERT release together

`answers.tar.bz2` ships ground truth for **every** CERT release side by side
(r2 through r6.2), and the ingest read all of it. Its master `insiders.csv`
has a `dataset` column naming the release each row belongs to:

| Release | Insiders in archive |
| --- | ---: |
| r5.2 | 99 |
| r4.2 | 70 |
| r6.1 | 5 |
| **r6.2** | **5** |
| others | 12 |

The result was **239 users labelled malicious** in a dataset that has 5. Some
~234 perfectly ordinary users were labelled insiders, which is on its own
enough to make the problem unlearnable. `script/dataset.py` now filters by
`RELEASE = "r6.2"`.

### 2. Scenario 3's victims were labelled as attackers

In scenario 3 the insider sends an alarming mass email, and the answer key also
records the **24 recipients' `View` events**. Those recipients are victims, not
insiders. Malicious events are now attributed only to the scenario's designated
insider; 50 collateral events are dropped.

### 3. Labels were derived by filling in windows

Ground truth was `[min_event, max_event]` per insider, with every day in
between marked malicious. Scenarios 2 and 4 unfold in short bursts over one to
two months, so this marked **131 user-days malicious when only 44 carry any
malicious activity** — two thirds of the "positives" were ordinary behaviour by
a user who happened to be an insider at some other point.

`script/dataset.py` now also emits `answers/malicious_days.csv`, the exact
event-derived day set, and `attach_context()` prefers it.

### 4. `n_file_to_removable` was always zero

```python
file_["to_removable_media"] == 1        # r6.2 stores "True"/"False" strings
```

This is `False` for every row of real data. Copying files to removable media is
the single most discriminative exfiltration signal in the dataset, and the
feature was dead. It now reports **584,737 events**.

### 5. `n_email_external` was always equal to `n_email_sent`

```python
~email["to"].str.contains("@corp.example.com")
```

r6.2's fictional company is **DTAA** (`@dtaa.com`); the string
`corp.example.com` appears nowhere, so every message counted as external. The
check now tests against `dtaa.com` across `to`, `cc` **and** `bcc` — r6.2 packs
multiple recipients into one `;`-separated field, and 28% of messages have more
than one.

### 6. `n_email_sent` counted received mail

`email.csv` contains both `Send` and `View` rows; `View` is mail received. Only
`Send` is user-initiated egress, which is what the exfiltration scenarios turn
on. Counting both inflated the feature by ~60%.

### 7. `n_email_attachments` summed a string column

The `attachments` column is a `;`-separated list of `path(size)` entries, not a
count. It is now parsed by counting entries.

### 8. `n_distinct_domains` counted URLs, not domains

r6.2 URLs carry unique per-visit path components — in a 300 k-row sample there
are **12,587 distinct URLs but only 456 distinct hosts**. A URL `nunique` was
very nearly a duplicate of `n_http_events` rather than a breadth-of-browsing
signal. Hosts are now extracted before counting (mean 24.1 vs 84.0 events/day).

### Result

Before, four of thirteen features were constant or degenerate against real
data. After:

```
DEAD FEATURES (always zero): none
```

---

## Known characteristics of the evaluation

Not bugs — properties of r6.2 that shape what results are achievable, and that
should be stated explicitly in any write-up.

**The positive class is extremely small.** 44 malicious user-days out of
1,394,010 (0.003%), from 5 insiders out of 4 000 users.

**The time split separates scenarios, not just days.** With a 70% global time
cutoff (2010-12-29):

| Split | Malicious days | Insiders |
| --- | ---: | --- |
| train | 6 | ACM2278 (s1), PLJ1771 (s3), MBG3183 (s5) |
| test | 38 | CMP2946 (s2), CDE1846 (s4) |

The supervised Random Forest baseline therefore trains on 6 positive days from
3 insiders and is evaluated on 2 **entirely unseen scenario types**. That is a
realistic and defensible design — it measures detection of novel insider
behaviour rather than memorisation — but it caps what a supervised baseline can
achieve, and it is precisely the argument for the unsupervised per-user
framework, which needs no positive labels at all.

Report both, and say which is which.
