# Dataset — CERT Insider Threat Test Dataset (r6.2)

This directory is where the raw dataset lives (`raw/`) and where it is staged before extraction. The dataset itself is **not committed to the repository** — it must be downloaded from the official source and ingested with `script/dataset.py`. This pipeline requires the **real** dataset; there is no synthetic or mock data mode.

## 1. Download

Download the following files manually from the dataset's landing page — they cannot be fetched automatically, since the host requires you to accept a usage agreement in your browser: <https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247>

You need two archives: `r6.2.tar.bz2`, which contains the five behavioural logs (`logon.csv`, `device.csv`, `file.csv`, `http.csv`, `email.csv`) and the monthly `LDAP/` org-structure snapshots, and `answers.tar.bz2`, which contains the official ground-truth answer key — one CSV per insider scenario, listing every malicious log line.

Also grab, if available, `CHECKSUMS-sha256.txt` (or `CHECKSUMS-sha1.txt`) and `SEI_Insider_README.txt`. The checksum file lets the ingestion script verify your download wasn't corrupted, and the SEI readme documents the scenario narratives in prose.

## 2. Place the files

Put the downloaded files directly in this `data/` directory, unmodified:

```
data/
├── r6.2.tar.bz2                 # ~23 GiB  (23,811,144,737 bytes)
├── answers.tar.bz2              # ~1.2 MiB (1,254,678 bytes)
├── CHECKSUMS-sha256.txt         # optional, but recommended
├── CHECKSUMS-sha1.txt           # optional
└── SEI_Insider_README.txt       # optional, background reading only
```

Do **not** extract the archives yourself — the ingestion script below does this for you and normalises the result into `data/raw/`.

## 3. Ingest

From the project root:

```bash
python script/dataset.py
```

This confirms `r6.2.tar.bz2` and `answers.tar.bz2` are present in `data/` (exiting with an error and generating nothing if either is missing), verifies them against the checksum files if present, extracts both archives to a temporary staging directory, locates the five log CSVs and the `LDAP/` folder inside them, copies everything into `data/raw/` and `data/raw/ldap/`, and parses the official answer key into ground truth.

Note that `answers.tar.bz2` ships the answer key for **every** CERT release at once (r2 through r6.2), so the parse is filtered to r6.2 — which has exactly **5 insiders**. It writes two files: `answers/insiders.csv`, the per-`(scenario, user)` `[start_date, end_date]` windows, and `answers/malicious_days.csv`, the exact set of **44 malicious `(user, day)` pairs** derived from the individual malicious events. The day file is the label set training uses; see [`../docs/data-pipeline.md`](../docs/data-pipeline.md) for why the windows alone over-label.

Useful flags:

```bash
python script/dataset.py --r6-archive /path/to/r6.2.tar.bz2 \
                                --answers-archive /path/to/answers.tar.bz2
python script/dataset.py --no-verify-checksums   # skip checksum check
python script/dataset.py --force                 # redo even if data/raw/ is populated
```

### Expected layout after ingestion

```
data/raw/
├── logon.csv           # ~230 MiB  (241,640,758 bytes)
├── device.csv          # ~133 MiB  (139,541,381 bytes)
├── file.csv            # ~1.24 GiB (1,331,296,612 bytes)
├── http.csv            # ~84.0 GiB (90,233,360,621 bytes)
├── email.csv           # ~7.53 GiB (8,084,765,283 bytes)
├── ldap/
│   ├── 2009-12.csv
│   ├── 2010-01.csv
│   └── ...
└── answers/
    ├── insiders.csv        # derived: user, scenario, threat_type, start_date, end_date, n_events
    └── malicious_days.csv  # derived: the exact 44 malicious (user, day) labels
```

### Reference: disk footprint

Sizes below are from a full r6.2 ingestion and are given so you can sanity-check your own
download/extraction and plan disk space — they are illustrative, not fixed (exact byte counts
can vary slightly by download mirror or release patch).

| Location | Size |
|---|---|
| `data/` (archives + checksums, pre-extraction) | ~22.2 GiB |
| `data/raw/` (extracted logs, excl. `ldap/`, `answers/`) | ~93.1 GiB |
| `data/raw/http.csv` | ~84.0 GiB (dominates total size) |
| `data/raw/email.csv` | ~7.53 GiB |

**Plan for ~120 GiB free** to hold the archives (`data/`) and their extraction (`data/raw/`)
simultaneously during ingestion; you can delete the two `.tar.bz2` archives afterwards to
reclaim the ~22 GiB they occupy, since `script/dataset.py` does not need them again unless
run with `--force`.

`data/artifacts/` is a separate, local-only directory (excluded from version control) used
for pipeline-generated output, not part of the raw dataset above. Two files in it matter:

| File | Size | What it is |
|---|---|---|
| `userday.parquet` | ~8.6 MiB | The raw logs reduced to one row per `(user, day)` |
| `dataset.parquet` | ~8.7 MiB | That table plus org context, labels and the train/test split |

`dataset.parquet` is self-contained. Once it exists, **nothing in the training or
evaluation path reads `data/raw/` again**, so the ~93 GiB of extracted logs can be deleted.

## After ingestion — running the pipeline

### Low-disk path (laptops)

Extracting the logs needs ~117 GiB of free disk. If you do not have it, leave
them compressed:

```bash
python script/dataset.py --logs-from-archive   # extracts LDAP (12 MiB) + the answer key only
python script/aggregate.py --cohort 400        # streams the logs from r6.2.tar.bz2, ~1 GB RAM
python script/prepare.py --cohort 400          # finds the cohort table on its own
```

`aggregate.py` streams from the archive whenever the extracted CSVs are absent.
That path is single-threaded, so allow one to three hours; it is a one-off. Drop
`--cohort 400` from both commands to build the full-population dataset the same
way (2.8 GB RAM at most).

### Extracted logs

The extracted logs are ~93 GiB and are never loaded into memory. Two one-time steps
reduce them to a dataset small enough to train off repeatedly:

```bash
python script/aggregate.py   # raw logs -> per-(user, day) counts   (2.8 GB RAM; ~3.3 min on 12 cores, ~15 min on 4)
python script/prepare.py     # counts   -> labelled dataset          (0.7 GB RAM; ~3 s)
python train.py              # trains off dataset.parquet            (2.6 GB RAM; ~7 min on 12 cores, ~21 min on 4)
streamlit run app.py         # launches the comparison dashboard
```

`script/aggregate.py` is the only step that reads the five big CSVs, and
`script/prepare.py` is the only step that reads `ldap/` and `answers/`. After both
have run, `data/raw/` is no longer needed at all.

For fast iteration, build a reproducible subset instead — every insider is kept and
the benign class is downsampled so all 46 roles stay represented:

```bash
python script/prepare.py --cohort 400 --seed 42
python train.py --dataset data/artifacts/dataset_cohort400_seed42.parquet   # ~20 s
```

`train.py` caches each step against the dataset it was built from — use
`python train.py --force` to guarantee a full rebuild.

### Reclaiming disk

Once `dataset.parquet` exists, the bulk data is only needed again if the **feature
set changes** — that is the one operation that has to re-read raw events. Decide
based on how likely that is:

```bash
# Always safe: the extracted CSVs are reproducible from the archive.
rm data/raw/{http,email,file,device,logon}.csv    # frees ~93 GiB

# Also delete the archive if you are done changing features.
rm data/r6.2.tar.bz2                              # frees ~22 GiB
```

Keeping `r6.2.tar.bz2` lets you rebuild after deleting the CSVs —
`python script/aggregate.py --from-archive` streams straight out of it without
unpacking. Deleting both means a feature change requires re-downloading (or
restoring from your own backup).

Two things must **not** be deleted: `data/raw/ldap/` (12 MiB) and
`data/raw/answers/` (12 KiB) are read by `script/prepare.py`, so removing them
means you can no longer re-label or re-split without re-ingesting. Keep
`answers.tar.bz2` (1.2 MiB) as the provenance for the ground truth — it costs
nothing.

## Notes

- `script/dataset.py` will not run without the real archives present in `data/`, and it never generates synthetic or placeholder data.
- The dataset is large (the full r6.2 release is tens of gigabytes); this repository never stores raw or extracted log data — only `data/raw/` on your local machine, which is excluded from version control (see `.gitignore`).
- Ground truth is expressed at day granularity (one row per `(user, day)`). The official answer key is line-level; `dataset.py` keeps only the events attributed to each scenario's **designated insider** — scenario 3's key also records the 24 recipients of the insider's mass email, and those are victims, not attackers — and reduces them to the exact days on which malicious activity occurred.
- r6.2 is a deliberately sparse needle-in-a-haystack release: **44 malicious user-days out of ~1.39 million**, from 5 insiders out of 4 000 users. With a 70% time split, scenarios 1/3/5 fall in train (6 positive days) and scenarios 2/4 fall entirely in test (38), so the supervised baseline is evaluated on unseen scenario types. State this when reporting results.
