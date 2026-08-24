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
├── r6.2.tar.bz2
├── answers.tar.bz2
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

This confirms `r6.2.tar.bz2` and `answers.tar.bz2` are present in `data/` (exiting with an error and generating nothing if either is missing), verifies them against the checksum files if present, extracts both archives to a temporary staging directory, locates the five log CSVs and the `LDAP/` folder inside them, copies everything into `data/raw/` and `data/raw/ldap/`, and parses the official answer key into the simplified per-`(scenario, user)` `[start_date, end_date]` window format written to `data/raw/answers/insiders.csv`.

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
├── logon.csv
├── device.csv
├── file.csv
├── http.csv
├── email.csv
├── ldap/
│   ├── 2009-12.csv
│   ├── 2010-01.csv
│   └── ...
└── answers/
    └── insiders.csv      # derived: user, scenario, threat_type, start_date, end_date, n_events
```

## After ingestion — running the pipeline

```bash
python train.py              # builds features, population RF, and the per-user framework
streamlit run app.py         # launches the comparison dashboard
```

`train.py` caches each step, so re-running after ingesting fresh data will pick up the new `data/raw/` contents — use `python train.py --force` to guarantee a full rebuild.

## Notes

- `script/dataset.py` will not run without the real archives present in `data/`, and it never generates synthetic or placeholder data.
- The dataset is large (the full r6.2 release is tens of gigabytes); this repository never stores raw or extracted log data — only `data/raw/` on your local machine, which is excluded from version control (see `.gitignore`).
- Ground truth is expressed at day granularity (one row per `(user, day)`), matching `src/preprocess.py::build_daily_features`. The official answer key is line-level; `dataset.py` collapses each scenario's malicious lines per user into the enclosing date range, which is what `attach_context()` matches against when labelling `is_malicious`.
