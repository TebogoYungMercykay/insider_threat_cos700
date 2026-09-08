# script/

| Script          | Purpose                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| `dataset.py`    | Ingest the official CERT r6.2 archives into `data/raw/`                     |
| `aggregate.py`  | Stream the raw logs once into `data/artifacts/userday.parquet`              |
| `prepare.py`    | Freeze counts + org context + labels + split into a self-contained dataset  |
| `experiment.py` | Run the detector comparison and write `results/<preset>/`                   |

Normal order: `dataset.py` -> `aggregate.py` -> `prepare.py`, then `train.py`
(for the dashboard) and/or `experiment.py` (for the comparison numbers).

## dataset.py

This script ingests the official CERT Insider Threat Test Dataset (r6.2) into `data/raw/`. It only works with the real dataset — there is no synthetic or mock data mode, so it exits with an error rather than generating anything if the required archives aren't present.

## Quick start

Download `r6.2.tar.bz2` and `answers.tar.bz2` from <https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247>, place them in `data/` (optionally alongside `CHECKSUMS-sha256.txt` / `CHECKSUMS-sha1.txt`), then run:

```bash
python script/dataset.py
```

This verifies the archives, extracts them, and populates `data/raw/` with the five log CSVs, the `LDAP/` org-structure snapshots, and a derived `answers/insiders.csv` ground-truth file.

Common flags:

```bash
python script/dataset.py --r6-archive PATH --answers-archive PATH   # non-default archive locations
python script/dataset.py --no-verify-checksums                      # skip checksum verification
python script/dataset.py --force                                    # re-run even if data/raw/ is already populated
```

## Where to look next

For download instructions and the expected `data/raw/` layout, see `data/README.md`. For a full walkthrough of every flag, output, and troubleshooting steps, see `RUNNING_DATASET.md`. For the reasoning behind how the ingestion converts the official line-level answer key into the per-user date ranges the rest of the pipeline expects, see the dataset notes in `docs/`.
