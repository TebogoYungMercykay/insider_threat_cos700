# Setup Guide

This guide covers environment setup, dataset preparation, and running the full pipeline (training + dashboard) for the Per-User Behavioural Baseline Framework for Insider Threat Detection.

## System Requirements

|            | Minimum                                                | Recommended                                                                               |
| ---------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **OS**     | Linux, macOS, or Windows (WSL2 recommended on Windows) | Linux                                                                                     |
| **Python** | 3.10                                                   | 3.11+                                                                                     |
| **RAM**    | 8 GB, laptop path peaks at 1.1 GB; full-population steps at 2.8 GB (one optional experiment: 5.5 GB) | 16 GB, to run several steps side by side |
| **Disk**   | ~30 GB: logs left compressed (`dataset.py --logs-from-archive`) | ~125 GB: logs extracted, so aggregation runs in parallel                   |
| **CPU**    | 2 cores / 4 threads (Intel Core i3 or similar)          | 8+ cores, `RandomForestClassifier` and `IsolationForest` use `n_jobs=-1`  |
| **GPU**    | Not required                                           | Not used, the framework is explicitly designed to run on consumer hardware without a GPU |

No internet access is required after the (optional) manual dataset download, training and the dashboard run entirely locally.

[Running the Experiments](running-the-experiments.md) gives the measured peak memory and time of every command, and the default laptop path.

### Python dependencies

Declared in `requirements.txt`:

```
numpy>=1.26
pandas>=2.0
scikit-learn>=1.3
imbalanced-learn>=0.12
shap>=0.44
joblib>=1.3
streamlit>=1.30
matplotlib>=3.8
```

## 1. Clone and install

```bash
git clone "https://github.com/TebogoYungMercykay/insider_threat_cos700.git"
cd insider_threat_cos700

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Prepare the dataset

You have two options. See **[`data/README.md`](../data/README.md)** for full details.

**Real data (recommended for the final research results):** download `r6.2.tar.bz2` and `answers.tar.bz2` from [kilthub.cmu.edu](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) and place them in `data/`, then:

```bash
python script/dataset.py
```

**Synthetic data (no download required, for development/demo):**

```bash
python script/dataset.py --synthetic
```

Either way, this populates `data/raw/` with the same file layout, so every later step is identical regardless of which source you used.

## 3. Aggregate the raw logs (real data only)

Extracted, the real r6.2 logs are ~93 GB, `http.csv` alone is 84 GB. They are never loaded into memory. Instead one streaming pass reduces them to a per-`(user, day)` table roughly 10,800x smaller, and everything downstream trains off that:

```bash
python script/aggregate.py               # 93.2 GB -> 8.6 MB: 2.8 GB RAM, ~15 min on 4 threads, ~3.3 min on 12 cores
python script/prepare.py                 # -> dataset.parquet (8.7 MB), 0.7 GB RAM, ~3 s
```

On a laptop with little free disk, ingest with `python script/dataset.py --logs-from-archive`
instead: only the LDAP snapshots and the answer key are extracted, and
`aggregate.py` streams the logs from `r6.2.tar.bz2` (single-threaded, a one-off
of one to three hours). `python script/aggregate.py --cohort 400` followed by
`python script/prepare.py --cohort 400` builds the 400-user dataset that way in
about 1 GB of RAM, without ever building the full table.

`aggregate.py` reads the five big CSVs; `prepare.py` reads `ldap/` and `answers/`
and attaches org context, labels and the train/test split. The result,
`data/artifacts/dataset.parquet`, is self-contained. After this, **nothing in the
training or evaluation path reads `data/raw/` again.**

Once it completes, the extracted CSVs are redundant and can be deleted to reclaim ~93 GB of disk:

```bash
rm data/raw/{http,email,file,device,logon}.csv
```

Keep `data/r6.2.tar.bz2`. If the feature definitions ever change, rebuild directly from the archive. It decompresses on the fly and never unpacks to disk:

```bash
python script/aggregate.py --from-archive --force
```

**For fast iteration**, build a reproducible subset instead. All five insiders are always kept and the benign class is downsampled by role-stratified sampling, so every one of the 46 roles stays represented:

```bash
python script/prepare.py --cohort 400 --seed 42
python train.py --dataset data/artifacts/dataset_cohort400_seed42.parquet
```

That cuts a training run from 7 minutes to 20 seconds while retaining all 44 malicious user-days, and takes 3 seconds to build because it never re-reads the raw logs.

The synthetic dataset is small enough that this step is optional, `train.py` falls back to in-memory loading when no aggregated table is present.

See **[`data-pipeline.md`](data-pipeline.md)** for how this works and what it fixed.

## 4. Train

```bash
python train.py
```

This runs a five-step, independently cached pipeline: loading the prepared dataset → population Random Forest → per-user framework (Profile Builder + Role-Aware Normaliser + Anomaly Scorer) → SHAP explainer → metadata manifest. Re-running only re-executes steps whose inputs changed.

Useful flags:

```bash
python train.py --dataset data/artifacts/dataset_cohort400_seed42.parquet  # train on a cohort
python train.py --max-users 100          # quick smoke-test on a user subset
python train.py --anomaly-method lof     # iforest | lof | ocsvm
python train.py --scope role             # personal | role | population
python train.py --statistic robust       # zscore (mean/std) | robust (median/IQR)
python train.py --no-shap                # skip SHAP (faster)
python train.py --force                  # ignore all caches, redo everything
python train.py --force-step 4           # force only step 4 onward
```

## 5. Compare against the other algorithms

`train.py` fits the two models the dashboard uses. The comparison experiment
that produces the dissertation's numbers is separate:

```bash
python script/experiment.py --preset core                # headline comparison
python script/experiment.py --preset baselines           # Noever's four classifiers + floors
python script/experiment.py --preset ablation            # scope x statistic x algorithm
python script/experiment.py --preset core --temporal     # + causal rolling windows
```

Each run writes `comparison.md`, `comparison.csv`, `budget_sweep.csv`,
`fpr_by_role.csv` and `run.json` (full provenance: dataset, split, features,
seed, platform, timings) into `results/<preset>/`.

Detectors are compared at a shared **alert budget** rather than a shared
threshold. See [`evaluation.md`](evaluation.md) for why, and for what the
benchmark's structure forces on the experiment.

## 6. Run the tests

```bash
pytest -m "not slow"    # model + evaluation + data logic, no raw-log scans
pytest                  # adds the tests that re-stream data/raw/ (~3 min)
```

## 7. Launch the dashboard

```bash
streamlit run app.py
```

Opens a local Streamlit server (default `http://localhost:8501`) with five pages: Overview, Risk Explorer, Method Comparison, Evaluation, About.

## Troubleshooting

| Symptom                                                         | Likely cause / fix                                                                                                                                                                                                      |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: Missing raw log files`                      | `data/raw/` is empty, run `python script/dataset.py` first (step 2 above).                                                                                                                                             |
| `No prepared dataset at ...`                                    | Run `python script/aggregate.py` then `python script/prepare.py` (step 3 above). `train.py` reads the prepared dataset, not the raw logs.                                                                               |
| `MemoryError: Refusing to load these raw logs into memory`      | Something asked for the in-memory loader on the real 93 GB dataset. Use `python script/aggregate.py` instead. See [`data-pipeline.md`](data-pipeline.md).                                                              |
| `python script/dataset.py` falls back to synthetic unexpectedly | `r6.2.tar.bz2` / `answers.tar.bz2` aren't at the expected path, pass `--r6-archive` / `--answers-archive` explicitly, or check they're directly under `data/`.                                                         |
| `CHECKSUM MISMATCH` on ingestion                                | The archive is corrupt/incomplete, re-download it, or pass `--no-verify-checksums` to proceed anyway if you're confident the file is fine.                                                                             |
| `streamlit run app.py` shows "not trained yet"                  | Run `python train.py` first. The dashboard reads cached artifacts from `data/artifacts/`, it does not train on the fly.                                                                                                |
| SHAP explainer unavailable                                      | Either `--no-shap` was used during training, or the `shap` package failed to import (see `pip install shap`). The dashboard still works without it, using top-\|z-score\|-feature explanations for the per-user model. |
| Out of memory on the full real dataset                          | Use `python train.py --max-users N` to work on a subset, or increase available RAM.                                                                                                                                     |
| `script/experiment.py` is slow with `--preset full`             | LOF and One-Class SVM are quadratic in the training set. Lower `--fit-subsample`, or use `--preset core`. The full 4 000-user `rf` fit alone takes ~7 min because SMOTE expands 990 765 rows to 1 981 518.               |
| A detector reports high recall but `tie_fraction` near 1.0       | Its scores are nearly all identical, so the alert budget was spent on a tiebreak lottery rather than on a ranking. `dt` does this on r6.2. The recall is real but says nothing about the detector.                       |
