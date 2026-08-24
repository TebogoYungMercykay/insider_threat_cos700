# Setup Guide

This guide covers environment setup, dataset preparation, and running the full pipeline (training + dashboard) for the Per-User Behavioural Baseline Framework for Insider Threat Detection.

## System Requirements

|            | Minimum                                                | Recommended                                                                               |
| ---------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **OS**     | Linux, macOS, or Windows (WSL2 recommended on Windows) | Linux                                                                                     |
| **Python** | 3.10                                                   | 3.11+                                                                                     |
| **RAM**    | 8 GB (synthetic default / small real subsets)          | 16 GB+ (full r6.2, all users)                                                             |
| **Disk**   | ~2 GB (synthetic data + artifacts)                     | ~30 GB free (official r6.2 + answers archives, extracted, plus cached artifacts)          |
| **CPU**    | Any dual-core x86_64/ARM64                             | Multi-core recommended — `RandomForestClassifier` and `IsolationForest` use `n_jobs=-1`   |
| **GPU**    | Not required                                           | Not used — the framework is explicitly designed to run on consumer hardware without a GPU |

No internet access is required after the (optional) manual dataset download — training and the dashboard run entirely locally.

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
git clone <this-repository-url>
cd insider_threat

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Prepare the dataset

You have two options — see **[`data/README.md`](../data/README.md)** for full details.

**Real data (recommended for the final research results):** download `r6.2.tar.bz2` and `answers.tar.bz2` from [kilthub.cmu.edu](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) and place them in `data/`, then:

```bash
python script/dataset.py
```

**Synthetic data (no download required, for development/demo):**

```bash
python script/dataset.py --synthetic
```

Either way, this populates `data/raw/` with the same file layout, so every later step is identical regardless of which source you used.

## 3. Train

```bash
python train.py
```

This runs a five-step, independently cached pipeline: feature engineering → population Random Forest → per-user framework (Profile Builder + Role-Aware Normaliser + Anomaly Scorer) → SHAP explainer → metadata manifest. Re-running only re-executes steps whose inputs changed.

Useful flags:

```bash
python train.py --max-users 100          # quick smoke-test on a user subset
python train.py --anomaly-method lof     # iforest | lof | ocsvm
python train.py --no-shap                # skip SHAP (faster)
python train.py --force                  # ignore all caches, redo everything
python train.py --force-step 4           # force only step 4 onward
```

## 4. Launch the dashboard

```bash
streamlit run app.py
```

Opens a local Streamlit server (default `http://localhost:8501`) with five pages: Overview, Risk Explorer, Method Comparison, Evaluation, About.

## Troubleshooting

| Symptom                                                         | Likely cause / fix                                                                                                                                                                                                      |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: Missing raw log files`                      | `data/raw/` is empty — run `python script/dataset.py` first (step 2 above).                                                                                                                                             |
| `python script/dataset.py` falls back to synthetic unexpectedly | `r6.2.tar.bz2` / `answers.tar.bz2` aren't at the expected path — pass `--r6-archive` / `--answers-archive` explicitly, or check they're directly under `data/`.                                                         |
| `CHECKSUM MISMATCH` on ingestion                                | The archive is corrupt/incomplete — re-download it, or pass `--no-verify-checksums` to proceed anyway if you're confident the file is fine.                                                                             |
| `streamlit run app.py` shows "not trained yet"                  | Run `python train.py` first — the dashboard reads cached artifacts from `data/artifacts/`, it does not train on the fly.                                                                                                |
| SHAP explainer unavailable                                      | Either `--no-shap` was used during training, or the `shap` package failed to import (see `pip install shap`) — the dashboard still works without it, using top-\|z-score\|-feature explanations for the per-user model. |
| Out of memory on the full real dataset                          | Use `python train.py --max-users N` to work on a subset, or increase available RAM.                                                                                                                                     |
