# Per-User Behavioural Baseline Framework for Insider Threat Detection

**COS700 Research Report** — University of Pretoria, 2026
Student: Selepe Sello · Supervisor: Prof. Jan Eloff

Implements the framework proposed in _"A Per-User Behavioural Baseline Framework for Insider Threat Detection in Resource-Constrained Environments"_: a lightweight, per-user, role-aware anomaly detection system, evaluated against a population-level Random Forest baseline, on the **CERT Insider Threat Test Dataset schema (r6.2)**.

---

## Why this project

Most deployed insider-threat detectors score every user against a single, population-wide "normal" model. That works for gross anomalies but produces persistent false positives for roles whose day-to-day activity is naturally elevated — system administrators, database administrators, IT support — while remaining blind to a low-and-slow insider whose activity never exceeds population-level thresholds even though it is far outside _their own_ historical pattern. This project implements and empirically evaluates a lightweight, per-user, role-aware alternative: profile each user (and, for cold-start users, their role group) individually, score deviations from _that_ baseline rather than from the population, and compare the result against a population-level Random Forest — the closest thing to current practice for this dataset (Noever, 2019) — on both detection recall and high-privilege false-positive rate.

**Key features**

- **Three-component per-user framework** (`src/baseline.py`): a Profile Builder (per-user mean/std per feature), a Role-Aware Normaliser (per-role fallback for users without enough history), and an Anomaly Scorer (Isolation Forest / LOF / One-Class SVM) fitted on _deviation_ vectors rather than raw activity magnitude.
- **Population-level Random Forest comparison baseline** (`PopulationBaseline`), trained on SMOTE-balanced, non-normalised features, so every result is reported against the same yardstick.
- **Real or synthetic data, same pipeline**: `script/dataset.py` ingests the official CERT r6.2 archives when present, or falls back to a schema-identical synthetic generator so the pipeline can be built and demonstrated without the multi-gigabyte official download.
- **Explainability**: SHAP `TreeExplainer` for the population RF, and top-|z-score|-feature explanations for the per-user framework — every flagged day comes with a reason, not just a score.
- **Proposal-defined evaluation metrics** (`src/evaluate.py`): recall, high-privilege subgroup false-positive rate, F1, ROC-AUC, and computational cost (fit time, inference latency, peak memory).
- **Interactive dashboard** (`app.py`): Overview, per-user Risk Explorer, side-by-side Method Comparison, and an Evaluation page with a live score threshold slider.

---

## The dataset

Evaluated against the **official CERT Insider Threat Test Dataset, release r6.2** (Lindauer, 2020), a synthetic-but-realistic corpus of organisational activity logs (logon, device, file, HTTP, email) with injected, ground-truth insider threat scenarios, together with monthly LDAP org-structure snapshots.

The dataset is not bundled in this repository (it must be downloaded under CMU's usage terms) — see **[`data/README.md`](data/README.md)** for the full download, placement, and ingestion instructions. In short:

```bash
# after placing r6.2.tar.bz2 and answers.tar.bz2 in data/ (see data/README.md)
python script/dataset.py
```

`script/dataset.py` auto-detects the official archives, verifies them against the published checksums, extracts them, and converts the official per-event answer key into ground truth. Note that `answers.tar.bz2` ships the answer key for *every* CERT release at once (r2 through r6.2), so the ingest filters to r6.2 — which has exactly **5 insiders and 44 malicious user-days**. It emits both per-user malicious windows and the exact event-derived day labels that training uses.

Because the extracted r6.2 logs are ~93 GB, they are reduced in one streaming pass to a compact per-`(user, day)` table before training — see **[`docs/data-pipeline.md`](docs/data-pipeline.md)**. If the archives aren't present, it transparently falls back to generating a schema-identical **synthetic** dataset instead, so the pipeline can still be built, run, and demonstrated end-to-end. The synthetic generator injects two attacker archetypes described in the proposal:

- **`burst`** — large, short-window activity spikes (catchable by _any_ baseline)
- **`low_slow`** — small, sustained per-user drift (the case a population baseline is expected to miss, and the framework's primary target)

plus a legitimate high-privilege cohort (`SysAdmin`, `DBAdmin`) with naturally elevated — but non-malicious — activity, to test the false-positive-reduction claim.

Both paths (real and synthetic) leave `data/raw/` in the same shape, so every downstream step (`train.py`, `app.py`) is completely agnostic to which one produced it.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# 0. Acquire data — real CERT archives if present in data/, else synthetic
#    (see data/README.md for how to get and place the real dataset)
python script/dataset.py
python script/dataset.py --n-users 800 --n-days 270   # larger synthetic run

# 1. Aggregate — one streaming pass over the raw logs (real data only)
#    93.2 GB / 135 M events -> 8.6 MB / 1.4 M user-days, in ~3.3 min.
#    Afterwards the extracted CSVs can be deleted to reclaim ~93 GB.
python script/aggregate.py
python script/aggregate.py --cohort 400  # reproducible subset; trains in 33 s

# 2. Train — population RF and per-user framework
python train.py                          # incremental: skips cached steps
python train.py --table data/artifacts/userday_cohort400_seed42.parquet
python train.py --max-users 100          # quick smoke-test
python train.py --anomaly-method lof     # iforest | lof | ocsvm
python train.py --no-shap                # skip SHAP (faster)
python train.py --force                  # nuke all caches, redo everything

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

## Documentation

| Doc                                                        | Covers                                                                                                                |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [`data/README.md`](data/README.md)                         | Dataset download, placement, ingestion (real + synthetic)                                                             |
| [`docs/setup-guide.md`](docs/setup-guide.md)               | System requirements, full setup/run walkthrough, troubleshooting                                                      |
| [`docs/data-pipeline.md`](docs/data-pipeline.md)           | How 93 GB of raw logs is trained on with 32 GB of RAM; cohort subsets; the correctness bugs this fixed                |
| [`docs/dataset-notes.md`](docs/dataset-notes.md)           | How the official answer key is converted to day-level ground truth, LDAP schema handling, extending to other releases |
| [`docs/COS700 Research Proposal - Sello Selepe.pdf`](docs) | The accepted research proposal this implementation follows                                                            |

## Project Structure

```
insider_threat/
├── app.py                      # Streamlit UI (Overview, Risk Explorer, Method Comparison, Evaluation, About)
├── train.py                    # Training pipeline with per-step caching (5 steps)
├── requirements.txt
├── tests/
│   └── test_pipeline.py        # Regression tests: shard invariance, feature semantics,
│                               # ground-truth integrity, cohort reproducibility
├── data/
│   ├── README.md               # Dataset download / ingestion instructions
│   ├── r6.2.tar.bz2            # (you provide) official log archive
│   ├── answers.tar.bz2         # (you provide) official answer-key archive
│   ├── raw/                    # CERT-schema logs (real or synthetic) — generated
│   │   ├── logon.csv  device.csv  file.csv  http.csv  email.csv
│   │   ├── ldap/               # monthly org-structure snapshots
│   │   └── answers/            # insiders.csv (windows) + malicious_days.csv (exact labels)
│   └── artifacts/              # userday.parquet (aggregated table) + cached models — generated
├── docs/
│   ├── setup-guide.md          # System requirements + setup walkthrough
│   ├── data-pipeline.md        # Streaming aggregation, cohorts, and the bugs it fixed
│   ├── dataset-notes.md        # Answer-key conversion & schema-handling notes
│   └── *.pdf                   # Research proposal, Gen-AI declaration
├── script/
│   ├── dataset.py              # real-data ingestion path + synthetic generator
│   └── aggregate.py            # one-time streaming reduction of the raw logs
└── src/
    ├── colors.py
    ├── schema.py               # feature set, raw-log layout, fast timestamp/URL parsing
    ├── aggregate.py            # parallel byte-range streaming aggregator (93 GB -> 8.6 MB)
    ├── cohort.py               # reproducible role-stratified subset selection
    ├── preprocess.py           # aggregated-table loading, context, labelling, time split
    ├── baseline.py             # Profile Builder, Role-Aware Normaliser, Anomaly
    │                           # Scorer (PerUserBaselineEngine); PopulationBaseline (RF)
    ├── explain.py              # SHAP TreeExplainer wrapper (population RF)
    └── evaluate.py             # Recall, subgroup FPR, F1, ROC-AUC, computational cost
```

## Architecture — Three Components (Section 4.3 of the proposal)

| Component             | Class                                 | Role                                                               |
| --------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| Profile Builder       | `BehaviouralProfiler`                 | Per-user (mean, std) per feature from the training window          |
| Role-Aware Normaliser | `BehaviouralProfiler` (role fallback) | Per-role (mean, std); used directly for cold-start users           |
| Anomaly Scorer        | `PerUserBaselineEngine`               | Isolation Forest / LOF / One-Class SVM fitted on deviation vectors |

A single, cheaply-trained global anomaly model becomes per-user-aware because its input space is already expressed as "how unusual for _this_ individual/role" rather than raw activity magnitude — this is what keeps fit time low enough for consumer hardware (no GPU required).

## Comparison Baseline (Section 4.4)

`PopulationBaseline` — a `RandomForestClassifier` trained on SMOTE-balanced, population-level (non-normalised) features, following Noever (2019)'s finding that Random Forest is the strongest population-level classifier for this dataset.

## Evaluation Metrics (Section 4.5)

Implemented in `src/evaluate.py`: Recall (malicious class), False Positive Rate (high-privilege subgroup and per-role breakdown), F1, ROC-AUC, and computational cost (fit time, inference latency, peak memory).

## A note on results with the synthetic dataset

Running the pipeline end-to-end on the bundled synthetic data does **not** automatically reproduce the false-positive-reduction result the proposal hypothesises. Because `PopulationBaseline` is a **supervised** Random Forest (per Section 4.4, following Noever), it can partially learn role-conditional "normal" ranges directly from labelled training examples — unlike the naive shared-threshold systems described in the proposal's Problem Statement, which motivated the research. On the current synthetic run, the per-user Isolation-Forest framework actually shows a _higher_ high-privilege false-positive rate than the RF baseline at the default threshold. This is a genuine, worth-reporting empirical finding, not a bug — and the risk-management plan in the proposal (Section 5.6) already anticipates exactly this kind of "marginal or inverted improvement" outcome and describes next steps (temporal/rolling-window features, contamination tuning, alternative anomaly algorithms). Use the Evaluation page's threshold slider and `--anomaly-method`/`--contamination` flags in `train.py` to explore this further.
