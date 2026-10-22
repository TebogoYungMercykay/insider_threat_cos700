# Per-User Behavioural Baseline Framework for Insider Threat Detection

**COS700 Research Report**, University of Pretoria, 2026
Student: Selepe Sello (uXXXXXXXX), Supervisor: Prof. Jan Eloff

Implements the framework proposed in _"A Per-User Behavioural Baseline Framework for Insider Threat Detection in Resource-Constrained Environments"_: a lightweight, per-user, role-aware anomaly detection system, evaluated against a population-level Random Forest baseline, on the **CERT Insider Threat Test Dataset schema (r6.2)**.

---

## Why this project

Most deployed insider-threat detectors score every user against a single, population-wide "normal" model. That works for gross anomalies but produces persistent false positives for roles whose day-to-day activity is naturally elevated, system administrators, database administrators, IT support, while remaining blind to a low-and-slow insider whose activity never exceeds population-level thresholds even though it is far outside _their own_ historical pattern. This project implements and empirically evaluates a lightweight, per-user, role-aware alternative: profile each user (and, for cold-start users, their role group) individually, score deviations from _that_ baseline rather than from the population, and compare the result against a population-level Random Forest. The closest thing to current practice for this dataset (Noever, 2019), on both detection recall and high-privilege false-positive rate.

**Key features**

- **Three-component per-user framework** (`src/baseline.py`): a Profile Builder (per-user mean/std per feature), a Role-Aware Normaliser (per-role fallback for users without enough history), and an Anomaly Scorer (Isolation Forest / LOF / One-Class SVM) fitted on _deviation_ vectors rather than raw activity magnitude.
- **Population-level Random Forest comparison baseline** (`PopulationBaseline`), trained on SMOTE-balanced, non-normalised features, so every result is reported against the same yardstick.
- **Real or synthetic data, same pipeline**: `script/dataset.py` ingests the official CERT r6.2 archives when present, or falls back to a schema-identical synthetic generator so the pipeline can be built and demonstrated without the multi-gigabyte official download.
- **Explainability**: SHAP `TreeExplainer` for the population RF, and top-|z-score|-feature explanations for the per-user framework, every flagged day comes with a reason, not just a score.
- **Proposal-defined evaluation metrics** (`src/evaluate.py`): recall, high-privilege subgroup false-positive rate, F1, ROC-AUC, and computational cost (fit time, inference latency, peak memory).
- **Interactive dashboard** (`app.py`): Overview, per-user Risk Explorer (each flagged day explained against the user's own typical day), side-by-side Method Comparison, an Evaluation page with a live alert-budget control, and a Findings page with the full-population results.

---

## The dataset

Evaluated against the **official CERT Insider Threat Test Dataset, release r6.2** (Lindauer, 2020), a synthetic-but-realistic corpus of organisational activity logs (logon, device, file, HTTP, email) with injected, ground-truth insider threat scenarios, together with monthly LDAP org-structure snapshots.

The dataset is not bundled in this repository (it must be downloaded under CMU's usage terms). See **[`data/README.md`](data/README.md)** for the full download, placement, and ingestion instructions. In short:

```bash
# after placing r6.2.tar.bz2 and answers.tar.bz2 in data/ (see data/README.md)
python script/dataset.py --logs-from-archive   # laptop: logs stay compressed (25 GB disk)
python script/dataset.py                       # workstation: extract the logs (125 GB disk)
```

`script/dataset.py` auto-detects the official archives, verifies them against the published checksums, extracts them, and converts the official per-event answer key into ground truth. Note that `answers.tar.bz2` ships the answer key for *every* CERT release at once (r2 through r6.2), so the ingest filters to r6.2, which has exactly **5 insiders and 44 malicious user-days**. It emits both per-user malicious windows and the exact event-derived day labels that training uses.

Because the extracted r6.2 logs are ~93 GB, they are reduced once to a compact per-`(user, day)` table and then frozen into a self-contained `dataset.parquet` (8.7 MB) carrying features, org context, labels and the train/test split. Training reads only that file. See **[`docs/data-pipeline.md`](docs/data-pipeline.md)**. If the archives aren't present, it transparently falls back to generating a schema-identical **synthetic** dataset instead, so the pipeline can still be built, run, and demonstrated end-to-end. The synthetic generator injects two attacker archetypes described in the proposal:

- **`burst`**. Large, short-window activity spikes (catchable by _any_ baseline)
- **`low_slow`**. Small, sustained per-user drift (the case a population baseline is expected to miss, and the framework's primary target)

plus a legitimate high-privilege cohort (`SysAdmin`, `DBAdmin`) with naturally elevated, but non-malicious, activity, to test the false-positive-reduction claim.

Both paths (real and synthetic) leave `data/raw/` in the same shape, so every downstream step (`train.py`, `app.py`) is completely agnostic to which one produced it.

---

## Hardware

The default way to run this project is on an ordinary laptop. The memory and
time of every step were measured, except the two that read the compressed
archive (marked *not measured*). Every step of the laptop path stays under
1.1 GB. Every full-population step stays under 3 GB, except one optional
experiment that needs 5.5 GB.

|              | Laptop, **the default**                    | Workstation (optional)                     |
| ------------ | ------------------------------------------- | ------------------------------------------ |
| **CPU**      | 2 cores / 4 threads (Intel Core i3 or similar) | 8+ cores                                |
| **RAM**      | 8 GB                                        | 16 GB                                      |
| **Free disk**| 30 GB, the logs stay compressed            | 125 GB, the logs are extracted (faster)   |
| **GPU**      | Not used                                    | Not used                                   |

How the numbers below were measured: each command ran pinned to 4 threads
(`taskset -c 0-3`), and peak memory is the proportional set size (PSS) of the
whole process tree, worker processes included. The CPU was a Core Ultra 5 225U,
so **allow up to twice the stated time on an i3**. Run one step at a time:
the peaks are per command, and two heavy steps side by side add up.

## Setup, disk 1 GB, RAM < 1 GB

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run on a laptop (default)

This path uses a reproducible 400-user cohort that keeps all five insiders
and all 44 malicious days. It runs every part of the system, and it is the
dataset behind the 45-detector ablation in `results/cohort400/`. The
headline numbers in the report come from the full population. They fit in
8 GB as well, but take longer; see [Full population](#full-population).

Place `r6.2.tar.bz2` and `answers.tar.bz2` in `data/` first, as described in
[`data/README.md`](data/README.md).

**1. Unpack only what is small**. Disk 25 GB, RAM < 0.5 GB, one read of the 24 GB archive (*not measured*)

```bash
python script/dataset.py --logs-from-archive
```

This extracts the LDAP snapshots (12 MB) and converts the answer key. The five
log files (93 GB once extracted) stay inside the archive.

**2. Reduce the logs for the cohort**. RAM 1.1 GB, 10 min from extracted CSVs; from the archive *not measured*

```bash
python script/aggregate.py --cohort 400
```

Every event in the archive is read, but only the cohort's user-days are kept.
Streaming from the compressed archive is single-threaded. Its time was not
measured, so allow one to three hours; it is a one-off.

**3. Freeze the dataset**. RAM 0.4 GB, seconds

```bash
python script/prepare.py --cohort 400
```

This writes `data/artifacts/dataset_cohort400_seed42.parquet`, identical row
for row to the cohort dataset sliced from the full population.

**4. Train the dashboard's models**. RAM 0.6 GB, 2 min

```bash
python train.py --dataset data/artifacts/dataset_cohort400_seed42.parquet
```

**5. Compare the detectors**. RAM 0.6 to 0.9 GB, 2 to 9 min

```bash
DS=data/artifacts/dataset_cohort400_seed42.parquet
python script/experiment.py --preset core --dataset $DS --out results/laptop   # 0.6 GB, 3 min
python script/experiment.py --preset full --dataset $DS --out results/laptop   # 0.9 GB, 9 min, all 45 detectors
python script/experiment.py --preset core --temporal --dataset $DS --out results/laptop   # 0.8 GB, 2 min
```

`--out results/laptop` keeps the cohort's tables apart from the
full-population ones in `results/`.

**6. Follow-up studies**. RAM 0.5 to 0.7 GB

```bash
python script/studies.py history --dataset $DS --out results/laptop/studies   # 0.5 GB, 4 min
python script/studies.py explain --dataset $DS --out results/laptop/studies   # 0.6 GB, 1.5 min
python script/studies.py seeds   --dataset $DS --out results/laptop/studies   # 0.6 GB, 18 min
```

**7. Dashboard**. RAM 0.5 GB, plus the browser

```bash
streamlit run app.py
```

**8. Tests**. RAM 0.9 GB, 1 min

```bash
pytest -m "not slow"
```

## Full population

These steps reproduce the report's headline numbers on all 4 000 users.
They fit in 8 GB, but take longer, so start them when you can leave the
laptop running. The steps and their order match the laptop path.

| Step | Command | Peak RAM | Time (4 threads) |
| ---- | ------- | -------- | ---------------- |
| Reduce the logs, extracted CSVs (needs 125 GB disk) | `python script/aggregate.py` | 2.8 GB | 15 min |
| Reduce the logs, from the archive (needs 25 GB disk) | `python script/aggregate.py` | ~2.2 GB¹ | *not measured*; allow 1 to 3 h |
| Freeze the dataset | `python script/prepare.py` | 0.7 GB | seconds |
| Train the dashboard's models | `python train.py` | 2.6 GB | 21 min |
| Headline comparison | `python script/experiment.py --preset core` | 2.6 GB | 29 min |
| Noever's classifiers + floors | `python script/experiment.py --preset baselines` | 1.5 GB | 25 min |
| Rolling-window features | `python script/experiment.py --preset core --temporal` | **5.5 GB** | 33 min |
| History length (SQ3) | `python script/studies.py history` | 1.9 GB | 21 min |
| Explanations (SQ5) | `python script/studies.py explain` | 1.8 GB | 20 min |
| Ten seeds | `python script/studies.py seeds` | ~2.6 GB | ~5 h, ten headline comparisons; run overnight |

The rolling-window experiment is the one step that needs a quiet machine. Its
39 features, three times the usual 13, enlarge the SMOTE-balanced Random
Forest's 2-million-row training set. On 8 GB, close the browser and other
programs first, or run it on the cohort instead (0.8 GB, see step 5 above).

`aggregate.py` reads the extracted CSVs when they are present and streams from
the archive when they are not. On a 4-thread laptop it starts 4 workers by
default, and each worker adds about 0.2 GB.

¹ Reading the archive is single-threaded. Its memory was estimated from the
single-worker CSV run (2.2 GB, 37 min of parsing on one core), which holds the
same running totals; decompression adds to the time, not to the memory.

The ten-seed row was not measured separately. Each seed reruns the headline
comparison's models, one detector at a time and on the same data, so the
memory matches that row; the time is roughly ten headline runs.

Then build the figures and the report, RAM 0.2 GB, seconds:

```bash
python report/make_figures.py
cd report && pdflatex Report && bibtex Report && pdflatex Report && pdflatex Report
```

## Documentation

| Doc                                                        | Covers                                                                                                                |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [`data/README.md`](data/README.md)                         | Dataset download, placement, ingestion (real + synthetic)                                                             |
| [`docs/setup-guide.md`](docs/setup-guide.md)               | System requirements, full setup/run walkthrough, troubleshooting                                                      |
| [`docs/data-pipeline.md`](docs/data-pipeline.md)           | How 93 GB of raw logs is trained on with 32 GB of RAM; cohort subsets; the correctness bugs this fixed                |
| [`docs/dataset-notes.md`](docs/dataset-notes.md)           | How the official answer key is converted to day-level ground truth, LDAP schema handling, extending to other releases |
| [`docs/evaluation.md`](docs/evaluation.md)                 | The comparison experiment: what is compared, how the operating point is made fair, and what the benchmark forces; the follow-up studies |
| [`report/Report.pdf`](report)                              | The COS700 research report (LaTeX source and figures script alongside)                                                |
| Research proposal                                          | The accepted proposal this implementation follows; kept with the author, not in the repository                        |

## Project Structure

```
insider_threat/
├── app.py                      # Streamlit UI (Overview, Risk Explorer, Method Comparison, Evaluation, About)
├── train.py                    # Training pipeline with per-step caching (5 steps)
├── requirements.txt
├── pytest.ini                  # `pytest -m "not slow"` skips the raw-log scans
├── results/                    # Experiment output, generated
│   └── <preset>/               # comparison.md/.csv, budget_sweep.csv,
│                               # fpr_by_role.csv, run.json (full provenance)
├── tests/
│   ├── test_pipeline.py        # Data regression tests: shard invariance, feature
│   │                           # semantics, ground-truth integrity, cohorts
│   ├── test_model.py           # Model + evaluation regression tests: scope
│   │                           # selection, cold start, score saturation, alert
│   │                           # budgets, tie handling, causal windows
│   └── test_studies.py         # Queue depth, Holm correction, history truncation,
│                               # answer-key replay, explanation scoring
├── data/
│   ├── README.md               # Dataset download / ingestion instructions
│   ├── r6.2.tar.bz2            # (you provide) official log archive
│   ├── answers.tar.bz2         # (you provide) official answer-key archive
│   ├── raw/                    # CERT-schema logs (real or synthetic), generated
│   │   ├── logon.csv  device.csv  file.csv  http.csv  email.csv
│   │   ├── ldap/               # monthly org-structure snapshots
│   │   └── answers/            # insiders.csv (windows) + malicious_days.csv (exact labels)
│   └── artifacts/              # userday.parquet (counts), dataset.parquet (trainable
│                               # dataset + manifest), cached models, generated
├── docs/
│   ├── setup-guide.md          # System requirements + setup walkthrough
│   ├── data-pipeline.md        # Streaming aggregation, cohorts, and the bugs it fixed
│   ├── dataset-notes.md        # Answer-key conversion & schema-handling notes
│   └── *.pdf                   # Research proposal, Gen-AI declaration
├── script/
│   ├── dataset.py              # real-data ingestion path + synthetic generator
│   ├── aggregate.py            # one-time streaming reduction of the raw logs
│   ├── prepare.py              # freeze counts + context + labels into one dataset
│   ├── experiment.py           # run the comparison and write the report
│   └── studies.py              # seeds / history (SQ3) / explain (SQ5) studies
├── report/
│   ├── Report.tex              # the COS700 research report
│   └── make_figures.py         # every data figure, from results/
└── src/
    ├── colors.py
    ├── schema.py               # feature set, raw-log layout, fast timestamp/URL parsing
    ├── aggregate.py            # parallel byte-range streaming aggregator (93 GB -> 8.6 MB)
    ├── cohort.py               # reproducible role-stratified subset selection
    ├── prepare.py              # context/label/split attachment + dataset manifest
    ├── preprocess.py           # aggregated-table loading, context, labelling, time split
    ├── baseline.py             # Profile Builder, Role-Aware Normaliser, Anomaly
    │                           # Scorer (PerUserBaselineEngine); PopulationBaseline (RF)
    ├── features.py             # causal trailing-window aggregates (--temporal)
    ├── detectors.py            # one fit/score interface over all 45 detectors
    ├── experiment.py           # the controlled comparison + report rendering
    ├── explain.py              # SHAP TreeExplainer wrapper (population RF)
    ├── studies.py              # seed robustness, history length, explanation fidelity
    └── evaluate.py             # alert-budget operating points, subgroup FPR,
                                # archetype recall, queue depth, detection delay, cost
```

## Architecture, Three Components (Section 4.3 of the proposal)

| Component             | Class                              | Role                                                                                             |
| --------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| Profile Builder       | `BehaviouralProfiler`              | Per-user mean, standard deviation **and percentiles** (p25/50/75/90/99) from the training window   |
| Role-Aware Normaliser | `BehaviouralProfiler(scope=...)`   | `population` / `role` / `personal` are one code path with one parameter changed, cold-start users fall back role-ward, and the three scopes give sub-question 2 its ablation |
| Anomaly Scorer        | `PerUserBaselineEngine`            | Isolation Forest / LOF / One-Class SVM fitted on the deviation vectors                             |

Deviations are available as a z-score (`mean`/`std`) or in robust form
(`median`/`IQR`). Behavioural counts are zero-inflated and right-skewed, so one
spike inflates the standard deviation and hides the next one; the robust form
does not have that failure mode. Both are evaluated rather than assumed.

A single, cheaply-trained global anomaly model becomes per-user-aware because its input space is already expressed as "how unusual for _this_ individual/role" rather than raw activity magnitude. This is what keeps fit time low enough for consumer hardware (no GPU required).

## Comparison Baseline (Section 4.4)

`PopulationBaseline`, a `RandomForestClassifier` trained on SMOTE-balanced, population-level (non-normalised) features, following Noever (2019)'s finding that Random Forest is the strongest population-level classifier for this dataset.

`src/detectors.py` widens this into the full comparison set, all fitted on the
same split and scored on the same rows:

- **Floors**. `random`, `volume` (rank by total activity). A detector that cannot beat "who was busiest" has not earned a place in a dissertation.
- **Population supervised**. `rf`, `dt`, `gb`, `lr`: the four classifiers Noever compares, so the baseline is a published operating point rather than one chosen here.
- **Population unsupervised**. `pop-iforest`, `pop-lof`, `pop-ocsvm`: the *same* anomaly algorithms the framework uses, on raw features instead of per-user deviations. This is the control that makes the headline result evidence about per-user normalisation rather than about Isolation Forest.
- **The framework**. Across all three scopes, both statistics, all three algorithms.

## Evaluation Metrics (Section 4.5)

Implemented in `src/evaluate.py`: Recall (malicious class), False Positive Rate (high-privilege subgroup and per-role breakdown), F1, ROC-AUC, and computational cost (fit time, inference latency, peak memory).

Three additions were forced by the benchmark, and [`docs/evaluation.md`](docs/evaluation.md) explains each:

- **Alert budgets instead of a shared threshold.** A Random Forest probability and an Isolation Forest path length are not the same unit, so thresholding both at 0.5 compares a probability against an arbitrary constant. Each detector instead flags the top *b*% of test rows, which is also how detection is deployed, since analyst teams have a fixed investigation capacity, not a fixed threshold.
- **Average precision as the primary metric.** At one malicious day in 31 000, ROC-AUC is near-saturated for every detector.
- **Insider-level recall and `alert_share_high_privilege`.** Day-level recall over 38 positives moves in 2.6-point steps; whether the *person* surfaced at all is more stable. And subgroup FPR understates alert fatigue, because high-privilege users are a small slice of the population. The share of the queue they occupy is what an analyst experiences.

## Results so far (CERT r6.2, full 4 000-user population)

Held-out time-based test split: **403 245 user-days, 38 malicious, 2 insiders**.
Alert budget 0.1% to 403 alerts, about three a day for five months across
4 000 users. High-privilege users are 2.43% of the population.

Reproduce with `python script/experiment.py --preset core` and
`--preset baselines`; the tables below come straight from `results/`.

| Detector                        | Family        | Recall (38 days) | Insiders | AP     | ROC-AUC | FPR (high-priv) | Alert share (high-priv) | Fit    | Peak RAM |
| ------------------------------- | ------------- | ---------------- | -------- | ------ | ------- | --------------- | ----------------------- | ------ | -------- |
| **`peruser-ocsvm-personal-zscore`** | **framework** | **0.553** (21)   | **2/2**  | 0.0383 | 0.9665  | 0.0014          | **3.5%**                | 7.9 s  | 1.3 GB   |
| `rf` (proposal's baseline)      | pop-sup       | 0.211 (8)        | 1/2      | 0.0077 | 0.6049  | 0.0093          | 22.6%                   | 447 s  | 1.9 GB   |
| `peruser-ocsvm-role-zscore`     | framework     | 0.079 (3)        | 1/2      | 0.0433 | 0.6748  | 0.0003          | 0.7%                    | 7.1 s  | not measured |
| `pop-ocsvm`                     | pop-unsup     | 0.079 (3)        | 1/2      | 0.0364 | 0.4455  | 0.0086          | 20.8%                   | 2.7 s  | 0.2 GB   |
| `lr`                            | pop-sup       | 0.053 (2)        | 1/2      | 0.0284 | 0.8643  | 0.0040          | 9.7%                    | 7 s    | not measured |
| `volume` (floor)                | floor         | 0.053 (2)        | 1/2      | 0.0284 | 0.8205  | 0.0016          | 4.0%                    | 0 s    | not measured |
| `peruser-iforest-personal-zscore` | framework   | 0.079 (3)        | 1/2      | 0.0030 | 0.9207  | 0.0032          | 7.7%                    | 7.7 s  | not measured |
| `gb`                            | pop-sup       | 0.026 (1)        | 1/2      | 0.0003 | 0.3792  | 0.0182          | 44.4%                   | 15 s   | not measured |
| `pop-iforest`                   | pop-unsup     | 0.000 (0)        | 0/2      | 0.0009 | 0.9352  | 0.0203          | 49.4%                   | 3.7 s  | not measured |
| `dt`                            | pop-sup       | 0.000 (0)        | 0/2      | 0.0001 | 0.5000  | 0.0011          | 2.7%                    | 70 s   | not measured |
| `random` (floor)                | floor         | 0.000 (0)        | 0/2      | 0.0001 | 0.4451  | 0.0008          | 2.0%                    | 0 s    | not measured |

These are early results on a benchmark with two insiders in the test window,
reported as they came out, including the parts that do not flatter the
hypothesis.

**The headline comparison supports the proposal.** Against the Random Forest
baseline the framework finds **2.6x the malicious days** (21 vs 8), is the only
detector to surface **both** insiders, spends **6.5x less** of the alert queue
on high-privilege users (3.5% vs 22.6%), and fits in **7.9 s instead of 447 s**.
The Random Forest is slow precisely because SMOTE-balancing six positive rows
against 990 765 negatives produces a 1 981 518-row training set.

**Sub-question 2 is answered cleanly.** High-privilege users are 2.4% of the
population but take 49% of `pop-iforest`'s alerts and 44% of `gb`'s. Every
per-user variant sits between 0.7% and 7.7%. `pop-iforest` and
`peruser-iforest-personal-zscore` are the *same algorithm* differing only in
reference frame, 49.4% versus 7.7%. That isolates the effect to the
normalisation, which is the point of the control.

**The Anomaly Scorer choice matters more than expected, and interacts with the
feature set.** Section 4.3 asks for Isolation Forest, LOF and One-Class SVM to
be compared and the best selected; that selection is not cosmetic here. On base
features One-Class SVM reaches 0.553 recall where Isolation Forest reaches
0.079. But adding causal rolling-window features (`--temporal`) reverses it:
Isolation Forest improves 9.2x (AP 0.0041 -> 0.0378) while One-Class SVM
*degrades* badly (recall 0.553 -> 0.053), which is the expected behaviour of an
RBF kernel when 13 features become 39 highly correlated ones. **Algorithm and
feature set have to be chosen together**, and the best overall configuration is
One-Class SVM on per-user z-scores with base features.

**A trivial volume ranker is a strong baseline on r6.2, and the literature
rarely reports one.** `volume` matches or beats `lr`, `gb`, `dt`, `pop-iforest`
and `random`. Scenario 2's mass download is simply the highest-volume user-day
in the test window. Any result on this benchmark that does not clear this floor
should be treated with suspicion.

**Two supervised baselines fail outright, instructively.** `dt` emits exactly
two distinct scores (0.0 and 1.0), trained to purity on SMOTE-balanced data it
produces hard labels and therefore *no ranking*, which its `tie_fraction` of
0.995 exposes. `gb` scores ROC-AUC 0.3792, worse than random. Both follow from
learning a class from six examples, which is what a chronological split of r6.2
leaves.

**Average precision and recall disagree, and both are reported.**
`peruser-ocsvm-role-zscore` has the higher AP (0.0433 vs 0.0383) but finds 3
malicious days to the personal variant's 21. AP is dominated by the extreme top
of the ranking; a detector that puts one day first and then nothing scores well
on it. This is why the report carries insider-level recall and the alert-budget
sweep rather than a single number.

**The magnitude ablation is a clean negative result.** Feeding
population-scaled raw magnitudes alongside per-user deviations (`...-mag`)
lowers AP and raises the high-privilege alert share 4-5x (7.7% -> 36.0%), it
re-creates precisely the population-baseline failure the research exists to
remove. Kept in the ablation for that reason.

### Follow-up studies (`python script/studies.py`)

**Ten seeds.** Re-drawing every random component (fit subsample, forests,
SMOTE, tie-break) ten times: the framework finds 20 to 21 of the 38 malicious
days and both insiders in every seed; the Random Forest 3 to 8 days and one
insider (its single-seed 8 was its best draw). 17 of 18 paired comparisons go
the same way in all ten seeds (Wilcoxon, Holm-adjusted p = 0.035, the floor
for this design): higher recall than every detector and a lower
high-privilege alert share than every population detector. Not every
comparison favours it: average precision ties with the same-algorithm control
(5 each), and the role-scoped variant beats it on AP and alert share while
finding a seventh of the days. The framework is also the only detector that
puts both insiders inside its first 200 alerts, the Random Forest leaves
CDE1846 ~24 000 alerts down.

**Sub-question 3, history needed.** Cutting every user's training history to
their most recent N active days: a personal baseline is *stable* after about
30 days (benign test days with any |z| > 3 fall from 34% at 3 days to 8.7%,
then flatten near 6%) and *useful* after about 60 (recall 0.34 and both
insiders, against 0.079 for the role baseline; 0.50 to 0.58 from 120 days). Below
30 days it is worse than the role baseline, and with the Isolation Forest
scorer a 3-day baseline puts 44% of alerts on administrators. The 14-day
cold-start default is too low; ~60 days is what the data supports.

**Sub-question 5, explanations against the answer key.** Replaying the
official answer key through the aggregator gives the features each malicious
day actually moved. The framework's top-named reason is one of them on 94.7%
of malicious test days (100% of its own alerts); a random explanation manages
29.6%. The same method against the population scores 52.6%. The Random
Forest's SHAP values score 2.6% ranked by |φ|, on low-probability days the
largest values are reasons *against* an alert, and 50.0% read for positive
contributions only (87.5% on its own 8 alerts). The dashboard now shows only
positive contributions.

The full write-up is the [research report](report/), built from these results.

Full tables, the alert-budget sweep and per-role breakdowns are in `results/`
(`core`, `core-temporal`, `baselines`, and a 45-detector `cohort400/full`
ablation). [`docs/evaluation.md`](docs/evaluation.md) explains the experimental
design and the two model defects this evaluation caught.
