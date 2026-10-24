# Running the Experiments

This guide is the measured, step-by-step way to reproduce every result: the default laptop path on a reproducible 400-user cohort, then the full 4 000-user population behind the report's headline numbers. Each step lists the peak memory and time it took. Installing the environment and getting the dataset are covered in the [Setup Guide](setup-guide.md) and [`data/README.md`](../data/README.md).

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
[`data/README.md`](../data/README.md).

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
