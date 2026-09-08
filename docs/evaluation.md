# Evaluation — Comparing Detectors Without Fooling Yourself

This document covers the experiment that answers the proposal's research
questions: what is compared, how the comparison is made fair, and which
decisions were forced by properties of the benchmark rather than chosen.

Run it with:

```bash
python script/experiment.py --preset core      # the headline three-way comparison
python script/experiment.py --preset full      # all 27 detectors
python script/experiment.py --preset ablation  # scope x statistic x algorithm
```

Every run writes `comparison.csv`, `comparison.md`, `budget_sweep.csv`,
`fpr_by_role.csv` and `run.json` into `results/<preset>/`. `run.json` records
the dataset, the split, the feature set, the seed, the platform and the
elapsed time, so a table in the dissertation can always be traced back to the
run that produced it.

## What the benchmark actually gives you

r6.2 is a harder evaluation target than its headline numbers suggest, and
three of its properties shape everything below.

**The positive class is 44 user-days in 1 394 010.** That is one in roughly
31 000. Accuracy is useless, ROC-AUC is close to useless — a detector can rank
the 1.39 million benign days beautifully and still put no insider near the top
— and day-level recall over 38 test positives moves in steps of 2.6
percentage points.

**The time-based split leaves six positive training rows.** The 70/30 split by
date puts 6 malicious user-days in the training window and 38 in the test
window. Every supervised classifier in the comparison is therefore learning
the malicious class from six examples. This is not a flaw in the split; it is
what a chronological split of this benchmark yields, and it is a fair
representation of the deployment problem, where the labels you have are
always from attacks that already happened.

**The archetypes are not spread across the split.** Classifying each insider
by the calendar span of their malicious window — the geometry is in the answer
key, fixed before any model runs — gives three burst insiders and two
low-and-slow ones:

| Archetype    | Insiders                  | Window span | Malicious days | Falls in |
| ------------ | ------------------------- | ----------- | -------------- | -------- |
| burst        | ACM2278, PLJ1771, MBG3183 | 0–6 days    | 6              | train    |
| low_and_slow | CMP2946, CDE1846          | 56–63 days  | 38             | test     |

Every burst insider acts before the split date; both low-and-slow insiders act
after it. So the test split measures low-and-slow detection **only**, and the
supervised baselines are being asked to generalise from burst training
examples to low-and-slow test ones.

This is reported rather than engineered away. Moving the split boundary to put
a burst insider in the test window would leave the supervised baselines with
no positive training rows at all, which trades one limitation for a worse one.
The constraint belongs in the write-up, not in the data.

## The comparison set

Detectors live in `src/detectors.py` behind one interface: fit on the training
split, return one score per row, higher meaning more suspicious. No detector
sets its own threshold.

| Family                    | Detectors                                  | Why it is in the comparison                                                     |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------- |
| `floor`                   | `random`, `volume`                         | A detector that cannot beat "rank everyone by how busy they were" has not earned a place in a dissertation |
| `population-supervised`   | `rf`, `dt`, `gb`, `lr`                     | The four classifiers Noever (2019) compares on this dataset, so the baseline is a published operating point rather than one chosen here |
| `population-unsupervised` | `pop-iforest`, `pop-lof`, `pop-ocsvm`      | The isolating control — see below                                               |
| `per-user-framework`      | `peruser-<algorithm>-<scope>-<statistic>`  | The proposed framework, across all three normalisation scopes                    |

### The control that makes the result mean something

`pop-iforest` runs the **same Isolation Forest** as the framework, on
standardised raw features instead of per-user deviation vectors. Holding the
algorithm fixed and changing only the reference frame is what makes the
headline comparison evidence about per-user normalisation rather than evidence
about Isolation Forest. Without it, "our Isolation Forest beat their Random
Forest" is a claim about two unrelated changes at once.

The same logic runs through the ablation preset: `population`, `role` and
`personal` scopes are one code path with one parameter changed, so the
difference between them is not confounded by anything else.

## Making the operating point fair

Two detectors emit incomparable numbers. A Random Forest emits a calibrated
probability; Isolation Forest emits a negated path length. Thresholding both
at 0.5 does not compare two detectors — it compares a probability against an
arbitrary constant.

So every threshold here is derived from an **alert budget**: each detector may
flag the top *b* fraction of test rows, and the question is who it spends
those alerts on. This is also how detection is deployed. An analyst team has a
fixed number of investigations per day, not a fixed decision threshold. The
default budget of 0.1% is about 400 alerts across five months and 4 000 users
— roughly three a day.

Alongside it the report carries two threshold-free metrics. **Average
precision** is the primary one: it integrates over every threshold and, unlike
ROC-AUC, does not saturate at a 1-in-31 000 positive rate.

### Ties, and why they needed handling

Scoring ties are the normal case here, not an edge case. A Random Forest
trained on six positives assigns probability 0.0 to almost every test row.

The first implementation flagged every row meeting the cut score, so that row
order could never decide the outcome. That let the Random Forest spend 40 360
alerts against a budget of 404 and then report the recall that bought. Taking
the first *k* in array order instead would make the metric depend on how the
Parquet file happened to be sorted.

Ties are therefore broken by a **seeded permutation**: exactly *k* rows are
flagged, the choice among equals is arbitrary, reproducible, and independent
of both the detector and the row order. A detector whose ranking is degenerate
at the cut is not rewarded for it — it gets a random sample of its own tied
group, which is what it would deliver in deployment.

The `tie_fraction` column reports how much of each detector's budget was spent
that way. A high value means the ranking, not the metric, is what ran out.

## Metrics

| Metric                       | What it answers                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| `avg_precision`              | Threshold-free ranking quality; the primary comparison metric                                  |
| `roc_auc`                    | Reported for continuity with prior work; near-saturated at this positive rate                  |
| `recall`, `precision`, `f1`  | At the shared alert budget                                                                     |
| `lift`                       | Precision over the base rate — how many times better than spending the budget at random        |
| `insider_recall`             | Did the detector surface this *person* at all? Coarser than day-level recall and far more stable over 38 positives |
| `recall_low_and_slow`        | Sub-question 1                                                                                 |
| `fpr_high_privilege`         | Sub-question 2                                                                                 |
| `alert_share_high_privilege` | Sub-question 2, as an analyst experiences it: what share of the queue is a power user doing their job? |
| `fit_seconds`, `score_ms_per_row`, `peak_memory_mb` | Sub-question 4                                                          |
| `tie_fraction`               | How much of the operating point was a coin toss                                                |

`alert_share_high_privilege` is reported next to `population_share_high_privilege`
because subgroup FPR alone understates the problem: high-privilege users are a
small slice of the population, so a low FPR can still mean they dominate the
alert queue.

## Evaluation always covers the full test split

Never a sample. With 38 positives in 403 245 rows, a 5 000-row sample contains
0.47 of them in expectation, so every metric computed on one is noise wearing
a number's clothes. Scoring the full split costs a few seconds.

## Two model defects this evaluation design caught

Both were found by the diagnostics above and fixed in the model, not worked
around in the data.

**The risk score clipped, destroying the ranking it existed to produce.**
`score_array()` min-max scaled the raw anomaly score against the 1st and
99.5th training percentiles and clipped to [0, 1]. Every test row above the
99.5th percentile therefore collapsed to exactly 1.0 — the entire top of the
ranking, which is the only part that matters at a 0.1% alert budget. The
framework was asked for 4 alerts and returned 174 indistinguishable rows.
Ranking now uses the native anomaly score (`raw_score()`), and the bounded
[0, 1] value is a display-only calibration that stays strictly ordered with
it. `test_calibrated_score_is_monotone_and_never_saturates` pins this.

**`tie_fraction` was off by one.** It counted the row *at* the cut as
arbitrary even when that row was uniquely determined, so a perfectly ranked
detector reported a tie fraction of 1/k instead of 0.
`test_tie_fraction_reports_how_much_of_the_budget_was_a_coin_toss` pins it.

## Temporal features

`--temporal` adds trailing per-user rolling means (7 and 30 active days) for
every feature. A single user-day row cannot express whether it is part of a
three-week drift or an isolated blip, which is structurally the thing
sub-question 1 asks about. The proposal's risk-management section anticipates
this extension explicitly.

Two properties keep it honest. The windows are **causal** — each covers the
current row and earlier rows of the same user only, so they can be computed
over the whole table before the split without leaking test information into
training (`assert_causal()` verifies this, and a test confirms the check
actually fails on a forward-looking window). And they are offered to **every**
detector, so any difference remains a difference in modelling strategy rather
than one side being handed better inputs.

### They help one algorithm and hurt another

The rolling windows are not a uniform improvement, and the direction depends
on the Anomaly Scorer:

| Detector (full population)        | AP base | AP + rolling | Recall base | Recall + rolling |
| --------------------------------- | ------- | ------------ | ----------- | ---------------- |
| `peruser-iforest-personal-robust` | 0.0041  | **0.0378**   | 0.079       | 0.079            |
| `peruser-iforest-personal-zscore` | 0.0030  | 0.0106       | 0.079       | 0.079            |
| `peruser-ocsvm-personal-zscore`   | 0.0383  | 0.0125       | **0.553**   | 0.053            |
| `volume` (floor)                  | 0.0284  | 0.0281       | 0.053       | 0.053            |
| `pop-iforest`                     | 0.0009  | 0.0006       | 0.000       | 0.000            |

Isolation Forest gains 9.2x from them; One-Class SVM loses most of its recall.
That is the expected behaviour of an RBF kernel when 13 features become 39
highly correlated ones — added dimensions dilute the distance metric — whereas
Isolation Forest samples features per split and tolerates the redundancy.

Two things follow. **Algorithm and feature set have to be selected together**,
not independently, so the ablation grid crosses them rather than fixing one.
And the population detectors gain nothing from the same features
(`volume` 0.0284 -> 0.0281, `pop-iforest` 0.0009 -> 0.0006), which is the
result worth reporting: behavioural drift is only informative *relative to an
individual's own baseline*.

The best configuration overall on r6.2 is One-Class SVM on per-user z-scores
with **base** features, and `train.py` defaults to it.

## Reproducibility

Every stochastic component takes the run's seed: the anomaly models, the
SMOTE resampling, the fit subsampling, and the tie-break permutation. LOF and
One-Class SVM are quadratic in the training set and are fitted on a seeded
subsample of 50 000 training rows — they cannot be fitted on 990 765 rows on
consumer hardware, and the proposal's feasibility claim is about consumer
hardware. Profiles are always built on the full training split, and scoring
always covers every test row. The subsample size is recorded in `run.json`.
