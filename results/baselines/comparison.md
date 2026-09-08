# Detector comparison

Dataset `dataset.parquet` · preset `baselines` · 9 detectors · 715.5s

- Train 990,765 rows / test 403,245 rows, split at 2010-12-29
- 38 malicious user-days in the test split (0.0094%)
- Features: 13 (base)
- Operating point: top 0.100% of test rows by score, applied identically to every detector

## Ground-truth archetypes

- **burst**: 3 insider(s) (ACM2278, MBG3183, PLJ1771), 0 malicious day(s) in the test split
- **low_and_slow**: 2 insider(s) (CDE1846, CMP2946), 38 malicious day(s) in the test split

> **No test data for: burst.** r6.2's insider windows are not spread evenly across the calendar, so a time-based split does not divide the archetypes between train and test. Every burst insider acts before the split date and every low-and-slow insider acts after it. The test split therefore measures low-and-slow detection only, and the supervised baselines are being asked to generalise from burst training examples to low-and-slow test ones. That is a property of the benchmark, not of the split fraction: moving the boundary to capture a burst insider in test would leave the supervised baselines with no positive training rows at all. It is reported rather than engineered away.

## Results

| detector    | family                  | roc_auc | avg_precision | recall | precision | lift     | insider_recall | n_insiders_detected | recall_low_and_slow | fpr_high_privilege | alert_share_high_privilege | tie_fraction | fit_seconds | score_ms_per_row |
| ----------- | ----------------------- | ------- | ------------- | ------ | --------- | -------- | -------------- | ------------------- | ------------------- | ------------------ | -------------------------- | ------------ | ----------- | ---------------- |
| pop-ocsvm   | population-unsupervised | 0.4455  | 0.0364        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0086             | 0.2084                     | 0.0000       | 3.3090      | 0.0242           |
| lr          | population-supervised   | 0.8643  | 0.0284        | 0.0526 | 0.0050    | 52.6700  | 0.5000         | 1                   | 0.0526              | 0.0040             | 0.0968                     | 0.0000       | 6.9190      | 0.0003           |
| volume      | floor                   | 0.8205  | 0.0284        | 0.0526 | 0.0050    | 52.6700  | 0.5000         | 1                   | 0.0526              | 0.0016             | 0.0397                     | 0.0000       | 0.0000      | 0.0000           |
| rf          | population-supervised   | 0.6049  | 0.0077        | 0.2105 | 0.0199    | 210.6500 | 0.5000         | 1                   | 0.2105              | 0.0093             | 0.2258                     | 0.2457       | 399.8070    | 0.0019           |
| pop-iforest | population-unsupervised | 0.9352  | 0.0009        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0203             | 0.4938                     | 0.0000       | 3.6550      | 0.0049           |
| gb          | population-supervised   | 0.3792  | 0.0003        | 0.0263 | 0.0025    | 26.3300  | 0.5000         | 1                   | 0.0263              | 0.0182             | 0.4442                     | 0.0000       | 15.0430     | 0.0007           |
| pop-lof     | population-unsupervised | 0.8313  | 0.0003        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0000             | 0.0000                     | 0.0050       | 17.7280     | 0.3763           |
| dt          | population-supervised   | 0.5000  | 0.0001        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0011             | 0.0273                     | 0.9950       | 70.4570     | 0.0001           |
| random      | floor                   | 0.4451  | 0.0001        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0008             | 0.0199                     | 0.0000       | 0.0000      | 0.0000           |


Ranked by average precision. `recall`, `precision` and the FPR columns are measured at the shared alert budget above; `roc_auc` and `avg_precision` are threshold-free. `tie_fraction` is the share of a detector's alerts drawn arbitrarily from a group of equal scores at the cut -- a high value means the ranking, not the metric, is what ran out.
