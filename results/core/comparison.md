# Detector comparison

Dataset `dataset.parquet` · preset `core` · 11 detectors · 609.5s

- Train 990,765 rows / test 403,245 rows, split at 2010-12-29
- 38 malicious user-days in the test split (0.0094%)
- Features: 13 (base)
- Operating point: top 0.100% of test rows by score, applied identically to every detector

## Ground-truth archetypes

- **burst**: 3 insider(s) (ACM2278, MBG3183, PLJ1771), 0 malicious day(s) in the test split
- **low_and_slow**: 2 insider(s) (CDE1846, CMP2946), 38 malicious day(s) in the test split

> **No test data for: burst.** r6.2's insider windows are not spread evenly across the calendar, so a time-based split does not divide the archetypes between train and test. Every burst insider acts before the split date and every low-and-slow insider acts after it. The test split therefore measures low-and-slow detection only, and the supervised baselines are being asked to generalise from burst training examples to low-and-slow test ones. That is a property of the benchmark, not of the split fraction: moving the boundary to capture a burst insider in test would leave the supervised baselines with no positive training rows at all. It is reported rather than engineered away.

## Results

| detector                            | family                  | roc_auc | avg_precision | recall | precision | lift     | insider_recall | n_insiders_detected | recall_low_and_slow | fpr_high_privilege | alert_share_high_privilege | tie_fraction | fit_seconds | score_ms_per_row |
| ----------------------------------- | ----------------------- | ------- | ------------- | ------ | --------- | -------- | -------------- | ------------------- | ------------------- | ------------------ | -------------------------- | ------------ | ----------- | ---------------- |
| peruser-ocsvm-role-zscore           | per-user-framework      | 0.6748  | 0.0433        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0003             | 0.0074                     | 0.0000       | 7.0770      | 0.0276           |
| peruser-ocsvm-personal-zscore       | per-user-framework      | 0.9665  | 0.0383        | 0.5526 | 0.0521    | 552.9700 | 1.0000         | 2                   | 0.5526              | 0.0014             | 0.0347                     | 0.0000       | 7.8590      | 0.0253           |
| pop-ocsvm                           | population-unsupervised | 0.4455  | 0.0364        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0086             | 0.2084                     | 0.0000       | 2.7340      | 0.0241           |
| volume                              | floor                   | 0.8205  | 0.0284        | 0.0526 | 0.0050    | 52.6700  | 0.5000         | 1                   | 0.0526              | 0.0016             | 0.0397                     | 0.0000       | 0.0000      | 0.0000           |
| rf                                  | population-supervised   | 0.6049  | 0.0077        | 0.2105 | 0.0199    | 210.6500 | 0.5000         | 1                   | 0.2105              | 0.0093             | 0.2258                     | 0.2457       | 446.8370    | 0.0018           |
| peruser-iforest-personal-robust     | per-user-framework      | 0.8529  | 0.0041        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0066             | 0.1613                     | 0.0000       | 7.8650      | 0.0074           |
| peruser-ocsvm-personal-robust       | per-user-framework      | 0.8566  | 0.0031        | 0.0263 | 0.0025    | 26.3300  | 0.5000         | 1                   | 0.0263              | 0.0002             | 0.0050                     | 0.0000       | 8.6900      | 0.0246           |
| peruser-iforest-personal-zscore     | per-user-framework      | 0.9207  | 0.0030        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0032             | 0.0769                     | 0.0000       | 7.7300      | 0.0083           |
| peruser-iforest-personal-zscore-mag | per-user-framework      | 0.9364  | 0.0018        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0148             | 0.3598                     | 0.0000       | 8.3390      | 0.0094           |
| peruser-iforest-role-zscore         | per-user-framework      | 0.9324  | 0.0012        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0007             | 0.0174                     | 0.0000       | 6.4190      | 0.0064           |
| pop-iforest                         | population-unsupervised | 0.9352  | 0.0009        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0203             | 0.4938                     | 0.0000       | 3.7300      | 0.0056           |


Ranked by average precision. `recall`, `precision` and the FPR columns are measured at the shared alert budget above; `roc_auc` and `avg_precision` are threshold-free. `tie_fraction` is the share of a detector's alerts drawn arbitrarily from a group of equal scores at the cut -- a high value means the ranking, not the metric, is what ran out.
