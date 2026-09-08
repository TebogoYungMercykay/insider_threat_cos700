# Detector comparison

Dataset `dataset.parquet` · preset `core` · 11 detectors · 727.8s

- Train 990,765 rows / test 403,245 rows, split at 2010-12-29
- 38 malicious user-days in the test split (0.0094%)
- Features: 39 (base + causal rolling windows)
- Operating point: top 0.100% of test rows by score, applied identically to every detector

## Ground-truth archetypes

- **burst**: 3 insider(s) (ACM2278, MBG3183, PLJ1771), 0 malicious day(s) in the test split
- **low_and_slow**: 2 insider(s) (CDE1846, CMP2946), 38 malicious day(s) in the test split

> **No test data for: burst.** r6.2's insider windows are not spread evenly across the calendar, so a time-based split does not divide the archetypes between train and test. Every burst insider acts before the split date and every low-and-slow insider acts after it. The test split therefore measures low-and-slow detection only, and the supervised baselines are being asked to generalise from burst training examples to low-and-slow test ones. That is a property of the benchmark, not of the split fraction: moving the boundary to capture a burst insider in test would leave the supervised baselines with no positive training rows at all. It is reported rather than engineered away.

## Results

| detector                            | family                  | roc_auc | avg_precision | recall | precision | lift     | insider_recall | n_insiders_detected | recall_low_and_slow | fpr_high_privilege | alert_share_high_privilege | tie_fraction | fit_seconds | score_ms_per_row |
| ----------------------------------- | ----------------------- | ------- | ------------- | ------ | --------- | -------- | -------------- | ------------------- | ------------------- | ------------------ | -------------------------- | ------------ | ----------- | ---------------- |
| peruser-ocsvm-role-zscore           | per-user-framework      | 0.7235  | 0.0387        | 0.1053 | 0.0099    | 105.3300 | 0.5000         | 1                   | 0.1053              | 0.0001             | 0.0025                     | 0.0000       | 13.6500     | 0.0441           |
| peruser-iforest-personal-robust     | per-user-framework      | 0.9116  | 0.0378        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0050             | 0.1216                     | 0.0000       | 12.1410     | 0.0098           |
| volume                              | floor                   | 0.8195  | 0.0281        | 0.0526 | 0.0050    | 52.6700  | 0.5000         | 1                   | 0.0526              | 0.0011             | 0.0273                     | 0.0000       | 0.0000      | 0.0001           |
| pop-ocsvm                           | population-unsupervised | 0.3583  | 0.0280        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0062             | 0.1514                     | 0.0000       | 5.1720      | 0.0382           |
| peruser-ocsvm-personal-zscore       | per-user-framework      | 0.9891  | 0.0125        | 0.0526 | 0.0050    | 52.6700  | 0.5000         | 1                   | 0.0526              | 0.0015             | 0.0372                     | 0.0000       | 15.1040     | 0.0431           |
| peruser-iforest-personal-zscore     | per-user-framework      | 0.9524  | 0.0106        | 0.0789 | 0.0074    | 78.9900  | 0.5000         | 1                   | 0.0789              | 0.0033             | 0.0794                     | 0.0000       | 12.3800     | 0.0095           |
| peruser-ocsvm-personal-robust       | per-user-framework      | 0.9732  | 0.0047        | 0.0263 | 0.0025    | 26.3300  | 0.5000         | 1                   | 0.0263              | 0.0005             | 0.0124                     | 0.0000       | 15.2300     | 0.0386           |
| peruser-iforest-personal-zscore-mag | per-user-framework      | 0.9399  | 0.0012        | 0.0263 | 0.0025    | 26.3300  | 0.5000         | 1                   | 0.0263              | 0.0148             | 0.3598                     | 0.0000       | 12.5830     | 0.0097           |
| rf                                  | population-supervised   | 0.5520  | 0.0009        | 0.1053 | 0.0099    | 105.3300 | 0.5000         | 1                   | 0.1053              | 0.0073             | 0.1787                     | 0.7072       | 493.9320    | 0.0019           |
| peruser-iforest-role-zscore         | per-user-framework      | 0.9444  | 0.0009        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0000             | 0.0000                     | 0.0000       | 11.2460     | 0.0080           |
| pop-iforest                         | population-unsupervised | 0.9180  | 0.0006        | 0.0000 | 0.0000    | 0.0000   | 0.0000         | 0                   | 0.0000              | 0.0098             | 0.2382                     | 0.0000       | 4.3330      | 0.0056           |


Ranked by average precision. `recall`, `precision` and the FPR columns are measured at the shared alert budget above; `roc_auc` and `avg_precision` are threshold-free. `tie_fraction` is the share of a detector's alerts drawn arbitrarily from a group of equal scores at the cut -- a high value means the ranking, not the metric, is what ran out.
