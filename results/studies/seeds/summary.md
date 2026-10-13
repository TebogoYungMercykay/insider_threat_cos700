# Seed robustness

Dataset `dataset.parquet` · 10 seeds [42, 1, 2, 3, 4, 5, 6, 7, 8, 9] · alert budget 0.10% · 5763.2s

Every random component is re-drawn per seed: the fit subsample, the forests, SMOTE and the tie-break at the cut. The split and the features are fixed.

## Mean ± standard deviation across seeds

| detector                        | avg_precision   | recall          | insider_recall  | alert_share_high_privilege | fit_seconds        |
| ------------------------------- | --------------- | --------------- | --------------- | -------------------------- | ------------------ |
| peruser-ocsvm-personal-zscore   | 0.0363 ± 0.0011 | 0.5473 ± 0.0111 | 1.0000 ± 0.0000 | 0.0340 ± 0.0020            | 10.3707 ± 5.0398   |
| peruser-ocsvm-role-zscore       | 0.0473 ± 0.0049 | 0.0789 ± 0.0000 | 0.5000 ± 0.0000 | 0.0052 ± 0.0008            | 9.4949 ± 6.6012    |
| peruser-iforest-personal-zscore | 0.0029 ± 0.0008 | 0.0763 ± 0.0083 | 0.5000 ± 0.0000 | 0.0742 ± 0.0152            | 8.0146 ± 1.1324    |
| pop-ocsvm                       | 0.0360 ± 0.0063 | 0.0789 ± 0.0000 | 0.5000 ± 0.0000 | 0.2042 ± 0.0165            | 2.9006 ± 0.5685    |
| pop-iforest                     | 0.0009 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4752 ± 0.0734            | 3.6004 ± 0.4065    |
| rf                              | 0.0049 ± 0.0027 | 0.1631 ± 0.0461 | 0.5000 ± 0.0000 | 0.2127 ± 0.0249            | 486.3532 ± 99.2178 |
| volume                          | 0.0284 ± 0.0000 | 0.0526 ± 0.0000 | 0.5000 ± 0.0000 | 0.0397 ± 0.0000            | 0.0000 ± 0.0000    |

## Paired Wilcoxon signed-rank tests vs `peruser-ocsvm-personal-zscore`

Positive `wins` mean the reference did better on that seed. `p_holm` is Holm-Bonferroni adjusted across every row.

| comparator                      | metric                     | reference_mean | comparator_mean | wins | losses | ties | p_value | p_holm | significant |
| ------------------------------- | -------------------------- | -------------- | --------------- | ---- | ------ | ---- | ------- | ------ | ----------- |
| peruser-ocsvm-role-zscore       | avg_precision              | 0.0363         | 0.0473          | 0    | 10     | 0    | 0.0020  | 0.0352 | True        |
| peruser-ocsvm-role-zscore       | recall                     | 0.5473         | 0.0789          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| peruser-ocsvm-role-zscore       | alert_share_high_privilege | 0.0340         | 0.0052          | 0    | 10     | 0    | 0.0020  | 0.0352 | True        |
| peruser-iforest-personal-zscore | avg_precision              | 0.0363         | 0.0029          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| peruser-iforest-personal-zscore | recall                     | 0.5473         | 0.0763          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| peruser-iforest-personal-zscore | alert_share_high_privilege | 0.0340         | 0.0742          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| pop-ocsvm                       | avg_precision              | 0.0363         | 0.0360          | 5    | 5      | 0    | 1.0000  | 1.0000 | False       |
| pop-ocsvm                       | recall                     | 0.5473         | 0.0789          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| pop-ocsvm                       | alert_share_high_privilege | 0.0340         | 0.2042          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| pop-iforest                     | avg_precision              | 0.0363         | 0.0009          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| pop-iforest                     | recall                     | 0.5473         | 0.0000          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| pop-iforest                     | alert_share_high_privilege | 0.0340         | 0.4752          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| rf                              | avg_precision              | 0.0363         | 0.0049          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| rf                              | recall                     | 0.5473         | 0.1631          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| rf                              | alert_share_high_privilege | 0.0340         | 0.2127          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| volume                          | avg_precision              | 0.0363         | 0.0284          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| volume                          | recall                     | 0.5473         | 0.0526          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |
| volume                          | alert_share_high_privilege | 0.0340         | 0.0397          | 10   | 0      | 0    | 0.0020  | 0.0352 | True        |

## Where each insider sits in the queue

`alerts_ahead` is the number of user-days ranked above the insider's best-ranked malicious day (threshold-free). `detected_runs` counts seeds in which the insider was flagged at the budget.

| detector                        | user    | alerts_ahead_median | detected_runs | delay_days_median |
| ------------------------------- | ------- | ------------------- | ------------- | ----------------- |
| peruser-iforest-personal-zscore | CDE1846 | 79.5                | 10            | 29.0              |
| peruser-iforest-personal-zscore | CMP2946 | 1239.0              | 0             |                   |
| peruser-ocsvm-personal-zscore   | CDE1846 | 84.5                | 10            | 29.0              |
| peruser-ocsvm-personal-zscore   | CMP2946 | 185.5               | 10            | 0.0               |
| peruser-ocsvm-role-zscore       | CDE1846 | 0.0                 | 10            | 29.0              |
| peruser-ocsvm-role-zscore       | CMP2946 | 5626.0              | 0             |                   |
| pop-iforest                     | CDE1846 | 1413.5              | 0             |                   |
| pop-iforest                     | CMP2946 | 3595.0              | 0             |                   |
| pop-ocsvm                       | CDE1846 | 0.0                 | 10            | 29.0              |
| pop-ocsvm                       | CMP2946 | 2174.5              | 0             |                   |
| rf                              | CDE1846 | 24367.0             | 0             |                   |
| rf                              | CMP2946 | 44.5                | 10            | 0.0               |
| volume                          | CDE1846 | 0.0                 | 10            | 49.0              |
| volume                          | CMP2946 | 30040.0             | 0             |                   |
