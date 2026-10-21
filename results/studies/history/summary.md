# History length (sub-question 3)

Dataset `dataset.parquet` · seeds [42, 1, 2] · alert budget 0.10% · 719.0s

Truncation: each user's most recent N active training days. The test window is unchanged.

## Baseline stability

`mean_shift_*` is the distance between the N-day and full-history per-user mean, in full-history standard deviations. `benign_day_exceedance` is the share of benign test days with any feature beyond |z| = 3.0 under the N-day profile.

| history_days | n_train_rows | mean_shift_median_sd | mean_shift_p90_sd | within_half_sd | log_std_ratio_median | share_zero_std | benign_day_exceedance | benign_cell_exceedance | share_scored_personal |
| ------------ | ------------ | -------------------- | ----------------- | -------------- | -------------------- | -------------- | --------------------- | ---------------------- | --------------------- |
| 3            | 12000        | 0.3542               | 0.8048            | 0.6963         | 0.3744               | 0.2488         | 0.3437                | 0.0379                 | 1.0000                |
| 7            | 28000        | 0.2405               | 0.5620            | 0.8571         | 0.2351               | 0.1208         | 0.1964                | 0.0220                 | 1.0000                |
| 14           | 56000        | 0.1614               | 0.3998            | 0.9560         | 0.1620               | 0.0681         | 0.1294                | 0.0143                 | 1.0000                |
| 30           | 119959       | 0.1153               | 0.2789            | 0.9949         | 0.1084               | 0.0383         | 0.0872                | 0.0096                 | 1.0000                |
| 60           | 239095       | 0.0795               | 0.1937            | 0.9997         | 0.0739               | 0.0228         | 0.0726                | 0.0080                 | 1.0000                |
| 90           | 357268       | 0.0561               | 0.1347            | 0.9999         | 0.0527               | 0.0175         | 0.0670                | 0.0074                 | 1.0000                |
| 120          | 474648       | 0.0422               | 0.1038            | 1.0000         | 0.0398               | 0.0129         | 0.0636                | 0.0070                 | 1.0000                |
| 180          | 706902       | 0.0243               | 0.0629            | 1.0000         | 0.0235               | 0.0060         | 0.0616                | 0.0068                 | 1.0000                |
| all          | 990765       | 0.0000               | 0.0000            | 1.0000         | 0.0000               | 0.0000         | 0.0594                | 0.0066                 | 1.0000                |

## Detection (mean across seeds)

| detector                        | history_days | avg_precision | recall | insider_recall | alert_share_high_privilege | fit_seconds |
| ------------------------------- | ------------ | ------------- | ------ | -------------- | -------------------------- | ----------- |
| peruser-ocsvm-personal-zscore   | 3            | 0.0005        | 0.0000 | 0.0000         | 0.0265                     | 0.5560      |
| peruser-iforest-personal-zscore | 3            | 0.0006        | 0.0000 | 0.0000         | 0.4433                     | 2.9563      |
| peruser-ocsvm-personal-zscore   | 7            | 0.0010        | 0.0000 | 0.0000         | 0.0157                     | 1.4553      |
| peruser-iforest-personal-zscore | 7            | 0.0011        | 0.0088 | 0.1667         | 0.2051                     | 3.7797      |
| peruser-ocsvm-personal-zscore   | 14           | 0.0036        | 0.0175 | 0.3333         | 0.0322                     | 4.0427      |
| peruser-iforest-personal-zscore | 14           | 0.0014        | 0.0000 | 0.0000         | 0.1216                     | 3.9770      |
| peruser-ocsvm-personal-zscore   | 30           | 0.0093        | 0.0526 | 0.5000         | 0.0331                     | 4.6307      |
| peruser-iforest-personal-zscore | 30           | 0.0019        | 0.0614 | 0.3333         | 0.0960                     | 4.2157      |
| peruser-ocsvm-personal-zscore   | 60           | 0.0223        | 0.3421 | 1.0000         | 0.0397                     | 4.6527      |
| peruser-iforest-personal-zscore | 60           | 0.0038        | 0.1667 | 0.5000         | 0.0719                     | 4.5023      |
| peruser-ocsvm-personal-zscore   | 90           | 0.0259        | 0.4386 | 1.0000         | 0.0397                     | 5.2730      |
| peruser-iforest-personal-zscore | 90           | 0.0028        | 0.0701 | 0.5000         | 0.0653                     | 4.9387      |
| peruser-ocsvm-personal-zscore   | 120          | 0.0297        | 0.5000 | 1.0000         | 0.0414                     | 5.6533      |
| peruser-iforest-personal-zscore | 120          | 0.0029        | 0.0877 | 0.5000         | 0.0720                     | 5.4127      |
| peruser-ocsvm-personal-zscore   | 180          | 0.0345        | 0.5789 | 1.0000         | 0.0339                     | 6.5067      |
| peruser-iforest-personal-zscore | 180          | 0.0023        | 0.0614 | 0.5000         | 0.0587                     | 6.2563      |
| peruser-ocsvm-personal-zscore   | all          | 0.0372        | 0.5526 | 1.0000         | 0.0339                     | 7.7247      |
| peruser-iforest-personal-zscore | all          | 0.0022        | 0.0701 | 0.5000         | 0.0620                     | 7.4057      |

Role-group reference (needs no personal history): AP 0.0454, recall 0.0789, insider recall 0.50, high-privilege alert share 0.0058.
