# Explanation fidelity (sub-question 5)

Dataset `dataset.parquet` · 38 malicious test days · explanations of length 3 over 13 features · 771.1s

Ground truth: the features the answer key's malicious events moved on that day, replayed through the dataset's own aggregator. `chance_*` is the analytic rate of a random 3-feature explanation.

## Overall

| view                    | explainer               | n_days | hit_at_1 | hit_at_3 | precision_at_3 | chance_hit_at_1 | chance_hit_at_3 | chance_precision_at_3 |
| ----------------------- | ----------------------- | ------ | -------- | -------- | -------------- | --------------- | --------------- | --------------------- |
| all malicious test days | personal z              | 38     | 0.9474   | 0.9474   | 0.6404         | 0.2955          | 0.6211          | 0.2955                |
| all malicious test days | role z                  | 38     | 0.3947   | 0.7368   | 0.4386         | 0.2955          | 0.6211          | 0.2955                |
| all malicious test days | population z            | 38     | 0.5263   | 0.9211   | 0.5263         | 0.2955          | 0.6211          | 0.2955                |
| all malicious test days | RF SHAP                 | 38     | 0.0263   | 0.5526   | 0.1842         | 0.2955          | 0.6211          | 0.2955                |
| all malicious test days | RF SHAP (positive only) | 38     | 0.5000   | 0.5263   | 0.2632         | 0.2955          | 0.6211          | 0.2955                |
| flagged by framework    | personal z              | 21     | 1.0000   | 1.0000   | 0.6349         | 0.2601          | 0.5801          | 0.2601                |
| flagged by framework    | role z                  | 21     | 0.3333   | 0.6667   | 0.3651         | 0.2601          | 0.5801          | 0.2601                |
| flagged by framework    | population z            | 21     | 0.3333   | 0.9524   | 0.4921         | 0.2601          | 0.5801          | 0.2601                |
| flagged by framework    | RF SHAP                 | 21     | 0.0476   | 0.4762   | 0.1587         | 0.2601          | 0.5801          | 0.2601                |
| flagged by framework    | RF SHAP (positive only) | 21     | 0.2857   | 0.3333   | 0.1429         | 0.2601          | 0.5801          | 0.2601                |
| flagged by RF           | personal z              | 8      | 1.0000   | 1.0000   | 0.5833         | 0.2885          | 0.6434          | 0.2885                |
| flagged by RF           | role z                  | 8      | 0.0000   | 0.3750   | 0.1250         | 0.2885          | 0.6434          | 0.2885                |
| flagged by RF           | population z            | 8      | 0.2500   | 0.7500   | 0.3333         | 0.2885          | 0.6434          | 0.2885                |
| flagged by RF           | RF SHAP                 | 8      | 0.0000   | 0.0000   | 0.0000         | 0.2885          | 0.6434          | 0.2885                |
| flagged by RF           | RF SHAP (positive only) | 8      | 0.8750   | 1.0000   | 0.5417         | 0.2885          | 0.6434          | 0.2885                |

## By insider (all malicious test days)

| explainer               | insider | n_days | hit_at_1 | hit_at_3 | precision_at_3 | chance_hit_at_1 | chance_hit_at_3 | chance_precision_at_3 |
| ----------------------- | ------- | ------ | -------- | -------- | -------------- | --------------- | --------------- | --------------------- |
| personal z              | CDE1846 | 9      | 1.0000   | 1.0000   | 0.9630         | 0.5385          | 0.9301          | 0.5385                |
| personal z              | CMP2946 | 29     | 0.9310   | 0.9310   | 0.5402         | 0.2202          | 0.5252          | 0.2202                |
| role z                  | CDE1846 | 9      | 1.0000   | 1.0000   | 0.9630         | 0.5385          | 0.9301          | 0.5385                |
| role z                  | CMP2946 | 29     | 0.2069   | 0.6552   | 0.2759         | 0.2202          | 0.5252          | 0.2202                |
| population z            | CDE1846 | 9      | 1.0000   | 1.0000   | 1.0000         | 0.5385          | 0.9301          | 0.5385                |
| population z            | CMP2946 | 29     | 0.3793   | 0.8966   | 0.3793         | 0.2202          | 0.5252          | 0.2202                |
| RF SHAP                 | CDE1846 | 9      | 0.0000   | 1.0000   | 0.3333         | 0.5385          | 0.9301          | 0.5385                |
| RF SHAP                 | CMP2946 | 29     | 0.0345   | 0.4138   | 0.1379         | 0.2202          | 0.5252          | 0.2202                |
| RF SHAP (positive only) | CDE1846 | 9      | 0.5556   | 0.5556   | 0.2963         | 0.5385          | 0.9301          | 0.5385                |
| RF SHAP (positive only) | CMP2946 | 29     | 0.4828   | 0.5172   | 0.2529         | 0.2202          | 0.5252          | 0.2202                |
