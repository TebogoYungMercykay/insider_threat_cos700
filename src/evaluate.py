##
# @file src/evaluate.py
# @brief Evaluation metrics for insider threat detection, per the research
#        proposal's Section 4.5 metric set.
#
# Computes:
#   - Recall (malicious class)              : primary metric
#   - False Positive Rate (high-privilege)   : subgroup fairness/utility metric
#   - F1-Score (malicious class)
#   - ROC-AUC
#   - Computational cost                    : fit / inference timing
##

import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix,
)


def _binary_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """@brief Threshold continuous risk scores into binary predictions."""
    return (scores >= threshold).astype(int)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    @brief Compute the false positive rate: FP / (FP + TN).

    @param y_true  Ground-truth binary labels.
    @param y_pred  Predicted binary labels.
    @return        FPR in [0, 1]; NaN if there are no negative examples.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")


# Matched case-insensitively against the 'role' column. Covers both the
# synthetic generator's exact role names ("SysAdmin", "DBAdmin") and the
# looser role-name strings that appear in the real CERT LDAP data (e.g.
# "Sys Admin", "Database Administrator", "IT Admin") — kept in sync with
# the equivalent pattern in src/preprocess.py::attach_context.
DEFAULT_HIGH_PRIVILEGE_PATTERN = r"sys.?admin|db.?admin|database.?admin|it.?admin|administrator"


def evaluate_predictions(
    results_df: pd.DataFrame,
    threshold: float = 0.5,
    high_privilege_roles: tuple[str, ...] | None = None,
    high_privilege_pattern: str = DEFAULT_HIGH_PRIVILEGE_PATTERN,
) -> dict:
    """
    @brief Compute the full proposal-defined metric set from a scored results frame.

    @param results_df             DataFrame with columns 'score', 'is_malicious',
                                  'role' (as produced by baseline.results_to_frame()).
    @param threshold              Score threshold for converting scores to predictions.
    @param high_privilege_roles   Optional explicit set of role names for the
                                  subgroup FPR metric (exact match). If None
                                  (default), roles are matched case-insensitively
                                  against high_privilege_pattern instead, which
                                  works against both synthetic and real CERT data.
    @param high_privilege_pattern Regex used when high_privilege_roles is None.
    @return                       Dict of aggregate and subgroup metrics.
    """
    y_true = results_df["is_malicious"].astype(int).to_numpy()
    scores = results_df["score"].to_numpy()
    y_pred = _binary_predictions(scores, threshold)

    metrics = {
        "threshold":  threshold,
        "n_rows":     len(results_df),
        "n_malicious_rows": int(y_true.sum()),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":    round(float(roc_auc_score(y_true, scores)), 4) if y_true.sum() > 0 else float("nan"),
        "fpr_overall": round(false_positive_rate(y_true, y_pred), 4),
    }

    if high_privilege_roles is not None:
        hp_mask = results_df["role"].isin(high_privilege_roles).to_numpy()
    else:
        hp_mask = results_df["role"].str.contains(
            high_privilege_pattern, case=False, regex=True, na=False
        ).to_numpy()
    if hp_mask.sum() > 0:
        metrics["fpr_high_privilege"] = round(
            false_positive_rate(y_true[hp_mask], y_pred[hp_mask]), 4)
        metrics["n_high_privilege_rows"] = int(hp_mask.sum())
    else:
        metrics["fpr_high_privilege"] = float("nan")
        metrics["n_high_privilege_rows"] = 0

    non_hp_mask = ~hp_mask
    if non_hp_mask.sum() > 0:
        metrics["fpr_non_high_privilege"] = round(
            false_positive_rate(y_true[non_hp_mask], y_pred[non_hp_mask]), 4)
    else:
        metrics["fpr_non_high_privilege"] = float("nan")

    # Per-role FPR breakdown, for the chart on the Evaluation page.
    role_fpr = {}
    for role, grp in results_df.groupby("role"):
        yt = grp["is_malicious"].astype(int).to_numpy()
        yp = _binary_predictions(grp["score"].to_numpy(), threshold)
        if len(grp) >= 5:
            role_fpr[role] = round(false_positive_rate(yt, yp), 4)
    metrics["fpr_by_role"] = role_fpr

    return metrics


def evaluate_recall_by_archetype(
    results_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict:
    """
    @brief Break recall down by attacker archetype (burst vs low-and-slow),
           directly answering research sub-question 1.

    @param results_df  Scored results (must include 'user', 'date', 'score').
    @param gt_df       Ground-truth window index with an 'archetype' column
                       (as produced by preprocess.load_ground_truth(), when
                       generated via the synthetic pipeline).
    @param threshold   Score threshold for binary predictions.
    @return            Dict: archetype -> recall.
    """
    if "archetype" not in gt_df.columns:
        return {}

    out = {}
    for archetype, grp in gt_df.groupby("archetype"):
        mask = pd.Series(False, index=results_df.index)
        for _, row in grp.iterrows():
            mask |= (
                (results_df["user"] == row["user"])
                & (results_df["date"] >= row["start_date"])
                & (results_df["date"] <= row["end_date"])
            )
        sub = results_df.loc[mask]
        if len(sub) == 0:
            continue
        y_pred = _binary_predictions(sub["score"].to_numpy(), threshold)
        out[archetype] = round(float(y_pred.sum() / len(sub)), 4)
    return out


def measure_computational_cost(fit_fn, score_fn) -> dict:
    """
    @brief Measure fit time, inference latency, and peak memory of a detector.

    @param fit_fn    Zero-arg callable that fits the model and returns it.
    @param score_fn  Zero-arg callable that scores a dataset given the fitted model.
    @return          Dict with fit_seconds, score_seconds, score_ms_per_row,
                     peak_memory_mb.
    """
    tracemalloc.start()

    t0 = time.perf_counter()
    model = fit_fn()
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = score_fn(model)
    score_seconds = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = len(results) if hasattr(results, "__len__") else 1
    return {
        "fit_seconds":       round(fit_seconds, 4),
        "score_seconds":     round(score_seconds, 4),
        "score_ms_per_row":  round((score_seconds / max(n, 1)) * 1000, 5),
        "peak_memory_mb":    round(peak / (1024 * 1024), 2),
    }
