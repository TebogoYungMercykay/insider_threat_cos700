##
# @file src/evaluate.py
# @brief Evaluation metrics for insider threat detection, per the research
#        proposal's Section 4.5 metric set.
#
# WHY THIS IS NOT A THIN WRAPPER OVER sklearn.metrics
# ---------------------------------------------------
# Two properties of the r6.2 benchmark make the naive framing useless, and
# both are handled here rather than by touching the data.
#
# 1. *Scores from different detectors are not on a common scale.* A Random
#    Forest emits a calibrated probability; Isolation Forest emits a negated
#    path-length. Thresholding both at 0.5 does not compare two detectors, it
#    compares a probability against an arbitrary constant. Every threshold in
#    this module is therefore derived from an **alert budget**: flag the top
#    b% of scored rows and ask who each detector spends those alerts on. That
#    is also how a SOC actually deploys a detector -- an analyst team has a
#    fixed number of investigations per day, not a fixed decision threshold.
#
# 2. *The positive class is 44 user-days in 1.39 million.* Day-level recall
#    over 38 test positives moves in steps of 2.6 percentage points, so it is
#    reported alongside two coarser, more stable views: insider-level
#    detection (did the detector surface this person at all?) and average
#    precision, which integrates over every threshold instead of one.
##

import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

## Fraction of scored rows a detector is allowed to flag, by default. 0.1% of
#  the r6.2 test split is ~403 alerts across five months and 4 000 users --
#  roughly three a day, which one analyst can actually work through.
DEFAULT_ALERT_BUDGET = 0.001

## Alert budgets swept in the comparison report, from "one analyst, part time"
#  to "a queue nobody will finish".
BUDGET_SWEEP = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)


def _binary_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """@brief Threshold continuous risk scores into binary predictions."""
    return (scores >= threshold).astype(int)


def select_top_k(scores: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """
    @brief Flag exactly the k highest-scoring rows, breaking ties neutrally.

    Tied scores are the normal case here, not an edge case: a Random Forest
    trained on six positives assigns probability 0.0 to almost every test row,
    and a deviation-based anomaly model repeats a score for every identical
    feature vector. Two obvious ways of handling that are both wrong.
    Flagging every tied row lets a detector spend 40 000 alerts against a
    budget of 400 and then claim the recall; flagging the first k in array
    order makes the metric depend on how the Parquet file happened to be
    sorted.

    So ties are broken by a seeded permutation: the choice among equals is
    arbitrary, reproducible, and independent of both the detector and the row
    order. A detector whose ranking is degenerate at the cut is not rewarded
    for it -- it gets a random sample of its own tied group, which is exactly
    what it would deliver in deployment.

    @param scores        Risk scores, higher = more anomalous.
    @param k             Number of rows to flag.
    @param random_state  Seed for the tiebreak.
    @return              Boolean mask with exactly min(k, len(scores)) True.
    """
    n = len(scores)
    k = int(min(max(k, 0), n))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    tiebreak = np.random.default_rng(random_state).random(n)
    order = np.lexsort((tiebreak, -np.asarray(scores, dtype=float)))
    mask[order[:k]] = True
    return mask


def flags_for_budget(scores: np.ndarray, budget: float,
                     random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    @brief Select the alert budget's worth of rows and describe the cut.

    @param scores        Risk scores, higher = more anomalous.
    @param budget        Fraction of rows to flag, in (0, 1].
    @param random_state  Seed for tie-breaking.
    @return              (boolean mask, diagnostics about the cut score).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = max(1, int(round(n * min(max(budget, 0.0), 1.0)))) if n else 0
    mask = select_top_k(scores, k, random_state)

    cut = float(scores[mask].min()) if mask.any() else float("inf")
    n_tied  = int((scores == cut).sum())
    n_above = int((scores > cut).sum())

    # Rows strictly above the cut were chosen by the ranking. The budget's
    # remainder comes from the group tied at the cut -- but only arbitrarily
    # if that group is larger than the remainder. When the tied group fits
    # entirely inside the budget, every one of its members is flagged and
    # nothing was decided by the tiebreak.
    n_needed    = max(k - n_above, 0)
    n_arbitrary = n_needed if n_tied > n_needed else 0

    return mask, {
        "threshold":      round(cut, 6),
        "n_tied_at_cut":  n_tied,
        # Share of the budget spent on a coin toss among equals. A high value
        # means the detector's ranking, not the metric, is what ran out.
        "tie_fraction":   round(float(n_arbitrary / max(k, 1)), 4),
    }


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
# "Sys Admin", "Database Administrator", "IT Admin") -- kept in sync with
# the equivalent pattern in src/prepare.py::HIGH_PRIVILEGE_PATTERN.
DEFAULT_HIGH_PRIVILEGE_PATTERN = r"sys.?admin|db.?admin|database.?admin|it.?admin|administrator"


def high_privilege_mask(df: pd.DataFrame,
                        roles: tuple[str, ...] | None = None,
                        pattern: str = DEFAULT_HIGH_PRIVILEGE_PATTERN) -> np.ndarray:
    """
    @brief Boolean mask selecting the high-privilege subgroup.

    Prefers the precomputed is_high_privilege column carried by a prepared
    dataset, so the subgroup definition cannot drift between the preparation
    stage and the evaluation stage. Falls back to matching role names.

    @param df       Frame with 'role' and optionally 'is_high_privilege'.
    @param roles    Explicit role names (exact match), or None.
    @param pattern  Regex used when `roles` is None and no column is present.
    @return         Boolean array, one entry per row.
    """
    if roles is not None:
        return df["role"].isin(roles).to_numpy()
    if "is_high_privilege" in df.columns:
        return df["is_high_privilege"].astype(bool).to_numpy()
    return df["role"].str.contains(pattern, case=False, regex=True, na=False).to_numpy()


def resolve_flags(scores: np.ndarray, budget: float | None,
                  threshold: float | None, random_state: int = 42
                  ) -> tuple[np.ndarray, dict]:
    """
    @brief Turn scores into a flagged mask, by explicit threshold or by budget.

    @param scores        Risk scores, higher = more anomalous.
    @param budget        Alert budget, used when `threshold` is None.
    @param threshold     Explicit decision threshold; takes precedence.
    @param random_state  Seed for tie-breaking under a budget.
    @return              (boolean mask, diagnostics about the cut).
    """
    scores = np.asarray(scores, dtype=float)
    if threshold is not None:
        mask = scores >= threshold
        cut = float(threshold)
        n_tied = int((scores == cut).sum())
        return mask, {"threshold": round(cut, 6), "n_tied_at_cut": n_tied,
                      "tie_fraction": round(float(n_tied / max(int(mask.sum()), 1)), 4)}
    return flags_for_budget(scores, budget if budget is not None else DEFAULT_ALERT_BUDGET,
                            random_state)


def evaluate_scores(
    df: pd.DataFrame,
    scores: np.ndarray,
    budget: float = DEFAULT_ALERT_BUDGET,
    threshold: float | None = None,
    high_privilege_roles: tuple[str, ...] | None = None,
    high_privilege_pattern: str = DEFAULT_HIGH_PRIVILEGE_PATTERN,
    random_state: int = 42,
) -> dict:
    """
    @brief Compute the proposal's Section 4.5 metric set for one detector.

    @param df                     Scored rows; needs 'is_malicious', 'role',
                                  'user', and ideally 'is_high_privilege'.
    @param scores                 Risk scores aligned to df's rows, higher =
                                  more anomalous.
    @param budget                 Alert budget used to pick the operating
                                  point when `threshold` is None.
    @param threshold              Explicit score threshold; overrides `budget`.
    @param high_privilege_roles   Explicit high-privilege role names, or None
                                  to use the dataset's own flag.
    @param high_privilege_pattern Regex fallback for the subgroup.
    @param random_state           Seed for tie-breaking.
    @return                       Dict of aggregate and subgroup metrics.
    """
    y_true = df["is_malicious"].astype(int).to_numpy()
    scores = np.asarray(scores, dtype=float)

    flagged, cut = resolve_flags(scores, budget, threshold, random_state)
    y_pred = flagged.astype(int)

    n_flagged = int(y_pred.sum())
    has_pos   = bool(y_true.sum())

    metrics = {
        "n_rows":           len(df),
        "n_malicious_rows": int(y_true.sum()),
        "alert_budget":     budget if threshold is None else float("nan"),
        **cut,
        "n_flagged":        n_flagged,
        "realised_budget":  round(n_flagged / max(len(df), 1), 6),
        # Threshold-free: the honest summary of a ranking on a 1-in-31 000
        # positive rate, where ROC-AUC alone flatters every detector.
        "roc_auc":          round(float(roc_auc_score(y_true, scores)), 4) if has_pos else float("nan"),
        "avg_precision":    round(float(average_precision_score(y_true, scores)), 6) if has_pos else float("nan"),
        # At the budgeted operating point.
        "recall":           round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision":        round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "f1":               round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "fpr_overall":      round(false_positive_rate(y_true, y_pred), 6),
    }

    # Lift over random: how many times better than spending the same budget
    # uniformly at random. 1.0 means the detector adds nothing.
    base_rate = y_true.sum() / max(len(df), 1)
    metrics["lift"] = (round(float(metrics["precision"] / base_rate), 2)
                       if base_rate > 0 else float("nan"))

    # Insider-level detection: the malicious test days belong to a handful of
    # people, and an analyst cares whether the person surfaced at all.
    if "user" in df.columns and has_pos:
        mal = df["is_malicious"].astype(bool).to_numpy()
        users = df["user"].astype(str).to_numpy()
        insiders = pd.unique(users[mal])
        caught = sorted({u for u in users[mal & flagged]})
        metrics["n_insiders"]          = int(len(insiders))
        metrics["n_insiders_detected"] = int(len(caught))
        metrics["insider_recall"]      = round(len(caught) / max(len(insiders), 1), 4)
        metrics["insiders_detected"]   = caught
    else:
        metrics["n_insiders"] = 0
        metrics["n_insiders_detected"] = 0
        metrics["insider_recall"] = float("nan")
        metrics["insiders_detected"] = []

    # Subgroup FPR: the metric behind research sub-question 2.
    hp_mask = high_privilege_mask(df, high_privilege_roles, high_privilege_pattern)
    metrics["n_high_privilege_rows"] = int(hp_mask.sum())
    metrics["fpr_high_privilege"] = (
        round(false_positive_rate(y_true[hp_mask], y_pred[hp_mask]), 6)
        if hp_mask.sum() else float("nan"))
    metrics["fpr_non_high_privilege"] = (
        round(false_positive_rate(y_true[~hp_mask], y_pred[~hp_mask]), 6)
        if (~hp_mask).sum() else float("nan"))
    # Share of the alert budget spent on high-privilege users. This is the
    # analyst-facing version of the same question: of everything the detector
    # put in the queue, how much of it is a power user doing their job?
    metrics["alert_share_high_privilege"] = (
        round(float(y_pred[hp_mask].sum() / n_flagged), 4) if n_flagged else float("nan"))
    metrics["population_share_high_privilege"] = round(float(hp_mask.mean()), 4)

    return metrics


def fpr_by_role(df: pd.DataFrame, flagged: np.ndarray, min_rows: int = 5) -> dict:
    """
    @brief Per-role false positive rate at a fixed operating point.

    @param df        Scored rows with a 'role' column.
    @param flagged   Boolean mask of flagged rows, aligned to df.
    @param min_rows  Roles with fewer rows than this are omitted as noise.
    @return          Dict {role: FPR}.
    """
    work = pd.DataFrame({
        "role": df["role"].to_numpy(),
        "y":    df["is_malicious"].astype(int).to_numpy(),
        "p":    np.asarray(flagged).astype(int),
    })
    out = {}
    for role, grp in work.groupby("role"):
        if len(grp) >= min_rows:
            out[str(role)] = round(
                false_positive_rate(grp["y"].to_numpy(), grp["p"].to_numpy()), 6)
    return out


# ---------------------------------------------------------------------------
# Attacker archetypes
# ---------------------------------------------------------------------------
#
# Research sub-question 1 asks specifically about "low-and-slow" behaviour, so
# the 44 malicious user-days have to be split into archetypes. r6.2 ships no
# archetype column, and inventing one by looking at how the detectors happen
# to score would be circular. The split below is derived only from the answer
# key's own window geometry, which is fixed before any model is fitted.

## An insider whose malicious activity spans more than this many days is
#  treated as low-and-slow rather than a burst.
LOW_AND_SLOW_SPAN_DAYS = 21


def classify_archetypes(gt_df: pd.DataFrame,
                        span_days: int = LOW_AND_SLOW_SPAN_DAYS) -> pd.DataFrame:
    """
    @brief Label each ground-truth insider window burst vs low-and-slow.

    Uses the calendar span between the first and last malicious event, which
    comes straight from the answer key. On r6.2 this separates scenario 2
    (CMP2946, 29 malicious days across ~8 weeks) and scenario 4 (CDE1846,
    9 days across ~9 weeks) from the three single-session exfiltrations.

    @param gt_df      Insider windows with 'user', 'start_date', 'end_date'.
    @param span_days  Threshold in days above which a window is low-and-slow.
    @return           Copy of gt_df with 'span_days' and 'archetype' columns.
    """
    out = gt_df.copy()
    if out.empty:
        out["span_days"] = []
        out["archetype"] = []
        return out

    start = pd.to_datetime(out["start_date"])
    end   = pd.to_datetime(out["end_date"])
    out["span_days"] = (end - start).dt.total_seconds() / 86400.0
    out["archetype"] = np.where(out["span_days"] > span_days, "low_and_slow", "burst")
    return out


def evaluate_recall_by_archetype(
    df: pd.DataFrame,
    flagged: np.ndarray,
    gt_df: pd.DataFrame,
    span_days: int = LOW_AND_SLOW_SPAN_DAYS,
) -> dict:
    """
    @brief Break malicious-day recall down by attacker archetype.

    Only rows *labelled malicious* count towards recall. Filling in each
    insider's whole window instead would score the benign days between their
    malicious sessions as misses, which penalises every detector for not
    flagging behaviour the answer key says was not malicious.

    @param df         Scored rows with 'user' and 'is_malicious'.
    @param flagged    Boolean mask of flagged rows, aligned to df.
    @param gt_df      Insider window index (from the dataset manifest).
    @param span_days  Burst / low-and-slow cut-off.
    @return           Dict {archetype: {recall, n_malicious_days, n_detected,
                       n_insiders, n_insiders_detected}}.
    """
    if gt_df is None or gt_df.empty or "user" not in df.columns:
        return {}

    arch = classify_archetypes(gt_df, span_days)
    user_archetype = dict(zip(arch["user"].astype(str), arch["archetype"]))

    mal = df["is_malicious"].astype(bool).to_numpy()
    if not mal.any():
        return {}

    users   = df["user"].astype(str).to_numpy()
    flagged = np.asarray(flagged, dtype=bool)

    work = pd.DataFrame({"user": users[mal], "flagged": flagged[mal]})
    work["archetype"] = work["user"].map(user_archetype).fillna("unknown")

    out = {}
    for archetype, grp in work.groupby("archetype"):
        out[str(archetype)] = {
            "recall":              round(float(grp["flagged"].mean()), 4),
            "n_malicious_days":    int(len(grp)),
            "n_detected":          int(grp["flagged"].sum()),
            "n_insiders":          int(grp["user"].nunique()),
            "n_insiders_detected": int(grp.loc[grp["flagged"], "user"].nunique()),
        }
    return out


def budget_sweep(df: pd.DataFrame, scores: np.ndarray,
                 budgets: tuple[float, ...] = BUDGET_SWEEP,
                 gt_df: pd.DataFrame | None = None,
                 random_state: int = 42) -> pd.DataFrame:
    """
    @brief Recompute the headline metrics across a range of alert budgets.

    A single operating point can flatter or bury a detector; the sweep shows
    how each one trades alert volume for detections, which is the decision an
    analyst team actually makes.

    @param df            Scored rows.
    @param scores        Risk scores aligned to df.
    @param budgets       Alert budgets to evaluate.
    @param gt_df         Insider windows, for the per-archetype columns.
    @param random_state  Seed for tie-breaking.
    @return              One row per budget.
    """
    rows = []
    for b in budgets:
        m = evaluate_scores(df, scores, budget=b, random_state=random_state)
        row = {
            "alert_budget":        b,
            "n_flagged":           m["n_flagged"],
            "tie_fraction":        m["tie_fraction"],
            "recall":              m["recall"],
            "precision":           m["precision"],
            "insider_recall":      m["insider_recall"],
            "n_insiders_detected": m["n_insiders_detected"],
            "fpr_high_privilege":  m["fpr_high_privilege"],
            "alert_share_high_privilege": m["alert_share_high_privilege"],
        }
        if gt_df is not None and not gt_df.empty:
            flagged, _ = flags_for_budget(scores, b, random_state)
            for name, stats in evaluate_recall_by_archetype(df, flagged, gt_df).items():
                row[f"recall_{name}"] = stats["recall"]
        rows.append(row)
    return pd.DataFrame(rows)


def measure_computational_cost(fit_fn, score_fn) -> dict:
    """
    @brief Measure fit time, inference latency, and peak memory of a detector.

    Answers research sub-question 4: whether per-user baseline construction is
    feasible on consumer hardware relative to population-level training.

    @param fit_fn    Zero-arg callable that fits the model and returns it.
    @param score_fn  One-arg callable that scores a dataset given the model.
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


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

def evaluate_predictions(results_df: pd.DataFrame, threshold: float = 0.5,
                         high_privilege_roles: tuple[str, ...] | None = None,
                         high_privilege_pattern: str = DEFAULT_HIGH_PRIVILEGE_PATTERN) -> dict:
    """
    @brief Metric set for a scored results frame carrying its own 'score' column.

    Retained for the Streamlit app, which works with the RiskResult frames
    produced by baseline.results_to_frame() rather than raw score arrays.

    @param results_df  Frame with 'score', 'is_malicious' and 'role'.
    @param threshold   Explicit decision threshold.
    @return            Metric dict, with the per-role FPR breakdown attached.
    """
    metrics = evaluate_scores(
        results_df, results_df["score"].to_numpy(), threshold=threshold,
        high_privilege_roles=high_privilege_roles,
        high_privilege_pattern=high_privilege_pattern,
    )
    metrics["fpr_by_role"] = fpr_by_role(
        results_df, results_df["score"].to_numpy() >= threshold)
    return metrics
