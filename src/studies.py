##
# @file src/studies.py
# @brief The follow-up studies the headline comparison leaves open.
#
# src/experiment.py answers "which detector is better, on this split, with
# this seed?". Three questions remain, and each one is the kind an examiner
# asks first:
#
#   * Seed robustness. The headline numbers come from one seed, and several
#     components are stochastic: the 50 000-row fit subsample the quadratic
#     anomaly models are trained on, the forests, SMOTE, and the tie-break at
#     the alert-budget cut. Re-drawing all of them across ten seeds shows
#     whether the result is a property of the method or of one draw. The
#     framework is compared with each baseline by a paired Wilcoxon
#     signed-rank test, Holm-Bonferroni corrected across every comparison.
#
#   * History length (sub-question 3). How many days of a user's own logs a
#     personal baseline needs before it is stable and useful. Each user's
#     training history is cut to their most recent N active days -- an
#     organisation that has been collecting logs for N days when monitoring
#     starts -- and the framework is refitted and rescored on the unchanged
#     test window.
#
#   * Explanation fidelity (sub-question 5). Whether the features an
#     explanation names are the ones the insider actually used. Ground truth
#     comes from the official answer key, replayed through the same per-event
#     aggregation that built the dataset, so "the right features" is decided
#     by what the malicious events changed rather than by judgement.
##

import csv
import io
import math
import os
import tarfile
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from src.aggregate import _consume, _to_frame
from src.baseline import BehaviouralProfiler
from src.detectors import (
    DEFAULT_FIT_SUBSAMPLE, PerUserDetector, SupervisedDetector, build_detectors,
)
from src.evaluate import DEFAULT_ALERT_BUDGET, flags_for_budget
from src.experiment import load_experiment_data, run_detector
from src.prepare import ground_truth_from_manifest
from src.schema import FEATURE_COLS, INTERNAL_EMAIL_DOMAIN, LOG_COLUMNS

## The detector the other studies measure against: the configuration the
#  ablation selects and train.py defaults to.
REFERENCE_DETECTOR = "peruser-ocsvm-personal-zscore"

## Detectors carried through the seed study: the framework, its two nearest
#  variants, the population controls that share its algorithms, the
#  proposal's Random Forest baseline, and the volume floor.
SEED_STUDY_DETECTORS = (
    REFERENCE_DETECTOR,
    "peruser-ocsvm-role-zscore",
    "peruser-iforest-personal-zscore",
    "pop-ocsvm",
    "pop-iforest",
    "rf",
    "volume",
)

## Ten seeds: the minimum for which a two-sided Wilcoxon signed-rank test on
#  paired runs can reach p < 0.005, which Holm-Bonferroni needs once the
#  comparisons are counted. 42 is included so the headline run is one draw.
DEFAULT_SEEDS = (42, 1, 2, 3, 4, 5, 6, 7, 8, 9)

## (metric, higher-is-better) pairs the paired tests run on: ranking quality,
#  sub-question 1, and sub-question 2.
TESTED_METRICS = (
    ("avg_precision", True),
    ("recall", True),
    ("alert_share_high_privilege", False),
)

## Per-run columns kept from the evaluation, in reading order.
SEED_COLUMNS = (
    "detector", "family", "avg_precision", "roc_auc", "recall", "precision",
    "f1", "lift", "insider_recall", "n_insiders_detected",
    "fpr_high_privilege", "alert_share_high_privilege", "tie_fraction",
    "fit_seconds", "score_ms_per_row", "peak_memory_mb",
)

## Personal-history lengths swept for sub-question 3, in active days. None is
#  the whole training window.
HISTORY_DAYS = (3, 7, 14, 30, 60, 90, 120, 180, None)

## (anomaly method, statistic) configurations swept over history length.
HISTORY_CONFIGS = (("ocsvm", "zscore"), ("iforest", "zscore"))

## |z| above which a benign day's feature counts as a false deviation.
EXCEEDANCE_Z = 3.0

## Explanation length an analyst reads.
EXPLANATION_K = 3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def select_detectors(names, feature_cols: list[str], random_state: int,
                     contamination: float = 0.02, min_history_days: int = 14,
                     fit_subsample: int | None = DEFAULT_FIT_SUBSAMPLE) -> list:
    """
    @brief Build named detectors from the full registry, in the order given.

    @param names             Detector names, as they appear in the results table.
    @param feature_cols      Features every detector receives.
    @param random_state      Seed.
    @param contamination     Anomaly-model contamination.
    @param min_history_days  Cold-start threshold for the framework.
    @param fit_subsample     Training-row cap for the quadratic models.
    @return                  Freshly constructed, unfitted detectors.
    """
    pool = {d.name: d for d in build_detectors(
        "full", feature_cols, contamination, min_history_days,
        fit_subsample, random_state)}
    missing = [n for n in names if n not in pool]
    if missing:
        raise ValueError(f"Unknown detector(s): {', '.join(missing)}")
    return [pool[n] for n in names]


def holm_adjust(pvalues) -> np.ndarray:
    """
    @brief Holm-Bonferroni adjusted p-values.

    Step-down: the i-th smallest p-value is multiplied by (m - i + 1), and the
    running maximum keeps the adjusted values monotone. A hypothesis is
    rejected at level alpha exactly when its adjusted value is <= alpha.

    @param pvalues  Raw p-values (NaN allowed, left as NaN).
    @return         Adjusted p-values in the input order, capped at 1.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.where(ok)[0][np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for i, j in enumerate(idx):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


# ---------------------------------------------------------------------------
# Study 1: seed robustness
# ---------------------------------------------------------------------------

def run_seed_study(dataset_path: str, seeds=DEFAULT_SEEDS,
                   detectors=SEED_STUDY_DETECTORS,
                   budget: float = DEFAULT_ALERT_BUDGET,
                   on_run=None) -> dict:
    """
    @brief Re-run the headline detectors across seeds and test the differences.

    The split and the features never change; only the random draws do. Each
    seed rebuilds every detector from scratch.

    @param dataset_path  Prepared dataset.
    @param seeds         Seeds to run.
    @param detectors     Detector names; the first is the reference.
    @param budget        Alert budget.
    @param on_run        Optional callback(seed, detector) for progress.
    @return              {"per_seed", "timing", "summary", "tests", "manifest"}.
    """
    t0 = time.perf_counter()
    train_df, test_df, feature_cols, manifest = load_experiment_data(dataset_path)
    gt_df = ground_truth_from_manifest(manifest)

    rows, timings = [], []
    for seed in seeds:
        for det in select_detectors(detectors, feature_cols, seed):
            if on_run:
                on_run(seed, det)
            out = run_detector(det, train_df, test_df, gt_df, budget, (budget,), seed)
            m = out["metrics"]
            rows.append({"seed": seed, **{c: m.get(c) for c in SEED_COLUMNS}})
            timings.append(out["timing"].assign(seed=seed))

    per_seed = pd.DataFrame(rows)
    return {
        "per_seed": per_seed,
        "timing":   pd.concat(timings, ignore_index=True),
        "summary":  summarise_seeds(per_seed),
        "tests":    paired_tests(per_seed, reference=detectors[0]),
        "manifest": {
            "study":            "seeds",
            "dataset":          os.path.basename(dataset_path),
            "seeds":            list(seeds),
            "detectors":        list(detectors),
            "reference":        detectors[0],
            "alert_budget":     budget,
            "tested_metrics":   [m for m, _ in TESTED_METRICS],
            "n_test_rows":      int(len(test_df)),
            "n_malicious_test": int(test_df["is_malicious"].sum()),
            "elapsed_seconds":  round(time.perf_counter() - t0, 1),
            "run_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def summarise_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Mean, standard deviation and range of every metric per detector.

    @param per_seed  One row per (seed, detector).
    @return          Long frame: detector, metric, n, mean, std, min, max.
    """
    metrics = [c for c in SEED_COLUMNS if c not in ("detector", "family")
               and c in per_seed.columns]
    long = per_seed.melt(id_vars=["detector"], value_vars=metrics,
                         var_name="metric").dropna(subset=["value"])
    long["value"] = long["value"].astype(float)
    out = (long.groupby(["detector", "metric"], sort=False)["value"]
           .agg(n="count", mean="mean", std="std", min="min", max="max")
           .reset_index())
    return out


def paired_tests(per_seed: pd.DataFrame, reference: str = REFERENCE_DETECTOR,
                 metrics=TESTED_METRICS, alpha: float = 0.05) -> pd.DataFrame:
    """
    @brief Paired Wilcoxon signed-rank tests, reference vs every other detector.

    Pairs are matched on seed, so the comparison is between two detectors
    that saw the same draw of every shared random component. Differences
    are oriented so a positive value always means the reference did better.
    Zero differences are dropped (Wilcoxon's original treatment); if every
    difference is zero the test is undefined and p is reported as 1.

    @param per_seed   One row per (seed, detector).
    @param reference  Detector every other one is compared with.
    @param metrics    (metric, higher_is_better) pairs.
    @param alpha      Family-wise error rate for the Holm correction.
    @return           One row per (comparator, metric), Holm-adjusted.
    """
    from scipy.stats import wilcoxon

    ref = per_seed[per_seed["detector"] == reference].set_index("seed")
    rows = []
    for other in per_seed["detector"].unique():
        if other == reference:
            continue
        oth = per_seed[per_seed["detector"] == other].set_index("seed")
        seeds = ref.index.intersection(oth.index)
        for metric, higher in metrics:
            a = ref.loc[seeds, metric].astype(float).to_numpy()
            b = oth.loc[seeds, metric].astype(float).to_numpy()
            diff = (a - b) if higher else (b - a)
            nonzero = diff[diff != 0]
            if len(nonzero) == 0:
                p = 1.0
            else:
                p = float(wilcoxon(nonzero, zero_method="wilcox",
                                   alternative="two-sided").pvalue)
            rows.append({
                "reference":        reference,
                "comparator":       other,
                "metric":           metric,
                "higher_is_better": higher,
                "n_seeds":          int(len(seeds)),
                "reference_mean":   float(np.mean(a)),
                "comparator_mean":  float(np.mean(b)),
                "median_advantage": float(np.median(diff)),
                "wins":             int((diff > 0).sum()),
                "losses":           int((diff < 0).sum()),
                "ties":             int((diff == 0).sum()),
                "p_value":          p,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"])
        out["significant"] = out["p_holm"] <= alpha
    return out


# ---------------------------------------------------------------------------
# Study 2: history length (sub-question 3)
# ---------------------------------------------------------------------------

def truncate_history(train_df: pd.DataFrame, n_days: int | None) -> pd.DataFrame:
    """
    @brief Keep each user's most recent `n_days` active days of training data.

    "Most recent" because a deployment's baseline is whatever it has
    collected up to the day monitoring starts, and the test window begins
    the day after the training window ends.

    @param train_df  Training split.
    @param n_days    Active days to keep per user, or None for everything.
    @return          Truncated training rows.
    """
    if n_days is None:
        return train_df.reset_index(drop=True)
    ordered = train_df.sort_values(["user", "date"], kind="mergesort")
    return (ordered.groupby("user", observed=True, sort=False)
            .tail(int(n_days)).reset_index(drop=True))


def _personal_min_history(n_days: int | None, default: int) -> int:
    """
    @brief Cold-start threshold for a truncated history.

    The study measures personal baselines built from N days, so the
    threshold is lowered to N when N is shorter than the default; otherwise
    every user would silently fall back to their role group and the sweep
    would be measuring the role baseline instead.
    """
    return default if n_days is None else max(1, min(default, int(n_days)))


def profile_stability(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      n_days: int | None, feature_cols: list[str],
                      min_history_days: int = 14,
                      full_profiler: BehaviouralProfiler | None = None) -> dict:
    """
    @brief How settled a personal baseline built from `n_days` is.

    Two views, because neither alone is enough:

    * Convergence -- how far the N-day per-user mean and standard deviation
      sit from the full-history estimates, in full-history standard
      deviations. The full history is not ground truth (behaviour drifts
      over eight months), so this measures settling, not correctness.

    * Benign exceedance -- the share of the *test window's* benign user-days
      that have at least one feature beyond |z| = 3 under the N-day profile.
      A short history underestimates spread, so ordinary days start to look
      extreme; this is the false-deviation load an analyst would inherit.

    @param train_df          Full training split.
    @param test_df           Test split.
    @param n_days            Personal history length, or None for all of it.
    @param feature_cols      Features profiled.
    @param min_history_days  Default cold-start threshold.
    @param full_profiler     Profiler already fitted on the full history.
    @return                  Dict of stability measures.
    """
    trunc = truncate_history(train_df, n_days)
    prof = BehaviouralProfiler(
        scope="personal", feature_cols=feature_cols,
        min_history_days=_personal_min_history(n_days, min_history_days),
    ).fit(trunc)
    full = full_profiler or BehaviouralProfiler(
        scope="personal", feature_cols=feature_cols,
        min_history_days=min_history_days).fit(train_df)

    users = full.user_stats_.index.intersection(prof.user_stats_.index)
    mean_cols = [f"{c}__mean" for c in feature_cols]
    std_cols  = [f"{c}__std" for c in feature_cols]
    mu_full = full.user_stats_.loc[users, mean_cols].to_numpy(float)
    sd_full = full.user_stats_.loc[users, std_cols].to_numpy(float)
    mu_n    = prof.user_stats_.loc[users, mean_cols].to_numpy(float)
    sd_n    = prof.user_stats_.loc[users, std_cols].to_numpy(float)

    varies = sd_full > 0
    shift  = np.abs(mu_n - mu_full)[varies] / sd_full[varies]
    both   = varies & (sd_n > 0)
    log_sd = np.abs(np.log(sd_n[both] / sd_full[both]))

    benign = test_df[~test_df["is_malicious"].astype(bool)]
    Z, used = prof.transform(benign)
    beyond = np.abs(Z) > EXCEEDANCE_Z

    return {
        "history_days":            "all" if n_days is None else int(n_days),
        "n_train_rows":            int(len(trunc)),
        "mean_shift_median_sd":    round(float(np.median(shift)), 4),
        "mean_shift_p90_sd":       round(float(np.quantile(shift, 0.9)), 4),
        "within_half_sd":          round(float(np.mean(shift <= 0.5)), 4),
        "log_std_ratio_median":    round(float(np.median(log_sd)), 4) if len(log_sd) else float("nan"),
        "share_zero_std":          round(float(np.mean(sd_n[varies] <= 0)), 4),
        "benign_day_exceedance":   round(float(beyond.any(axis=1).mean()), 4),
        "benign_cell_exceedance":  round(float(beyond.mean()), 5),
        "share_scored_personal":   round(float(np.mean(np.asarray(used) == "personal")), 4),
    }


def run_history_study(dataset_path: str, history_days=HISTORY_DAYS,
                      configs=HISTORY_CONFIGS, seeds=(42, 1, 2),
                      budget: float = DEFAULT_ALERT_BUDGET,
                      min_history_days: int = 14, on_run=None) -> dict:
    """
    @brief Sweep personal-history length and measure stability and detection.

    @param dataset_path      Prepared dataset.
    @param history_days      History lengths to sweep (None = all).
    @param configs           (method, statistic) pairs to refit at each length.
    @param seeds             Seeds per configuration.
    @param budget            Alert budget.
    @param min_history_days  Default cold-start threshold.
    @param on_run            Optional callback(label) for progress.
    @return                  {"detection", "stability", "reference", "manifest"}.
    """
    t0 = time.perf_counter()
    train_df, test_df, feature_cols, manifest = load_experiment_data(dataset_path)
    gt_df = ground_truth_from_manifest(manifest)

    full_profiler = BehaviouralProfiler(
        scope="personal", feature_cols=feature_cols,
        min_history_days=min_history_days).fit(train_df)

    stability, detection = [], []
    for n in history_days:
        label = "all" if n is None else int(n)
        if on_run:
            on_run(f"history={label}: profile stability")
        stability.append(profile_stability(train_df, test_df, n, feature_cols,
                                           min_history_days, full_profiler))
        trunc = truncate_history(train_df, n)
        for method, statistic in configs:
            for seed in seeds:
                name = f"peruser-{method}-personal-{statistic}"
                if on_run:
                    on_run(f"history={label}: {name} seed={seed}")
                det = PerUserDetector(
                    name, method=method, scope="personal", statistic=statistic,
                    min_history_days=_personal_min_history(n, min_history_days),
                    random_state=seed, feature_cols=feature_cols)
                out = run_detector(det, trunc, test_df, gt_df, budget, (budget,), seed)
                m = out["metrics"]
                timing = out["timing"].set_index("user")
                detection.append({
                    "history_days": label, "detector": name, "seed": seed,
                    "n_train_rows": int(len(trunc)),
                    **{c: m.get(c) for c in SEED_COLUMNS if c not in ("detector", "family")},
                    **{f"alerts_ahead_{u}": int(r["alerts_ahead"])
                       for u, r in timing.iterrows()},
                })

    # The role-group baseline needs no personal history at all, so it is the
    # floor a short personal history has to beat to be worth having.
    reference = []
    for seed in seeds:
        name = "peruser-ocsvm-role-zscore"
        if on_run:
            on_run(f"reference: {name} seed={seed}")
        det = PerUserDetector(name, method="ocsvm", scope="role", statistic="zscore",
                              min_history_days=min_history_days, random_state=seed,
                              feature_cols=feature_cols)
        out = run_detector(det, train_df, test_df, gt_df, budget, (budget,), seed)
        m = out["metrics"]
        reference.append({"detector": name, "seed": seed,
                          **{c: m.get(c) for c in SEED_COLUMNS
                             if c not in ("detector", "family")}})

    return {
        "detection": pd.DataFrame(detection),
        "stability": pd.DataFrame(stability),
        "reference": pd.DataFrame(reference),
        "manifest": {
            "study":            "history",
            "dataset":          os.path.basename(dataset_path),
            "history_days":     ["all" if n is None else int(n) for n in history_days],
            "configs":          [f"{m}-{s}" for m, s in configs],
            "seeds":            list(seeds),
            "alert_budget":     budget,
            "exceedance_z":     EXCEEDANCE_Z,
            "min_history_days": min_history_days,
            "truncation":       "each user's most recent N active training days",
            "elapsed_seconds":  round(time.perf_counter() - t0, 1),
            "run_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


# ---------------------------------------------------------------------------
# Study 3: explanation fidelity (sub-question 5)
# ---------------------------------------------------------------------------

def malicious_event_features(archive_path: str, release: str = "r6.2",
                             org_domain: str = INTERNAL_EMAIL_DOMAIN) -> pd.DataFrame:
    """
    @brief Feature counts contributed by the answer key's malicious events alone.

    Every answer-key row is a verbatim copy of a malicious log line with its
    log type prefixed. Replaying those rows through the aggregator's own
    per-event handler gives, for each malicious (user, day), exactly which
    feature columns the malicious activity moved -- the same definitions the
    dataset was built with, so there is no separate mapping to drift.

    The http, email and file rows carry one extra field the logs do not: a
    narrative label (e.g. "jobsearch") placed immediately before `content`.
    It is dropped so each row matches the log schema the handler expects.

    @param archive_path  Path to answers.tar.bz2.
    @param release       CERT release whose answer files are read.
    @param org_domain    The organisation's email domain.
    @return              Frame of user, date and FEATURE_COLS (malicious counts).
    """
    counts, domains = {}, {}
    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            name = os.path.basename(member.name)
            if not (member.isfile() and name.startswith(f"{release}-")
                    and name.endswith(".csv")):
                continue
            text = tar.extractfile(member).read().decode("utf-8", "replace")
            by_kind = defaultdict(list)
            for row in csv.reader(io.StringIO(text)):
                if not row or row[0] not in LOG_COLUMNS:
                    continue
                kind, fields = row[0], row[1:]
                if len(fields) == len(LOG_COLUMNS[kind]) + 1:
                    fields = fields[:-2] + fields[-1:]
                by_kind[kind].append(fields)
            for kind, rows in by_kind.items():
                _consume(kind, rows, None, counts, domains, org_domain)
    return _to_frame(counts, domains)


def top_k_features(matrix: np.ndarray, names: list[str], k: int = EXPLANATION_K,
                   positive_only: bool = False) -> list[list[str]]:
    """
    @brief The k features with the largest absolute contribution, per row.

    Stable ordering, so equal contributions resolve by column order and the
    result is reproducible.

    @param matrix         Contributions [n_rows x n_features].
    @param names          Feature names, aligned to the columns.
    @param k              Features to keep.
    @param positive_only  Name only features pushing the score up; a row with
                          fewer than k of them gets a shorter explanation
                          rather than being padded with arbitrary features.
    @return               One list of feature names per row.
    """
    m = np.asarray(matrix, dtype=float)
    order = np.argsort(-np.abs(m), axis=1, kind="stable")
    if not positive_only:
        return [[names[j] for j in row[:k]] for row in order]
    return [[names[j] for j in row if m[i, j] > 0][:k] for i, row in enumerate(order)]


def chance_hit_rate(n_relevant: int, n_features: int, k: int) -> float:
    """
    @brief Probability that k features drawn at random include a relevant one.

    @param n_relevant  Relevant features.
    @param n_features  Features in total.
    @param k           Features drawn.
    @return            1 - C(n - g, k) / C(n, k).
    """
    if n_relevant <= 0:
        return 0.0
    if n_features - n_relevant < k:
        return 1.0
    return 1.0 - math.comb(n_features - n_relevant, k) / math.comb(n_features, k)


def score_explanations(explained: list[list[str]], truth: list[set],
                       n_features: int, k: int = EXPLANATION_K) -> pd.DataFrame:
    """
    @brief Per-row fidelity of a set of explanations against ground truth.

    @param explained   Ranked feature names per row.
    @param truth       Set of ground-truth features per row.
    @param n_features  Size of the feature space, for the chance baseline.
    @param k           Explanation length.
    @return            hit@1, hit@k, precision@k and their chance values.
    """
    rows = []
    for names, gt in zip(explained, truth):
        top = names[:k]
        rows.append({
            "hit_at_1":            float(bool(top) and top[0] in gt),
            f"hit_at_{k}":         float(any(f in gt for f in top)),
            f"precision_at_{k}":   (sum(f in gt for f in top) / k) if top else 0.0,
            "chance_hit_at_1":     len(gt) / n_features,
            f"chance_hit_at_{k}":  chance_hit_rate(len(gt), n_features, k),
            f"chance_precision_at_{k}": len(gt) / n_features,
            "n_relevant":          len(gt),
        })
    return pd.DataFrame(rows)


def run_explanation_study(dataset_path: str, answers_archive: str,
                          budget: float = DEFAULT_ALERT_BUDGET,
                          random_state: int = 42, min_history_days: int = 14,
                          org_domain: str = INTERNAL_EMAIL_DOMAIN,
                          on_step=None) -> dict:
    """
    @brief Score each system's explanations of the test window's malicious days.

    Four explainers, each the thing an analyst would actually be shown:

      personal z    the framework (top |z| against the user's own history)
      role z        the same, against the user's role group
      population z  the same, against the whole organisation -- the control
                    that isolates the reference frame from the method
      RF SHAP       per-row SHAP values of the proposal's Random Forest

    plus the analytic chance rate of a random k-feature explanation.

    @param dataset_path      Prepared dataset.
    @param answers_archive   Path to answers.tar.bz2.
    @param budget            Alert budget, for the "flagged only" breakdown.
    @param random_state      Seed for the Random Forest and SMOTE.
    @param min_history_days  Cold-start threshold.
    @param org_domain        Organisation email domain.
    @param on_step           Optional callback(label) for progress.
    @return                  {"per_day", "summary", "case_study", "manifest"}.
    """
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.perf_counter()
    step = on_step or (lambda _label: None)
    train_df, test_df, feature_cols, manifest = load_experiment_data(dataset_path)

    step("replaying the answer key through the aggregator")
    truth_df = malicious_event_features(answers_archive, org_domain=org_domain)
    truth_df["date"] = pd.to_datetime(truth_df["date"]).dt.normalize()

    mal = test_df[test_df["is_malicious"].astype(bool)].copy()
    mal["date"] = pd.to_datetime(mal["date"]).dt.normalize()
    mal = mal.merge(truth_df.rename(columns={c: f"{c}__mal" for c in feature_cols}),
                    on=["user", "date"], how="left")
    truth = [{c for c in feature_cols if row.get(f"{c}__mal", 0) and row[f"{c}__mal"] > 0}
             for row in mal.to_dict("records")]

    explainers: dict[str, np.ndarray] = {}
    profilers = {}
    for scope in ("personal", "role", "population"):
        step(f"{scope} deviations")
        prof = BehaviouralProfiler(scope=scope, statistic="zscore",
                                   min_history_days=min_history_days,
                                   feature_cols=feature_cols).fit(train_df)
        explainers[f"{scope} z"] = prof.transform(mal)[0]
        profilers[scope] = prof

    step("fitting the Random Forest baseline for SHAP")
    rf = SupervisedDetector(
        "rf", RandomForestClassifier(n_estimators=300, random_state=random_state,
                                     n_jobs=-1),
        feature_cols=feature_cols, random_state=random_state).fit(train_df)
    step("computing SHAP values")
    import shap
    raw = shap.TreeExplainer(rf.estimator).shap_values(
        mal[feature_cols].to_numpy(dtype=float))
    if isinstance(raw, list):
        raw = raw[1] if len(raw) > 1 else raw[0]
    elif np.ndim(raw) == 3:
        raw = raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    explainers["RF SHAP"] = np.asarray(raw, dtype=float)
    # SHAP ranks by |phi| above, as the dashboard does. A negative value is a
    # reason *against* the alert, so the variant below names only features
    # that push towards "malicious" -- the conclusion should not hinge on
    # which reading an analyst takes.
    explainers["RF SHAP (positive only)"] = explainers["RF SHAP"]

    # Which malicious days each system actually put in front of an analyst.
    step("scoring the test window for the flagged-only breakdown")
    flagged = {}
    fw = PerUserDetector(REFERENCE_DETECTOR, method="ocsvm", scope="personal",
                         statistic="zscore", min_history_days=min_history_days,
                         random_state=random_state, feature_cols=feature_cols).fit(train_df)
    for label, scorer in (("framework", fw), ("rf", rf)):
        mask, _ = flags_for_budget(np.asarray(scorer.score(test_df), dtype=float),
                                   budget, random_state)
        hit = test_df.loc[mask & test_df["is_malicious"].astype(bool).to_numpy(),
                          ["user", "date"]].copy()
        hit["date"] = pd.to_datetime(hit["date"]).dt.normalize()
        flagged[label] = set(map(tuple, hit.to_numpy()))
    keys = list(zip(mal["user"], mal["date"]))

    per_day = []
    for name, matrix in explainers.items():
        ranked = top_k_features(matrix, feature_cols,
                                positive_only=name.endswith("(positive only)"))
        scored = score_explanations(ranked, truth, len(feature_cols))
        scored.insert(0, "explainer", name)
        scored.insert(1, "user", mal["user"].to_numpy())
        scored.insert(2, "date", mal["date"].to_numpy())
        scored["top_features"] = ["; ".join(r[:EXPLANATION_K]) for r in ranked]
        scored["truth_features"] = ["; ".join(sorted(t)) for t in truth]
        scored["flagged_by_framework"] = [k in flagged["framework"] for k in keys]
        scored["flagged_by_rf"] = [k in flagged["rf"] for k in keys]
        per_day.append(scored)
    per_day = pd.concat(per_day, ignore_index=True)

    k = EXPLANATION_K
    metric_cols = ["hit_at_1", f"hit_at_{k}", f"precision_at_{k}",
                   "chance_hit_at_1", f"chance_hit_at_{k}", f"chance_precision_at_{k}"]
    views = [("all malicious test days", per_day)]
    views += [(f"flagged by {who}", per_day[per_day[f"flagged_by_{col}"]])
              for who, col in (("framework", "framework"), ("RF", "rf"))]
    summary = []
    for view, frame in views:
        for (name, user), grp in [((n, "all"), g) for n, g in frame.groupby("explainer", sort=False)] + \
                                 [((n, u), g) for (n, u), g in frame.groupby(["explainer", "user"], sort=False)]:
            summary.append({"view": view, "explainer": name, "insider": user,
                            "n_days": int(len(grp)),
                            **{c: round(float(grp[c].mean()), 4) for c in metric_cols}})
    summary = pd.DataFrame(summary)

    case = _case_study(mal, {n: m for n, m in explainers.items()
                             if not n.endswith("(positive only)")},
                       profilers["personal"], fw, feature_cols)

    return {
        "per_day":    per_day,
        "summary":    summary,
        "case_study": case,
        "manifest": {
            "study":            "explain",
            "dataset":          os.path.basename(dataset_path),
            "answers_archive":  os.path.basename(answers_archive),
            "explainers":       list(explainers),
            "explanation_k":    EXPLANATION_K,
            "n_features":       len(feature_cols),
            "n_malicious_test_days": int(len(mal)),
            "alert_budget":     budget,
            "random_state":     random_state,
            "elapsed_seconds":  round(time.perf_counter() - t0, 1),
            "run_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def _case_study(mal: pd.DataFrame, explainers: dict, personal: BehaviouralProfiler,
                framework: PerUserDetector, feature_cols: list[str]) -> pd.DataFrame:
    """
    @brief What an analyst sees for each insider's highest-scored malicious day.

    For every feature: the day's value, the user's own typical (median) and
    90th-percentile day, and each explainer's contribution. This is the
    "additional context" sub-question 5 asks about, laid side by side.

    @param mal           The malicious test days, aligned to the explainers.
    @param explainers    {name: contribution matrix} aligned to `mal`.
    @param personal      Fitted personal-scope profiler.
    @param framework     Fitted framework detector (for picking the day).
    @param feature_cols  Features.
    @return              Long frame: one row per (insider, feature).
    """
    scores = np.asarray(framework.score(mal), dtype=float)
    rows = []
    for user in sorted(mal["user"].unique()):
        idx = np.where(mal["user"].to_numpy() == user)[0]
        i = idx[np.argmax(scores[idx])]
        profile = personal.profile_for(user, mal["role"].iloc[i])
        for j, col in enumerate(feature_cols):
            rows.append({
                "user": user, "date": pd.Timestamp(mal["date"].iloc[i]).date(),
                "role": mal["role"].iloc[i], "feature": col,
                "value": float(mal[col].iloc[i]),
                "user_p50": float(profile.loc[col, "p50"]),
                "user_p90": float(profile.loc[col, "p90"]),
                **{name: round(float(m[i, j]), 4) for name, m in explainers.items()},
                "malicious_count": float(mal[f"{col}__mal"].iloc[i])
                if pd.notna(mal[f"{col}__mal"].iloc[i]) else 0.0,
            })
    return pd.DataFrame(rows)
