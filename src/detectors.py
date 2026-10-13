##
# @file src/detectors.py
# @brief One interface over every detector in the comparison.
#
# WHY THIS EXISTS
# ---------------
# The research question is comparative: does a per-user baseline beat the
# population-level practice it replaces? A comparison is only worth reporting
# if the things being compared differ in exactly the one respect under study,
# so every detector here is fitted on the same training split, scored on the
# same test split, handed the same feature columns, and reduced to the same
# output -- a single array of "higher = more suspicious" scores, with no
# detector allowed to define its own threshold.
#
# THE COMPARISON SET
# ------------------
# * Supervised population classifiers (Section 4.4). Random Forest is the
#   proposal's designated baseline; Decision Tree, Gradient Boosting and
#   Logistic Regression are the rest of the set Noever compares on this
#   dataset, so the baseline is a published operating point rather than one
#   chosen here. All are SMOTE-balanced on the training split only.
#
# * Unsupervised population anomaly detectors. The *same three algorithms*
#   the framework uses for its Anomaly Scorer, fitted on standardised raw
#   features instead of per-user deviations. This is the control that isolates
#   the contribution the research actually claims: if the framework beats
#   these, per-user normalisation is doing the work, not Isolation Forest.
#
# * The per-user framework itself, across the population / role / personal
#   scopes and both statistics.
#
# * Two floors -- total activity volume, and seeded random scores. A detector
#   that cannot beat "rank everyone by how busy they were" has not earned a
#   place in a dissertation, and on a 1-in-31 000 positive rate that is a
#   surprisingly easy trap to fall into unnoticed.
##

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from src.baseline import PerUserBaselineEngine
from src.preprocess import FEATURE_COLS
from src.progress import substep, warn

## Rows above which LOF and One-Class SVM are fitted on a seeded subsample.
#  Both are quadratic in the training set; at 990k rows neither finishes on
#  consumer hardware, and the proposal's feasibility claim is about consumer
#  hardware. Scoring still covers every test row.
DEFAULT_FIT_SUBSAMPLE = 50_000

## SMOTE is skipped below this many positive training rows -- it interpolates
#  between neighbouring positives and cannot run without at least two.
MIN_SMOTE_POSITIVES = 2


class Detector:
    """
    @brief Base class: fit on the training split, emit one score per row.

    @var name    Identifier used in the results table.
    @var family  Grouping used when the report is broken down by approach.
    @var config  Parameters worth recording alongside a result.
    """

    family = "detector"

    def __init__(self, name: str, feature_cols: list[str] | None = None,
                 **config) -> None:
        self.name         = name
        self.feature_cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
        self.config       = config
        self.fit_seconds_ = 0.0

    def fit(self, train_df: pd.DataFrame) -> "Detector":
        """@brief Fit on the training split. @return self."""
        raise NotImplementedError

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """@brief Risk score per row, higher = more anomalous."""
        raise NotImplementedError

    def describe(self) -> dict:
        """@brief Detector identity and configuration, for the results manifest."""
        return {"detector": self.name, "family": self.family,
                "n_features": len(self.feature_cols), **self.config}


# ---------------------------------------------------------------------------
# Supervised population classifiers (Section 4.4)
# ---------------------------------------------------------------------------

def _resample(X: np.ndarray, y: np.ndarray, random_state: int = 42
              ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    @brief SMOTE-balance a training set, per Section 4.2.

    Applied to the training split only, after the time-based split, so no
    synthetic row can influence the test window.

    @param X             Training features.
    @param y             Binary labels.
    @param random_state  Seed.
    @return              (X_balanced, y_balanced, info).
    """
    n_pos = int(y.sum())
    if n_pos < MIN_SMOTE_POSITIVES:
        warn(f"only {n_pos} positive training row(s) — SMOTE skipped")
        return X, y, {"smote": False, "n_positives": n_pos, "n_rows": len(X)}

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=random_state, k_neighbors=min(5, n_pos - 1))
    X_bal, y_bal = sm.fit_resample(X, y)
    return X_bal, y_bal, {"smote": True, "n_positives": n_pos, "n_rows": len(X_bal)}


class SupervisedDetector(Detector):
    """
    @brief A population-level supervised classifier: one shared decision
           boundary for every user and role.
    """

    family = "population-supervised"

    def __init__(self, name: str, estimator, scale: bool = False,
                 smote: bool = True, feature_cols: list[str] | None = None,
                 random_state: int = 42, **config) -> None:
        """
        @param name          Identifier for the results table.
        @param estimator     Unfitted sklearn classifier exposing predict_proba.
        @param scale         Standardise features before fitting (linear models).
        @param smote         Apply SMOTE to the training split.
        @param feature_cols  Features to train on.
        @param random_state  Seed for SMOTE. The estimator carries its own.
        """
        super().__init__(name, feature_cols, scale=scale, smote=smote, **config)
        self.estimator    = estimator
        self.scale        = scale
        self.smote        = smote
        self.random_state = random_state
        self.scaler_      = None

    def fit(self, train_df: pd.DataFrame) -> "SupervisedDetector":
        t0 = time.perf_counter()
        X = train_df[self.feature_cols].to_numpy(dtype=float)
        y = train_df["is_malicious"].astype(int).to_numpy()

        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            X = self.scaler_.transform(X)

        info = {"smote": False, "n_positives": int(y.sum()), "n_rows": len(X)}
        if self.smote:
            X, y, info = _resample(X, y, self.random_state)

        with substep(f"{self.name}: fitting on {len(X):,} rows "
                     f"({info['n_positives']} real positives)") as s:
            self.estimator.fit(X, y)
            s.set_detail("SMOTE-balanced" if info["smote"] else "no resampling")

        self.config.update({f"train_{k}": v for k, v in info.items()})
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=float)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        proba = self.estimator.predict_proba(X)
        # A classifier that saw one class only exposes a single column.
        if proba.shape[1] == 1:
            return np.zeros(len(df))
        return proba[:, 1]


# ---------------------------------------------------------------------------
# Unsupervised population anomaly detectors -- the isolating control
# ---------------------------------------------------------------------------

class PopulationAnomalyDetector(Detector):
    """
    @brief The framework's own anomaly algorithms, fitted on standardised raw
           features rather than per-user deviations.

    Standardisation here is population-wide and label-free, which is precisely
    the "single shared organisational baseline" the problem statement
    describes. Holding the algorithm fixed and changing only the reference
    frame is what turns the headline comparison into evidence about per-user
    normalisation rather than about Isolation Forest.
    """

    family = "population-unsupervised"

    def __init__(self, name: str, method: str, contamination: float = 0.02,
                 fit_subsample: int | None = DEFAULT_FIT_SUBSAMPLE,
                 random_state: int = 42,
                 feature_cols: list[str] | None = None) -> None:
        super().__init__(name, feature_cols, method=method,
                         contamination=contamination, fit_subsample=fit_subsample)
        self.method        = method
        self.contamination = contamination
        self.fit_subsample = fit_subsample
        self.random_state  = random_state
        self.scaler_       = None
        self.model_        = None

    def fit(self, train_df: pd.DataFrame) -> "PopulationAnomalyDetector":
        from src.baseline import _make_anomaly_model

        t0 = time.perf_counter()
        X = train_df[self.feature_cols].to_numpy(dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)

        if self.fit_subsample and len(X) > self.fit_subsample:
            rng = np.random.default_rng(self.random_state)
            X = X[np.sort(rng.choice(len(X), size=self.fit_subsample, replace=False))]

        self.model_ = _make_anomaly_model(self.method, self.contamination,
                                          self.random_state)
        with substep(f"{self.name}: fitting {self.method.upper()} on "
                     f"{len(X):,} standardised rows") as s:
            self.model_.fit(X)
            s.set_detail(f"{X.shape[1]} features")

        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler_.transform(df[self.feature_cols].to_numpy(dtype=float))
        if self.method == "iforest":
            return -self.model_.score_samples(X)
        return -self.model_.decision_function(X)


# ---------------------------------------------------------------------------
# The proposed framework
# ---------------------------------------------------------------------------

class PerUserDetector(Detector):
    """@brief The per-user behavioural baseline framework (Section 4.3)."""

    family = "per-user-framework"

    def __init__(self, name: str, method: str = "iforest",
                 scope: str = "personal", statistic: str = "zscore",
                 include_magnitude: bool = False,
                 contamination: float = 0.02, min_history_days: int = 14,
                 fit_subsample: int | None = DEFAULT_FIT_SUBSAMPLE,
                 random_state: int = 42,
                 feature_cols: list[str] | None = None) -> None:
        super().__init__(name, feature_cols, method=method, scope=scope,
                         statistic=statistic, include_magnitude=include_magnitude,
                         contamination=contamination,
                         min_history_days=min_history_days,
                         fit_subsample=fit_subsample)
        self.engine = PerUserBaselineEngine(
            method=method, contamination=contamination,
            min_history_days=min_history_days, scope=scope, statistic=statistic,
            include_magnitude=include_magnitude,
            fit_subsample=fit_subsample, random_state=random_state,
            feature_cols=self.feature_cols,
        )

    def fit(self, train_df: pd.DataFrame) -> "PerUserDetector":
        t0 = time.perf_counter()
        self.engine.fit(train_df)
        self.fit_seconds_ = time.perf_counter() - t0

        history = self.engine.profiler.user_history_len_
        n_personal = int((history >= self.engine.min_history_days).sum())
        self.config.update({
            "n_users_profiled":   int(len(history)),
            "n_personal_baseline": n_personal,
            "n_coldstart_fallback": int(len(history) - n_personal),
        })
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        # The native anomaly score, not the calibrated display value: the
        # display scale is bounded and would tie the whole top of the ranking.
        return self.engine.raw_score(df)


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------

class VolumeDetector(Detector):
    """
    @brief Ranks user-days by total raw activity. The "just look at who was
           busiest" heuristic every real deployment starts from.
    """

    family = "floor"

    def __init__(self, name: str = "volume",
                 feature_cols: list[str] | None = None) -> None:
        super().__init__(name, feature_cols)

    def fit(self, train_df: pd.DataFrame) -> "VolumeDetector":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].to_numpy(dtype=float).sum(axis=1)


class RandomDetector(Detector):
    """@brief Seeded random scores: the value of a detector that has none."""

    family = "floor"

    def __init__(self, name: str = "random", random_state: int = 42,
                 feature_cols: list[str] | None = None) -> None:
        super().__init__(name, feature_cols, random_state=random_state)
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame) -> "RandomDetector":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.random_state).random(len(df))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _supervised(name: str, feature_cols, random_state: int) -> SupervisedDetector:
    """
    @brief Build one of Noever's four population classifiers by short name.

    Gradient boosting uses sklearn's histogram implementation rather than the
    exact one: SMOTE balancing a 990k-row split produces ~2M rows, which the
    exact learner cannot fit in reasonable time on consumer hardware. The
    histogram learner is the same algorithm family with binned splits, and
    keeping it in the comparison is better than reporting a gap it was never
    run to fill.

    @param name          "rf" | "dt" | "gb" | "lr".
    @param feature_cols  Features to train on.
    @param random_state  Seed.
    @return              Configured SupervisedDetector.
    """
    if name == "rf":
        return SupervisedDetector(
            "rf", RandomForestClassifier(n_estimators=300, random_state=random_state,
                                         n_jobs=-1),
            feature_cols=feature_cols, random_state=random_state,
            algorithm="RandomForest")
    if name == "dt":
        return SupervisedDetector(
            "dt", DecisionTreeClassifier(random_state=random_state),
            feature_cols=feature_cols, random_state=random_state,
            algorithm="DecisionTree")
    if name == "gb":
        return SupervisedDetector(
            "gb", HistGradientBoostingClassifier(random_state=random_state),
            feature_cols=feature_cols, random_state=random_state,
            algorithm="HistGradientBoosting")
    if name == "lr":
        return SupervisedDetector(
            "lr", LogisticRegression(max_iter=1000, random_state=random_state),
            scale=True, feature_cols=feature_cols, random_state=random_state,
            algorithm="LogisticRegression")
    raise ValueError(f"Unknown supervised detector: {name}")


def build_detectors(preset: str = "core", feature_cols: list[str] | None = None,
                    contamination: float = 0.02, min_history_days: int = 14,
                    fit_subsample: int | None = DEFAULT_FIT_SUBSAMPLE,
                    random_state: int = 42) -> list[Detector]:
    """
    @brief Assemble the detector set for a comparison run.

    @param preset            "core"      -> the headline three-way comparison
                             "baselines" -> Noever's four classifiers + floors
                             "ablation"  -> the framework across scope,
                                            statistic and anomaly algorithm
                             "full"      -> everything above.
    @param feature_cols      Features every detector receives.
    @param contamination     Expected anomalous proportion, for the anomaly models.
    @param min_history_days  Cold-start threshold for the framework.
    @param fit_subsample     Training-row cap for the quadratic anomaly models.
    @param random_state      Seed.
    @return                  Detectors, in report order.
    """
    fc = list(feature_cols) if feature_cols else list(FEATURE_COLS)
    common = dict(feature_cols=fc, contamination=contamination,
                  fit_subsample=fit_subsample, random_state=random_state)

    def peruser(scope: str, statistic: str, method: str,
                magnitude: bool = False) -> PerUserDetector:
        suffix = "-mag" if magnitude else ""
        return PerUserDetector(
            f"peruser-{method}-{scope}-{statistic}{suffix}", method=method,
            scope=scope, statistic=statistic, include_magnitude=magnitude,
            min_history_days=min_history_days, **common)

    if preset == "core":
        # One-Class SVM on per-user deviations is the configuration the
        # ablation selects (Section 4.3 asks for exactly this choice to be made
        # empirically), so it is carried here alongside its own
        # population-scope control and the Isolation Forest variants.
        return [
            VolumeDetector(feature_cols=fc),
            _supervised("rf", fc, random_state),
            PopulationAnomalyDetector("pop-iforest", "iforest", **common),
            PopulationAnomalyDetector("pop-ocsvm", "ocsvm", **common),
            peruser("role", "zscore", "iforest"),
            peruser("personal", "zscore", "iforest"),
            peruser("personal", "robust", "iforest"),
            peruser("personal", "zscore", "iforest", magnitude=True),
            peruser("role", "zscore", "ocsvm"),
            peruser("personal", "zscore", "ocsvm"),
            peruser("personal", "robust", "ocsvm"),
        ]

    if preset == "baselines":
        return [
            RandomDetector(feature_cols=fc, random_state=random_state),
            VolumeDetector(feature_cols=fc),
            *[_supervised(n, fc, random_state) for n in ("rf", "dt", "gb", "lr")],
            PopulationAnomalyDetector("pop-iforest", "iforest", **common),
            PopulationAnomalyDetector("pop-lof", "lof", **common),
            PopulationAnomalyDetector("pop-ocsvm", "ocsvm", **common),
        ]

    if preset == "ablation":
        return [peruser(scope, statistic, method, magnitude)
                for method in ("iforest", "lof", "ocsvm")
                for scope in ("population", "role", "personal")
                for statistic in ("zscore", "robust")
                for magnitude in (False, True)]

    if preset == "full":
        return (build_detectors("baselines", fc, contamination, min_history_days,
                                fit_subsample, random_state)
                + build_detectors("ablation", fc, contamination, min_history_days,
                                  fit_subsample, random_state))

    raise ValueError(f"Unknown preset: {preset!r}")


PRESETS = ("core", "baselines", "ablation", "full")
