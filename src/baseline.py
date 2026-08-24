##
# @file src/baseline.py
# @brief Per-user behavioural baseline framework and population-level baseline
#        for insider threat detection.
#
# Implements the three-component architecture from the research proposal
# (Section 4.3):
#   1. Profile Builder        -> per-user historical mean/std per feature
#   2. Role-Aware Normaliser  -> per-role-group mean/std per feature
#      (used directly for users with an insufficient personal history —
#      the "cold-start" fallback — and as the comparison baseline that
#      demonstrates the false-positive reduction claim)
#   3. Anomaly Scorer         -> unsupervised outlier model (Isolation Forest /
#      Local Outlier Factor / One-Class SVM) fitted on the *deviation* vectors
#      produced by (1)+(2), rather than on raw feature magnitudes. This is
#      what makes a single, cheaply-trained global anomaly model behave as a
#      per-user-aware detector: the input space it sees is already expressed
#      in "how unusual for this individual/role" terms.
#
# A separate PopulationBaseline class implements the population-level
# Random Forest classifier described in Section 4.4, used as the direct
# comparison point for all evaluation metrics.
##

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from src.preprocess import FEATURE_COLS
from src.progress import Heartbeat, substep

ANOMALY_METHODS = ("iforest", "lof", "ocsvm")


@dataclass
class RiskResult:
    """
    @brief Holds a single per-(user, day) risk assessment.

    @var user            User identifier.
    @var date            Date of the assessed activity (Timestamp).
    @var role            User's role at assessment time.
    @var score           Risk score in [0, 1]; higher = more anomalous.
    @var method          Scoring method: "population" | "peruser".
    @var baseline_used   "personal" | "role_group"  (per-user method only).
    @var top_features    Ranked list of (feature, deviation) driving the score.
    @var is_malicious    Ground-truth label, if available.
    """
    user:          str
    date:          pd.Timestamp
    role:          str
    score:         float
    method:        str
    baseline_used: str = ""
    top_features:  list = field(default_factory=list)
    is_malicious:  bool = False


def _minmax(x: np.ndarray) -> np.ndarray:
    """@brief Min-max scale an array to [0, 1], guarding against a flat input."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Component 1 + 2: Profile Builder / Role-Aware Normaliser
# ---------------------------------------------------------------------------

class BehaviouralProfiler:
    """
    @brief Builds per-user and per-role-group behavioural baselines from the
           training split, and converts raw feature rows into deviation
           (z-score) vectors relative to the appropriate baseline.
    """

    def __init__(self, min_history_days: int = 14) -> None:
        """
        @param min_history_days  Minimum number of a user's own training-split
                                 days required before their personal baseline
                                 is trusted; users below this threshold fall
                                 back to their role-group baseline
                                 (cold-start handling, Section 4.3).
        """
        self.min_history_days = min_history_days
        self.user_stats_: pd.DataFrame | None = None   # indexed by user
        self.role_stats_: pd.DataFrame | None = None   # indexed by role
        self.user_history_len_: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "BehaviouralProfiler":
        """
        @brief Compute per-user and per-role (mean, std) for every feature.

        Vectorised via groupby-agg (no per-row Python loop) so fitting stays
        fast even for hundreds of thousands of training rows.

        @param train_df  Training-split rows (user, date, FEATURE_COLS..., role).
        @return          self.
        """
        self.user_history_len_ = train_df.groupby("user").size()

        user_agg = train_df.groupby("user")[FEATURE_COLS].agg(["mean", "std"])
        user_agg.columns = [f"{c}__{s}" for c, s in user_agg.columns]
        self.user_stats_ = user_agg.fillna(0.0)

        role_agg = train_df.groupby("role")[FEATURE_COLS].agg(["mean", "std"])
        role_agg.columns = [f"{c}__{s}" for c, s in role_agg.columns]
        self.role_stats_ = role_agg.fillna(0.0)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        @brief Convert each row's raw features into a per-user/role z-score
               deviation vector.

        Vectorised: joins the appropriate baseline (personal, with a
        role-group fallback for cold-start users) onto every row via
        DataFrame.merge rather than iterating row by row.

        @param df  DataFrame with 'user', 'role', and FEATURE_COLS columns.
        @return    (deviation_matrix [n_rows x n_features], baseline_used per row)
        """
        n = len(df)
        history_len = df["user"].map(self.user_history_len_).fillna(0)
        use_personal = (history_len >= self.min_history_days).to_numpy()

        personal = df[["user"]].merge(
            self.user_stats_, left_on="user", right_index=True, how="left"
        )
        role_fallback = df[["role"]].merge(
            self.role_stats_, left_on="role", right_index=True, how="left"
        )

        Z = np.zeros((n, len(FEATURE_COLS)), dtype=float)
        raw = df[FEATURE_COLS].to_numpy()
        for j, col in enumerate(FEATURE_COLS):
            mean_p = personal[f"{col}__mean"].to_numpy()
            std_p  = personal[f"{col}__std"].to_numpy()
            mean_r = role_fallback[f"{col}__mean"].to_numpy()
            std_r  = role_fallback[f"{col}__std"].to_numpy()

            mean = np.where(use_personal, mean_p, mean_r)
            std  = np.where(use_personal, std_p, std_r)
            std  = np.where((std == 0) | np.isnan(std), 1.0, std)
            mean = np.nan_to_num(mean, nan=0.0)

            Z[:, j] = (raw[:, j] - mean) / std

        used = np.where(use_personal, "personal", "role_group").tolist()
        return Z, used


# ---------------------------------------------------------------------------
# Component 3: Anomaly Scorer
# ---------------------------------------------------------------------------

def _make_anomaly_model(method: str, contamination: float):
    """
    @brief Instantiate the requested unsupervised anomaly detection model.

    @param method         "iforest" | "lof" | "ocsvm".
    @param contamination  Expected proportion of anomalous rows.
    @return               Unfitted sklearn estimator.
    """
    if method == "iforest":
        return IsolationForest(n_estimators=200, contamination=contamination,
                                random_state=42, n_jobs=-1)
    if method == "lof":
        return LocalOutlierFactor(n_neighbors=25, contamination=contamination, novelty=True)
    if method == "ocsvm":
        return OneClassSVM(kernel="rbf", nu=min(max(contamination, 0.001), 0.5), gamma="scale")
    raise ValueError(f"Unknown anomaly method: {method}")


class PerUserBaselineEngine:
    """
    @brief Full per-user behavioural baseline framework: Profile Builder +
           Role-Aware Normaliser + Anomaly Scorer, combined behind one
           `.score()` call — the direct analogue of ForensicSearchEngine
           in the COS783 keyword-search system.
    """

    def __init__(self, method: str = "iforest", contamination: float = 0.02,
                 min_history_days: int = 14) -> None:
        self.method        = method
        self.contamination = contamination
        self.profiler       = BehaviouralProfiler(min_history_days=min_history_days)
        self.anomaly_model_ = None
        self._raw_scores_train_: np.ndarray | None = None
        self.fit_seconds_   = 0.0

    def fit(self, train_df: pd.DataFrame) -> "PerUserBaselineEngine":
        """
        @brief Fit the profiler and the anomaly model on the training split.

        @param train_df  Training-split rows.
        @return          self.
        """
        t0 = time.perf_counter()
        with substep(f"fitting per-user/role-group profiles ({len(train_df):,} training rows)") as s:
            self.profiler.fit(train_df)
            Z_train, _ = self.profiler.transform(train_df)
            s.set_detail(f"{len(self.profiler.user_stats_):,} user profiles, {len(self.profiler.role_stats_)} role groups")

        self.anomaly_model_ = _make_anomaly_model(self.method, self.contamination)
        with substep(f"fitting {self.method.upper()} anomaly model on deviation vectors") as s:
            with Heartbeat(f"{self.method.upper()} fit"):
                self.anomaly_model_.fit(Z_train)
            s.set_detail(f"{Z_train.shape[0]:,} rows x {Z_train.shape[1]} features")

        raw = self._raw_anomaly_score(Z_train)
        self._raw_scores_train_ = raw
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def _raw_anomaly_score(self, Z: np.ndarray) -> np.ndarray:
        """
        @brief Return the model's raw anomaly score (higher = more anomalous)
               for a deviation matrix, using each estimator's native scoring.
        """
        if self.method == "iforest":
            return -self.anomaly_model_.score_samples(Z)
        if self.method == "lof":
            return -self.anomaly_model_.decision_function(Z)
        if self.method == "ocsvm":
            return -self.anomaly_model_.decision_function(Z)
        raise ValueError(self.method)

    def score(self, df: pd.DataFrame, top_n_features: int = 3) -> list[RiskResult]:
        """
        @brief Score every row in df and return ranked, explained results.

        Risk scores are min-max scaled against the training score
        distribution so that scores are comparable across calls. Per-row
        explanations are the top |z-score| features — cheap, fully
        interpretable, and available even for anomaly models (Isolation
        Forest, LOF) that TreeExplainer-style SHAP does not cleanly support.

        @param df              Rows to score (must include 'user','role','date').
        @param top_n_features  Number of top contributing features to report.
        @return                List of RiskResult, in input row order.
        """
        Z, used = self.profiler.transform(df)
        raw = self._raw_anomaly_score(Z)

        train_raw = self._raw_scores_train_
        lo, hi = np.percentile(train_raw, 1), np.percentile(train_raw, 99.5)
        scaled = np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

        results = []
        users = df["user"].to_numpy()
        roles = df["role"].to_numpy()
        dates = df["date"].to_numpy()
        mal   = df["is_malicious"].to_numpy() if "is_malicious" in df.columns else np.zeros(len(df), bool)

        for i in range(len(df)):
            order = np.argsort(-np.abs(Z[i]))[:top_n_features]
            top = [(FEATURE_COLS[j], float(Z[i, j])) for j in order]
            results.append(RiskResult(
                user=users[i], date=pd.Timestamp(dates[i]), role=roles[i],
                score=float(scaled[i]), method="peruser", baseline_used=used[i],
                top_features=top, is_malicious=bool(mal[i]),
            ))
        return results


# ---------------------------------------------------------------------------
# Population-level baseline (Section 4.4)
# ---------------------------------------------------------------------------

class PopulationBaseline:
    """
    @brief Population-level Random Forest classifier — the current-practice
           comparison point. Trained on raw (non-normalised) features, since
           tree ensembles do not require feature scaling; every user and
           role is evaluated against the same shared decision boundary,
           which is exactly the limitation the per-user framework addresses.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, class_weight=None,
            random_state=random_state, n_jobs=-1,
        )
        self.fit_seconds_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PopulationBaseline":
        """
        @brief Fit the Random Forest on (typically SMOTE-balanced) training data.

        @param X  Feature matrix.
        @param y  Binary malicious-day labels.
        @return   self.
        """
        t0 = time.perf_counter()
        with substep(f"fitting RandomForestClassifier ({X.shape[0]:,} rows x {X.shape[1]} features)") as s:
            with Heartbeat("RandomForest fit"):
                self.model.fit(X, y)
            s.set_detail(f"{self.model.n_estimators} trees")
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def score(self, df: pd.DataFrame) -> list[RiskResult]:
        """
        @brief Score rows using the predicted probability of the malicious class.

        @param df  Rows to score (must include FEATURE_COLS, 'user','role','date').
        @return    List of RiskResult in input row order.
        """
        X = df[FEATURE_COLS].to_numpy()
        proba = self.model.predict_proba(X)[:, 1]

        importances = self.model.feature_importances_
        results = []
        users = df["user"].to_numpy()
        roles = df["role"].to_numpy()
        dates = df["date"].to_numpy()
        mal   = df["is_malicious"].to_numpy() if "is_malicious" in df.columns else np.zeros(len(df), bool)

        top_idx = np.argsort(-importances)[:3]
        top_global = [(FEATURE_COLS[j], float(importances[j])) for j in top_idx]

        for i in range(len(df)):
            results.append(RiskResult(
                user=users[i], date=pd.Timestamp(dates[i]), role=roles[i],
                score=float(proba[i]), method="population", baseline_used="population",
                top_features=top_global, is_malicious=bool(mal[i]),
            ))
        return results


def results_to_frame(results: list[RiskResult]) -> pd.DataFrame:
    """
    @brief Convert a list of RiskResult into a flat DataFrame for display/eval.

    @param results  List of RiskResult objects.
    @return         DataFrame with one row per result.
    """
    return pd.DataFrame([{
        "user": r.user, "date": r.date, "role": r.role, "score": r.score,
        "method": r.method, "baseline_used": r.baseline_used,
        "top_features": "; ".join(f"{f} ({z:+.2f}σ)" for f, z in r.top_features),
        "is_malicious": r.is_malicious,
    } for r in results])
