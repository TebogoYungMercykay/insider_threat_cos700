##
# @file src/baseline.py
# @brief Per-user behavioural baseline framework and population-level baseline
#        for insider threat detection.
#
# Implements the three-component architecture from the research proposal
# (Section 4.3):
#
#   1. Profile Builder        -> per-user descriptive statistics (mean,
#      standard deviation and percentiles) computed from that user's own
#      training-window activity. Users with less history than
#      `min_history_days` fall back to their role group, which is the
#      cold-start handling the proposal calls for.
#
#   2. Role-Aware Normaliser  -> the same statistics computed per role group.
#      Exposed as a first-class *scope* rather than only as a fallback, so the
#      three reference frames a row can be measured against --
#      population / role / personal -- are the same code path with one
#      parameter changed. That is what makes sub-question 2 answerable: the
#      role-aware and population-normalised variants differ in nothing else.
#
#   3. Anomaly Scorer         -> an unsupervised outlier model (Isolation
#      Forest / Local Outlier Factor / One-Class SVM) fitted on the
#      *deviation* vectors produced by (1)+(2) rather than on raw feature
#      magnitudes. This is what makes a single, cheaply-trained global model
#      behave as a per-user-aware detector: the space it sees is already
#      expressed in "how unusual for this individual" terms.
#
# A separate PopulationBaseline class implements the population-level
# Random Forest classifier described in Section 4.4. src/detectors.py wraps
# both of these, plus the other classifiers from the comparison, behind one
# interface so the experiment harness can treat them identically.
##

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.preprocess import FEATURE_COLS
from src.progress import Heartbeat, substep

ANOMALY_METHODS = ("iforest", "lof", "ocsvm")

## Reference distribution a row's features are measured against.
BASELINE_SCOPES = ("population", "role", "personal")

## Percentiles retained on every profile. p25/p75 drive the robust scale;
#  p50/p90/p99 are the descriptive statistics Section 4.3 asks the Profile
#  Builder to store, and the app surfaces them when explaining a flag.
PROFILE_PERCENTILES = (0.25, 0.50, 0.75, 0.90, 0.99)

## Ratio converting an interquartile range into a standard-deviation-scale
#  estimate for a normal distribution.
IQR_TO_SIGMA = 1.349


@dataclass
class RiskResult:
    """
    @brief Holds a single per-(user, day) risk assessment.

    @var user            User identifier.
    @var date            Date of the assessed activity (Timestamp).
    @var role            User's role at assessment time.
    @var score           Calibrated risk score in [0, 1], for display.
    @var raw_score       The detector's native score. Ranking and every
                         evaluation metric use this, never `score`.
    @var method          Scoring method: "population" | "peruser".
    @var baseline_used   "personal" | "role_group" | "population".
    @var top_features    Ranked list of (feature, deviation) driving the score.
    @var is_malicious    Ground-truth label, if available.
    """
    user:          str
    date:          pd.Timestamp
    role:          str
    score:         float
    raw_score:     float
    method:        str
    baseline_used: str = ""
    top_features:  list = field(default_factory=list)
    is_malicious:  bool = False


# ---------------------------------------------------------------------------
# Components 1 + 2: Profile Builder / Role-Aware Normaliser
# ---------------------------------------------------------------------------

def _grouped_stats(df: pd.DataFrame, key: str, feature_cols: list[str]) -> pd.DataFrame:
    """
    @brief Descriptive statistics for every feature, grouped by one key.

    Vectorised through groupby-agg and groupby-quantile; no per-group Python
    loop, so this stays workable on the full 990k-row training split.

    @param df            Rows to profile.
    @param key           Column to group by ('user' or 'role').
    @param feature_cols  Features to profile.
    @return              Frame indexed by `key`, with `{feature}__{stat}`
                         columns for mean, std and each PROFILE_PERCENTILES.
    """
    grouped = df.groupby(key, observed=True)[feature_cols]

    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{c}__{s}" for c, s in agg.columns]

    q = grouped.quantile(list(PROFILE_PERCENTILES)).unstack()
    q.columns = [f"{c}__p{int(round(p * 100))}" for c, p in q.columns]

    return agg.join(q).fillna(0.0)


def _global_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """
    @brief The same descriptive statistics over the whole population.

    @param df            Rows to profile.
    @param feature_cols  Features to profile.
    @return              Series of `{feature}__{stat}` values.
    """
    out = {}
    for col in feature_cols:
        values = df[col].to_numpy(dtype=float)
        out[f"{col}__mean"] = float(np.mean(values)) if len(values) else 0.0
        out[f"{col}__std"]  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for p in PROFILE_PERCENTILES:
            out[f"{col}__p{int(round(p * 100))}"] = (
                float(np.quantile(values, p)) if len(values) else 0.0)
    return pd.Series(out, dtype=float)


class BehaviouralProfiler:
    """
    @brief Builds population, role-group and per-user behavioural baselines
           from the training split, and converts raw feature rows into
           deviation vectors relative to the requested reference frame.
    """

    def __init__(self, scope: str = "personal", min_history_days: int = 14,
                 statistic: str = "zscore", include_magnitude: bool = False,
                 feature_cols: list[str] | None = None) -> None:
        """
        @param scope             Reference distribution each row is measured
                                 against: "population" (one shared baseline,
                                 i.e. current practice), "role" (role-aware
                                 normalisation), or "personal" (the user's own
                                 history, falling back to their role group and
                                 then the population).
        @param min_history_days  Minimum number of a user's own training-split
                                 days before their personal baseline is
                                 trusted (cold-start threshold, Section 4.3).
                                 Ignored unless scope == "personal".
        @param statistic         "zscore"  -> (x - mean) / std
                                 "robust"  -> (x - median) / (IQR / 1.349).
                                 Behavioural counts are zero-inflated and
                                 right-skewed, so a single spike inflates std
                                 and hides the next one; the robust form is
                                 immune to that and is what makes a
                                 low-and-slow drift visible.
        @param include_magnitude Append population-standardised raw features to
                                 the deviation vector.

                                 A pure deviation discards magnitude: a user
                                 whose baseline is 2 file events doing 8 scores
                                 exactly like a user whose baseline is 200
                                 doing 800. The first is noise, the second is
                                 an exfiltration. Across 4 000 users there are
                                 always many people having a personally
                                 unusual but trivially small day, and at a
                                 0.1% alert budget they crowd out the real
                                 one. Carrying both channels lets the Anomaly
                                 Scorer require a day to be unusual *for this
                                 person* and materially large before it ranks
                                 it highly.
        @param feature_cols      Features to profile (defaults to FEATURE_COLS).
        """
        if scope not in BASELINE_SCOPES:
            raise ValueError(f"Unknown baseline scope: {scope!r}")
        if statistic not in ("zscore", "robust"):
            raise ValueError(f"Unknown statistic: {statistic!r}")

        self.scope             = scope
        self.min_history_days  = min_history_days
        self.statistic         = statistic
        self.include_magnitude = include_magnitude
        self.feature_cols      = list(feature_cols) if feature_cols else list(FEATURE_COLS)

        self.user_stats_:       pd.DataFrame | None = None
        self.role_stats_:       pd.DataFrame | None = None
        self.population_stats_: pd.Series | None    = None
        self.user_history_len_: pd.Series | None    = None

    @property
    def output_columns(self) -> list[str]:
        """
        @brief Names of the columns transform() emits, in order.

        @return  Deviation column names, plus magnitude columns when enabled.
        """
        cols = [f"{c}__dev" for c in self.feature_cols]
        if self.include_magnitude:
            cols += [f"{c}__mag" for c in self.feature_cols]
        return cols

    # -- fitting ------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "BehaviouralProfiler":
        """
        @brief Compute per-user, per-role and population profiles.

        All three scopes are always computed. They are cheap relative to the
        anomaly model, and holding all three lets the ablation switch
        reference frames without refitting anything else -- which is the only
        way the population / role / personal comparison isolates the variable
        it claims to.

        @param train_df  Training-split rows (user, role, feature columns).
        @return          self.
        """
        self.user_history_len_ = train_df.groupby("user", observed=True).size()
        self.user_stats_       = _grouped_stats(train_df, "user", self.feature_cols)
        self.role_stats_       = _grouped_stats(train_df, "role", self.feature_cols)
        self.population_stats_ = _global_stats(train_df, self.feature_cols)
        return self

    # -- centre / scale extraction -----------------------------------------

    def _center_scale_columns(self) -> tuple[str, str, str]:
        """
        @brief Column suffixes providing the centre and scale for the
               configured statistic.

        @return  (centre suffix, lower suffix, upper suffix). For "zscore" the
                 lower/upper pair is unused and returned as the std column
                 twice.
        """
        if self.statistic == "zscore":
            return "mean", "std", "std"
        return "p50", "p25", "p75"

    def _stats_for(self, frame: pd.DataFrame, source: pd.DataFrame | pd.Series,
                   key: str | None) -> tuple[np.ndarray, np.ndarray]:
        """
        @brief Look up the (centre, scale) each row of `frame` is measured against.

        The population profile is one row of numbers shared by everybody, so it
        is returned as a 1-D array of length n_features and left to broadcast.
        Materialising it as a full n x n_features tile would cost ~300 MB on
        the 990k-row training split for no information.

        @param frame   Rows being transformed.
        @param source  Profile table (indexed by `key`) or a population Series.
        @param key     Join column, or None for the population profile.
        @return        (centre, scale) -- [n_rows x n_features] when keyed,
                       [n_features] when population-wide.
        """
        centre_suffix, lo_suffix, hi_suffix = self._center_scale_columns()

        if key is None:
            centre = np.array([source[f"{c}__{centre_suffix}"] for c in self.feature_cols],
                              dtype=float)
            if self.statistic == "zscore":
                scale = np.array([source[f"{c}__std"] for c in self.feature_cols],
                                 dtype=float)
            else:
                lo = np.array([source[f"{c}__{lo_suffix}"] for c in self.feature_cols])
                hi = np.array([source[f"{c}__{hi_suffix}"] for c in self.feature_cols])
                scale = (hi - lo) / IQR_TO_SIGMA
            return centre, scale

        joined = frame[[key]].merge(source, left_on=key, right_index=True, how="left")
        centre = joined[[f"{c}__{centre_suffix}" for c in self.feature_cols]].to_numpy(float)
        if self.statistic == "zscore":
            scale = joined[[f"{c}__std" for c in self.feature_cols]].to_numpy(float)
        else:
            lo = joined[[f"{c}__{lo_suffix}" for c in self.feature_cols]].to_numpy(float)
            hi = joined[[f"{c}__{hi_suffix}" for c in self.feature_cols]].to_numpy(float)
            scale = (hi - lo) / IQR_TO_SIGMA
        return centre, scale

    # -- transform ----------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        @brief Convert raw feature rows into deviation vectors.

        A scale of zero means the reference distribution is degenerate for
        that feature -- typically a count the user has never once recorded.
        It is floored at 1.0 rather than dropped, which makes the deviation
        equal the raw count: the first time somebody who has never written to
        removable media writes one file, that is a deviation of 1.0, not a
        division by zero and not a silent zero.

        @param df  Rows with 'user', 'role' and the feature columns.
        @return    (deviation matrix [n_rows x n_features], baseline used per row)
        """
        if self.user_stats_ is None:
            raise RuntimeError("BehaviouralProfiler.transform() before fit()")

        n   = len(df)
        raw = df[self.feature_cols].to_numpy(dtype=float)

        pop_centre, pop_scale = self._stats_for(df, self.population_stats_, None)

        if self.scope == "population":
            # 1-D; broadcasts across rows in the arithmetic below.
            centre, scale = pop_centre, pop_scale
            used = ["population"] * n
        else:
            role_centre, role_scale = self._stats_for(df, self.role_stats_, "role")
            # A role absent from the training split leaves NaNs behind; those
            # rows fall through to the population profile.
            role_known = ~np.isnan(role_centre).any(axis=1)
            centre = np.where(role_known[:, None], role_centre, pop_centre)
            scale  = np.where(role_known[:, None], role_scale,  pop_scale)
            used   = np.where(role_known, "role_group", "population")

            if self.scope == "personal":
                user_centre, user_scale = self._stats_for(df, self.user_stats_, "user")
                history = df["user"].map(self.user_history_len_).fillna(0).to_numpy()
                use_personal = ((history >= self.min_history_days)
                                & ~np.isnan(user_centre).any(axis=1))
                centre = np.where(use_personal[:, None], user_centre, centre)
                scale  = np.where(use_personal[:, None], user_scale,  scale)
                used   = np.where(use_personal, "personal", used)
            used = used.tolist()

        centre = np.nan_to_num(centre, nan=0.0)
        scale  = np.nan_to_num(scale,  nan=0.0)
        scale  = np.where(scale <= 0.0, 1.0, scale)

        Z = (raw - centre) / scale
        if self.include_magnitude:
            # Note: under scope="population" with the z-score statistic this
            # channel is identical to the deviation, so those ablation cells
            # are expected to duplicate. They are kept for grid completeness.
            mag_scale = np.where(pop_scale <= 0.0, 1.0,
                                 np.nan_to_num(pop_scale, nan=0.0))
            magnitude = (raw - np.nan_to_num(pop_centre, nan=0.0)) / mag_scale
            Z = np.hstack([Z, np.broadcast_to(magnitude, (n, len(self.feature_cols)))])
        return Z, used

    # -- introspection ------------------------------------------------------

    def profile_for(self, user: str, role: str = "") -> pd.DataFrame:
        """
        @brief The stored descriptive profile for one user, for display.

        @param user  User identifier.
        @param role  Role to fall back to when the user has no profile.
        @return      Frame indexed by feature with mean/std/percentile columns.
        """
        if self.user_stats_ is not None and user in self.user_stats_.index:
            row = self.user_stats_.loc[user]
        elif self.role_stats_ is not None and role in self.role_stats_.index:
            row = self.role_stats_.loc[role]
        else:
            row = self.population_stats_

        stats = ["mean", "std"] + [f"p{int(round(p * 100))}" for p in PROFILE_PERCENTILES]
        return pd.DataFrame(
            {stat: [float(row.get(f"{c}__{stat}", 0.0)) for c in self.feature_cols]
             for stat in stats},
            index=self.feature_cols,
        )


# ---------------------------------------------------------------------------
# Component 3: Anomaly Scorer
# ---------------------------------------------------------------------------

def _make_anomaly_model(method: str, contamination: float, random_state: int = 42):
    """
    @brief Instantiate the requested unsupervised anomaly detection model.

    @param method         "iforest" | "lof" | "ocsvm".
    @param contamination  Expected proportion of anomalous rows.
    @param random_state   Seed for the stochastic estimators.
    @return               Unfitted sklearn estimator.
    """
    if method == "iforest":
        return IsolationForest(n_estimators=200, contamination=contamination,
                               random_state=random_state, n_jobs=-1)
    if method == "lof":
        return LocalOutlierFactor(n_neighbors=25, contamination=contamination,
                                  novelty=True, n_jobs=-1)
    if method == "ocsvm":
        return OneClassSVM(kernel="rbf", nu=min(max(contamination, 1e-3), 0.5),
                           gamma="scale")
    raise ValueError(f"Unknown anomaly method: {method}")


class PerUserBaselineEngine:
    """
    @brief The full framework: Profile Builder + Role-Aware Normaliser +
           Anomaly Scorer behind one fit/score pair.
    """

    def __init__(self, method: str = "iforest", contamination: float = 0.02,
                 min_history_days: int = 14, scope: str = "personal",
                 statistic: str = "zscore", include_magnitude: bool = False,
                 fit_subsample: int | None = None, random_state: int = 42,
                 feature_cols: list[str] | None = None) -> None:
        """
        @param method            Anomaly Scorer algorithm.
        @param contamination     Expected proportion of anomalous rows.
        @param min_history_days  Cold-start threshold (see BehaviouralProfiler).
        @param scope             Normalisation scope for the ablation.
        @param statistic         "zscore" | "robust".
        @param include_magnitude Also feed the anomaly model population-scaled
                                 raw magnitudes (see BehaviouralProfiler).
        @param fit_subsample     Cap on training rows used to fit the anomaly
                                 model. LOF and One-Class SVM are quadratic in
                                 the training set and cannot be fitted on
                                 990k rows on consumer hardware; a seeded
                                 subsample keeps them in the comparison
                                 instead of dropping them from it. Profiles
                                 are always built on the full training split.
        @param random_state      Seed for the subsample and the estimators.
        @param feature_cols      Features to use (defaults to FEATURE_COLS).
        """
        self.method            = method
        self.contamination     = contamination
        self.min_history_days  = min_history_days
        self.scope             = scope
        self.statistic         = statistic
        self.include_magnitude = include_magnitude
        self.fit_subsample     = fit_subsample
        self.random_state      = random_state
        self.feature_cols      = list(feature_cols) if feature_cols else list(FEATURE_COLS)

        self.profiler = BehaviouralProfiler(
            scope=scope, min_history_days=min_history_days,
            statistic=statistic, include_magnitude=include_magnitude,
            feature_cols=self.feature_cols,
        )
        self.anomaly_model_ = None
        self._raw_scores_train_:   np.ndarray | None = None
        self._sorted_train_scores_: np.ndarray | None = None
        self.fit_seconds_ = 0.0

    # -- fit ----------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "PerUserBaselineEngine":
        """
        @brief Fit the profiler and the anomaly model on the training split.

        @param train_df  Training-split rows.
        @return          self.
        """
        t0 = time.perf_counter()
        with substep(f"building {self.scope} profiles ({len(train_df):,} training rows)") as s:
            self.profiler.fit(train_df)
            Z_train, _ = self.profiler.transform(train_df)
            s.set_detail(f"{len(self.profiler.user_stats_):,} user profiles, "
                         f"{len(self.profiler.role_stats_)} role groups")

        Z_fit = Z_train
        if self.fit_subsample and len(Z_train) > self.fit_subsample:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(Z_train), size=self.fit_subsample, replace=False)
            Z_fit = Z_train[np.sort(idx)]

        self.anomaly_model_ = _make_anomaly_model(
            self.method, self.contamination, self.random_state)
        with substep(f"fitting {self.method.upper()} on deviation vectors") as s:
            with Heartbeat(f"{self.method.upper()} fit"):
                self.anomaly_model_.fit(Z_fit)
            s.set_detail(f"{Z_fit.shape[0]:,} rows x {Z_fit.shape[1]} features"
                         + (f" (subsampled from {len(Z_train):,})"
                            if len(Z_fit) != len(Z_train) else ""))

        self._raw_scores_train_  = self._raw_anomaly_score(Z_fit)
        self._sorted_train_scores_ = np.sort(self._raw_scores_train_)
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    # -- scoring ------------------------------------------------------------

    def _raw_anomaly_score(self, Z: np.ndarray) -> np.ndarray:
        """
        @brief The model's native anomaly score, negated so higher = more
               anomalous for every estimator.
        """
        if self.method == "iforest":
            return -self.anomaly_model_.score_samples(Z)
        if self.method in ("lof", "ocsvm"):
            return -self.anomaly_model_.decision_function(Z)
        raise ValueError(self.method)

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        @brief The native anomaly score per row -- the ranking signal.

        Everything that compares or thresholds detectors uses this. It is
        deliberately not squashed into [0, 1]: any bounded rescaling has to
        decide what to do with scores beyond the training range, and the
        obvious choice -- clip -- collapses the entire top of the ranking to
        one value. That is the region the detector exists to order.

        @param df  Rows to score.
        @return    Array aligned to df's rows, higher = more anomalous.
        """
        Z, _ = self.profiler.transform(df)
        return self._raw_anomaly_score(Z)

    def score_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        @brief A calibrated risk score in [0, 1) for display.

        Reads as "this day is more extreme than X% of the training window".
        Scores beyond anything seen in training would all sit at 1.0 under a
        plain empirical percentile, so the excess is mapped through a
        saturating-but-strictly-increasing tail into the top percent. The
        result stays ordered by the underlying anomaly score, which keeps a
        sorted risk queue meaningful.

        @param df  Rows to score.
        @return    Array in [0, 1) aligned to df's rows.
        """
        return self.calibrate(self.raw_score(df))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        """
        @brief Map native anomaly scores onto a [0, 1) display scale.

        @param raw  Native anomaly scores.
        @return     Calibrated scores in [0, 1).
        """
        train = self._sorted_train_scores_
        raw   = np.asarray(raw, dtype=float)
        if train is None or len(train) == 0:
            return np.zeros_like(raw)

        percentile = np.searchsorted(train, raw, side="right") / len(train)
        base = np.minimum(percentile, 1.0) * 0.99

        top    = float(train[-1])
        spread = max(top - float(np.median(train)), 1e-9)
        excess = np.maximum(raw - top, 0.0) / spread
        return base + 0.01 * (excess / (1.0 + excess))

    def deviations(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        @brief The deviation matrix and per-row baseline used, for explanations.

        @param df  Rows to transform.
        @return    (deviation matrix, baseline used per row).
        """
        return self.profiler.transform(df)

    def score(self, df: pd.DataFrame, top_n_features: int = 3) -> list[RiskResult]:
        """
        @brief Score every row and return ranked, explained results.

        Per-row explanations are the top |deviation| features -- cheap, fully
        interpretable, and available even for the anomaly models that
        TreeExplainer-style SHAP does not cleanly support.

        @param df              Rows to score (needs 'user', 'role', 'date').
        @param top_n_features  Number of contributing features to report.
        @return                List of RiskResult, in input row order.
        """
        Z, used = self.profiler.transform(df)
        raw     = self._raw_anomaly_score(Z)
        scaled  = self.calibrate(raw)

        users = df["user"].to_numpy()
        roles = df["role"].to_numpy()
        dates = df["date"].to_numpy()
        mal   = (df["is_malicious"].to_numpy() if "is_malicious" in df.columns
                 else np.zeros(len(df), bool))

        names = self.profiler.output_columns
        results = []
        for i in range(len(df)):
            order = np.argsort(-np.abs(Z[i]))[:top_n_features]
            results.append(RiskResult(
                user=users[i], date=pd.Timestamp(dates[i]), role=roles[i],
                score=float(scaled[i]), raw_score=float(raw[i]),
                method="peruser", baseline_used=used[i],
                top_features=[(names[j], float(Z[i, j])) for j in order],
                is_malicious=bool(mal[i]),
            ))
        return results


# ---------------------------------------------------------------------------
# Population-level baseline (Section 4.4)
# ---------------------------------------------------------------------------

class PopulationBaseline:
    """
    @brief Population-level Random Forest classifier -- the current-practice
           comparison point. Trained on raw (non-normalised) features, since
           tree ensembles do not require feature scaling; every user and role
           is evaluated against the same shared decision boundary, which is
           exactly the limitation the per-user framework addresses.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42,
                 feature_cols: list[str] | None = None) -> None:
        self.feature_cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
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
        with substep(f"fitting RandomForestClassifier "
                     f"({X.shape[0]:,} rows x {X.shape[1]} features)") as s:
            with Heartbeat("RandomForest fit"):
                self.model.fit(X, y)
            s.set_detail(f"{self.model.n_estimators} trees")
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def score_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        @brief Predicted probability of the malicious class, per row.

        @param df  Rows to score.
        @return    Array of probabilities aligned to df's rows.
        """
        return self.model.predict_proba(df[self.feature_cols].to_numpy())[:, 1]

    def score(self, df: pd.DataFrame) -> list[RiskResult]:
        """
        @brief Score rows using the predicted probability of the malicious class.

        @param df  Rows to score (needs the features, 'user', 'role', 'date').
        @return    List of RiskResult in input row order.
        """
        proba = self.score_array(df)
        importances = self.model.feature_importances_

        top_idx = np.argsort(-importances)[:3]
        top_global = [(self.feature_cols[j], float(importances[j])) for j in top_idx]

        users = df["user"].to_numpy()
        roles = df["role"].to_numpy()
        dates = df["date"].to_numpy()
        mal   = (df["is_malicious"].to_numpy() if "is_malicious" in df.columns
                 else np.zeros(len(df), bool))

        return [RiskResult(
            user=users[i], date=pd.Timestamp(dates[i]), role=roles[i],
            score=float(proba[i]), raw_score=float(proba[i]),
            method="population", baseline_used="population",
            top_features=top_global, is_malicious=bool(mal[i]),
        ) for i in range(len(df))]


def results_to_frame(results: list[RiskResult]) -> pd.DataFrame:
    """
    @brief Convert a list of RiskResult into a flat DataFrame for display/eval.

    @param results  List of RiskResult objects.
    @return         DataFrame with one row per result.
    """
    return pd.DataFrame([{
        "user": r.user, "date": r.date, "role": r.role, "score": r.score,
        "raw_score": r.raw_score, "method": r.method,
        "baseline_used": r.baseline_used,
        "top_features": "; ".join(f"{f} ({z:+.2f}s)" for f, z in r.top_features),
        "is_malicious": r.is_malicious,
    } for r in results])
