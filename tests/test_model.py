##
# @file tests/test_model.py
# @brief Regression tests for the detection framework and its evaluation.
#
# tests/test_pipeline.py covers getting the data right. This file covers not
# fooling ourselves about the model. Most of these pin a specific way a
# comparison on a 1-in-31 000 positive rate can quietly produce a number that
# looks like a result and is not one:
#
#   * a bounded score that clips, tying the whole top of the ranking
#   * an alert budget that a tied score silently overspends by 1000x
#   * a "recall" that comes from evaluating on a sample containing no positives
#   * a rolling feature that reads forward across the split
#
# Every test here runs on small synthetic frames and needs no CERT data.
#
# Run:  pytest tests/test_model.py -q
##

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.baseline import (
    BehaviouralProfiler, PerUserBaselineEngine, PopulationBaseline,
    PROFILE_PERCENTILES, results_to_frame,
)
from src.detectors import build_detectors
from src.evaluate import (
    classify_archetypes, evaluate_recall_by_archetype, evaluate_scores,
    flags_for_budget, high_privilege_mask, select_top_k,
)
from src.features import add_rolling_features, assert_causal, rolling_column_names
from src.preprocess import FEATURE_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _frame(n_users: int = 12, n_days: int = 40, seed: int = 0) -> pd.DataFrame:
    """
    @brief A small user-day frame with the real column contract.

    @param n_users  Distinct users.
    @param n_days   Days per user.
    @param seed     RNG seed.
    @return         Frame with user, date, role, features, labels and split.
    """
    rng   = np.random.default_rng(seed)
    users = [f"U{i:03d}" for i in range(n_users)]
    roles = ["Analyst", "ITAdmin", "Salesman"]

    rows = []
    for i, user in enumerate(users):
        role = roles[i % len(roles)]
        for d in range(n_days):
            rows.append({"user": user, "role": role,
                         "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=d)})
    df = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        df[col] = rng.poisson(4, len(df)).astype(float)

    df["is_high_privilege"] = df["role"].eq("ITAdmin")
    df["is_malicious"] = False
    df["split"] = np.where(df["date"] < pd.Timestamp("2010-02-01"), "train", "test")
    return df


@pytest.fixture
def frame() -> pd.DataFrame:
    return _frame()


# ---------------------------------------------------------------------------
# Profile Builder / Role-Aware Normaliser  (proposal Section 4.3)
# ---------------------------------------------------------------------------

def test_profile_stores_the_percentiles_the_proposal_specifies(frame):
    """Section 4.3 asks for mean, standard deviation *and* percentiles."""
    profiler = BehaviouralProfiler(scope="personal").fit(frame)
    profile = profiler.profile_for("U000")

    expected = {"mean", "std"} | {f"p{int(round(p * 100))}" for p in PROFILE_PERCENTILES}
    assert expected <= set(profile.columns)
    assert list(profile.index) == FEATURE_COLS
    # Percentiles must be ordered within each feature.
    assert (profile["p25"] <= profile["p50"]).all()
    assert (profile["p50"] <= profile["p75"]).all()
    assert (profile["p75"] <= profile["p90"]).all()


def test_each_scope_uses_its_own_reference_distribution(frame):
    """population / role / personal must be three different baselines."""
    deviations = {}
    for scope in ("population", "role", "personal"):
        profiler = BehaviouralProfiler(scope=scope, min_history_days=1).fit(frame)
        Z, used = profiler.transform(frame)
        deviations[scope] = Z
        assert set(used) == {{"population": "population", "role": "role_group",
                              "personal": "personal"}[scope]}

    assert not np.allclose(deviations["population"], deviations["role"])
    assert not np.allclose(deviations["role"], deviations["personal"])


def test_personal_scope_falls_back_to_role_group_for_cold_start(frame):
    """A user below the history threshold gets their role group, not silence."""
    thin = frame[frame["user"] != "U000"].copy()
    newcomer = frame[frame["user"] == "U000"].head(3)          # 3 days of history
    train = pd.concat([thin, newcomer], ignore_index=True)

    profiler = BehaviouralProfiler(scope="personal", min_history_days=14).fit(train)
    _, used = profiler.transform(train)

    used = pd.Series(used, index=train.index)
    assert set(used[train["user"] == "U000"]) == {"role_group"}
    assert set(used[train["user"] == "U001"]) == {"personal"}


def test_a_never_before_seen_event_is_a_deviation_not_a_divide_by_zero(frame):
    """
    A feature a user has never recorded has zero spread. Flooring the scale at
    1.0 makes the first occurrence read as its own raw count, which is the
    point: one write to removable media by someone who has never done it is
    the signal, and it must not become NaN or a silent zero.
    """
    train = frame.copy()
    train["n_file_to_removable"] = 0.0

    profiler = BehaviouralProfiler(scope="personal", min_history_days=1).fit(train)

    probe = train.head(1).copy()
    probe["n_file_to_removable"] = 3.0
    Z, _ = profiler.transform(probe)

    j = FEATURE_COLS.index("n_file_to_removable")
    assert np.isfinite(Z).all()
    assert Z[0, j] == pytest.approx(3.0)


def test_robust_statistic_survives_a_spike_that_hides_the_next_one(frame):
    """
    One huge day inflates the standard deviation, so the *following* elevated
    days score as unremarkable. The median/IQR form does not have that
    failure, which is why it is offered as an alternative rather than a
    cosmetic option.
    """
    train = frame[frame["user"] == "U000"].copy()
    train["n_http_upload"] = 1.0
    train.loc[train.index[0], "n_http_upload"] = 500.0        # the spike

    probe = train.head(1).copy()
    probe["n_http_upload"] = 8.0                              # a mild elevation

    j = FEATURE_COLS.index("n_http_upload")
    z_dev = BehaviouralProfiler(scope="personal", min_history_days=1,
                                statistic="zscore").fit(train).transform(probe)[0][0, j]
    r_dev = BehaviouralProfiler(scope="personal", min_history_days=1,
                                statistic="robust").fit(train).transform(probe)[0][0, j]

    assert abs(z_dev) < 1.0        # buried by the inflated standard deviation
    assert abs(r_dev) > abs(z_dev)


# ---------------------------------------------------------------------------
# Scoring: ranking must not be destroyed by presentation
# ---------------------------------------------------------------------------

def test_calibrated_score_is_monotone_and_never_saturates(frame):
    """
    Regression: the display score used to min-max scale against the training
    range and clip at 1.0, which collapsed every test row above the 99.5th
    training percentile onto one value -- exactly the rows a detector exists
    to rank. Ranking now comes from raw_score, and the display scale must
    stay strictly ordered with it.
    """
    engine = PerUserBaselineEngine(method="iforest").fit(frame)

    probe = frame.copy()
    probe.loc[probe.index[:25], FEATURE_COLS] = 1000.0        # far beyond training

    raw = engine.raw_score(probe)
    cal = engine.score_array(probe)

    assert cal.max() < 1.0
    assert np.all(np.diff(cal[np.argsort(raw)]) >= -1e-12)
    # The extreme rows are identical, so they legitimately tie with each other
    # and with nothing else.
    assert len(np.unique(cal)) >= len(np.unique(raw)) - 1


def test_detector_ranks_on_raw_score_not_the_display_scale(frame):
    """The comparison must consume the unbounded score, or ties reappear."""
    [detector] = [d for d in build_detectors("core")
                  if d.name == "peruser-iforest-personal-zscore"]
    detector.fit(frame)

    probe = frame.copy()
    probe.loc[probe.index[:25], FEATURE_COLS] = 1000.0
    assert np.allclose(detector.score(probe), detector.engine.raw_score(probe))


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------

def test_top_k_flags_exactly_k_even_when_scores_are_tied():
    """
    Regression: a Random Forest trained on six positives scores almost every
    row 0.0. Flagging every row that meets the cut let it spend 40 360 alerts
    against a budget of 404 and then report the recall that bought.
    """
    scores = np.array([0.9, 0.5] + [0.0] * 98)
    for k in (1, 2, 5, 50):
        assert select_top_k(scores, k).sum() == k
    # The genuinely top-ranked rows are always chosen.
    mask = select_top_k(scores, 5)
    assert mask[0] and mask[1]


def test_tie_breaking_is_reproducible_and_ignores_row_order():
    """Ties are resolved by seed, never by how the Parquet file was sorted."""
    scores = np.array([0.0] * 50 + [1.0])
    a = select_top_k(scores, 10, random_state=7)
    b = select_top_k(scores, 10, random_state=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, select_top_k(scores, 10, random_state=8))


def test_tie_fraction_reports_how_much_of_the_budget_was_a_coin_toss():
    """A degenerate ranking must stay visible in the results table."""
    ranked = np.arange(100, dtype=float)
    _, clean = flags_for_budget(ranked, 0.1)
    assert clean["tie_fraction"] == 0.0

    flat = np.zeros(100)
    flat[0] = 1.0
    _, degenerate = flags_for_budget(flat, 0.1)
    assert degenerate["tie_fraction"] == pytest.approx(0.9)


def test_budget_is_respected_across_detectors(frame):
    """Every detector spends the same number of alerts, whatever its scale."""
    test_df = frame.copy()
    test_df.loc[test_df.index[:4], "is_malicious"] = True

    for scores in (np.random.default_rng(0).random(len(test_df)),
                   np.zeros(len(test_df)),
                   np.linspace(0, 1e9, len(test_df))):
        m = evaluate_scores(test_df, scores, budget=0.05)
        assert m["n_flagged"] == max(1, round(len(test_df) * 0.05))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metric_arithmetic_on_a_hand_checked_case():
    """Ten rows, two malicious, a budget of four, three of which are hits."""
    df = pd.DataFrame({
        "user":         [f"u{i}" for i in range(10)],
        "role":         ["Analyst"] * 10,
        "is_malicious": [True, True, False, False, False, False, False, False, False, False],
        "is_high_privilege": [False] * 10,
    })
    scores = np.array([0.99, 0.98, 0.97, 0.96, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    m = evaluate_scores(df, scores, budget=0.4)
    assert m["n_flagged"] == 4
    assert m["recall"] == pytest.approx(1.0)          # both malicious flagged
    assert m["precision"] == pytest.approx(0.5)       # 2 of 4 flagged are hits
    assert m["lift"] == pytest.approx(2.5)            # precision / base rate 0.2
    assert m["n_insiders_detected"] == 2
    assert m["insider_recall"] == pytest.approx(1.0)


def test_high_privilege_subgroup_prefers_the_datasets_own_flag():
    """The subgroup definition must not drift between prepare and evaluate."""
    df = pd.DataFrame({
        "role": ["Mystery Role", "Analyst"],
        "is_high_privilege": [True, False],
    })
    assert list(high_privilege_mask(df)) == [True, False]

    # Without the column, fall back to matching the role name.
    fallback = pd.DataFrame({"role": ["ITAdmin", "Analyst"]})
    assert list(high_privilege_mask(fallback)) == [True, False]


def test_alert_share_exposes_who_the_budget_is_spent_on():
    """
    Sub-question 2 is about false alerts on power users. FPR alone understates
    it, because high-privilege users are a small slice of the population; the
    share of the *queue* they occupy is what an analyst experiences.
    """
    df = pd.DataFrame({
        "user":         [f"u{i}" for i in range(10)],
        "role":         ["ITAdmin"] * 2 + ["Analyst"] * 8,
        "is_malicious": [False] * 10,
        "is_high_privilege": [True] * 2 + [False] * 8,
    })
    scores = np.array([1.0, 1.0] + [0.0] * 8)         # both alerts on admins

    m = evaluate_scores(df, scores, budget=0.2)
    assert m["alert_share_high_privilege"] == pytest.approx(1.0)
    assert m["population_share_high_privilege"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Archetypes  (research sub-question 1)
# ---------------------------------------------------------------------------

def test_archetype_split_comes_from_the_answer_key_geometry():
    """
    Derived from window span alone, fixed before any model runs. Deciding it
    from how detectors happened to score would make sub-question 1 circular.
    """
    gt = pd.DataFrame({
        "user":       ["BURST", "SLOW"],
        "start_date": ["2010-08-18", "2011-02-02"],
        "end_date":   ["2010-08-24", "2011-03-30"],
    })
    arch = classify_archetypes(gt).set_index("user")["archetype"]
    assert arch["BURST"] == "burst"
    assert arch["SLOW"] == "low_and_slow"


def test_archetype_recall_counts_only_labelled_malicious_days():
    """
    Benign days between an insider's malicious sessions are not misses.
    Counting them would penalise every detector for correctly ignoring
    behaviour the answer key says was not malicious.
    """
    df = pd.DataFrame({
        "user":         ["SLOW"] * 4,
        "role":         ["Analyst"] * 4,
        "is_malicious": [True, False, True, False],
    })
    gt = pd.DataFrame({"user": ["SLOW"], "start_date": ["2011-02-02"],
                       "end_date": ["2011-03-30"]})

    flagged = np.array([True, False, False, True])
    out = evaluate_recall_by_archetype(df, flagged, gt)

    assert out["low_and_slow"]["n_malicious_days"] == 2      # not 4
    assert out["low_and_slow"]["n_detected"] == 1
    assert out["low_and_slow"]["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def test_rolling_features_read_only_backwards(frame):
    """A window that peeked forward would leak the test split into training."""
    out, added = add_rolling_features(frame, FEATURE_COLS)
    assert added == rolling_column_names(FEATURE_COLS)
    assert_causal(out, FEATURE_COLS)


def test_causality_check_actually_catches_a_forward_looking_window(frame):
    """The guard is only worth having if it fails on a real violation."""
    out, _ = add_rolling_features(frame, FEATURE_COLS)
    # A centred window sees the future; the check must reject it.
    out["n_logon_roll7"] = (out.groupby("user")["n_logon"]
                            .transform(lambda s: s.rolling(7, center=True,
                                                           min_periods=1).mean()))
    with pytest.raises(AssertionError):
        assert_causal(out, FEATURE_COLS)


def test_rolling_windows_do_not_bleed_between_users(frame):
    """Each user's trailing window is their own history, nobody else's."""
    out, _ = add_rolling_features(frame, ["n_logon"], (7,))
    first_rows = out.groupby("user").head(1)
    assert np.allclose(first_rows["n_logon_roll7"], first_rows["n_logon"])


# ---------------------------------------------------------------------------
# The detector contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["core", "baselines"])
def test_every_detector_honours_the_same_interface(frame, preset):
    """One fit signature, one score signature, finite scores, right length."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    test  = frame[frame["split"] == "test"].reset_index(drop=True)
    train.loc[train.index[:6], "is_malicious"] = True

    for detector in build_detectors(preset, fit_subsample=200):
        detector.fit(train)
        scores = np.asarray(detector.score(test), dtype=float)
        assert scores.shape == (len(test),), detector.name
        assert np.isfinite(scores).all(), detector.name
        assert "family" in detector.describe(), detector.name


def test_the_framework_is_deterministic_under_a_fixed_seed(frame):
    """A reported result has to be reproducible from the seed alone."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    test  = frame[frame["split"] == "test"].reset_index(drop=True)

    a = PerUserBaselineEngine(method="iforest", random_state=42).fit(train).raw_score(test)
    b = PerUserBaselineEngine(method="iforest", random_state=42).fit(train).raw_score(test)
    assert np.array_equal(a, b)


def test_profiles_are_built_from_training_rows_only(frame):
    """
    The per-user baseline is fitted on the training window. If test rows
    reached the profiler, a user's baseline would already contain the
    behaviour being scored.
    """
    train = frame[frame["split"] == "train"]
    test  = frame[frame["split"] == "test"].copy()
    test[FEATURE_COLS] = 999.0

    profiler = BehaviouralProfiler(scope="personal", min_history_days=1).fit(train)
    baseline_mean = profiler.user_stats_.loc["U000", "n_logon__mean"]

    assert baseline_mean == pytest.approx(
        train.loc[train["user"] == "U000", "n_logon"].mean())
    # And the untouched profile makes the inflated test rows look extreme.
    Z, _ = profiler.transform(test[test["user"] == "U000"])
    assert Z[:, FEATURE_COLS.index("n_logon")].min() > 10


def test_results_frame_carries_both_scores(frame):
    """The app displays the calibrated score but must be able to rank by raw."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    engine = PerUserBaselineEngine(method="iforest").fit(train)
    out = results_to_frame(engine.score(train.head(20)))
    assert {"score", "raw_score"} <= set(out.columns)
    assert out["score"].between(0, 1).all()


def test_magnitude_channel_separates_trivial_from_material_deviations(frame):
    """
    A pure deviation vector discards scale: 2 -> 8 and 200 -> 800 are both
    "6 sigma" if each user's own spread says so. Across thousands of users the
    trivially small ones crowd out the material one at a tight alert budget,
    which is what the magnitude channel exists to prevent.
    """
    # Give each user spread proportional to their level, so a raw value of 8
    # is exactly as many standard deviations above U000's mean as 800 is above
    # U001's. A constant column would floor the scale at 1.0 and accidentally
    # preserve magnitude, which is not the case under test.
    train = frame.copy()
    for user, mean, sd in (("U000", 2.0, 1.0), ("U001", 200.0, 100.0)):
        rows = train["user"] == user
        shape = np.arange(int(rows.sum()), dtype=float)
        shape = (shape - shape.mean()) / shape.std(ddof=1)
        train.loc[rows, "n_file_events"] = mean + sd * shape

    quiet = train[train["user"] == "U000"].head(1).copy()
    quiet["n_file_events"] = 8.0            # trivially small, personally odd
    loud = train[train["user"] == "U001"].head(1).copy()
    loud["n_file_events"] = 800.0           # materially large, personally odd
    probe = pd.concat([quiet, loud], ignore_index=True)

    j = FEATURE_COLS.index("n_file_events")

    plain = BehaviouralProfiler(scope="personal", min_history_days=1).fit(train)
    Z, _ = plain.transform(probe)
    assert Z[0, j] == pytest.approx(Z[1, j])          # indistinguishable

    with_mag = BehaviouralProfiler(scope="personal", min_history_days=1,
                                   include_magnitude=True).fit(train)
    Zm, _ = with_mag.transform(probe)
    assert Zm.shape[1] == 2 * len(FEATURE_COLS)
    assert with_mag.output_columns[j].endswith("__dev")
    assert with_mag.output_columns[len(FEATURE_COLS) + j].endswith("__mag")
    # The deviation half still ties; the magnitude half separates them.
    assert Zm[0, j] == pytest.approx(Zm[1, j])
    assert Zm[1, len(FEATURE_COLS) + j] > Zm[0, len(FEATURE_COLS) + j]


def test_explanations_name_the_right_column_when_magnitude_is_on(frame):
    """Doubling the deviation matrix must not desync the feature names."""
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    engine = PerUserBaselineEngine(method="iforest", include_magnitude=True).fit(train)
    valid = set(engine.profiler.output_columns)
    for result in engine.score(train.head(10)):
        assert {name for name, _ in result.top_features} <= valid


def test_framework_at_population_scope_reduces_to_the_population_detector(frame):
    """
    The ablation's claim is that population / role / personal differ in nothing
    but the reference distribution. If that holds, the framework configured at
    population scope with the z-score statistic *is* the population anomaly
    detector, and their scores must agree.

    They do, but not because the inputs are byte-identical: the profiler uses
    the sample standard deviation (ddof=1) while StandardScaler uses the
    population one (ddof=0). That is a single global scalar -- the same factor
    for every feature -- and both estimators absorb it. Isolation Forest splits
    at a uniform point within each feature's observed range, so a uniform
    rescale leaves every split fraction unchanged; an RBF One-Class SVM with
    gamma="scale" divides by the data variance, which cancels it.

    This is what makes the headline comparison attributable to per-user
    normalisation rather than to the choice of anomaly algorithm.
    """
    from src.detectors import PerUserDetector, PopulationAnomalyDetector

    train = frame[frame["split"] == "train"].reset_index(drop=True)
    test  = frame[frame["split"] == "test"].reset_index(drop=True)

    common = dict(contamination=0.02, fit_subsample=None, random_state=42,
                  feature_cols=FEATURE_COLS)

    # The inputs differ by exactly one global scalar, and nothing else.
    from sklearn.preprocessing import StandardScaler
    X = train[FEATURE_COLS].to_numpy(float)
    sk = StandardScaler().fit(X).transform(X)
    pf, _ = BehaviouralProfiler(scope="population", statistic="zscore").fit(train).transform(train)
    nonzero = np.abs(sk) > 1e-9
    ratios = pf[nonzero] / sk[nonzero]
    assert np.allclose(ratios, ratios[0], rtol=1e-9)

    # And the scores agree regardless, for both estimators.
    for method in ("iforest", "ocsvm"):
        population = PopulationAnomalyDetector("pop", method, **common).fit(train)
        framework  = PerUserDetector("fw", method=method, scope="population",
                                     statistic="zscore", **common).fit(train)
        assert np.allclose(population.score(test), framework.score(test),
                           atol=1e-8), method
