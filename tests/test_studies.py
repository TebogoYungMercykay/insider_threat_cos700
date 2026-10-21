##
# @file tests/test_studies.py
# @brief Regression tests for the follow-up studies and the queue metrics.
#
# Each test pins one way the studies could report a number that is not what
# it claims to be: a queue position that disagrees with the alert budget, a
# multiple-comparison correction that is too lenient, a "history length" that
# is quietly measuring the role baseline, or an explanation scored against a
# ground truth that the answer key's extra label column has shifted.
#
# Run:  pytest tests/test_studies.py -q
##

import io
import os
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluate import flags_for_budget, insider_timing, rank_rows, select_top_k
from src.studies import (
    _personal_min_history, chance_hit_rate, holm_adjust, malicious_event_features,
    paired_tests, profile_stability, score_explanations, top_k_features,
    truncate_history,
)
from src.preprocess import FEATURE_COLS
from tests.test_model import _frame


# ---------------------------------------------------------------------------
# Queue position and detection delay
# ---------------------------------------------------------------------------

def test_rank_agrees_with_the_alert_budget_even_through_ties():
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.5, 0.7])
    rank = rank_rows(scores, random_state=7)
    for k in range(1, len(scores) + 1):
        assert np.array_equal(rank <= k, select_top_k(scores, k, random_state=7))


def test_insider_timing_counts_the_alerts_ahead_and_the_days_to_detection():
    dates = pd.to_datetime(["2011-01-01", "2011-01-05", "2011-01-09",
                            "2011-01-02", "2011-01-03", "2011-01-04"])
    df = pd.DataFrame({
        "user": ["A", "A", "A", "B", "C", "D"],
        "date": dates,
        "is_malicious": [True, True, True, False, False, False],
    })
    # A's first malicious day ranks 4th, the second 1st, the third 6th.
    scores = np.array([0.5, 0.9, 0.1, 0.8, 0.7, 0.2])
    t = insider_timing(df, scores, budget=2 / 6).set_index("user")
    assert t.loc["A", "best_rank"] == 1
    assert t.loc["A", "alerts_ahead"] == 0
    assert t.loc["A", "n_detected"] == 1
    assert t.loc["A", "detection_delay_days"] == 4.0

    # A budget too small to reach the insider leaves the delay undefined
    # rather than zero, while the queue position is still reported.
    late = np.array([0.1, 0.2, 0.05, 0.9, 0.8, 0.7])
    missed = insider_timing(df, late, budget=1 / 6).set_index("user")
    assert missed.loc["A", "n_detected"] == 0
    assert np.isnan(missed.loc["A", "detection_delay_days"])
    assert missed.loc["A", "alerts_ahead"] == 3


# ---------------------------------------------------------------------------
# Paired tests and the multiple-comparison correction
# ---------------------------------------------------------------------------

def test_holm_adjustment_matches_a_hand_computed_case():
    p = [0.01, 0.04, 0.03, 0.005]
    # Sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert np.allclose(holm_adjust(p), [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm_adjust([np.nan, 0.01])[0])


def test_paired_tests_orient_lower_is_better_metrics_correctly():
    seeds = range(10)
    per_seed = pd.DataFrame(
        [{"seed": s, "detector": "ref", "avg_precision": 0.5 + s * 1e-3,
          "recall": 0.5, "alert_share_high_privilege": 0.03} for s in seeds]
        + [{"seed": s, "detector": "other", "avg_precision": 0.1 + s * 2e-3,
            "recall": 0.5, "alert_share_high_privilege": 0.30 + s * 1e-3} for s in seeds])
    t = paired_tests(per_seed, reference="ref").set_index("metric")

    # Higher AP and a *lower* alert share are both wins for the reference.
    assert t.loc["avg_precision", "wins"] == 10
    assert t.loc["alert_share_high_privilege", "wins"] == 10
    assert t.loc["avg_precision", "p_value"] < 0.005
    # Identical recall is a tie, not evidence.
    assert t.loc["recall", "ties"] == 10 and t.loc["recall", "p_value"] == 1.0
    assert not t.loc["recall", "significant"]


# ---------------------------------------------------------------------------
# History-length study (sub-question 3)
# ---------------------------------------------------------------------------

def test_truncation_keeps_each_users_most_recent_days_only():
    df = _frame(n_users=5, n_days=30)
    train = df[df["split"] == "train"]
    cut = truncate_history(train, 7)
    assert cut.groupby("user").size().max() == 7
    for user, grp in cut.groupby("user"):
        latest = train[train["user"] == user]["date"].nlargest(7)
        assert set(grp["date"]) == set(latest)
    assert len(truncate_history(train, None)) == len(train)


def test_short_histories_are_profiled_personally_not_by_role():
    # A 3-day history with a 14-day cold-start threshold would send every user
    # to the role baseline, and the sweep would measure the wrong thing.
    assert _personal_min_history(3, 14) == 3
    assert _personal_min_history(60, 14) == 14
    assert _personal_min_history(None, 14) == 14

    df = _frame(n_users=6, n_days=40)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    stats = profile_stability(train, test, 3, FEATURE_COLS)
    assert stats["share_scored_personal"] == 1.0


def test_a_shorter_history_moves_further_from_the_full_profile():
    df = _frame(n_users=8, n_days=80)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    short = profile_stability(train, test, 5, FEATURE_COLS)
    full  = profile_stability(train, test, None, FEATURE_COLS)
    assert full["mean_shift_median_sd"] == 0.0
    assert short["mean_shift_median_sd"] > full["mean_shift_median_sd"]


# ---------------------------------------------------------------------------
# Explanation study (sub-question 5)
# ---------------------------------------------------------------------------

def _answers_archive(tmp_path) -> str:
    """@brief A tiny answers archive in the official row layout, label included."""
    rows = "\n".join([
        'logon,"{A}","02/01/2011 22:15:00","ABC0001","PC-1","Logon"',
        'device,"{B}","02/01/2011 22:16:00","ABC0001","PC-1","R:\\","Connect"',
        'file,"{C}","02/01/2011 22:17:00","ABC0001","PC-1","R:\\x.doc","File Write",'
        '"True","False","jobsearch","D0-CF"',
        'email,"{D}","02/02/2011 10:00:00","ABC0001","PC-1","me@home.net","","",'
        '"abc@dtaa.com","Send","1234","C:\\a.doc(10);C:\\b.doc(20)","jobsearch","hi"',
        'http,"{E}","02/02/2011 10:05:00","ABC0001","PC-1","http://jobs.com/x",'
        '"WWW Upload","jobsearch","FF"',
    ]) + "\n"
    path = tmp_path / "answers.tar.bz2"
    with tarfile.open(path, "w:bz2") as tar:
        for name, text in (("answers/r6.2-9.csv", rows),
                           ("answers/r5.2-1.csv", rows.replace("ABC0001", "ZZZ9999"))):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_answer_key_replay_recovers_exactly_the_features_the_events_moved(tmp_path):
    f = malicious_event_features(_answers_archive(tmp_path)).set_index(["user", "date"])
    assert set(f.index.get_level_values("user")) == {"ABC0001"}   # other releases ignored

    day1 = f.loc[("ABC0001", pd.Timestamp("2011-02-01"))]
    assert day1["n_logon"] == 1 and day1["n_afterhours_logon"] == 1
    assert day1["n_device_connect"] == 1
    assert day1["n_file_events"] == 1 and day1["n_file_to_removable"] == 1

    # The label column is dropped before parsing, so the email's size and
    # attachments land in the right slots instead of being shifted by one.
    day2 = f.loc[("ABC0001", pd.Timestamp("2011-02-02"))]
    assert day2["n_email_sent"] == 1 and day2["n_email_external"] == 1
    assert day2["n_email_attachments"] == 2 and day2["total_email_size"] == 1234
    assert day2["n_http_upload"] == 1 and day2["n_distinct_domains"] == 1
    assert day2["n_logon"] == 0


def test_explanations_are_ranked_by_magnitude_and_scored_against_truth():
    names = ["a", "b", "c", "d"]
    ranked = top_k_features(np.array([[0.1, -5.0, 2.0, 0.0]]), names, k=2)
    assert ranked == [["b", "c"]]
    # A reason against the alert is not named, and nothing pads the gap.
    assert top_k_features(np.array([[0.1, -5.0, 2.0, 0.0]]), names, k=3,
                          positive_only=True) == [["c", "a"]]

    s = score_explanations([["b", "c", "a"]], [{"c", "d"}], n_features=4, k=3).iloc[0]
    assert s["hit_at_1"] == 0.0 and s["hit_at_3"] == 1.0
    assert s["precision_at_3"] == pytest.approx(1 / 3)


def test_chance_rate_is_the_hypergeometric_complement():
    assert chance_hit_rate(1, 13, 3) == pytest.approx(3 / 13)
    assert chance_hit_rate(0, 13, 3) == 0.0
    assert chance_hit_rate(11, 13, 3) == 1.0
