##
# @file src/features.py
# @brief Causal temporal feature construction.
#
# WHY
# ---
# A user-day row says what somebody did on one day. Nothing in it says whether
# that day is part of a three-week drift or an isolated blip, so a detector
# reading one row at a time is structurally unable to see the low-and-slow
# behaviour that research sub-question 1 is about. The proposal anticipates
# this in its risk management section: if the per-user framework does not
# improve on the baseline, "the anomaly scoring component will be extended to
# incorporate temporal sequence features (rolling window aggregates)".
#
# This module builds those aggregates. Two properties keep it honest:
#
#   * They are *causal*. Every window covers the current row and earlier rows
#     of the same user only, so a row's features never depend on the future.
#     That is what allows them to be computed once over the whole table, before
#     the train/test split, without leaking test information into training.
#
#   * They are *offered to every detector*. The rolling columns are ordinary
#     feature columns; the experiment harness hands the same set to the
#     population baselines and to the per-user framework. Any difference the
#     comparison reports is therefore a difference in modelling strategy, not
#     one side being handed better inputs.
##

import numpy as np
import pandas as pd

from src.schema import FEATURE_COLS

## Trailing windows, in *active days* (rows present for that user) rather than
#  calendar days. A user-day table only holds days a person was active, so an
#  employee on leave does not silently dilute their own baseline with zeros
#  that were never observed.
DEFAULT_WINDOWS = (7, 30)


def rolling_column_names(feature_cols: list[str] | None = None,
                         windows: tuple[int, ...] = DEFAULT_WINDOWS) -> list[str]:
    """
    @brief Names of the rolling columns produced for a feature set.

    @param feature_cols  Base features (defaults to FEATURE_COLS).
    @param windows       Trailing window lengths.
    @return              Column names, in the order add_rolling_features writes them.
    """
    cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
    return [f"{c}_roll{w}" for w in windows for c in cols]


def add_rolling_features(df: pd.DataFrame,
                         feature_cols: list[str] | None = None,
                         windows: tuple[int, ...] = DEFAULT_WINDOWS
                         ) -> tuple[pd.DataFrame, list[str]]:
    """
    @brief Append trailing per-user rolling means for every feature.

    The frame is sorted by (user, date) first, so "trailing" is well defined
    regardless of the input order, and the result is returned in that sorted
    order.

    @param df            User-day rows with 'user', 'date' and the features.
    @param feature_cols  Base features to aggregate (defaults to FEATURE_COLS).
    @param windows       Trailing window lengths in active days.
    @return              (frame with rolling columns appended, new column names).
    """
    cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
    out  = df.sort_values(["user", "date"], kind="mergesort").reset_index(drop=True)

    grouped = out.groupby("user", observed=True, sort=False)[cols]
    added = []
    for w in windows:
        rolled = grouped.rolling(window=w, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True).sort_index()
        names  = [f"{c}_roll{w}" for c in cols]
        rolled.columns = names
        out = pd.concat([out, rolled], axis=1)
        added.extend(names)

    return out, added


def assert_causal(df: pd.DataFrame, feature_cols: list[str],
                  windows: tuple[int, ...] = DEFAULT_WINDOWS) -> None:
    """
    @brief Verify that rolling columns depend on no future row.

    Recomputes the rolling columns from a truncated prefix of each user's
    history and checks the prefix rows are unchanged. A window that peeked
    forward would disagree.

    @param df            Frame already carrying the rolling columns.
    @param feature_cols  Base features.
    @param windows       Windows that were applied.
    @raises AssertionError  If any rolling value changes when the future is removed.
    """
    ordered = df.sort_values(["user", "date"], kind="mergesort").reset_index(drop=True)
    half = len(ordered) // 2
    prefix, _ = add_rolling_features(ordered.iloc[:half][["user", "date"] + feature_cols],
                                     feature_cols, windows)
    for col in rolling_column_names(feature_cols, windows):
        a = ordered[col].to_numpy()[:half]
        b = prefix[col].to_numpy()
        if not np.allclose(a, b, equal_nan=True):
            raise AssertionError(f"rolling column {col} depends on future rows")
