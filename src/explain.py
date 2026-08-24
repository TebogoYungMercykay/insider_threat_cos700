##
# @file src/explain.py
# @brief SHAP-based explainability for the population Random Forest baseline.
#
# SHAP's TreeExplainer gives exact, fast Shapley-value decompositions for
# tree ensembles, which is why it is used here for PopulationBaseline. The
# per-user anomaly framework (IsolationForest / LOF / OneClassSVM) is instead
# explained via the interpretable z-score deviation vector produced directly
# by BehaviouralProfiler (see RiskResult.top_features in src/baseline.py) —
# SHAP's tree support for IsolationForest is experimental and its
# model-agnostic KernelExplainer is far too slow for interactive use, so the
# deviation-vector ranking is used instead as a deliberately faster and
# equally actionable substitute for the anomaly-model side of the framework.
##

import numpy as np
import pandas as pd

from src.preprocess import FEATURE_COLS
from src.progress import Heartbeat, substep


def build_rf_explainer(rf_model):
    """
    @brief Construct a SHAP TreeExplainer for a fitted RandomForestClassifier.

    @param rf_model  Fitted sklearn RandomForestClassifier.
    @return          shap.TreeExplainer instance, or None if shap is unavailable.
    """
    try:
        import shap
        with substep("building SHAP TreeExplainer") as s:
            with Heartbeat("SHAP TreeExplainer build"):
                explainer = shap.TreeExplainer(rf_model)
            s.set_detail(f"{rf_model.n_estimators} trees")
        return explainer
    except Exception:
        return None


def explain_rows(explainer, df: pd.DataFrame, top_n: int = 3) -> list[list[tuple]]:
    """
    @brief Compute per-row top-N SHAP feature contributions for the malicious class.

    @param explainer  A fitted shap.TreeExplainer (or None).
    @param df         Rows to explain (must contain FEATURE_COLS).
    @param top_n      Number of top contributing features to return per row.
    @return           List (one per row) of [(feature, shap_value), ...],
                      or [] per row if explainer is None.
    """
    if explainer is None or len(df) == 0:
        return [[] for _ in range(len(df))]

    X = df[FEATURE_COLS].to_numpy()
    with substep(f"computing SHAP contributions for {len(df):,} row(s)") as s:
        with Heartbeat("SHAP contribution computation"):
            raw = explainer.shap_values(X)
        s.set_detail(f"{len(df):,} rows explained")

    # Newer shap versions return a single (n, features, classes) array;
    # older versions return a list of per-class arrays. Normalise to the
    # positive-class (malicious) 2-D contribution matrix either way.
    if isinstance(raw, list):
        contrib = raw[1] if len(raw) > 1 else raw[0]
    elif raw.ndim == 3:
        contrib = raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    else:
        contrib = raw

    out = []
    for i in range(len(df)):
        order = np.argsort(-np.abs(contrib[i]))[:top_n]
        out.append([(FEATURE_COLS[j], float(contrib[i, j])) for j in order])
    return out
