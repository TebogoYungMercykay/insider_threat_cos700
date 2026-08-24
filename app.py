##
# @file app.py
# @brief Streamlit UI — Per-User Behavioural Baseline Framework for Insider Threat Detection.
#
# Requires train.py to have been run first:
#   python train.py
#
# Then launch:
#   streamlit run app.py
##

import os
import sys
import time
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))

from src.colors import (
    HEX_AMBER  as COLOR_AMBER,
    HEX_BLUE   as COLOR_BLUE,
    HEX_GREEN  as COLOR_GREEN,
    HEX_RED    as COLOR_RED,
    HEX_BG_BLUE   as COLOR_BG_BLUE,
    HEX_BG_GREEN  as COLOR_BG_GREEN,
    HEX_BG_RED    as COLOR_BG_RED,
    HEX_BG_AMBER  as COLOR_BG_AMBER,
)
from src.preprocess import FEATURE_COLS

ROOT          = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT, "data", "artifacts")

PATHS = {
    "daily_df":   os.path.join(ARTIFACTS_DIR, "daily_df.pkl"),
    "gt_df":      os.path.join(ARTIFACTS_DIR, "gt_df.pkl"),
    "population": os.path.join(ARTIFACTS_DIR, "population_rf.pkl"),
    "peruser":    os.path.join(ARTIFACTS_DIR, "peruser_engine.pkl"),
    "shap":       os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"),
    "meta":       os.path.join(ARTIFACTS_DIR, "meta.json"),
}

st.set_page_config(page_title="Insider Threat — Per-User Baseline", layout="wide")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');
    body, .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
    code, .mono  {{ font-family: 'JetBrains Mono', monospace; }}

    .section-header {{
        color: {COLOR_BLUE}; font-size: 1.05rem; font-weight: 600;
        border-bottom: 2px solid {COLOR_BLUE}; padding-bottom: 4px; margin-bottom: 12px;
    }}
    .info-box {{
        background-color: {COLOR_BG_BLUE}; border-left: 4px solid {COLOR_BLUE};
        padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;
    }}
    .warning-box {{
        background-color: {COLOR_BG_AMBER}; border-left: 4px solid {COLOR_AMBER};
        padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;
    }}
    .danger-box {{
        background-color: {COLOR_BG_RED}; border-left: 4px solid {COLOR_RED};
        padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;
    }}
    .success-box {{
        background-color: {COLOR_BG_GREEN}; border-left: 4px solid {COLOR_GREEN};
        padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;
    }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_BLUE}; font-weight: 700; }}

    .result-card {{
        border: 1px solid rgba(21,101,192,0.3); border-radius: 6px;
        padding: 14px 16px; margin-bottom: 10px; background-color: rgba(21,101,192,0.05);
    }}
    .result-rank {{
        color: {COLOR_BLUE}; font-size: 0.78rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .cat-tag {{
        display: inline-block; background: rgba(21,101,192,0.2); color: {COLOR_BLUE};
        font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 3px; margin-left: 8px;
    }}
    .hp-tag {{
        display: inline-block; background: rgba(245,158,11,0.25); color: {COLOR_AMBER};
        font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 3px; margin-left: 6px;
    }}
    .banner-wrap {{
        background: #0d1117; border: 1px solid {COLOR_BLUE}; border-radius: 8px;
        padding: 28px 32px 20px 32px; margin-bottom: 24px; overflow: hidden;
    }}
    .banner-title {{ color: #fff; font-size: 1.15rem; font-weight: 700; margin: 0 0 4px 0; }}
    .banner-subtitle {{ color: #8b949e; font-size: 0.85rem; margin: 0; }}
    .banner-tagline {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #4a7cc7;
        margin: 8px 0 14px 0; letter-spacing: 0.05em;
    }}
    .banner-divider {{ border: none; border-top: 1px solid rgba(21,101,192,0.35); margin: 14px 0; }}
    .banner-badge {{
        display: inline-block; margin-top: 14px; padding: 3px 12px; border: 1px solid {COLOR_BLUE};
        border-radius: 3px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.1em; color: {COLOR_BLUE};
    }}
</style>
""", unsafe_allow_html=True)


def show_banner() -> None:
    """@brief Render the project banner."""
    st.markdown(f"""
    <div class="banner-wrap">
        <p class="banner-title">🛡️ &nbsp;Per-User Behavioural Baseline Framework</p>
        <div class="banner-tagline">INSIDER&nbsp;THREAT&nbsp;DETECTION &nbsp;•&nbsp; ROLE-AWARE&nbsp;ANOMALY&nbsp;SCORING</div>
        <hr class="banner-divider">
        <p class="banner-subtitle">
            COS700 Research Report &mdash; University of Pretoria, 2026 &nbsp;|&nbsp; Supervisor: Prof. Jan Eloff
        </p>
        <span class="banner-badge">PROTOTYPE v1.0</span>
    </div>
    """, unsafe_allow_html=True)


def _artifacts_exist() -> bool:
    required = ["daily_df", "gt_df", "population", "peruser"]
    return all(os.path.exists(PATHS[k]) for k in required)


def show_not_trained_error() -> None:
    st.error("**No trained artifacts found.**")
    st.markdown(f"""
    <div class="danger-box">
        Run the training pipeline first, then restart Streamlit:<br><br>
        <code>python script/dataset.py</code>  (ingests real CERT logs from data/, or generates synthetic data if absent)<br>
        <code>python train.py</code><br><br>
        For a quick smoke-test:<br>
        <code>python train.py --max-users 100 --no-shap</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    @brief Load all training artifacts.

    @return  (daily_df, gt_df, population_baseline, peruser_engine, shap_explainer_or_None)
    """
    import joblib
    with st.status("Loading trained artifacts…", expanded=True) as status:
        status.write("Loading engineered feature table (daily_df.pkl)…")
        daily_df = joblib.load(PATHS["daily_df"])
        status.write("Loading ground-truth insider windows (gt_df.pkl)…")
        gt_df = joblib.load(PATHS["gt_df"])
        status.write("Loading Population (RF) baseline…")
        population = joblib.load(PATHS["population"])
        status.write("Loading Per-User Framework…")
        peruser = joblib.load(PATHS["peruser"])
        if os.path.exists(PATHS["shap"]):
            status.write("Loading SHAP explainer…")
            shap_explainer = joblib.load(PATHS["shap"])
        else:
            shap_explainer = None
        status.update(label="Artifacts loaded.", state="complete", expanded=False)
    return daily_df, gt_df, population, peruser, shap_explainer


def _load_meta() -> dict:
    if os.path.exists(PATHS["meta"]):
        with open(PATHS["meta"]) as f:
            return json.load(f)
    return {}


def _dark_fig(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a5a")
    return fig, ax


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(daily_df, meta: dict) -> None:
    show_banner()

    cols = st.columns(5)
    cols[0].metric("User-Days Indexed", f"{len(daily_df):,}")
    cols[1].metric("Users",             f"{daily_df['user'].nunique():,}")
    cols[2].metric("Roles",             daily_df["role"].nunique())
    cols[3].metric("Malicious Rows",    int(daily_df["is_malicious"].sum()))
    if meta.get("trained_at"):
        cols[4].metric("Trained At", meta["trained_at"])

    st.markdown('<p class="section-header">How It Works</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""<div class="info-box">
            <b>1 · Profile Builder</b><br>
            Per-user historical mean/std computed from each user's own training-window
            activity. Users with insufficient history fall back to their role-group
            baseline (cold-start handling).
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="info-box">
            <b>2 · Role-Aware Normaliser</b><br>
            Features are z-scored relative to the user's own role-peer group,
            not the whole organisation — a database administrator's bulk file access
            is only "unusual" if it's unusual <i>for a database administrator</i>.
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div class="info-box">
            <b>3 · Anomaly Scorer</b><br>
            An unsupervised outlier model ({meta.get('anomaly_method','iforest').upper()})
            scores each user-day's <i>deviation vector</i> — so one cheaply-trained
            global model behaves as a per-user-aware detector.
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Role Distribution</p>', unsafe_allow_html=True)
    counts = daily_df.drop_duplicates("user")["role"].value_counts().sort_values(ascending=True)
    fig, ax = _dark_fig(9, max(3, len(counts) * 0.45 + 0.6))
    bars = ax.barh(counts.index, counts.values, color=COLOR_BLUE)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color="white", fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Users")
    ax.grid(axis="x", color="#1e2d45", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def page_risk_explorer(daily_df, population, peruser, shap_explainer) -> None:
    """@brief Per-user risk timeline: population score vs per-user score, with explanations."""
    st.subheader("Per-User Risk Explorer")

    users = sorted(daily_df["user"].unique())
    col_u, col_m = st.columns([2, 1])
    with col_u:
        user = st.selectbox("Select a user", users)
    with col_m:
        method = st.radio("Score with", ["Both", "Population (RF)", "Per-User Framework"], horizontal=False)

    user_df = daily_df[daily_df["user"] == user].sort_values("date").reset_index(drop=True)
    role = user_df["role"].iloc[0]
    is_hp = bool(user_df["is_high_privilege"].iloc[0])

    st.markdown(
        f'<span class="cat-tag">{role}</span>'
        + ('<span class="hp-tag">HIGH-PRIVILEGE</span>' if is_hp else ""),
        unsafe_allow_html=True,
    )

    from src.baseline import results_to_frame
    from src.explain import explain_rows

    pop_res = results_to_frame(population.score(user_df)) if method in ("Both", "Population (RF)") else None
    per_res = results_to_frame(peruser.score(user_df)) if method in ("Both", "Per-User Framework") else None

    fig, ax = _dark_fig(11, 3.2)
    if pop_res is not None:
        ax.plot(pop_res["date"], pop_res["score"], color=COLOR_RED, label="Population (RF)", linewidth=1.6)
    if per_res is not None:
        ax.plot(per_res["date"], per_res["score"], color=COLOR_BLUE, label="Per-User Framework", linewidth=1.6)
    mal_days = user_df.loc[user_df["is_malicious"], "date"]
    for d in mal_days:
        ax.axvspan(d, d + pd.Timedelta(days=1), color=COLOR_AMBER, alpha=0.15)
    ax.set_ylabel("Risk score")
    ax.set_title(f"Risk score over time — {user}", fontsize=10)
    ax.legend(facecolor="#0d1117", labelcolor="white", fontsize=8)
    ax.grid(color="#1e2d45", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown('<p class="section-header">Highest-Risk Days</p>', unsafe_allow_html=True)
    display_res = per_res if per_res is not None else pop_res
    top = display_res.sort_values("score", ascending=False).head(8).copy()

    # For rows scored by the population RF, replace the generic global
    # feature-importance explanation with a real per-row SHAP explanation.
    if display_res is pop_res and shap_explainer is not None:
        exp_rows = user_df.set_index("date").loc[top["date"]].reset_index()
        shap_expl = explain_rows(shap_explainer, exp_rows)
        top["top_features"] = [
            "; ".join(f"{f} ({v:+.3f})" for f, v in row) or "—" for row in shap_expl
        ]

    for _, row in top.iterrows():
        mal_flag = ' <span class="hp-tag" style="background:rgba(211,47,47,0.25);color:#ff6b6b;">MALICIOUS DAY</span>' if row["is_malicious"] else ""
        st.markdown(f"""
        <div class="result-card">
            <span class="result-rank">{row['date'].strftime('%Y-%m-%d')}</span>
            <span class="cat-tag">{row['method']}</span>
            <span style="color:#8b949e;font-size:0.8rem;">
                &nbsp;|&nbsp; Baseline: {row['baseline_used']}
                &nbsp;|&nbsp; Score: <b style="color:{COLOR_BLUE}">{row['score']:.3f}</b>
            </span>{mal_flag}
            <div style="font-family:monospace;font-size:0.8rem;color:#cdd;margin-top:6px;">
                Top contributing features: {row['top_features'] if row['top_features'] else '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_compare(daily_df, population, peruser) -> None:
    """@brief Side-by-side population vs per-user scoring across a chosen role/day slice."""
    st.subheader("Method Comparison — Population Baseline vs Per-User Framework")
    st.markdown(
        '<div class="info-box">This page reproduces the central claim of the research proposal: '
        'compare how each method scores the <b>same</b> activity for high-privilege vs standard users.</div>',
        unsafe_allow_html=True,
    )

    test_df = daily_df[daily_df["split"] == "test"]
    role_filter = st.multiselect("Filter by role", sorted(daily_df["role"].unique()),
                                  default=["SysAdmin", "DBAdmin", "Employee"])
    sub = test_df[test_df["role"].isin(role_filter)] if role_filter else test_df
    n_sample = st.slider("Sample size", 20, min(2000, max(20, len(sub))), min(300, len(sub)))
    sample = sub.sample(n=min(n_sample, len(sub)), random_state=42)

    from src.baseline import results_to_frame
    t0 = time.perf_counter()
    pop_res = results_to_frame(population.score(sample))
    t_pop = time.perf_counter() - t0

    t0 = time.perf_counter()
    per_res = results_to_frame(peruser.score(sample))
    t_per = time.perf_counter() - t0

    c1, c2 = st.columns(2)
    c1.metric("Population (RF) scoring time", f"{t_pop*1000:.1f} ms", f"{len(sample)} rows")
    c2.metric("Per-User Framework scoring time", f"{t_per*1000:.1f} ms", f"{len(sample)} rows")

    merged = pop_res[["user", "date", "role", "score"]].rename(columns={"score": "population_score"})
    merged = merged.merge(
        per_res[["user", "date", "score", "baseline_used"]].rename(columns={"score": "peruser_score"}),
        on=["user", "date"], how="inner",
    )
    merged["role_high_priv"] = merged["role"].isin(["SysAdmin", "DBAdmin"])

    fig, ax = _dark_fig(9, 5.5)
    colors = np.where(merged["role_high_priv"], COLOR_AMBER, COLOR_BLUE)
    ax.scatter(merged["population_score"], merged["peruser_score"], c=colors, s=18, alpha=0.7)
    ax.set_xlabel("Population (RF) score")
    ax.set_ylabel("Per-User Framework score")
    ax.set_title("Score comparison  (amber = high-privilege role)", fontsize=10)
    ax.grid(color="#1e2d45", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown('<p class="section-header">Mean Score by Role</p>', unsafe_allow_html=True)
    role_means = merged.groupby("role")[["population_score", "peruser_score"]].mean().sort_values("population_score")
    fig, ax = _dark_fig(9, max(3, len(role_means) * 0.4))
    y = np.arange(len(role_means))
    ax.barh(y - 0.2, role_means["population_score"], height=0.4, color=COLOR_RED, label="Population (RF)")
    ax.barh(y + 0.2, role_means["peruser_score"], height=0.4, color=COLOR_BLUE, label="Per-User Framework")
    ax.set_yticks(y)
    ax.set_yticklabels(role_means.index)
    ax.legend(facecolor="#0d1117", labelcolor="white", fontsize=8)
    ax.grid(axis="x", color="#1e2d45", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def page_evaluate(daily_df, gt_df, population, peruser) -> None:
    st.subheader("Evaluation — Population Baseline vs Per-User Framework")
    st.markdown(
        '<div class="info-box">Evaluated on the <b>held-out time-based test split</b> '
        '(no leakage: both models are fit on the training window only). Adjust the '
        'decision threshold to explore different operating points.</div>',
        unsafe_allow_html=True,
    )

    test_df = daily_df[daily_df["split"] == "test"].reset_index(drop=True)

    col_t, col_s = st.columns(2)
    threshold = col_t.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    sample_n  = col_s.slider("Test rows to evaluate", 200, len(test_df), min(5000, len(test_df)))

    if st.button("Run Evaluation", type="primary"):
        from src.baseline import results_to_frame
        from src.evaluate import evaluate_predictions, evaluate_recall_by_archetype, measure_computational_cost

        sample = test_df.sample(n=min(sample_n, len(test_df)), random_state=42)

        with st.status(f"Evaluating on {len(sample):,} sampled test rows…", expanded=True) as status:
            status.write(f"Scoring Population (RF) baseline on {len(sample):,} rows…")
            pop_res = results_to_frame(population.score(sample))
            status.write(f"Scoring Per-User Framework on {len(sample):,} rows…")
            per_res = results_to_frame(peruser.score(sample))
            status.write("Computing primary metrics (recall, precision, F1, ROC-AUC, FPR)…")
            pop_metrics = evaluate_predictions(pop_res, threshold=threshold)
            per_metrics = evaluate_predictions(per_res, threshold=threshold)
            status.write("Breaking recall down by attacker archetype…")
            pop_arch = evaluate_recall_by_archetype(pop_res, gt_df, threshold=threshold)
            per_arch = evaluate_recall_by_archetype(per_res, gt_df, threshold=threshold)
            status.update(label="Evaluation complete.", state="complete", expanded=False)

        st.markdown('<p class="section-header">Primary Metrics</p>', unsafe_allow_html=True)
        comp = pd.DataFrame({
            "Population (RF)": {k: pop_metrics[k] for k in
                                 ["recall", "precision", "f1", "roc_auc", "fpr_overall", "fpr_high_privilege"]},
            "Per-User Framework": {k: per_metrics[k] for k in
                                    ["recall", "precision", "f1", "roc_auc", "fpr_overall", "fpr_high_privilege"]},
        })
        st.dataframe(comp, width="stretch")

        m1, m2, m3 = st.columns(3)
        delta_recall = per_metrics["recall"] - pop_metrics["recall"]
        delta_fpr_hp = per_metrics["fpr_high_privilege"] - pop_metrics["fpr_high_privilege"]
        m1.metric("Recall — Population", f"{pop_metrics['recall']:.3f}")
        m2.metric("Recall — Per-User", f"{per_metrics['recall']:.3f}", f"{delta_recall:+.3f}")
        m3.metric("FPR (high-privilege) Δ", f"{delta_fpr_hp:+.3f}",
                   help="Negative is better: per-user framework flags fewer high-privilege false positives.")

        st.markdown('<p class="section-header">Recall by Attacker Archetype</p>', unsafe_allow_html=True)
        arch_df = pd.DataFrame({"Population (RF)": pop_arch, "Per-User Framework": per_arch}).fillna(0)
        if not arch_df.empty:
            st.dataframe(arch_df, width="stretch")
            st.caption("`low_slow` insiders are the ones the research question targets: small, sustained "
                       "per-user drift that a population-level model is expected to miss.")
        else:
            st.info("No labelled malicious windows fall inside the sampled test rows — increase sample size.")

        st.markdown('<p class="section-header">False Positive Rate by Role</p>', unsafe_allow_html=True)
        role_fpr_df = pd.DataFrame({
            "Population (RF)": pd.Series(pop_metrics["fpr_by_role"]),
            "Per-User Framework": pd.Series(per_metrics["fpr_by_role"]),
        }).fillna(0).sort_values("Population (RF)")
        fig, ax = _dark_fig(9, max(3, len(role_fpr_df) * 0.4))
        y = np.arange(len(role_fpr_df))
        ax.barh(y - 0.2, role_fpr_df["Population (RF)"], height=0.4, color=COLOR_RED, label="Population (RF)")
        ax.barh(y + 0.2, role_fpr_df["Per-User Framework"], height=0.4, color=COLOR_BLUE, label="Per-User Framework")
        ax.set_yticks(y)
        ax.set_yticklabels(role_fpr_df.index)
        ax.set_xlabel("False Positive Rate")
        ax.legend(facecolor="#0d1117", labelcolor="white", fontsize=8)
        ax.grid(axis="x", color="#1e2d45", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown('<p class="section-header">Computational Cost</p>', unsafe_allow_html=True)
        cost_pop = measure_computational_cost(
            fit_fn=lambda: population, score_fn=lambda m: m.score(sample))
        cost_per = measure_computational_cost(
            fit_fn=lambda: peruser, score_fn=lambda m: m.score(sample))
        cost_df = pd.DataFrame({"Population (RF)": cost_pop, "Per-User Framework": cost_per})
        st.dataframe(cost_df, width="stretch")
        st.caption("Fit time above reflects re-use of the already-trained model (0s); see train.py console "
                   "output for original fit-from-scratch timings.")

        with st.expander("Raw metrics JSON"):
            st.json({"population": pop_metrics, "peruser": per_metrics})


def page_about(meta: dict) -> None:
    show_banner()
    st.markdown(f"""
    ### About This System

    This prototype implements the **Per-User Behavioural Baseline Framework** proposed in the
    COS700 research report *"A Per-User Behavioural Baseline Framework for Insider Threat
    Detection in Resource-Constrained Environments"* (Selepe Sello, supervised by Prof. Jan Eloff,
    University of Pretoria, 2026).

    | Artifact info | Value |
    |---|---|
    | **User-days indexed** | {meta.get('n_rows', '?'):,} |
    | **Users** | {meta.get('n_users', '?')} |
    | **Anomaly Scorer method** | {meta.get('anomaly_method', '?')} |
    | **SHAP explainer** | {"Yes" if meta.get("shap_available") else "No"} |
    | **Last trained** | {meta.get('trained_at', 'unknown')} |

    #### Architecture

    | Component | Responsibility |
    |-----------|---------------|
    | **`train.py`** | Caching pipeline: feature engineering → population RF → per-user framework → SHAP → manifest |
    | **`app.py`** | Streamlit UI: risk explorer, method comparison, evaluation |
    | **`src/preprocess.py`** | Raw CERT-schema log ingestion, per-(user, day) feature engineering, time-based split |
    | **`src/baseline.py`** | `BehaviouralProfiler` (Profile Builder + Role-Aware Normaliser), `PerUserBaselineEngine`, `PopulationBaseline` |
    | **`src/explain.py`** | SHAP TreeExplainer wrapper for the population RF baseline |
    | **`src/evaluate.py`** | Recall, subgroup FPR, F1, ROC-AUC, computational cost |

    #### Data

    Built against the **official CERT Insider Threat Test Dataset r6.2**
    (`logon.csv`, `device.csv`, `file.csv`, `http.csv`, `email.csv`, monthly LDAP
    org-structure snapshots). The real dataset must be downloaded manually from
    [kilthub.cmu.edu](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
    (Lindauer, 2020) and placed in `data/` — see `data/README.md`. `script/dataset.py`
    then verifies, extracts, and converts the official answer key automatically.
    If the real archives aren't present, it falls back to a **schema-identical synthetic
    dataset** with two injected attacker archetypes (`burst`, `low_slow`) and a legitimate
    high-privilege cohort (SysAdmin/DBAdmin), so the full pipeline can be developed and
    demonstrated without the official multi-gigabyte download.

    #### Forensic / Security Relevance

    - Role-aware normalisation targets the false-positive problem for high-privilege users
      (database administrators, system operators) described in the proposal's Problem Statement.
    - The Anomaly Scorer operates on *deviation* vectors (not raw magnitudes), so a single
      cheaply-trained model implicitly becomes per-user-aware — this is what keeps the
      framework's fit/inference cost low enough for consumer hardware.
    - SHAP explanations (population model) and top-deviating-feature explanations
      (per-user model) give an analyst a reason for every flagged day, not just a score.
    """)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("Insider Threat Detection")
st.sidebar.markdown("**COS700 — Per-User Behavioural Baseline Framework**")
st.sidebar.markdown("University of Pretoria, 2026")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "Overview", "Risk Explorer", "Method Comparison", "Evaluation", "About",
])

if not _artifacts_exist():
    show_not_trained_error()

daily_df, gt_df, population, peruser, shap_explainer = load_artifacts()
meta = _load_meta()

if page == "Overview":
    page_overview(daily_df, meta)
elif page == "Risk Explorer":
    page_risk_explorer(daily_df, population, peruser, shap_explainer)
elif page == "Method Comparison":
    page_compare(daily_df, population, peruser)
elif page == "Evaluation":
    page_evaluate(daily_df, gt_df, population, peruser)
elif page == "About":
    page_about(meta)
