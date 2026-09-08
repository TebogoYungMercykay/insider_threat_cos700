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
    all_roles = sorted(daily_df["role"].unique())
    # Default to the roles the comparison is actually about: the high-privilege
    # group plus the largest ordinary one. Hard-coded names would silently
    # produce an empty selection, since r6.2's LDAP writes "ITAdmin" rather
    # than the synthetic generator's "SysAdmin"/"DBAdmin".
    hp_roles = sorted(daily_df.loc[daily_df["is_high_privilege"], "role"].unique())
    common   = daily_df["role"].value_counts().index.tolist()
    default_roles = (hp_roles + [r for r in common if r not in hp_roles][:2])[:4]
    role_filter = st.multiselect("Filter by role", all_roles, default=default_roles)
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
    """@brief Population baseline vs per-user framework on the full test split."""
    st.subheader("Evaluation — Population Baseline vs Per-User Framework")
    st.markdown(
        '<div class="info-box">Evaluated on the <b>entire held-out time-based test '
        'split</b> — no leakage (both models are fit on the training window only) '
        'and no sampling. Sampling would be fatal here: the malicious class is '
        'roughly one user-day in thirty thousand, so a few thousand sampled rows '
        'contain no positives at all and every metric computed on them is noise. '
        'Detectors are compared at a shared <b>alert budget</b> rather than at a '
        'shared threshold, because a Random Forest probability and an anomaly '
        'score are not the same unit.</div>',
        unsafe_allow_html=True,
    )

    test_df = daily_df[daily_df["split"] == "test"].reset_index(drop=True)

    col_b, col_i = st.columns(2)
    budget = col_b.select_slider(
        "Alert budget (share of test rows an analyst team can investigate)",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        value=0.001, format_func=lambda b: f"{b * 100:.2f}%")
    col_i.metric("Alerts this buys", f"{max(1, round(len(test_df) * budget)):,}",
                 f"of {len(test_df):,} test rows")

    if st.button("Run Evaluation", type="primary"):
        from src.evaluate import (
            evaluate_recall_by_archetype, evaluate_scores, flags_for_budget,
            fpr_by_role, measure_computational_cost,
        )

        with st.status(f"Scoring all {len(test_df):,} test rows…", expanded=True) as status:
            status.write("Scoring the population (RF) baseline…")
            pop_scores = population.score_array(test_df)
            status.write("Scoring the per-user framework…")
            per_scores = peruser.raw_score(test_df)
            status.write("Computing metrics at the shared alert budget…")
            pop_metrics = evaluate_scores(test_df, pop_scores, budget=budget)
            per_metrics = evaluate_scores(test_df, per_scores, budget=budget)
            pop_flags, _ = flags_for_budget(pop_scores, budget)
            per_flags, _ = flags_for_budget(per_scores, budget)
            status.update(label="Evaluation complete.", state="complete", expanded=False)

        st.markdown('<p class="section-header">Primary Metrics</p>', unsafe_allow_html=True)
        rows = ["roc_auc", "avg_precision", "recall", "precision", "f1",
                "lift", "insider_recall", "n_insiders_detected",
                "fpr_overall", "fpr_high_privilege", "alert_share_high_privilege",
                "n_flagged", "tie_fraction"]
        st.dataframe(pd.DataFrame({
            "Population (RF)":    {k: pop_metrics[k] for k in rows},
            "Per-User Framework": {k: per_metrics[k] for k in rows},
        }), width="stretch")
        st.caption(
            "`roc_auc` and `avg_precision` are threshold-free; everything else is "
            "measured at the alert budget above. `lift` is precision over the "
            "base rate — how many times better than spending the same budget at "
            "random. `tie_fraction` is the share of alerts drawn arbitrarily "
            "from equal scores; a high value means the ranking ran out."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Recall — Population", f"{pop_metrics['recall']:.3f}")
        m2.metric("Recall — Per-User", f"{per_metrics['recall']:.3f}",
                  f"{per_metrics['recall'] - pop_metrics['recall']:+.3f}")
        m3.metric("FPR (high-privilege) Δ",
                  f"{per_metrics['fpr_high_privilege'] - pop_metrics['fpr_high_privilege']:+.4f}",
                  help="Negative is better: the per-user framework spends fewer "
                       "of its alerts on high-privilege users doing their job.")

        st.markdown('<p class="section-header">Recall by Attacker Archetype</p>',
                    unsafe_allow_html=True)
        arch_pop = evaluate_recall_by_archetype(test_df, pop_flags, gt_df)
        arch_per = evaluate_recall_by_archetype(test_df, per_flags, gt_df)
        if arch_pop or arch_per:
            st.dataframe(pd.DataFrame({
                "Population (RF)":    {k: v["recall"] for k, v in arch_pop.items()},
                "Per-User Framework": {k: v["recall"] for k, v in arch_per.items()},
            }).fillna(0), width="stretch")
            st.caption(
                "Archetypes come from the answer key's own window geometry: an "
                "insider whose malicious activity spans more than three weeks is "
                "low-and-slow. On r6.2 every burst insider acts before the split "
                "date and both low-and-slow insiders act after it, so the test "
                "split measures low-and-slow detection only — and the supervised "
                "baseline is being asked to generalise from burst training "
                "examples to low-and-slow test ones."
            )
        else:
            st.info("No labelled malicious days fall in the test split.")

        st.markdown('<p class="section-header">False Positive Rate by Role</p>',
                    unsafe_allow_html=True)
        role_fpr_df = pd.DataFrame({
            "Population (RF)":    pd.Series(fpr_by_role(test_df, pop_flags)),
            "Per-User Framework": pd.Series(fpr_by_role(test_df, per_flags)),
        }).fillna(0).sort_values("Population (RF)")
        role_fpr_df = role_fpr_df[role_fpr_df.sum(axis=1) > 0].tail(20)
        if len(role_fpr_df):
            fig, ax = _dark_fig(9, max(3, len(role_fpr_df) * 0.4))
            y = np.arange(len(role_fpr_df))
            ax.barh(y - 0.2, role_fpr_df["Population (RF)"], height=0.4,
                    color=COLOR_RED, label="Population (RF)")
            ax.barh(y + 0.2, role_fpr_df["Per-User Framework"], height=0.4,
                    color=COLOR_BLUE, label="Per-User Framework")
            ax.set_yticks(y)
            ax.set_yticklabels(role_fpr_df.index)
            ax.set_xlabel("False Positive Rate")
            ax.legend(facecolor="#0d1117", labelcolor="white", fontsize=8)
            ax.grid(axis="x", color="#1e2d45", linewidth=0.5, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        else:
            st.info("No role raised a false positive at this alert budget.")

        st.markdown('<p class="section-header">Computational Cost</p>',
                    unsafe_allow_html=True)
        cost = pd.DataFrame({
            "Population (RF)": measure_computational_cost(
                lambda: population, lambda m: m.score_array(test_df)),
            "Per-User Framework": measure_computational_cost(
                lambda: peruser, lambda m: m.raw_score(test_df)),
        })
        st.dataframe(cost, width="stretch")
        st.caption("Fit time here is zero because both models are already trained; "
                   "see the train.py console output for fit-from-scratch timings, "
                   "or run `python script/experiment.py` for measured fit cost.")

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
    | **Normalisation scope** | {meta.get('scope', 'personal')} |
    | **Deviation statistic** | {meta.get('statistic', 'zscore')} |
    | **SHAP explainer** | {"Yes" if meta.get("shap_available") else "No"} |
    | **Last trained** | {meta.get('trained_at', 'unknown')} |

    #### Architecture

    | Component | Responsibility |
    |-----------|---------------|
    | **`train.py`** | Caching pipeline: feature engineering → population RF → per-user framework → SHAP → manifest |
    | **`app.py`** | Streamlit UI: risk explorer, method comparison, evaluation |
    | **`src/preprocess.py`** | Raw CERT-schema log ingestion, per-(user, day) feature engineering, time-based split |
    | **`src/baseline.py`** | `BehaviouralProfiler` (Profile Builder + Role-Aware Normaliser), `PerUserBaselineEngine`, `PopulationBaseline` |
    | **`src/features.py`** | Causal trailing-window aggregates (no forward leakage) |
    | **`src/detectors.py`** | One fit/score interface over all comparison detectors |
    | **`src/experiment.py`** | The controlled comparison + report rendering |
    | **`src/explain.py`** | SHAP TreeExplainer wrapper for the population RF baseline |
    | **`src/evaluate.py`** | Alert-budget operating points, subgroup FPR, archetype recall, cost |

    #### Running the full comparison

    This dashboard shows the two headline models. The complete comparison
    against Noever's four population classifiers, the population-level anomaly
    controls and the framework's own ablation is run from the command line:

    ```
    python script/experiment.py --preset full
    ```

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
