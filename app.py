##
# @file app.py
# @brief Streamlit UI, Per-User Behavioural Baseline Framework for Insider Threat Detection.
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
import random
from urllib.parse import quote

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

st.set_page_config(page_title="Insider Threat, Per-User Baseline", layout="wide")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    body, .stApp {{ font-family: 'Inter', sans-serif; }}
    code, .mono  {{ font-family: 'JetBrains Mono', monospace; }}
    h1, h2, h3   {{ letter-spacing: -0.015em; }}

    .section-header {{
        color: #4a7cc7; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; margin: 28px 0 12px 0;
    }}
    .info-box, .warning-box, .danger-box, .success-box {{
        background-color: #161b22; border: 1px solid #1e2d45; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 14px; line-height: 1.6;
    }}
    .warning-box {{ border-color: {COLOR_AMBER}; }}
    .danger-box  {{ border-color: {COLOR_RED}; }}
    .success-box {{ border-color: {COLOR_GREEN}; }}
    div[data-testid="stMetric"] {{
        background-color: #161b22; border: 1px solid #1e2d45; border-radius: 12px; padding: 16px 20px;
    }}
    div[data-testid="stMetricLabel"] {{ color: #8b949e; }}
    div[data-testid="stMetricValue"] {{ color: #e6edf3; font-weight: 700; }}

    .result-card {{
        border: 1px solid #1e2d45; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 10px; background-color: #161b22;
    }}
    .result-rank {{
        color: #e6edf3; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700;
    }}
    .cat-tag {{
        display: inline-block; background: #1c2330; color: #4a7cc7; border: 1px solid #2a3a5a;
        font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 6px; margin-left: 8px;
    }}
    .hp-tag {{
        display: inline-block; border: 1px solid {COLOR_AMBER}; color: {COLOR_AMBER};
        font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 6px; margin-left: 6px;
    }}
    .mal-tag {{
        display: inline-block; background: {COLOR_AMBER}; color: #1a1203;
        font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-left: 6px;
    }}
    .banner-wrap {{
        background: #161b22; border: 1px solid #1e2d45; border-radius: 12px;
        padding: 28px 32px 24px 32px; margin-bottom: 24px; overflow: hidden;
    }}
    .banner-title {{ color: #e6edf3; font-size: 1.6rem; font-weight: 700; letter-spacing: -0.015em; margin: 0 0 4px 0; }}
    .banner-subtitle {{ color: #8b949e; font-size: 0.9rem; margin: 0; }}
    .banner-tagline {{
        font-size: 0.78rem; font-weight: 600; color: #4a7cc7;
        margin: 8px 0 14px 0; letter-spacing: 0.12em; text-transform: uppercase;
    }}
    .banner-divider {{ border: none; border-top: 1px solid #1e2d45; margin: 14px 0; }}
    .banner-badge {{
        display: inline-block; margin-top: 14px; padding: 3px 10px; border: 1px solid #2a3a5a;
        border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        font-weight: 500; color: #8b949e;
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
    # No UI in here: Streamlit replays elements created inside a cached
    # function on every rerun, which left a "Loading…" box on every page.
    import joblib
    daily_df   = joblib.load(PATHS["daily_df"])
    gt_df      = joblib.load(PATHS["gt_df"])
    population = joblib.load(PATHS["population"])
    peruser    = joblib.load(PATHS["peruser"])
    shap_explainer = joblib.load(PATHS["shap"]) if os.path.exists(PATHS["shap"]) else None
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
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
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
            <b>1. Profile Builder</b><br>
            Per-user historical mean/std computed from each user's own training-window
            activity. Users with insufficient history fall back to their role-group
            baseline (cold-start handling).
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="info-box">
            <b>2. Role-Aware Normaliser</b><br>
            Features are z-scored relative to the user's own role-peer group,
            not the whole organisation, a database administrator's bulk file access
            is only "unusual" if it's unusual <i>for a database administrator</i>.
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div class="info-box">
            <b>3. Anomaly Scorer</b><br>
            An unsupervised outlier model ({meta.get('anomaly_method','iforest').upper()})
            scores each user-day's <i>deviation vector</i>, so one cheaply-trained
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
    ax.grid(axis="x", color="#1e2d45", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


@st.cache_data(show_spinner=False)
def budget_flags(_population, _peruser, _test_df, n_rows: int, budget: float = 0.001) -> dict:
    """
    @brief Which test user-days each model alerts on at the alert budget.

    The same tie-broken selection the Evaluation page and the experiments
    use, so the counts shown per user add up to the evaluation's numbers.
    `n_rows` is the cache key; the leading-underscore arguments are not hashed.

    @return  {"population"|"peruser": (flag mask, display-score cut)}.
    """
    from src.evaluate import flags_for_budget
    out = {}
    for name, model in (("population", _population), ("peruser", _peruser)):
        ranking = (model.score_array(_test_df) if name == "population"
                   else model.raw_score(_test_df))
        flags, _ = flags_for_budget(ranking, budget)
        display = model.score_array(_test_df)
        out[name] = (flags, float(display[flags].min()))
    return out


def page_risk_explorer(daily_df, population, peruser, shap_explainer) -> None:
    """@brief Per-user risk timeline: population score vs per-user score, with explanations."""
    st.subheader("Per-User Risk Explorer")

    users = sorted(daily_df["user"].unique())
    col_u, col_m = st.columns([2, 1])
    with col_u:
        # ?user=CMP2946 preselects a user, so a specific case is one link away.
        wanted = st.query_params.get("user")
        user = st.selectbox("Select a user", users,
                            index=users.index(wanted) if wanted in users else 0)
    with col_m:
        method = st.radio("Score with", ["Both", "Population (RF)", "Per-User Framework"], horizontal=False)
        # Training days are scored in-sample, so the framework's percentile
        # score spans the whole range there by construction; the held-out
        # window is where a departure from the user's normal is visible.
        test_only = st.checkbox("Test window only", value=True)

    user_df = daily_df[daily_df["user"] == user].sort_values("date").reset_index(drop=True)
    if test_only and (user_df["split"] == "test").any():
        user_df = user_df[user_df["split"] == "test"].reset_index(drop=True)
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
    per_objs = peruser.score(user_df) if method in ("Both", "Per-User Framework") else None
    per_res = results_to_frame(per_objs) if per_objs is not None else None

    fig, ax = _dark_fig(11, 3.2)
    if pop_res is not None:
        ax.plot(pop_res["date"], pop_res["score"], color=COLOR_RED, label="Population (RF)", linewidth=1.6)
    if per_res is not None:
        ax.plot(per_res["date"], per_res["score"], color=COLOR_BLUE, label="Per-User Framework", linewidth=1.6)
    mal_days = user_df.loc[user_df["is_malicious"], "date"]
    for d in mal_days:
        ax.axvspan(d, d + pd.Timedelta(days=1), color=COLOR_AMBER, alpha=0.15)

    # Each model's alert line at the 0.1% budget, and how many of this
    # user's test days fall above it -- the scores themselves are on
    # different scales, the alerts are not.
    test_df = daily_df[daily_df["split"] == "test"].reset_index(drop=True)
    flags = budget_flags(population, peruser, test_df, len(test_df))
    in_user = (test_df["user"] == user).to_numpy()
    counts = {}
    for name, res, colour in (("population", pop_res, COLOR_RED), ("peruser", per_res, COLOR_BLUE)):
        mask, cut = flags[name]
        counts[name] = int(mask[in_user].sum())
        if res is not None:
            ax.axhline(cut, color=colour, linewidth=0.9, linestyle=":")
    ax.set_ylabel("Risk score")
    ax.set_title(f"Risk score over time, {user}", fontsize=10)
    ax.legend(facecolor="#0d1117", labelcolor="white", fontsize=8)
    ax.grid(color="#1e2d45", linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    n_mal_test = int(test_df.loc[in_user, "is_malicious"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts on this user, Population (RF)", counts["population"])
    c2.metric("Alerts on this user, Per-User Framework", counts["peruser"])
    c3.metric("Malicious test days (answer key)", n_mal_test)
    st.caption("Dotted lines mark each model's alert line: the top 0.1% of all "
               f"{len(test_df):,} test user-days, the budget used throughout the evaluation.")

    st.markdown('<p class="section-header">Highest-Risk Days</p>', unsafe_allow_html=True)
    display_res = per_res if per_res is not None else pop_res
    top = display_res.sort_values("score", ascending=False).head(8).copy()

    # For rows scored by the population RF, replace the generic global
    # feature-importance explanation with a real per-row SHAP explanation.
    if display_res is pop_res and shap_explainer is not None:
        exp_rows = user_df.set_index("date").loc[top["date"]].reset_index()
        shap_expl = explain_rows(shap_explainer, exp_rows)
        top["top_features"] = [
            "; ".join(f"{f} ({v:+.3f})" for f, v in row) or "none" for row in shap_expl
        ]

    # For the per-user framework, say what each deviation is relative to: the
    # value today against this person's own typical and 90th-percentile day.
    # A bare z-score tells an analyst *that* a day is unusual; this tells them
    # what to go and look at.
    if display_res is per_res:
        profile = peruser.profiler.profile_for(user, role)
        by_date = {pd.Timestamp(r.date): r for r in per_objs}
        values  = user_df.set_index("date")
        lines = []
        for d in top["date"]:
            parts = []
            for name, z in by_date[pd.Timestamp(d)].top_features:
                feat = name.split("__")[0]
                parts.append(
                    f"<b>{feat}</b> = {values.loc[d, feat]:,.0f} "
                    f"(usually {profile.loc[feat, 'p50']:,.0f}, 90th pct "
                    f"{profile.loc[feat, 'p90']:,.0f}; {z:+.1f}σ)")
            lines.append("<br>".join(parts) or "none")
        top["top_features"] = lines

    for _, row in top.iterrows():
        mal_flag = ' <span class="mal-tag">MALICIOUS DAY</span>' if row["is_malicious"] else ""
        st.markdown(f"""
        <div class="result-card">
            <span class="result-rank">{row['date'].strftime('%Y-%m-%d')}</span>
            <span class="cat-tag">{row['method']}</span>
            <span style="color:#8b949e;font-size:0.8rem;">
                &nbsp;|&nbsp; Baseline: {row['baseline_used']}
                &nbsp;|&nbsp; Score: <b style="color:{COLOR_BLUE}">{row['score']:.3f}</b>
            </span>{mal_flag}
            <div style="font-family:monospace;font-size:0.8rem;color:#cdd;margin-top:6px;">
                Top contributing features: {row['top_features'] if row['top_features'] else 'none'}
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_compare(daily_df, population, peruser) -> None:
    """@brief Who each model spends its alert budget on, role by role."""
    st.subheader("Method Comparison. Who Gets the Alerts?")
    st.markdown(
        '<div class="info-box">The two models\' scores are in different units (a '
        'probability and an anomaly score), so they are compared where it matters: '
        'given the same alert budget, <b>whose activity does each model put in front '
        'of an analyst?</b> A role that receives far more alerts than its share of '
        'user-days is being flagged for what it normally does.</div>',
        unsafe_allow_html=True,
    )

    test_df = daily_df[daily_df["split"] == "test"].reset_index(drop=True)
    budget = st.select_slider(
        "Alert budget", options=[0.0005, 0.001, 0.005, 0.01], value=0.001,
        format_func=lambda b: f"{b * 100:.2f}%  ({max(1, round(len(test_df) * b)):,} alerts)")
    flags = budget_flags(population, peruser, test_df, len(test_df), budget)
    pop_f, per_f = flags["population"][0], flags["peruser"][0]
    hp = test_df["is_high_privilege"].astype(bool).to_numpy()
    mal = test_df["is_malicious"].astype(bool).to_numpy()
    k = int(pop_f.sum())

    c = st.columns(4)
    c[0].metric("High-privilege share of user-days", f"{hp.mean() * 100:.1f}%")
    c[1].metric("High-priv alerts, RF",
                f"{int((pop_f & hp).sum())} of {k}", f"{(pop_f & hp).sum() / k * 100:.0f}%",
                delta_color="off")
    c[2].metric("High-priv alerts, Per-User",
                f"{int((per_f & hp).sum())} of {k}", f"{(per_f & hp).sum() / k * 100:.0f}%",
                delta_color="off")
    c[3].metric("Malicious days caught (RF → Per-User)",
                f"{int((pop_f & mal).sum())} → {int((per_f & mal).sum())}")

    st.markdown('<p class="section-header">Alerts by Role</p>', unsafe_allow_html=True)
    roles = test_df["role"].astype(str).to_numpy()
    frame = pd.DataFrame({
        "Population (RF)":    pd.Series(roles[pop_f]).value_counts(),
        "Per-User Framework": pd.Series(roles[per_f]).value_counts(),
    }).fillna(0)
    frame["Fair share"] = pd.Series(roles).value_counts(normalize=True).reindex(frame.index) * k
    frame = frame.loc[frame[["Population (RF)", "Per-User Framework"]].max(axis=1)
                      .sort_values(ascending=False).index[:10]].iloc[::-1]

    fig, ax = _dark_fig(9, max(3, len(frame) * 0.48))
    y = np.arange(len(frame))
    ax.barh(y + 0.2, frame["Population (RF)"], height=0.38, color=COLOR_RED, label="Population (RF)")
    ax.barh(y - 0.2, frame["Per-User Framework"], height=0.38, color=COLOR_BLUE, label="Per-User Framework")
    ax.scatter(frame["Fair share"], y, marker="|", s=260, color="white", zorder=3,
               label="Alerts expected from the role's share of user-days")
    ax.set_yticks(y)
    ax.set_yticklabels(frame.index)
    ax.set_xlabel(f"Alerts at the {budget * 100:.2f}% budget")
    ax.legend(facecolor="#0d1117", labelcolor="white", fontsize=8, loc="lower right")
    ax.grid(axis="x", color="#1e2d45", linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown('<p class="section-header">High-Privilege Users Each Model Flagged</p>',
                unsafe_allow_html=True)
    users = test_df["user"].astype(str).to_numpy()
    hp_table = pd.DataFrame({
        "Population (RF) alerts":    pd.Series(users[pop_f & hp]).value_counts(),
        "Per-User Framework alerts": pd.Series(users[per_f & hp]).value_counts(),
    }).fillna(0).astype(int).sort_values("Population (RF) alerts", ascending=False)
    hp_table["Malicious test days"] = [int(mal[users == u].sum()) for u in hp_table.index]
    if len(hp_table):
        st.dataframe(hp_table, width="stretch")
        st.caption("Pick any of these users in the Risk Explorer to see why.")
    else:
        st.info("Neither model flagged a high-privilege user at this budget.")


def page_evaluate(daily_df, gt_df, population, peruser) -> None:
    """@brief Population baseline vs per-user framework on the full test split."""
    st.subheader("Evaluation. Population Baseline vs Per-User Framework")
    st.markdown(
        '<div class="info-box">Evaluated on the <b>entire held-out time-based test '
        'split</b>, no leakage (both models are fit on the training window only) '
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
            "base rate, how many times better than spending the same budget at "
            "random. `tie_fraction` is the share of alerts drawn arbitrarily "
            "from equal scores; a high value means the ranking ran out."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Recall, Population", f"{pop_metrics['recall']:.3f}")
        m2.metric("Recall, Per-User", f"{per_metrics['recall']:.3f}",
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
                "split measures low-and-slow detection only, and the supervised "
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
            ax.grid(axis="x", color="#1e2d45", linewidth=0.5)
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


RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "report", "figures")


def _figure(name: str, caption: str) -> None:
    """@brief Show one of the report's figures, or say how to build it."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not built yet, run `python report/make_figures.py`.")


def page_findings() -> None:
    """@brief The research findings, read from results/ rather than recomputed."""
    st.subheader("Research Findings. Full CERT r6.2 Population")
    st.markdown(
        '<div class="info-box">Everything on this page is read from the experiment '
        'outputs in <code>results/</code>, produced on the <b>full 4,000-user dataset</b> '
        '(403,245 test user-days, 38 malicious, 2 insiders) at an alert budget of '
        '<b>0.1%</b>, 403 alerts over five months. The other pages run on a 400-user '
        'cohort so they stay interactive.</div>', unsafe_allow_html=True)

    core_path = os.path.join(RESULTS_DIR, "core", "comparison.csv")
    if not os.path.exists(core_path):
        st.warning("No results yet, run `python script/experiment.py --preset core`.")
        return
    core = pd.read_csv(core_path).set_index("detector")
    fw, rf = core.loc["peruser-ocsvm-personal-zscore"], core.loc["rf"]

    st.markdown('<p class="section-header">Framework vs the proposal\'s Random Forest baseline</p>',
                unsafe_allow_html=True)
    c = st.columns(4)
    n_mal = int(fw["n_malicious_rows"])
    c[0].metric("Malicious days found", f"{round(fw['recall'] * n_mal)} / {n_mal}",
                f"RF: {round(rf['recall'] * n_mal)} / {n_mal}")
    c[1].metric("Insiders surfaced", f"{int(fw['n_insiders_detected'])} / {int(fw['n_insiders'])}",
                f"RF: {int(rf['n_insiders_detected'])} / {int(rf['n_insiders'])}")
    c[2].metric("Alerts on high-privilege users", f"{fw['alert_share_high_privilege'] * 100:.1f}%",
                f"RF: {rf['alert_share_high_privilege'] * 100:.1f}%", delta_color="off")
    c[3].metric("Training time", f"{fw['fit_seconds']:.1f} s",
                f"RF: {rf['fit_seconds']:.0f} s", delta_color="off")

    tabs = st.tabs(["SQ1 Low-and-slow", "SQ2 High-privilege", "Robustness",
                    "SQ3 History", "SQ5 Explanations"])
    with tabs[0]:
        _figure("timeline", "The two test-window insiders, scored against their own "
                "history (blue) and by the same One-Class SVM on population-scaled "
                "features (green). Only the reference frame differs.")
        _figure("budget_sweep", "Recall at every alert budget.")
    with tabs[1]:
        _figure("alert_share", "Share of each detector's 403 alerts spent on IT "
                "administrators, who are 2.4% of the test user-days.")
    with tabs[2]:
        path = os.path.join(RESULTS_DIR, "studies", "seeds", "tests.csv")
        _figure("seeds", "Ten seeds per detector; the black tick is the mean.")
        if os.path.exists(path):
            tests = pd.read_csv(path)
            st.dataframe(tests[["comparator", "metric", "reference_mean", "comparator_mean",
                                "wins", "losses", "p_holm", "significant"]],
                         width="stretch", hide_index=True)
            st.caption("Paired Wilcoxon signed-rank tests of the framework against each "
                       "detector across seeds, Holm-Bonferroni corrected.")
    with tabs[3]:
        _figure("history", "Detection and stability against days of personal history.")
    with tabs[4]:
        _figure("explain", "How often each system's explanation names a feature the "
                "insider actually used, scored against the official answer key.")


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
    | **`app.py`** | Streamlit UI: risk explorer, method comparison, evaluation, findings |
    | **`src/preprocess.py`** | Raw CERT-schema log ingestion, per-(user, day) feature engineering, time-based split |
    | **`src/baseline.py`** | `BehaviouralProfiler` (Profile Builder + Role-Aware Normaliser), `PerUserBaselineEngine`, `PopulationBaseline` |
    | **`src/features.py`** | Causal trailing-window aggregates (no forward leakage) |
    | **`src/detectors.py`** | One fit/score interface over all comparison detectors |
    | **`src/experiment.py`** | The controlled comparison + report rendering |
    | **`src/explain.py`** | SHAP TreeExplainer wrapper for the population RF baseline |
    | **`src/evaluate.py`** | Alert-budget operating points, subgroup FPR, archetype recall, queue depth and detection delay, cost |
    | **`src/studies.py`** | Follow-up studies: seed robustness, history length (SQ3), explanation fidelity (SQ5) |

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
    (Lindauer, 2020) and placed in `data/`. See `data/README.md`. `script/dataset.py`
    then verifies, extracts, and converts the official answer key automatically.
    If the real archives aren't present, it falls back to a **schema-identical synthetic
    dataset** with two injected attacker archetypes (`burst`, `low_slow`) and a legitimate
    high-privilege cohort (SysAdmin/DBAdmin), so the full pipeline can be developed and
    demonstrated without the official multi-gigabyte download.

    #### Forensic / Security Relevance

    - Role-aware normalisation targets the false-positive problem for high-privilege users
      (database administrators, system operators) described in the proposal's Problem Statement.
    - The Anomaly Scorer operates on *deviation* vectors (not raw magnitudes), so a single
      cheaply-trained model implicitly becomes per-user-aware. This is what keeps the
      framework's fit/inference cost low enough for consumer hardware.
    - SHAP explanations (population model) and top-deviating-feature explanations
      (per-user model) give an analyst a reason for every flagged day, not just a score.
    """)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = ["Overview", "Risk Explorer", "Method Comparison", "Evaluation", "Findings", "About"]

# Lucide icons (lucide.dev, ISC licence), one per page in PAGES order.
_ICONS = {
    "shield":            '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "Overview":          '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "Risk Explorer":     '<circle cx="10" cy="7" r="4"/><path d="M10.3 15H7a4 4 0 0 0-4 4v2"/><circle cx="17" cy="17" r="3"/><path d="m21 21-1.9-1.9"/>',
    "Method Comparison": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "Evaluation":        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "Findings":          '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" x2="4" y1="22" y2="15"/>',
    "About":             '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
}


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml," + quote(svg)


def _icon_uri(name: str) -> str:
    return _svg_uri('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
                    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + _ICONS[name] + "</svg>")


@st.cache_data
def _network_background() -> str:
    """
    @brief A still frame of the presentation's background: employees as a faint
           network, three of them flagged in amber. Seeded, so it never changes.
    """
    rng = random.Random(7)
    pts = [(rng.uniform(0, 1920), rng.uniform(0, 1080), rng.uniform(.4, 1)) for _ in range(110)]
    parts = []
    for i, (x1, y1, z1) in enumerate(pts):
        for x2, y2, z2 in pts[i + 1:]:
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5
            if d < 173:
                parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                             f'stroke="#4A7CC7" stroke-opacity="{(1 - d / 173) * .08 * min(z1, z2):.3f}"/>')
    for x, y, z in pts:
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{1.3 + z * 1.2:.1f}" fill="#4A7CC7" fill-opacity="{.28 * z:.2f}"/>')
    for x, y, _ in rng.sample(pts, 3):
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3" fill="#F59E0B" fill-opacity=".55"/>'
                     f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" fill="none" stroke="#F59E0B" stroke-opacity=".18"/>')
    return _svg_uri('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080" '
                    'preserveAspectRatio="xMidYMid slice">' + "".join(parts) + "</svg>")


# Streamlit renders the page list as a radio group; these rules restyle it as a
# navigation menu with one icon per page, matched to PAGES by position.
st.markdown(f"""
<style>
    .stApp {{ background-image: url("{_network_background()}"); background-size: cover; background-attachment: fixed; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    .nav-brand {{ display: flex; align-items: center; gap: 10px; color: #e6edf3; font-size: 1.05rem; font-weight: 650; }}
    .nav-brand i {{ width: 22px; height: 22px; background: #4a7cc7;
                    -webkit-mask: url("{_icon_uri('shield')}") center / contain no-repeat; mask: url("{_icon_uri('shield')}") center / contain no-repeat; }}
    .nav-meta {{ color: #8b949e; font-size: 0.85rem; margin: 6px 0 22px 32px; }}
    .nav-eyebrow {{ color: #4a7cc7; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
                    text-transform: uppercase; margin: 0 0 8px 2px; }}

    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; width: 100%; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {{
        width: 100%; padding: 9px 12px; border-radius: 8px; border: 1px solid transparent; margin: 0;
    }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: #1c2330; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{
        background: #1c2330; border-color: #1565c0;
    }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {{
        width: 18px; height: 18px; margin-right: 10px; border: none; border-radius: 0; background: #8b949e;
    }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child > div {{ display: none; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {{ background: #4a7cc7; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: 0.95rem; color: #8b949e; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: #e6edf3; font-weight: 600; }}
    {"".join(
        f'[data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type({i}) > div > div > div:first-child '
        f'{{ -webkit-mask: url("{_icon_uri(name)}") center / contain no-repeat; mask: url("{_icon_uri(name)}") center / contain no-repeat; }}'
        for i, name in enumerate(PAGES, 1))}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<div class="nav-brand"><i></i>Insider Threat Detection</div>'
    '<div class="nav-meta">COS700, University of Pretoria</div>'
    '<div class="nav-eyebrow">Navigation</div>',
    unsafe_allow_html=True,
)
# ?page=Findings opens a page directly, which makes a page linkable.
_wanted = st.query_params.get("page", PAGES[0])
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed",
                        index=PAGES.index(_wanted) if _wanted in PAGES else 0)

if not _artifacts_exist():
    show_not_trained_error()

with st.spinner("Loading trained artifacts…"):
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
elif page == "Findings":
    page_findings()
elif page == "About":
    page_about(meta)
