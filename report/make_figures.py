#!/usr/bin/env python3
##
# @file report/make_figures.py
# @brief Build every data figure in the report from results/.
#
# Each figure reads the CSVs an experiment wrote, so a number in a figure can
# always be traced to the run that produced it. The one exception is the risk
# timeline, which needs per-day scores the experiment does not persist; it
# fits the two detectors it compares (a few seconds each) and caches the
# scores in figures/data/.
#
# Usage (from the repository root):
#   python report/make_figures.py            # every figure whose inputs exist
#   python report/make_figures.py timeline   # one figure
##

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.join(ROOT, "results")
STUDIES = os.path.join(RESULTS, "studies")
OUT     = os.path.join(HERE, "figures")
CACHE   = os.path.join(OUT, "data")

# Reference categorical palette, light steps, in its fixed order. Validated
# for adjacent CVD separation; aqua and yellow sit under 3:1 on the surface,
# so every series that uses them is direct-labelled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

# Colour follows the detector, in every figure.
DETECTOR_STYLE = {
    "peruser-ocsvm-personal-zscore": ("Framework (personal, OCSVM)", BLUE),
    "rf":                            ("Random Forest (proposal baseline)", ORANGE),
    "pop-ocsvm":                     ("Population OCSVM (control)", AQUA),
    "pop-iforest":                   ("Population iForest (control)", YELLOW),
    "volume":                        ("Volume floor", MUTED),
}

FAMILY_COLOUR = {
    "per-user-framework":      BLUE,
    "population-supervised":   ORANGE,
    "population-unsupervised": ORANGE,
    "floor":                   MUTED,
}

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":          9,
    "axes.titlesize":     9.5,
    "axes.labelsize":     9,
    "axes.edgecolor":     MUTED,
    "axes.labelcolor":    INK,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRID,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "-",
    "xtick.color":        INK_2,
    "ytick.color":        INK_2,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "lines.linewidth":    1.6,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
})


def _save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.pdf")
    fig.savefig(path)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def _label(name: str) -> str:
    return DETECTOR_STYLE.get(name, (name, MUTED))[0]


# ---------------------------------------------------------------------------
# Headline comparison
# ---------------------------------------------------------------------------

def fig_budget_sweep() -> None:
    """Recall against alert budget: what each detector buys per alert."""
    sweep = pd.read_csv(os.path.join(RESULTS, "core", "budget_sweep.csv"))
    n_test = 403_245
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for name, (label, colour) in reversed(list(DETECTOR_STYLE.items())):
        s = sweep[sweep["detector"] == name].sort_values("alert_budget")
        if s.empty:
            continue
        lw = 2.2 if name == "peruser-ocsvm-personal-zscore" else 1.5
        ax.plot(s["alert_budget"] * 100, s["recall"], color=colour, lw=lw,
                marker="o", ms=4, label=label, zorder=3 if colour == BLUE else 2)
    ax.axvline(0.1, color=INK_2, lw=0.8)
    ax.text(0.1 * 1.08, 0.97, "headline budget\n0.1% = 403 alerts", color=INK_2,
            fontsize=7.5, va="top")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.05, 0.1, 0.5, 1, 5])
    ax.set_xticklabels([f"{b:g}%\n({round(n_test * b / 100):,})" for b in
                        [0.01, 0.05, 0.1, 0.5, 1, 5]])
    ax.set_xlabel("Alert budget: share of test user-days flagged (alerts)")
    ax.set_ylabel("Recall (38 malicious days)")
    ax.set_ylim(-0.02, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.0, 0.83))
    _save(fig, "budget_sweep")


def fig_alert_share() -> None:
    """Who the alert queue is spent on: share of alerts on high-privilege users."""
    core = pd.read_csv(os.path.join(RESULTS, "core", "comparison.csv"))
    base = pd.read_csv(os.path.join(RESULTS, "baselines", "comparison.csv"))
    df = pd.concat([core, base[~base["detector"].isin(core["detector"])]],
                   ignore_index=True)
    df = df.sort_values("alert_share_high_privilege")
    pop_share = float(df["population_share_high_privilege"].iloc[0])

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    y = np.arange(len(df))
    colours = [FAMILY_COLOUR.get(f, MUTED) for f in df["family"]]
    ax.barh(y, df["alert_share_high_privilege"] * 100, height=0.62,
            color=colours, zorder=2)
    for yi, v in zip(y, df["alert_share_high_privilege"] * 100):
        ax.text(v + 0.6, yi, f"{v:.1f}%", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(pop_share * 100, color=INK, lw=0.9, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["detector"], fontsize=7.5, family="monospace")
    ax.set_xlabel("Share of the 0.1% alert queue spent on high-privilege users (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(df["alert_share_high_privilege"]) * 100 + 9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Per-user framework"),
                       Patch(color=ORANGE, label="Population detectors"),
                       Patch(color=MUTED, label="Floors"),
                       Line2D([], [], color=INK, lw=0.9,
                              label=f"High-privilege share of users ({pop_share * 100:.2f}%)")],
              loc="lower right")
    _save(fig, "alert_share")


# ---------------------------------------------------------------------------
# Risk timeline: same algorithm, two reference frames
# ---------------------------------------------------------------------------

def _timeline_scores() -> pd.DataFrame:
    path = os.path.join(CACHE, "timeline.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])

    from src.detectors import PerUserDetector, PopulationAnomalyDetector
    from src.experiment import load_experiment_data

    train, test, fc, _ = load_experiment_data(
        os.path.join(ROOT, "data", "artifacts", "dataset.parquet"))
    out = test[["user", "date", "is_malicious"]].copy()
    for col, det in (
        ("framework", PerUserDetector("peruser-ocsvm-personal-zscore", method="ocsvm",
                                      scope="personal", statistic="zscore",
                                      feature_cols=fc)),
        ("population", PopulationAnomalyDetector("pop-ocsvm", "ocsvm", feature_cols=fc)),
    ):
        det.fit(train)
        s = np.asarray(det.score(test), dtype=float)
        # Queue percentile within the test window: comparable across the two
        # detectors, and 99.9 is exactly the headline alert-budget cut.
        out[col] = pd.Series(s).rank(pct=True).to_numpy() * 100
    keep = out[out["user"].isin(["CMP2946", "CDE1846"])]
    os.makedirs(CACHE, exist_ok=True)
    keep.to_csv(path, index=False)
    return keep


def fig_timeline() -> None:
    """The two low-and-slow insiders, scored against their own baseline vs the population's."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = _timeline_scores()
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)
    for ax, user in zip(axes, ["CMP2946", "CDE1846"]):
        u = df[df["user"] == user].sort_values("date")
        mal = u["is_malicious"].astype(bool)
        for d in u.loc[mal, "date"]:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                       color=ORANGE, alpha=0.2, lw=0, zorder=0)
        # Position in the queue as "top x%": log scale, best at the top, so the
        # 0.1% alert line and every day below it stay on the chart.
        for col, colour, lw, z in (("population", AQUA, 1.1, 2),
                                   ("framework", BLUE, 1.5, 3)):
            top = np.clip(100.0 - u[col].to_numpy(), 0.002, 100)
            ax.plot(u["date"], top, color=colour, lw=lw, zorder=z)
        ax.axhline(0.1, color=INK, lw=0.9, zorder=4)
        ax.set_yscale("log")
        ax.set_ylim(100, 0.002)
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
        ax.set_ylabel("Top x% of queue")
        caught = int(((u["framework"] >= 99.9) & mal).sum())
        caught_pop = int(((u["population"] >= 99.9) & mal).sum())
        ax.set_title(f"{user}  ·  malicious days inside the 0.1% alert line: "
                     f"framework {caught}/{int(mal.sum())}, population {caught_pop}/{int(mal.sum())}",
                     loc="left", fontsize=8.5, color=INK)
    handles = [Line2D([], [], color=BLUE, lw=1.5),
               Line2D([], [], color=AQUA, lw=1.1),
               Line2D([], [], color=INK, lw=0.9),
               Patch(color=ORANGE, alpha=0.3)]
    labels = ["Framework: deviation from own history",
              "Same OCSVM on population-scaled features",
              "Alert line (0.1% budget)", "Labelled malicious day"]
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.07))
    fig.autofmt_xdate()
    _save(fig, "timeline")


# ---------------------------------------------------------------------------
# Follow-up studies
# ---------------------------------------------------------------------------

SEED_ORDER = ["peruser-ocsvm-personal-zscore", "peruser-ocsvm-role-zscore",
              "peruser-iforest-personal-zscore", "pop-ocsvm", "pop-iforest",
              "rf", "volume"]


def fig_seeds() -> None:
    """Each detector's spread across ten seeds, for the three tested metrics."""
    df = pd.read_csv(os.path.join(STUDIES, "seeds", "per_seed.csv"))
    panels = [("recall", "Recall at 0.1%"),
              ("avg_precision", "Average precision"),
              ("alert_share_high_privilege", "High-privilege alert share")]
    order = [d for d in SEED_ORDER if d in set(df["detector"])][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, panels):
        for i, det in enumerate(order):
            v = df.loc[df["detector"] == det, metric].astype(float).to_numpy()
            if metric == "alert_share_high_privilege":
                v = v * 100
            colour = BLUE if det.startswith("peruser") else (
                MUTED if det == "volume" else ORANGE)
            jitter = rng.uniform(-0.16, 0.16, len(v))
            ax.scatter(v, i + jitter, s=14, color=colour, alpha=0.75, lw=0, zorder=3)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.2, zorder=4)
        ax.set_title(title + (" (%)" if metric.startswith("alert") else ""), loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, family="monospace", fontsize=7)
    _save(fig, "seeds")


def fig_history() -> None:
    """Sub-question 3: what a personal baseline needs, in days of history."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    det = pd.read_csv(os.path.join(STUDIES, "history", "detection.csv"))
    stab = pd.read_csv(os.path.join(STUDIES, "history", "stability.csv"))
    ref = pd.read_csv(os.path.join(STUDIES, "history", "reference.csv"))
    full_days = 248   # mean active training days per user; plotted position of "all"

    def xpos(v):
        return full_days if str(v) == "all" else float(v)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.7))
    styles = {"peruser-ocsvm-personal-zscore": (BLUE, "Personal baseline, OCSVM"),
              "peruser-iforest-personal-zscore": (AQUA, "Personal baseline, iForest")}
    panels = (("avg_precision", axes[0, 0], "Average precision", 1),
              ("recall", axes[0, 1], "Recall at the 0.1% budget", 1),
              ("alert_share_high_privilege", axes[1, 0],
               "Alerts on high-privilege users (%)", 100))
    for metric, ax, title, scale in panels:
        for name, (colour, label) in styles.items():
            g = det[det["detector"] == name].copy()
            g["x"] = g["history_days"].map(xpos)
            agg = g.groupby("x")[metric].agg(["mean", "min", "max"]).reset_index()
            ax.fill_between(agg["x"], agg["min"] * scale, agg["max"] * scale,
                            color=colour, alpha=0.15, lw=0)
            ax.plot(agg["x"], agg["mean"] * scale, color=colour, marker="o", ms=3.5)
        ax.axhline(ref[metric].mean() * scale, color=INK, lw=0.9)
        ax.set_title(title, loc="left")
    stab = stab.copy()
    stab["x"] = stab["history_days"].map(xpos)
    axes[1, 1].plot(stab["x"], stab["benign_day_exceedance"] * 100, color=INK_2,
                    marker="o", ms=3.5)
    axes[1, 1].set_title("Benign test days with any |z| > 3 (%)", loc="left")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([3, 7, 14, 30, 60, 120, full_days])
        ax.set_xticklabels(["3", "7", "14", "30", "60", "120", "all"])
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Days of personal history")
    fig.legend([Line2D([], [], color=BLUE, marker="o", ms=3.5),
                Line2D([], [], color=AQUA, marker="o", ms=3.5),
                Line2D([], [], color=INK, lw=0.9)],
               ["Personal baseline, One-Class SVM", "Personal baseline, Isolation Forest",
                "Role baseline (no personal history needed)"],
               loc="upper center", ncol=3, fontsize=7.3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    _save(fig, "history")


def fig_explain() -> None:
    """Sub-question 5: does the explanation name what the insider actually did?"""
    from matplotlib.lines import Line2D

    s = pd.read_csv(os.path.join(STUDIES, "explain", "summary.csv"))
    s = s[(s["view"] == "all malicious test days") & (s["insider"] == "all")]
    names = {"personal z": "Framework (own history)", "role z": "Role group z-score",
             "population z": "Population z-score",
             "RF SHAP (positive only)": "RF SHAP, positive only",
             "RF SHAP": "RF SHAP, by $|\\phi|$"}
    order = [e for e in names if e in set(s["explainer"])][::-1]
    s = s.set_index("explainer").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), sharey=True)
    y = np.arange(len(order))
    colours = [BLUE if e == "personal z" else (ORANGE if e.startswith("RF") else MUTED)
               for e in order]
    for ax, metric, title in ((axes[0], "hit_at_1", "Top reason correct (%)"),
                              (axes[1], "hit_at_3", "Any of top 3 correct (%)")):
        ax.barh(y, s[metric] * 100, height=0.62, color=colours, zorder=2)
        for yi, v in zip(y, s[metric] * 100):
            ax.text(v + 1.5, yi, f"{v:.0f}", va="center", fontsize=7.5, color=INK_2)
        ax.axvline(float(s[f"chance_{metric}"].iloc[0]) * 100, color=INK, lw=0.9, zorder=3)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 105)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[e] for e in order], fontsize=7.8)
    fig.legend([Line2D([], [], color=INK, lw=0.9)], ["A random three-feature explanation"],
               loc="lower center", bbox_to_anchor=(0.62, -0.06), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "explain")


FIGURES = {
    "budget_sweep": fig_budget_sweep,
    "alert_share":  fig_alert_share,
    "timeline":     fig_timeline,
    "seeds":        fig_seeds,
    "history":      fig_history,
    "explain":      fig_explain,
}


def main(argv: list[str]) -> int:
    wanted = argv or list(FIGURES)
    for name in wanted:
        try:
            FIGURES[name]()
        except FileNotFoundError as e:
            print(f"  skipped {name}: {os.path.relpath(e.filename, ROOT)} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
