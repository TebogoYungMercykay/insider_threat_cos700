##
# @file src/experiment.py
# @brief The controlled comparison experiment (proposal Section 4.1).
#
# Fits every detector on one training split, scores every detector on one test
# split, and reduces the lot to a table. The design rules it enforces:
#
#   * One split, fixed in the dataset. The train/test boundary is a date
#     chosen at prepare time and frozen into the Parquet file, so no detector
#     can be tuned against the window it is scored on.
#
#   * One feature set. Whatever columns the run is configured with, every
#     detector gets all of them.
#
#   * One operating point, chosen by alert budget rather than by threshold,
#     plus threshold-free metrics. See src/evaluate.py for why.
#
#   * The full test split, never a sample. 38 malicious user-days sit in
#     403 245 test rows; a 5 000-row sample contains 0.47 of them in
#     expectation, so any metric computed on a sample is noise wearing a
#     number's clothes.
##

import json
import os
import platform
import time

import numpy as np
import pandas as pd

from src.detectors import Detector, build_detectors
from src.evaluate import (
    BUDGET_SWEEP, DEFAULT_ALERT_BUDGET, budget_sweep, evaluate_recall_by_archetype,
    evaluate_scores, flags_for_budget, fpr_by_role, insider_timing,
)
from src.features import add_rolling_features, rolling_column_names
from src.prepare import ground_truth_from_manifest, load_prepared
from src.preprocess import FEATURE_COLS
from src.progress import elapsed_str, kv, ok, substep, warn

## Columns promoted to the headline table, in reading order.
HEADLINE_COLUMNS = [
    "detector", "family", "roc_auc", "avg_precision", "recall", "precision",
    "lift", "insider_recall", "n_insiders_detected", "recall_low_and_slow",
    "recall_burst", "fpr_high_privilege", "alert_share_high_privilege",
    "tie_fraction", "fit_seconds", "score_ms_per_row",
]


def load_experiment_data(dataset_path: str, temporal: bool = False,
                         max_users: int | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """
    @brief Load a prepared dataset and derive the experiment's feature set.

    @param dataset_path  Prepared dataset .parquet.
    @param temporal      Append causal rolling-window aggregates.
    @param max_users     Cap on distinct users, for smoke runs.
    @return              (train_df, test_df, feature_cols, manifest).
    """
    df, manifest = load_prepared(dataset_path)

    if max_users:
        keep = set(sorted(df["user"].unique())[:max_users])
        df = df[df["user"].isin(keep)].reset_index(drop=True)

    feature_cols = list(FEATURE_COLS)
    if temporal:
        with substep("adding causal rolling-window features") as s:
            df, added = add_rolling_features(df, FEATURE_COLS)
            feature_cols += added
            s.set_detail(f"{len(added)} columns, windows over trailing active days")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, test_df, feature_cols, manifest


def run_detector(detector: Detector, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 gt_df: pd.DataFrame, budget: float = DEFAULT_ALERT_BUDGET,
                 budgets: tuple[float, ...] = BUDGET_SWEEP,
                 random_state: int = 42) -> dict:
    """
    @brief Fit one detector, score the full test split, and evaluate it.

    Peak memory is sampled with tracemalloc around fit and score together,
    which answers the feasibility half of sub-question 4 without needing a
    separate profiling run.

    @param detector  Detector to run.
    @param train_df  Training split.
    @param test_df   Test split (scored in full).
    @param gt_df     Insider windows, for the archetype breakdown.
    @param budget    Headline alert budget.
    @param budgets   Budgets for the sweep.
    @param random_state  Seed for the tie-break at the alert-budget cut.
    @return          {"metrics": ..., "sweep": DataFrame, "role_fpr": ...,
                     "scores": ndarray, "timing": DataFrame}.
    """
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()
    detector.fit(train_df)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = np.asarray(detector.score(test_df), dtype=float)
    score_seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not np.isfinite(scores).all():
        warn(f"{detector.name}: non-finite scores replaced with the minimum")
        finite_min = np.nanmin(scores[np.isfinite(scores)]) if np.isfinite(scores).any() else 0.0
        scores = np.nan_to_num(scores, nan=finite_min, posinf=finite_min, neginf=finite_min)

    metrics = evaluate_scores(test_df, scores, budget=budget, random_state=random_state)
    flagged, _ = flags_for_budget(scores, budget, random_state)
    arch    = evaluate_recall_by_archetype(test_df, flagged, gt_df)

    row = {**detector.describe(), **metrics}
    for name, stats in arch.items():
        row[f"recall_{name}"]             = stats["recall"]
        row[f"n_malicious_days_{name}"]   = stats["n_malicious_days"]
        row[f"n_detected_{name}"]         = stats["n_detected"]
    row.update({
        "fit_seconds":      round(fit_seconds, 3),
        "score_seconds":    round(score_seconds, 3),
        "score_ms_per_row": round(score_seconds / max(len(test_df), 1) * 1000, 5),
        "peak_memory_mb":   round(peak / (1024 * 1024), 1),
    })

    return {
        "metrics":  row,
        "sweep":    budget_sweep(test_df, scores, budgets, gt_df,
                                 random_state).assign(detector=detector.name),
        "role_fpr": fpr_by_role(test_df, flagged),
        "scores":   scores,
        "timing":   insider_timing(test_df, scores, budget,
                                   random_state).assign(detector=detector.name),
    }


def run_comparison(dataset_path: str, preset: str = "core", temporal: bool = False,
                   budget: float = DEFAULT_ALERT_BUDGET,
                   budgets: tuple[float, ...] = BUDGET_SWEEP,
                   contamination: float = 0.02, min_history_days: int = 14,
                   fit_subsample: int | None = None, max_users: int | None = None,
                   random_state: int = 42, on_detector=None) -> dict:
    """
    @brief Run the full comparison and return every table it produced.

    @param dataset_path      Prepared dataset to evaluate on.
    @param preset            Detector set (see detectors.build_detectors).
    @param temporal          Give every detector rolling-window features too.
    @param budget            Headline alert budget.
    @param budgets           Budgets for the sweep.
    @param contamination     Anomaly-model contamination.
    @param min_history_days  Cold-start threshold for the framework.
    @param fit_subsample     Training-row cap for the quadratic models.
    @param max_users         Cap on users, for smoke runs.
    @param random_state      Seed.
    @param on_detector       Optional callback(index, total, detector) for progress.
    @return                  {"results", "sweep", "role_fpr", "manifest"}.
    """
    from src.detectors import DEFAULT_FIT_SUBSAMPLE

    t_total = time.perf_counter()
    train_df, test_df, feature_cols, manifest = load_experiment_data(
        dataset_path, temporal=temporal, max_users=max_users)
    gt_df = ground_truth_from_manifest(manifest)

    if fit_subsample is None:
        fit_subsample = DEFAULT_FIT_SUBSAMPLE

    detectors = build_detectors(
        preset, feature_cols=feature_cols, contamination=contamination,
        min_history_days=min_history_days, fit_subsample=fit_subsample,
        random_state=random_state)

    rows, sweeps, timings, role_fprs = [], [], [], {}
    for i, detector in enumerate(detectors, 1):
        if on_detector:
            on_detector(i, len(detectors), detector)
        out = run_detector(detector, train_df, test_df, gt_df, budget, budgets,
                           random_state)
        rows.append(out["metrics"])
        sweeps.append(out["sweep"])
        timings.append(out["timing"])
        role_fprs[detector.name] = out["role_fpr"]

    results = pd.DataFrame(rows)
    # Rank by average precision: on a 1-in-31 000 positive rate it is the only
    # threshold-free metric that does not saturate, and it is what separates
    # detectors that put insiders near the top of the queue from ones that
    # merely order the whole table plausibly.
    if "avg_precision" in results.columns:
        results = results.sort_values("avg_precision", ascending=False,
                                      ignore_index=True)

    run_manifest = {
        "dataset":          os.path.basename(dataset_path),
        "dataset_manifest": {k: manifest.get(k) for k in
                             ("n_rows", "n_users", "split_cutoff", "n_train",
                              "n_test", "n_malicious", "n_malicious_train",
                              "n_malicious_test", "n_insiders", "label_source")},
        "preset":           preset,
        "temporal":         temporal,
        "feature_cols":     feature_cols,
        "n_features":       len(feature_cols),
        "alert_budget":     budget,
        "budget_sweep":     list(budgets),
        "contamination":    contamination,
        "min_history_days": min_history_days,
        "fit_subsample":    fit_subsample,
        "max_users":        max_users,
        "random_state":     random_state,
        "n_detectors":      len(detectors),
        "n_train_rows":     int(len(train_df)),
        "n_test_rows":      int(len(test_df)),
        "n_malicious_test": int(test_df["is_malicious"].sum()),
        "archetypes":       _archetype_summary(gt_df, test_df),
        "elapsed_seconds":  round(time.perf_counter() - t_total, 1),
        "run_at":           time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform":         f"{platform.system()} {platform.machine()}, "
                            f"Python {platform.python_version()}",
        "cpu_count":        os.cpu_count(),
    }

    return {"results": results, "sweep": pd.concat(sweeps, ignore_index=True),
            "timing": pd.concat(timings, ignore_index=True),
            "role_fpr": role_fprs, "manifest": run_manifest}


def _archetype_summary(gt_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    @brief Record how the ground-truth insiders split across archetypes and
           how many of their malicious days actually land in the test window.

    @param gt_df    Insider windows.
    @param test_df  Test split.
    @return         {archetype: {n_insiders, users, n_malicious_test_days}}.
    """
    from src.evaluate import classify_archetypes

    if gt_df is None or gt_df.empty:
        return {}

    arch = classify_archetypes(gt_df)
    mal  = test_df[test_df["is_malicious"]]
    out  = {}
    for archetype, grp in arch.groupby("archetype"):
        users = sorted(grp["user"].astype(str))
        out[str(archetype)] = {
            "n_insiders":           len(users),
            "users":                users,
            "n_malicious_test_days": int(mal["user"].astype(str).isin(users).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """
    @brief Render a DataFrame as a GitHub-flavoured Markdown table.

    Hand-rolled rather than pulling in `tabulate` for one call: the report is
    the only place a Markdown table is needed, and a dissertation repository
    is easier to reproduce with one fewer dependency to pin.

    @param df         Frame to render.
    @param float_fmt  Format applied to floating-point cells.
    @return           Markdown table.
    """
    def cell(v) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else float_fmt.format(v)
        return "" if v is None else str(v)

    head = [str(c) for c in df.columns]
    body = [[cell(v) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(head[i]), *(len(r[i]) for r in body)) if body else len(head[i])
              for i in range(len(head))]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    return "\n".join([line(head),
                      "| " + " | ".join("-" * w for w in widths) + " |",
                      *(line(r) for r in body)])


def headline_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Reduce the full results frame to the columns worth reading first.

    @param results  Full results table.
    @return         Headline view, columns present only if they were produced.
    """
    cols = [c for c in HEADLINE_COLUMNS if c in results.columns]
    return results[cols]


def write_report(outcome: dict, out_dir: str) -> dict:
    """
    @brief Write every table and the run manifest to disk.

    @param outcome  Result of run_comparison().
    @param out_dir  Destination directory.
    @return         {label: path} of everything written.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = outcome["results"]

    paths = {
        "results_csv":  os.path.join(out_dir, "comparison.csv"),
        "sweep_csv":    os.path.join(out_dir, "budget_sweep.csv"),
        "role_fpr_csv": os.path.join(out_dir, "fpr_by_role.csv"),
        "timing_csv":   os.path.join(out_dir, "insider_timing.csv"),
        "manifest":     os.path.join(out_dir, "run.json"),
        "report_md":    os.path.join(out_dir, "comparison.md"),
    }

    results.to_csv(paths["results_csv"], index=False)
    outcome["sweep"].to_csv(paths["sweep_csv"], index=False)
    outcome["timing"].to_csv(paths["timing_csv"], index=False)
    pd.DataFrame(outcome["role_fpr"]).to_csv(paths["role_fpr_csv"])
    with open(paths["manifest"], "w") as f:
        json.dump(outcome["manifest"], f, indent=2, default=str)
    with open(paths["report_md"], "w") as f:
        f.write(render_markdown(outcome))

    return paths


def render_markdown(outcome: dict) -> str:
    """
    @brief Render the comparison as a Markdown report.

    @param outcome  Result of run_comparison().
    @return         Markdown document.
    """
    m       = outcome["manifest"]
    results = outcome["results"]
    lines = [
        "# Detector comparison",
        "",
        f"Dataset `{m['dataset']}` · preset `{m['preset']}` · "
        f"{m['n_detectors']} detectors · {m['elapsed_seconds']}s",
        "",
        f"- Train {m['n_train_rows']:,} rows / test {m['n_test_rows']:,} rows, "
        f"split at {m['dataset_manifest'].get('split_cutoff')}",
        f"- {m['n_malicious_test']} malicious user-days in the test split "
        f"({m['n_malicious_test'] / max(m['n_test_rows'], 1) * 100:.4f}%)",
        f"- Features: {m['n_features']}"
        + (" (base + causal rolling windows)" if m["temporal"] else " (base)"),
        f"- Operating point: top {m['alert_budget'] * 100:.3f}% of test rows "
        f"by score, applied identically to every detector",
        "",
    ]

    if m.get("archetypes"):
        lines += ["## Ground-truth archetypes", ""]
        for name, info in sorted(m["archetypes"].items()):
            lines.append(f"- **{name}**: {info['n_insiders']} insider(s) "
                         f"({', '.join(info['users'])}), "
                         f"{info['n_malicious_test_days']} malicious day(s) in the test split")
        empty = [n for n, i in m["archetypes"].items()
                 if i["n_malicious_test_days"] == 0]
        if empty:
            lines += ["",
                      f"> **No test data for: {', '.join(sorted(empty))}.** "
                      "r6.2's insider windows are not spread evenly across the "
                      "calendar, so a time-based split does not divide the "
                      "archetypes between train and test. Every burst insider "
                      "acts before the split date and every low-and-slow "
                      "insider acts after it. The test split therefore measures "
                      "low-and-slow detection only, and the supervised "
                      "baselines are being asked to generalise from burst "
                      "training examples to low-and-slow test ones. That is a "
                      "property of the benchmark, not of the split fraction: "
                      "moving the boundary to capture a burst insider in test "
                      "would leave the supervised baselines with no positive "
                      "training rows at all. It is reported rather than "
                      "engineered away."]
        lines.append("")

    lines += ["## Results", "",
              _md_table(headline_table(results)), ""]
    lines += ["",
              "Ranked by average precision. `recall`, `precision` and the FPR "
              "columns are measured at the shared alert budget above; "
              "`roc_auc` and `avg_precision` are threshold-free. "
              "`tie_fraction` is the share of a detector's alerts drawn "
              "arbitrarily from a group of equal scores at the cut -- a high "
              "value means the ranking, not the metric, is what ran out.",
              ""]
    return "\n".join(lines)
