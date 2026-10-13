#!/usr/bin/env python3
##
# @file script/studies.py
# @brief Run the follow-up studies and write their reports.
#
# Three studies, each answering a question the single-seed comparison leaves
# open (see src/studies.py for the design of each):
#
#   seeds    -- do the headline differences survive re-drawing every random
#               component? Ten seeds, paired Wilcoxon, Holm-corrected.
#   history  -- sub-question 3: how many days of history a personal baseline
#               needs before it is stable and useful.
#   explain  -- sub-question 5: do explanations name the features the insider
#               actually used, scored against the official answer key?
#
# Usage:
#   python script/studies.py history
#   python script/studies.py explain
#   python script/studies.py seeds                 # ~80 min, dominated by the RF
#   python script/studies.py seeds --seeds 42 1 2  # quicker, too few for the tests
#   python script/studies.py all
#
# Output lands in results/studies/<study>/, each with a run.json manifest.
##

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from src.colors import amber, blue, dim, green
from src.experiment import _md_table
from src.progress import divider, elapsed_str, header, ok, step
from src.studies import (
    DEFAULT_SEEDS, EXPLANATION_K, SEED_STUDY_DETECTORS, run_explanation_study,
    run_history_study, run_seed_study,
)

DEFAULT_DATASET = os.path.join(ROOT, "data", "artifacts", "dataset.parquet")
DEFAULT_ANSWERS = os.path.join(ROOT, "data", "answers.tar.bz2")
DEFAULT_OUT_DIR = os.path.join(ROOT, "results", "studies")


def _write(out_dir: str, frames: dict, manifest: dict, markdown: str) -> None:
    """@brief Write a study's tables, manifest and Markdown summary."""
    os.makedirs(out_dir, exist_ok=True)
    for name, frame in frames.items():
        path = os.path.join(out_dir, f"{name}.csv")
        frame.to_csv(path, index=False)
        ok(f"{name:12s} -> {os.path.relpath(path, ROOT)}")
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(markdown)
    ok(f"{'summary':12s} -> {os.path.relpath(os.path.join(out_dir, 'summary.md'), ROOT)}")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _seeds_markdown(out: dict) -> str:
    m = out["manifest"]
    s = out["summary"]
    wide = s.pivot(index="detector", columns="metric", values="mean")
    sd   = s.pivot(index="detector", columns="metric", values="std")
    cols = ["avg_precision", "recall", "insider_recall",
            "alert_share_high_privilege", "fit_seconds"]
    table = pd.DataFrame({
        c: [f"{wide.loc[d, c]:.4f} ± {sd.loc[d, c]:.4f}" for d in wide.index]
        for c in cols if c in wide.columns
    }, index=wide.index).reindex(m["detectors"]).reset_index()

    tests = out["tests"][["comparator", "metric", "reference_mean", "comparator_mean",
                          "wins", "losses", "ties", "p_value", "p_holm", "significant"]]
    timing = (out["timing"].groupby(["detector", "user"])
              .agg(alerts_ahead_median=("alerts_ahead", "median"),
                   detected_runs=("n_detected", lambda v: int((v > 0).sum())),
                   delay_days_median=("detection_delay_days", "median"))
              .reset_index())
    return "\n".join([
        "# Seed robustness", "",
        f"Dataset `{m['dataset']}` · {len(m['seeds'])} seeds {m['seeds']} · "
        f"alert budget {m['alert_budget'] * 100:.2f}% · {m['elapsed_seconds']}s", "",
        "Every random component is re-drawn per seed: the fit subsample, the "
        "forests, SMOTE and the tie-break at the cut. The split and the "
        "features are fixed.", "",
        "## Mean ± standard deviation across seeds", "",
        _md_table(table), "",
        f"## Paired Wilcoxon signed-rank tests vs `{m['reference']}`", "",
        "Positive `wins` mean the reference did better on that seed. "
        "`p_holm` is Holm-Bonferroni adjusted across every row.", "",
        _md_table(tests), "",
        "## Where each insider sits in the queue", "",
        "`alerts_ahead` is the number of user-days ranked above the insider's "
        "best-ranked malicious day (threshold-free). `detected_runs` counts "
        "seeds in which the insider was flagged at the budget.", "",
        _md_table(timing, "{:.1f}"), "",
    ])


def _history_markdown(out: dict) -> str:
    m = out["manifest"]
    det = out["detection"]
    agg = (det.groupby(["detector", "history_days"], sort=False)
           [["avg_precision", "recall", "insider_recall",
             "alert_share_high_privilege", "fit_seconds"]]
           .mean().reset_index())
    ref = out["reference"][["avg_precision", "recall", "insider_recall",
                            "alert_share_high_privilege"]].mean()
    return "\n".join([
        "# History length (sub-question 3)", "",
        f"Dataset `{m['dataset']}` · seeds {m['seeds']} · "
        f"alert budget {m['alert_budget'] * 100:.2f}% · {m['elapsed_seconds']}s", "",
        f"Truncation: {m['truncation']}. The test window is unchanged.", "",
        "## Baseline stability", "",
        "`mean_shift_*` is the distance between the N-day and full-history "
        "per-user mean, in full-history standard deviations. "
        f"`benign_day_exceedance` is the share of benign test days with any "
        f"feature beyond |z| = {m['exceedance_z']} under the N-day profile.", "",
        _md_table(out["stability"]), "",
        "## Detection (mean across seeds)", "",
        _md_table(agg), "",
        "Role-group reference (needs no personal history): "
        f"AP {ref['avg_precision']:.4f}, recall {ref['recall']:.4f}, "
        f"insider recall {ref['insider_recall']:.2f}, "
        f"high-privilege alert share {ref['alert_share_high_privilege']:.4f}.", "",
    ])


def _explain_markdown(out: dict) -> str:
    m = out["manifest"]
    k = m["explanation_k"]
    s = out["summary"]
    overall = s[s["insider"] == "all"].drop(columns="insider")
    per_insider = s[(s["insider"] != "all") & (s["view"] == "all malicious test days")]
    return "\n".join([
        "# Explanation fidelity (sub-question 5)", "",
        f"Dataset `{m['dataset']}` · {m['n_malicious_test_days']} malicious test "
        f"days · explanations of length {k} over {m['n_features']} features · "
        f"{m['elapsed_seconds']}s", "",
        "Ground truth: the features the answer key's malicious events moved on "
        "that day, replayed through the dataset's own aggregator. `chance_*` is "
        f"the analytic rate of a random {k}-feature explanation.", "",
        "## Overall", "",
        _md_table(overall), "",
        "## By insider (all malicious test days)", "",
        _md_table(per_insider.drop(columns="view")), "",
    ])


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_seeds(args) -> None:
    step(1, f"Seed robustness — {len(args.seeds)} seeds × "
            f"{len(SEED_STUDY_DETECTORS)} detectors")

    def on_run(seed, det):
        print(f"  {dim(f'[seed {seed}]')} {blue(det.name)}", flush=True)

    out = run_seed_study(args.dataset, seeds=tuple(args.seeds), budget=args.budget,
                         on_run=on_run)
    _write(os.path.join(args.out, "seeds"),
           {"per_seed": out["per_seed"], "summary": out["summary"],
            "tests": out["tests"], "timing": out["timing"]},
           out["manifest"], _seeds_markdown(out))


def run_history(args) -> None:
    step(2, "History length (sub-question 3)")
    out = run_history_study(args.dataset, budget=args.budget,
                            on_run=lambda label: print(f"  {dim(label)}", flush=True))
    _write(os.path.join(args.out, "history"),
           {"detection": out["detection"], "stability": out["stability"],
            "reference": out["reference"]},
           out["manifest"], _history_markdown(out))


def run_explain(args) -> None:
    step(3, "Explanation fidelity (sub-question 5)")
    if not os.path.exists(args.answers):
        print(f"  {amber('No answer key at')} {args.answers} — skipped.")
        return
    out = run_explanation_study(args.dataset, args.answers, budget=args.budget,
                                on_step=lambda label: print(f"  {dim(label)}", flush=True))
    _write(os.path.join(args.out, "explain"),
           {"per_day": out["per_day"], "summary": out["summary"],
            "case_study": out["case_study"]},
           out["manifest"], _explain_markdown(out))


def main() -> int:
    """@brief Parse arguments and run the requested studies."""
    from src.evaluate import DEFAULT_ALERT_BUDGET

    parser = argparse.ArgumentParser(
        description="Run the follow-up studies (seeds, history, explain).")
    parser.add_argument("study", choices=("seeds", "history", "explain", "all"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, metavar="PATH")
    parser.add_argument("--answers", default=DEFAULT_ANSWERS, metavar="PATH",
                        help="answers.tar.bz2, for the explanation study.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--budget", type=float, default=DEFAULT_ALERT_BUDGET)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, metavar="DIR")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"\n  No prepared dataset at {os.path.relpath(args.dataset, ROOT)}.")
        print(f"  Build it with:  {amber('python script/prepare.py')}\n")
        return 1

    t0 = time.perf_counter()
    header("COS700 — Follow-up Studies")
    # Shortest first, so the long seed study never holds the others up.
    order = ("history", "explain", "seeds") if args.study == "all" else (args.study,)
    for name in order:
        {"seeds": run_seeds, "history": run_history, "explain": run_explain}[name](args)

    print()
    divider("═", 70)
    print(green("  Studies complete.") + f"  Total elapsed: {elapsed_str(t0)}")
    divider("═", 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
