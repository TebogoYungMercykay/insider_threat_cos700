##
# @file train.py
# @brief Training pipeline for the Per-User Behavioural Baseline Framework.
#
# Each step is independently cached. Re-running only executes steps whose
# inputs have changed since the last run. Use --force to bypass all caches.
#
# Usage:
#   python train.py                         # skip any steps already up-to-date
#   python train.py --max-users 100         # quick smoke-test
#   python train.py --anomaly-method lof    # iforest | lof | ocsvm
#   python train.py --no-shap               # skip SHAP explainer (RF only)
#   python train.py --force                 # ignore all caches, redo everything
#   python train.py --force-step 4          # force only step 4 (and its dependents)
##

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.colors import blue, green, red, amber, dim
from src.progress import (
    step as _step, skip as _skip, ok as _ok, warn as _warn, err as _err,
    kv as _kv, divider as _divider, elapsed_str as _elapsed, Heartbeat,
)

ROOT          = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(ROOT, "data", "artifacts")

PATHS = {
    "daily_df":     os.path.join(ARTIFACTS_DIR, "daily_df.pkl"),
    "gt_df":        os.path.join(ARTIFACTS_DIR, "gt_df.pkl"),
    "population":   os.path.join(ARTIFACTS_DIR, "population_rf.pkl"),
    "peruser":      os.path.join(ARTIFACTS_DIR, "peruser_engine.pkl"),
    "shap":         os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"),
    "meta":         os.path.join(ARTIFACTS_DIR, "meta.json"),
    "cache_state":  os.path.join(ARTIFACTS_DIR, ".cache_state.json"),
}

_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"


def _header() -> None:
    _divider("═", 70)
    print(blue(_BOLD + "  COS700 — Per-User Behavioural Baseline Framework" + _RESET))
    _divider("═", 70)
    print()


def _load_state() -> dict:
    if os.path.exists(PATHS["cache_state"]):
        try:
            with open(PATHS["cache_state"]) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    with open(PATHS["cache_state"], "w") as f:
        json.dump(state, f, indent=2)


def _file_sig(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    s = os.stat(path)
    return f"{s.st_size}:{s.st_mtime:.2f}"


def _raw_sig(max_users) -> str:
    raw_dir = os.path.join(ROOT, "data", "raw")
    files = ["logon.csv", "device.csv", "file.csv", "http.csv", "email.csv"]
    sig = "|".join(_file_sig(os.path.join(raw_dir, f)) for f in files)
    return f"{sig}|max_users={max_users}"


def _step_ok(state: dict, step: str, fingerprint: dict) -> bool:
    saved = state.get(step)
    if not saved:
        return False
    for path in saved.get("_files", []):
        if not os.path.exists(path):
            return False
    return {k: v for k, v in saved.items() if k != "_files"} == fingerprint


def _mark_done(state: dict, step: str, fingerprint: dict, files: list[str]) -> None:
    state[step] = {**fingerprint, "_files": files}
    _save_state(state)


def train(
    max_users:       int | None = None,
    anomaly_method:  str = "iforest",
    contamination:   float = 0.03,
    min_history_days: int = 14,
    use_shap:        bool = True,
    force_from:      int = 99,
) -> None:
    """
    @brief Execute the full training pipeline with per-step caching.

    Steps: load & engineer features -> time split -> fit population RF
    baseline (SMOTE-balanced) -> fit per-user baseline framework -> fit SHAP
    explainer -> write metadata manifest.

    @param max_users        Cap on distinct users loaded (None = all).
    @param anomaly_method    "iforest" | "lof" | "ocsvm" for the per-user
                            framework's Anomaly Scorer component.
    @param contamination     Expected proportion of anomalous rows.
    @param min_history_days  Minimum per-user training days before the
                            personal baseline is trusted over the role-group
                            baseline (cold-start threshold).
    @param use_shap          If False, skip fitting the SHAP explainer.
    @param force_from        Re-run all steps >= this number, ignoring cache.
    """
    import joblib
    import numpy as np

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    t_total = time.perf_counter()
    state   = _load_state()

    _header()
    print(f"  {dim('Checking cache state ...')}")

    if force_from <= 5:
        for s in [f"step{i}" for i in range(force_from, 6)]:
            state.pop(s, None)
        _save_state(state)
        print(f"  {amber(f'Cache invalidated from step {force_from} onward.')}")

    # Step 1: Load raw logs and build the per-(user, day) feature table
    S1_FP    = {"raw_sig": _raw_sig(max_users)}
    S1_FILES = [PATHS["daily_df"], PATHS["gt_df"]]

    if _step_ok(state, "step1", S1_FP):
        _skip(1, "Loading & Engineering Features")
        df    = joblib.load(PATHS["daily_df"])
        gt_df = joblib.load(PATHS["gt_df"])
    else:
        _step(1, "Loading Raw Logs & Engineering Behavioural Features")
        t0 = time.perf_counter()

        from src.preprocess import (
            load_raw_logs, build_daily_features, load_org_structure,
            load_ground_truth, attach_context, time_split,
        )
        logs = load_raw_logs()
        if max_users:
            keep = set(logs["logon"]["user"].drop_duplicates().head(max_users))
            logs = {k: v[v["user"].isin(keep)].reset_index(drop=True) for k, v in logs.items()}

        daily = build_daily_features(logs)
        org   = load_org_structure()
        gt_df = load_ground_truth()
        df    = attach_context(daily, org, gt_df)
        df    = time_split(df)

        _kv("Users",              df["user"].nunique())
        _kv("User-day rows",      f"{len(df):,}")
        _kv("Train / Test rows",  f"{(df['split']=='train').sum():,} / {(df['split']=='test').sum():,}")
        _kv("Malicious rows",     int(df["is_malicious"].sum()))
        _kv("High-privilege rows", int(df["is_high_privilege"].sum()))
        _kv("Elapsed", _elapsed(t0))
        _ok("Feature table built")

        joblib.dump(df, PATHS["daily_df"])
        joblib.dump(gt_df, PATHS["gt_df"])
        for s in ["step2", "step3", "step4", "step5"]:
            state.pop(s, None)
        _mark_done(state, "step1", S1_FP, S1_FILES)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # Step 2: Fit population-level Random Forest baseline (Section 4.4)
    S2_FP    = {"n_train": len(train_df), "sig": _file_sig(PATHS["daily_df"])}
    S2_FILES = [PATHS["population"]]

    if _step_ok(state, "step2", S2_FP):
        _skip(2, "Fitting Population Baseline (Random Forest)")
        population = joblib.load(PATHS["population"])
    else:
        _step(2, "Fitting Population Baseline  (Random Forest, SMOTE-balanced)")
        t0 = time.perf_counter()

        from imblearn.over_sampling import SMOTE
        from src.baseline import PopulationBaseline
        from src.preprocess import FEATURE_COLS

        X = train_df[FEATURE_COLS].to_numpy()
        y = train_df["is_malicious"].astype(int).to_numpy()
        n_pos = int(y.sum())

        if n_pos >= 2:
            sm = SMOTE(random_state=42, k_neighbors=min(5, n_pos - 1))
            with Heartbeat("SMOTE resampling"):
                X_bal, y_bal = sm.fit_resample(X, y)
        else:
            _warn("Fewer than 2 positive training rows — skipping SMOTE.")
            X_bal, y_bal = X, y

        _kv("Positive rows before SMOTE", n_pos)
        _kv("Rows after SMOTE",           f"{len(X_bal):,}")

        population = PopulationBaseline(n_estimators=300).fit(X_bal, y_bal)
        _kv("RF fit time", f"{population.fit_seconds_:.2f}s")
        _kv("Elapsed", _elapsed(t0))
        _ok("Population baseline fitted")

        joblib.dump(population, PATHS["population"])
        for s in ["step5"]:
            state.pop(s, None)
        _mark_done(state, "step2", S2_FP, S2_FILES)

    # Step 3: Fit per-user behavioural baseline framework (Section 4.3)
    S3_FP    = {
        "n_train": len(train_df), "sig": _file_sig(PATHS["daily_df"]),
        "method": anomaly_method, "contamination": contamination,
        "min_history_days": min_history_days,
    }
    S3_FILES = [PATHS["peruser"]]

    if _step_ok(state, "step3", S3_FP):
        _skip(3, "Fitting Per-User Baseline Framework")
        peruser = joblib.load(PATHS["peruser"])
    else:
        _step(3, f"Fitting Per-User Baseline Framework  "
                 f"(Profile Builder + Role-Aware Normaliser + {anomaly_method.upper()})")
        t0 = time.perf_counter()

        from src.baseline import PerUserBaselineEngine

        peruser = PerUserBaselineEngine(
            method=anomaly_method, contamination=contamination,
            min_history_days=min_history_days,
        ).fit(train_df)

        n_personal = int((peruser.profiler.user_history_len_ >= min_history_days).sum())
        _kv("Users with personal baseline", n_personal)
        _kv("Users on role-group fallback",
            train_df["user"].nunique() - n_personal)
        _kv("Anomaly model fit time", f"{peruser.fit_seconds_:.3f}s")
        _kv("Elapsed", _elapsed(t0))
        _ok("Per-user framework fitted")

        joblib.dump(peruser, PATHS["peruser"])
        _mark_done(state, "step3", S3_FP, S3_FILES)

    # Step 4: Fit SHAP explainer for the population baseline
    shap_ok = False
    if not use_shap:
        _skip(4, "SHAP Explainer", reason="--no-shap")
    else:
        S4_FP    = {"pop_sig": _file_sig(PATHS["population"])}
        S4_FILES = [PATHS["shap"]]

        if _step_ok(state, "step4", S4_FP):
            _skip(4, "SHAP Explainer")
            shap_ok = True
        else:
            _step(4, "Fitting SHAP TreeExplainer  (population baseline)")
            t0 = time.perf_counter()
            try:
                from src.explain import build_rf_explainer
                explainer = build_rf_explainer(population.model)
                if explainer is None:
                    raise RuntimeError("shap package unavailable")
                joblib.dump(explainer, PATHS["shap"])
                _kv("Elapsed", _elapsed(t0))
                _ok("SHAP explainer fitted and saved")
                shap_ok = True
                _mark_done(state, "step4", S4_FP, S4_FILES)
            except Exception as exc:
                _err(f"SHAP fitting failed — {exc}")
                _warn("App will fall back to feature-importance explanations only.")

    # Step 5: Metadata manifest (always refreshed)
    _step(5, "Writing Metadata Manifest")

    meta = {
        "n_users":            int(df["user"].nunique()),
        "n_rows":             int(len(df)),
        "n_train":            int(len(train_df)),
        "n_test":             int(len(test_df)),
        "n_malicious_rows":   int(df["is_malicious"].sum()),
        "n_insider_windows":  int(len(gt_df)),
        "anomaly_method":     anomaly_method,
        "contamination":      contamination,
        "min_history_days":   min_history_days,
        "shap_available":     shap_ok,
        "trained_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_users_arg":      max_users,
    }
    with open(PATHS["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    for k, v in meta.items():
        _kv(f"  {k}", str(v))
    _ok("Metadata written to  meta.json")

    print()
    _divider("═", 70)
    print(green(_BOLD + "  Pipeline complete." + _RESET) + f"  Total elapsed: {_elapsed(t_total)}")
    _divider("─", 70)
    print(f"  Artifacts       : {blue(ARTIFACTS_DIR)}")
    print(f"  SHAP explainer  : {green('fitted') if shap_ok else amber('unavailable')}")
    print()
    print(f"  {dim('Run the app:')}")
    print(f"    {amber(_BOLD + 'streamlit run app.py' + _RESET)}")
    print()
    _divider("═", 70)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the Per-User Behavioural Baseline Framework.\n"
            "Each step is independently cached — only changed steps re-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-users", type=int, default=None, metavar="N",
                         help="Limit to first N users (for quick smoke-tests).")
    parser.add_argument("--anomaly-method", choices=["iforest", "lof", "ocsvm"],
                         default="iforest", help="Anomaly Scorer algorithm.")
    parser.add_argument("--contamination", type=float, default=0.03,
                         help="Expected proportion of anomalous rows.")
    parser.add_argument("--min-history-days", type=int, default=14,
                         help="Cold-start threshold for personal vs role-group baseline.")
    parser.add_argument("--no-shap", action="store_true",
                         help="Skip fitting the SHAP explainer.")
    parser.add_argument("--force", action="store_true",
                         help="Ignore all caches and redo every step from scratch.")
    parser.add_argument("--force-step", type=int, default=99, metavar="N",
                         help="Force re-run from step N onward.")
    args = parser.parse_args()

    force_from = 1 if args.force else args.force_step

    train(
        max_users=args.max_users,
        anomaly_method=args.anomaly_method,
        contamination=args.contamination,
        min_history_days=args.min_history_days,
        use_shap=not args.no_shap,
        force_from=force_from,
    )
