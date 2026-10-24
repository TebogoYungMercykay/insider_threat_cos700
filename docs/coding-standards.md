# Coding Standards

How code, tests and documentation are written in this repository. The aim is that every number in the report can be traced to code a reviewer can read, and that the evaluation cannot quietly flatter a detector.

### Table of Contents

1. [Project structure](#project-structure)
2. [Python style](#python-style)
3. [Comments and docstrings](#comments-and-docstrings)
4. [Correctness rules](#correctness-rules)
5. [Tests](#tests)
6. [The dashboard](#the-dashboard)
7. [Git, branches and commits](#git-branches-and-commits)
8. [Documentation](#documentation)

## Project structure

```
insider_threat_cos700/
├── app.py                      # Streamlit dashboard: Overview, Risk Explorer, Method
│                               # Comparison, Evaluation, Findings, About
├── train.py                    # trains the dashboard's two models, with per-step caching
├── Dockerfile                  # the dashboard image
├── deploy.sh                   # AWS deployment, see docs/deployment.md
├── requirements.txt            # dependencies
├── constraints.txt             # pinned versions for the image
├── pytest.ini                  # `pytest -m "not slow"` skips the raw-log scans
├── .streamlit/config.toml      # dashboard theme
├── data/
│   ├── README.md               # dataset download and ingestion
│   ├── r6.2.tar.bz2            # (you provide) official log archive
│   ├── answers.tar.bz2         # (you provide) official answer-key archive
│   ├── raw/                    # CERT-schema logs, LDAP snapshots and answers, generated
│   └── artifacts/              # userday.parquet, dataset.parquet, trained models, generated
├── docs/                       # guides, see the README's documentation table
│   └── assets/                 # images used by the docs
├── report/
│   ├── Report.tex              # the COS700 research report
│   ├── references.bib
│   ├── make_figures.py         # every data figure, from results/
│   └── figures/                # generated figures and dashboard screenshots
├── results/                    # experiment output, generated
│   └── <preset>/               # comparison tables, budget sweep, per-insider timing,
│                               # run.json with full provenance
├── script/                     # command-line entry points
│   ├── dataset.py              # verify and extract r6.2, convert the answer key
│   ├── aggregate.py            # one-time streaming reduction of the raw logs
│   ├── prepare.py              # freeze counts, context, labels and split into one dataset
│   ├── experiment.py           # the controlled comparison
│   └── studies.py              # seeds, history (SQ3) and explanation (SQ5) studies
├── src/                        # the importable library
│   ├── schema.py               # feature set, raw-log layout, fast timestamp and URL parsing
│   ├── aggregate.py            # parallel byte-range streaming aggregator
│   ├── cohort.py               # reproducible role-stratified subsets
│   ├── prepare.py              # context, labels, split and the dataset manifest
│   ├── preprocess.py           # loading the aggregated table, labelling, the time split
│   ├── baseline.py             # Profile Builder, Role-Aware Normaliser, Anomaly Scorer,
│   │                           # PerUserBaselineEngine, PopulationBaseline (Random Forest)
│   ├── features.py             # causal trailing-window features (--temporal)
│   ├── detectors.py            # one fit and score interface over every detector
│   ├── evaluate.py             # alert budgets, metrics, queue depth, detection delay, cost
│   ├── experiment.py           # running the comparison and rendering its tables
│   ├── explain.py              # SHAP TreeExplainer wrapper for the Random Forest
│   ├── studies.py              # the three follow-up studies
│   ├── progress.py             # console progress output
│   └── colors.py               # the shared colour palette
└── tests/
    ├── test_pipeline.py        # shard invariance, feature semantics, ground truth, cohorts
    ├── test_model.py           # scopes, cold start, saturation, budgets, ties, causal windows
    └── test_studies.py         # queue depth, Holm correction, history truncation, replay
```

Logic lives in `src/` and is imported. Anything that parses arguments or prints progress for a person lives in `script/`, `train.py` or `app.py`. A new experiment is a function in `src/` with a thin entry point in `script/`, not a new standalone script.

## Python style

- Python 3.14, the version the image uses. Use modern syntax such as `str | None` and built-in generics.
- PEP 8 naming. `snake_case` for functions and variables, `CamelCase` for classes, `UPPER_CASE` for module constants.
- Four-space indentation. Keep lines in `src/` and `script/` under about 100 characters. The dashboard's inline HTML strings are the one exception.
- Type hints on every public function.
- Align a block of related assignments when that makes it easier to scan, as the existing code does.
- Prefer vectorised NumPy and pandas over Python loops. The exception is the aggregator's inner loop in `src/aggregate.py`, which deliberately uses flat list arithmetic because it runs over every log event.
- Constants that shape a result, such as the alert budget, the cold-start threshold or the fit subsample, are named module constants or parameters with defaults, never literals buried in a function.

## Comments and docstrings

Every module opens with a header block that names the file, says what it is for and, where it matters, why it is built the way it is.

```python
##
# @file src/evaluate.py
# @brief Evaluation metrics for insider threat detection.
#
# WHY THIS IS NOT A THIN WRAPPER OVER sklearn.metrics
# ---------------------------------------------------
# The reasoning that a reader cannot get from the code itself.
##
```

Public functions and classes use the same tags in their docstrings.

```python
def flags_for_budget(scores: np.ndarray, budget: float,
                     random_state: int = 42) -> tuple[np.ndarray, dict]:
    """
    @brief Select the alert budget's worth of rows and describe the cut.

    @param scores        Risk scores, higher = more anomalous.
    @param budget        Fraction of rows to flag, in (0, 1].
    @param random_state  Seed for tie-breaking.
    @return              (boolean mask, diagnostics about the cut score).
    """
```

Comments document the code, not the task that produced it.

- Write a comment only for a non-obvious reason, a constraint or a trap, and use comments as section separators in long files.
- Do not narrate what the next line does, restate the code, or teach the language.
- Before keeping a comment, ask whether it would still be true if only the code existed. If it describes a conversation, a review or a change ("now uses", "fixed"), it belongs in the commit message instead.
- Review notes and to-do markers never reach the main branches.

## Correctness rules

These rules exist because each one was broken once and produced a number that looked like a result and was not.

1. **Never sample the test split.** Metrics are computed on every test row. A sample of a few thousand rows holds no malicious day at all.
2. **Detectors return scores, never decisions.** A detector is fitted on the training split and returns one score per test row. Thresholds come only from the alert budget in `src/evaluate.py`, so every detector is judged at the same operating point.
3. **Ties are broken by a seeded permutation.** Exactly the budget is flagged, and the tie fraction is reported. Never flag every row tied at the cut.
4. **Nothing learns from the test window.** Profiles, scalers and SMOTE are fitted on training rows only, and every rolling feature is causal. A feature that reads across the split must fail a test.
5. **Every random component takes a seed, and the seed is recorded** in the run's `run.json`.
6. **Rank by the model's native score.** Display calibrations, such as the dashboard's bounded score, never feed a ranking.
7. **One code path per idea.** The population, role and personal references are one implementation with a `scope` parameter, so comparing them changes nothing else.
8. **Results are generated, never edited.** `results/` is written only by `script/experiment.py` and `script/studies.py`, and the report's figures only by `report/make_figures.py`. Every number quoted in the report or the docs must trace to a file in `results/` or to a recorded run.

## Tests

- Tests use pytest and live in `tests/`. Tests that read the raw logs are marked `@pytest.mark.slow`.
- Run `pytest -m "not slow"` before every commit. It needs no CERT data and finishes in about a minute. The slow tests need the extracted logs.
- Every bug fix comes with a regression test that fails without the fix. Most of the existing tests pin exactly such a failure, from clipped scores to answer-key replays that shift columns.
- Test behaviour through the public functions, on small hand-made frames whose right answer can be worked out by hand.

## The dashboard

- Colours come from `src/colors.py` and mean the same thing everywhere. Blue is the per-user framework, red the Random Forest and amber a malicious day, in the dashboard, the report figures and the presentation.
- Expensive work is cached with Streamlit's cache decorators, so moving between pages does not recompute it.
- The interactive pages use the models saved by `train.py`. The Findings page only reads `results/` and never reruns an experiment.

## Git, branches and commits

- `master` holds reviewed releases and `develop` collects finished work. Branch from `develop` as `feat/...`, `fix/...` or `docs/...`, open a pull request into `develop`, and merge `develop` into `master` after review.
- Commit subjects use a conventional prefix, `feat:`, `fix:`, `docs:`, `chore:`, `test:` or `refactor:`, then a short imperative description in lower case, for example `feat: restyle the dashboard to match the presentation`.
- Pull request descriptions are short, a one-line summary and a few bullets of changes.
- Never commit data, trained models, archives, private documents or credentials. `data/raw/`, `data/artifacts/`, the dataset archives and the PDFs in `docs/` are gitignored, and deployment credentials stay in your AWS CLI configuration.

## Documentation

- Guides live in `docs/` as Markdown, linked from the README's documentation table.
- Prose uses plain punctuation, commas and full stops, with no dash punctuation, and British spelling as in the report (behaviour, normalise).
- Numbers in the docs follow the same rule as the report. They come from `results/` or a recorded run, never from memory.
