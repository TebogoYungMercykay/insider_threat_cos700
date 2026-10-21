# Dataset Notes, CERT r6.2 Ingestion and Ground-Truth Conversion

This note documents how `script/dataset.py` turns the official CERT r6.2 distribution into the format `src/preprocess.py` consumes, and why. It's intended as a reference for anyone auditing the pipeline or extending it to another CERT release.

## Source archives

| Archive           | Role                                                                                                                                                                                                                                                                                          |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `r6.2.tar.bz2`    | Background + malicious activity logs, indistinguishable from each other at this level: `logon.csv`, `device.csv`, `file.csv`, `http.csv`, `email.csv`, plus monthly `LDAP/*.csv` org-structure snapshots.                                                                                     |
| `answers.tar.bz2` | The ground-truth answer key, **one CSV per insider scenario**, each row a verbatim copy of a malicious log line from the files above (with the log type prefixed as the first field), e.g.: `file,{R8A0-...},01/05/2010 16:11:39,ABC0001,PC-0001,R:\keylogger.exe,File Write,True,False,...` |

## Log schema (consistent across the five log types)

Every log type places `id`, `date`, `user` as the first three data columns (after the leading `type` field in answer-key files), then type-specific columns:

| Log type | Columns (after id, date, user)                                                        |
| -------- | ------------------------------------------------------------------------------------- |
| `logon`  | `pc`, `activity` (`Logon`/`Logoff`)                                                   |
| `device` | `pc`, `file_tree`, `activity` (`Connect`/`Disconnect`)                                |
| `file`   | `pc`, `filename`, `activity`, `to_removable_media`, `from_removable_media`, `content` |
| `http`   | `pc`, `url`, `activity`, `content`                                                    |
| `email`  | `pc`, `to`, `cc`, `bcc`, `from`, `activity`, `size`, `attachments`, `content`         |

`src/preprocess.py::build_daily_features` only needs `id`/`date`/`user` plus a handful of type-specific columns to build the per-(user, day) behavioural feature table (`FEATURE_COLS`), so the ingestion step doesn't need to normalise every column. It copies the CSVs through unmodified.

## Why the answer key needs converting

`src/preprocess.py::attach_context` labels each `(user, day)` row as malicious by checking whether it falls inside a **date range** for that user (`is_malicious`), because the rest of the pipeline (feature engineering, the per-user framework, evaluation) all operate at day-granularity, not individual-event-granularity.

The official answer key, however, is **line-level**: it lists every individual malicious log line, not a start/end date per user. So `dataset.py::build_ground_truth_from_answers` does the conversion:

1. Parse every row of every `answers/*.csv` file, keeping just `(scenario, user, date)` from the first four CSV fields.
2. Group by `(scenario, user)`.
3. Collapse each group to `start_date = min(date)`, `end_date = max(date)`. The window spanning that user's malicious activity within that scenario.
4. Write `user, scenario, threat_type, start_date, end_date, n_events` to `data/raw/answers/insiders.csv`, the same simplified format the synthetic generator (`generate_synthetic`) already produces.

`threat_type` is set to `scenario_<N>` (the scenario's numeric suffix in the answer-key filename, e.g. `r6.2-3.csv` → scenario `3`). The official answer-key files also embed a short narrative label per row (e.g. `"AA Keylogger"`, `"AA Upset Peers"`) describing the scenario's story beat, useful for a human reading the raw data, but not machine-actionable at the group level this pipeline uses, so it isn't parsed out separately.

This is a **simplification**, not a loss of the primary evaluation signal: day-granularity labelling is what the framework is evaluated against throughout (Section 4.5 of the proposal, recall, subgroup FPR, F1, ROC-AUC are all computed per `(user, day)`). If line-level ground truth is needed for finer-grained follow-up work, `n_events` (how many malicious lines fall in each window) is preserved in `insiders.csv` as a starting point, and `build_ground_truth_from_answers` can be extended to also emit the individual `(user, date, scenario)` events instead of just the collapsed window.

## LDAP schema handling

The official LDAP snapshot columns are resolved by name (case-insensitive, with a small alias table. See `_LDAP_COLUMN_ALIASES` in `src/preprocess.py`) rather than assumed to be in a fixed position, because different CERT release packagings have used slightly different column names/orders (e.g. `business_unit` vs `department`, `user_id` vs `user`). Any extra columns present in the real data (`project`, `team`, `email`, etc.) are ignored. They aren't used by `build_daily_features` or the baseline models.

## High-privilege role matching

`is_high_privilege` (used for the subgroup false-positive-rate metric) is computed via a case-insensitive regex match against the `role` column (`sys.?admin|db.?admin|database.?admin|it.?admin|administrator`. See `src/preprocess.py::attach_context` and `src/evaluate.py::DEFAULT_HIGH_PRIVILEGE_PATTERN`) rather than an exact string list, since the synthetic generator's role names (`SysAdmin`, `DBAdmin`) won't necessarily match the real dataset's role-name strings verbatim.

## Extending to another release (r4.2, r5.2, ...)

The ingestion functions (`ingest_r6_archive`, `build_ground_truth_from_answers`) search recursively for the expected filenames rather than assuming a fixed archive layout, so pointing `--r6-archive` / `--answers-archive` at a different release's archives should work unchanged as long as that release ships the same five log files, an `LDAP/` snapshot folder, and an `answers/` directory of per-scenario CSVs in the same row layout. Releases with a materially different log or answer schema would need a corresponding update to `build_daily_features` and/or `build_ground_truth_from_answers`.
