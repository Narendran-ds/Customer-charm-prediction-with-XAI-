# ISSUES.md

Full issue backlog from the 2026-09-04 audit (3 parallel reviews: `src/` + `run_pipeline/`,
`app/app.py`, repo/setup hygiene). Grouped by root cause. Check items off as they're fixed and
re-verified — see `CLAUDE.md` for the working conventions this backlog assumes.

Severity: **Blocking** (pipeline/app cannot run) · **High** (silent wrong behavior / real bug) ·
**Medium** (correctness or robustness gap) · **Low** (style/hygiene)

**Status as of 2026-09-04: all Blocking/High items and most Medium/Low items fixed and verified**
(pipeline run end-to-end on a rebuilt venv, 11 pytest tests passing, Streamlit app manually
exercised in-browser for both the manual form and batch upload, including error paths). See
`CLAUDE.md` change log for what changed. Items still open are marked below with why they were
left for a later pass.

---

## RC1 — Feature encoding duplicated 3x, no persisted schema

**Fixed**: added `src/features.py` as the single canonical `build_features`/schema module, used by
training, the app's batch mode, and single-row manual prediction. `train_model.py` now persists
`models/feature_schema.json` (column list + training medians/modes). A real regression this
surfaced during browser testing — `build_features` on a single manually-entered row would see only
one category per field and silently drop the dummy column under `drop_first=True` — was caught and
fixed by encoding each categorical column against its known fixed domain
(`features.CATEGORICAL_DOMAINS`) before one-hot encoding, and is covered by a regression test
(`tests/test_features.py::test_build_features_single_row_encodes_full_category_domain`).

- [x] **High** `app/app.py:25-26` — batch mode never coerced `TotalCharges` to numeric before
  `pd.get_dummies`. Fixed: `build_features` coerces `tenure`/`MonthlyCharges`/`TotalCharges` to
  numeric first (generalized past just `TotalCharges` after browser testing found the same bug
  could hit `MonthlyCharges`).
- [x] **High** `app/app.py:25-26` — unseen categorical values were silently dropped. Fixed:
  `features.unseen_columns` surfaces them and the app shows an `st.warning` before aligning.
- [x] **Medium** `app/app.py:26` vs `src/data_preprocessing.py:27` — `drop_first` mismatch. Fixed:
  both now call the same `build_features(df, drop_first=True)`.
- [x] **High** `src/utils.py:33-41` (`preprocess_single_input`) — missing features defaulted to
  raw `0`. Fixed: goes through `align_to_schema`, which fills from persisted training medians/modes.
- [x] **Medium** `src/explainability.py:26` — trusted the CSV's column order. Fixed: now loads the
  persisted held-out test set by index and transforms it directly (schema drift would now surface
  as a scaler shape-mismatch error rather than silently misaligning).
- [x] **Medium** `src/data_preprocessing.py:27` — no persisted encoder/schema. Fixed: see above.

---

## RC2 — Import-time side effects / duplicated EDA code

- [x] **Blocking** `src/data_preprocessing.py:48-142` — duplicated `eda.py` body removed entirely.
- [x] **High** `src/eda.py` — wrapped in `main()` behind `if __name__ == "__main__":`.
- [x] **Medium** `src/train_model.py` — wrapped in `main()` behind `if __name__ == "__main__":`.
- [ ] **Low** `src/eda.py` vs `src/data_preprocessing.py` still independently clean the raw data
  (EDA still doesn't drop duplicates/`customerID` before analysis). Left as-is this pass — EDA is
  meant to look at the full raw dataset, and unifying it with training's cleaning would couple two
  scripts with different purposes. Revisit only if the two ever disagree on `Churn`/`TotalCharges`
  handling specifically.

---

## RC3 — Broken/non-portable environment & repo hygiene

- [x] **Blocking** `.venv/` — rebuilt from scratch with `py -3.11 -m venv .venv` + `pip install -r
  requirements.txt` on this machine; old broken venv removed only after the new one was verified.
- [x] **Blocking** `run_pipeline/run_pipeline.py:20-21` — now resolves `.venv/Scripts/python.exe`
  if present, else falls back to `sys.executable`; pipeline steps invoked as `-m src.<module>`.
- [x] **High** `run_pipeline/run_pipeline.py:8-16` — `subprocess.run` now wrapped in try/except.
- [x] **Blocking** `.gitignore` added (`.venv/`, `__pycache__/`, `*.pyc`, editor/OS cruft) before
  git was touched.
- [x] **Blocking** `requirements.txt` — re-saved as plain UTF-8 (also added `pytest` for the new
  test suite).
- [x] **Medium** README step 2 now says `.venv` consistently with what the scripts expect.
- [x] **Medium** `app/app.py:17` — model/scaler/schema paths now resolved from `BASE_DIR`
  (repo root), independent of the process's current working directory.
- [x] **Low** Broken `reports/shap_force_example.png` reference replaced with the real
  `explainability_reports/shap_summary.png`.
- [x] **Low** `LICENSE` (MIT) added to match the README's existing claim.
- [x] **Low** README's usage section rewritten to match the actual Streamlit UI (sidebar
  uploader + nav radio, reactive updates, no Predict button).
- [ ] **Low** `requirements.txt` is still an uncurated `pip freeze`-style list (transitive deps
  mixed with direct ones) — tracked in Stretch below; re-pinning risks breaking the exact-version
  reproducibility the audit relied on, so deferred rather than done opportunistically.

---

## RC4 — Weak error handling / silent failure discipline

- [x] **High** `src/explainability.py` bare `except:` (x2) — removed; replaced the whole
  TreeExplainer/generic-Explainer branch with a single `shap.Explainer(...)` callable-API call.
- [x] **High** `src/explainability.py:46-48` — dead waterfall fallback removed (see above fix).
- [x] **Medium** `src/explainability.py:30,32` — no longer applicable; the callable `Explainer` API
  returns a consistent `Explanation` object instead of a raw, possibly-3D `shap_values` array.
- [x] **Medium** `src/explainability.py:43` — `idx` now clamped via `min(idx, len(X) - 1)`, with a
  `ValueError` if the test set is empty.
- [x] **Low** `src/explainability.py:70` — `webbrowser.open` now opt-in via `--open-browser`.
- [x] **High** `src/utils.py:43-49` (`save_object`) — now raises on failure instead of swallowing.
- [ ] **Medium** `pd.read_excel`/`pd.read_csv`/`pickle.load` in the CLI pipeline scripts
  (`data_preprocessing.py`, `eda.py`, `train_model.py`, `explainability.py`) still raise raw
  tracebacks on I/O failure. Left as-is: these are developer-run CLI scripts, not end-user-facing —
  a Python traceback naming the missing file is acceptable there. The user-facing surface
  (`app/app.py`) was fixed (see below).
- [x] **Medium** `app/app.py` CSV validation — missing required columns and empty uploads now show
  a friendly `st.error`/`st.warning` instead of crashing; the whole batch-prediction path is
  wrapped in try/except. Browser-testing this surfaced a real related bug (a malformed
  `MonthlyCharges`/`tenure` value crashed the KPI calculation with a raw `TypeError` traceback) —
  fixed by coercing and dropping invalid rows with a warning before computing KPIs.

---

## Other issues (not part of a shared root cause)

- [ ] **Medium** `src/data_preprocessing.py:17-18` — `TotalCharges` median is still imputed before
  the train/test split exists (structural: the split only happens later, from the saved CSV, in
  `train_model.py`). Left open: fixing this properly means restructuring the pipeline so imputation
  fits only on the training fold, which is a bigger structural change than this pass's scope for an
  ~11-row effect. Worth a dedicated follow-up if the dataset changes meaningfully.
- [x] **Medium** `src/eda.py` VIF — now computed with `statsmodels.tools.add_constant`.
- [x] **Medium** `src/eda.py` `mutual_info_classif` — `random_state=42` added.
- [x] **Low** `src/eda.py` Cramér's V double loop — now iterates unique pairs only.
- [x] **Low** `src/train_model.py` `LogisticRegression` — `random_state=42` added.
- [x] **Low** `src/train_model.py` `XGBClassifier(use_label_encoder=False, ...)` — removed.
- [x] **Medium** `src/train_model.py` no cross-validation — 5-fold `StratifiedKFold` +
  `cross_val_score` now drives model selection; the held-out test set is used once, only to report.
- [x] **Medium** `src/train_model.py` test split not persisted — `models/test_indices.json` added.
- [x] **Low** `src/train_model.py` results round-trip — simplified in `select_best_model`.
- [x] **Low** `src/__init__.py` import-time `logging.basicConfig()` — removed.
- [x] **Low** Emoji-prefixed prints — removed from every file touched this pass.
- [ ] **Low** `BASE_DIR` path-resolution boilerplate is still repeated per-file (now consistent, via
  `sys.path.insert(0, BASE_DIR)` + `from src.x import y`, rather than centralized into one helper).
  Left as-is: each script needing a different subset of imports made a single shared helper more
  awkward than it's worth for four call sites.
- [x] **Low** `app/app.py` manual-entry form — now exposes the full raw feature set via selectboxes
  built from `features.CATEGORICAL_DOMAINS`, instead of hardcoding ~28 of ~30 features to `0`.
- [x] **Low** `app/app.py` threshold/risk-bucket inconsistency — `RISK_BINS` now ends its "Medium"
  bucket exactly at `CHURN_THRESHOLD`, so "High" risk and a "Churn" prediction always agree.
- [x] **Low** `app/app.py` unused `pickle` import — removed.
- [ ] **Low** `app/app.py` SHAP explainer caching (`@st.cache_resource`) — deliberately **not**
  added: the explainer's background data is a numpy array that differs per upload/manual entry, and
  `st.cache_resource`'s default hashing of that array would either fail or defeat the point of
  caching. Wrapped construction in try/except for robustness instead; caching is a genuine
  nice-to-have, not fixed this pass.
- [x] **Low** No tests / no LICENSE — 11 pytest tests added under `tests/` (including the
  single-row-encoding regression test) and `LICENSE` added. CI and packaging remain in Stretch.

---

## Stretch / not this pass

Explicitly out of scope for the current fix-the-project pass — tracked here so they aren't
silently dropped, not because they don't matter:

- [ ] Packaging: `pyproject.toml` / `setup.py` so `src` is pip-installable instead of relying on
  cwd-relative imports.
- [ ] CI: GitHub Actions workflow running lint + pytest on push/PR.
- [ ] Pre-commit hooks (formatting, lint).
- [ ] `requirements.txt` curation: trim to direct dependencies only, let pip resolve transitives.
- [ ] Broader test coverage beyond the pure-function pieces (e.g. model-quality regression tests,
  Streamlit UI tests).
- [ ] Fully centralize `BASE_DIR`/path-resolution boilerplate into a shared helper.
- [ ] Restructure the pipeline so `TotalCharges` imputation fits only on the training fold.
