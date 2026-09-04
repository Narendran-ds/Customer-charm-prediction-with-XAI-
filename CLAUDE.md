# CLAUDE.md

Guidance for Claude Code (and future you) working in this repository.

## What this project is

A Telco customer-churn classifier (LogisticRegression / RandomForest / XGBoost) with SHAP + LIME
explainability and a Streamlit demo app. Originally generated with ChatGPT as a first project;
currently being audited end-to-end for correctness, robustness, and hygiene issues (see `ISSUES.md`).

## Layout

- `data/` — raw Telco xlsx + cleaned CSV produced by `src/data_preprocessing.py`
- `src/` — pipeline code: `data_preprocessing.py`, `eda.py`, `train_model.py`, `explainability.py`, `utils.py`
- `run_pipeline/run_pipeline.py` — orchestrates the full pipeline + launches the app
- `app/app.py` — Streamlit demo (batch CSV upload + manual single-customer form)
- `models/` — pickled trained model + scaler (+ feature schema, once added)
- `reports/` / `explainability_reports/` — generated EDA plots and SHAP/LIME output
- `tests/` — pytest coverage for the pure-function pieces of the pipeline

## Working conventions for this project

- Run pipeline stages with `python -m src.<module>` (or `python src/<module>.py`) from the repo
  root, not from inside `src/` — several scripts resolve paths relative to the repo root.
- Every pipeline module must be `if __name__ == "__main__":` guarded — importing a module must
  never trigger disk I/O or plot generation as a side effect (this was previously violated).
- There is exactly one place that turns a raw customer row into model features — do not
  reimplement `pd.get_dummies`/reindex logic in a new call site. Reuse the canonical feature
  builder + persisted schema (see `ISSUES.md` root cause 1).
- Regenerating `models/*.pkl` invalidates `reports/*.png` and `explainability_reports/*` —
  re-run the full pipeline after any change to preprocessing, training, or the train/test split,
  don't hand-patch downstream artifacts.
- `.venv/` is local-machine-specific and must never be committed (see `.gitignore`). If the venv
  ever breaks, rebuild with `py -3.11 -m venv .venv && pip install -r requirements.txt`.

## Change log

- 2026-09-04 — Full audit completed (3 parallel Explore agents over `src/`, `app/app.py`, and
  repo hygiene). ~25 concrete issues found and filed in `ISSUES.md`. Plan approved; fix-test loop
  starting now. Confirmed Python 3.11 available locally (`py -3.11`) to rebuild the broken
  committed `.venv`.
- 2026-09-04 — Fix-test loop completed. Highlights (full detail in `ISSUES.md`):
  - Rebuilt `.venv` from scratch (old one pointed at a different machine/user and couldn't run);
    fixed `requirements.txt`'s UTF-16 encoding that broke `pip install`.
  - Added `src/features.py` as the single canonical feature-encoding + schema module, replacing
    three independent `pd.get_dummies` implementations across training/app/manual-input. Caught
    and fixed a real single-row-encoding bug this surfaced (manually-entered categorical values
    were silently discarded) via a regression test.
  - Removed a duplicated `eda.py` block accidentally pasted into `data_preprocessing.py`; added
    `__main__` guards project-wide so importing a pipeline module has no side effects.
  - `train_model.py` now uses cross-validated model selection, persists the held-out test set and
    feature schema, and drops a removed/deprecated XGBoost kwarg.
  - `explainability.py` now explains only the held-out test set and uses a single unified SHAP
    API call instead of a bare-except fallback that (per the installed shap version) could never
    actually succeed.
  - Rewrote `app/app.py`: fixed path resolution, added upload validation (missing columns, empty
    file, malformed numeric values) with friendly errors instead of tracebacks, and expanded the
    manual-entry form to expose the full customer profile instead of hardcoding ~28 of 30 features
    to 0.
  - Verified end-to-end: full pipeline run on the rebuilt venv, 11 new pytest tests passing, and
    the Streamlit app manually exercised in-browser (manual form, valid/invalid/empty batch CSVs).
  - Initialized git (`.gitignore` added first) so this and future changes are tracked/revertible.
  - Remaining open items (mostly deliberate scope cuts) are listed in `ISSUES.md`'s "Other issues"
    and "Stretch" sections — CI, packaging, and a couple of low-impact methodological notes.
