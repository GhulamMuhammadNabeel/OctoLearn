# OctoLearn Architecture Guide

Welcome to the complete architectural reference for **OctoLearn** — an enterprise-grade AutoML library built for transparency, robustness, and ease of use. This document explains *how* the library is built, *why* specific design choices were made, and *how to extend* it.

---

## 1. System Overview

OctoLearn follows a **Pipeline Orchestration** pattern. The central `AutoML` class acts as the conductor, coordinating specialized workers (Profiler, Cleaner, Trainer, ReportGenerator, etc.) to transform raw data into a production-ready model and a comprehensive PDF report.

### High-Level Data Flow

```
Raw DataFrame (X, y)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AutoML Orchestrator (core.py)                │
│                                                                 │
│  1. Validate Inputs                                             │
│  2. Sample (if large dataset)                                   │
│  3. Profile Raw Data ──────────────────► DatasetProfile (raw)  │
│  4. Train/Test Split (stratified)                               │
│  5. Clean Train Data ──────────────────► DatasetProfile (clean) │
│  6. Transform Test Data (no leakage)                            │
│  7. Feature Engineering                                         │
│  8. Train Models + Optuna Tuning ──────► Best Model             │
│  9. Generate PDF Report ───────────────► octolearn_report.pdf   │
└─────────────────────────────────────────────────────────────────┘
```

### Why a Pipeline Orchestrator?

The orchestrator pattern keeps each component **single-responsibility** and **independently testable**. You can swap out the cleaner, trainer, or report generator without touching the others. It also makes the execution order explicit and auditable.

---

## 2. Directory Structure

```
OctoLearn/
├── octolearn/
│   ├── core.py                    # AutoML orchestrator + config dataclasses
│   ├── config.py                  # Global constants (Optuna, model registry)
│   ├── profiling/
│   │   └── data_profiler.py       # Statistical analysis → DatasetProfile
│   ├── preprocessing/
│   │   └── auto_cleaner.py        # Imputation, encoding, scaling
│   ├── models/
│   │   ├── model_trainer.py       # Multi-model training + Optuna
│   │   └── registry.py            # Model versioning and persistence
│   ├── experiments/
│   │   ├── report_generator.py    # PDF report (ReportLab)
│   │   └── plot_generator.py      # matplotlib/seaborn visualizations
│   ├── evaluation/
│   │   └── metrics.py             # Scoring functions
│   ├── utils/
│   │   └── helpers.py             # Logging, utilities
│   ├── fonts/                     # ShantellSans TTF font files
│   └── images/                    # logo.png
├── tests/
│   └── test_complete_pipeline.py  # Integration test suite (16 tests)
├── ARCHITECTURE.md                # This file
└── README.md                      # User-facing documentation
```

---

## 3. Configuration System (`core.py`)

### Design: Dataclasses over kwargs

OctoLearn uses Python `@dataclass` objects instead of a flat list of keyword arguments. This provides:

- **Type safety**: IDE autocomplete and type checkers work correctly
- **Grouping**: Related settings are co-located (e.g., all Optuna settings in `OptimizationConfig`)
- **Defaults**: Each field has a sensible default, so `AutoML()` works out of the box
- **Discoverability**: Users can explore configs with `help(OptimizationConfig)`

### Config Objects

| Class | Key Fields | Rationale |
|-------|-----------|-----------|
| `DataConfig` | `sample_size=5000`, `test_size=0.2`, `stratify_target=True` | Sampling prevents OOM on large datasets; 20% test is standard |
| `ProfilingConfig` | `detect_outliers=True`, `analyze_interactions=True` | Both are expensive; can be disabled for speed |
| `PreprocessingConfig` | `imputer_strategy`, `scaler='standard'`, `encoder_strategy` | Sensible defaults; user can override per-column |
| `ModelingConfig` | `n_models=5`, `models_to_train=None`, `evaluation_metric=None` | Auto-selects best models; metric auto-detected from task type |
| `OptimizationConfig` | `optuna_trials_per_model=20`, `optuna_timeout_seconds=300` | 20 trials is a good speed/quality tradeoff; timeout prevents runaway |
| `ReportingConfig` | `report_detail='detailed'`, `visuals_limit=10` | Detailed by default; brief for quick runs |
| `ParallelConfig` | `n_jobs=1`, `backend='loky'` | n_jobs=1 prevents Windows CPU oversubscription with Optuna |

### Why `n_jobs=1` for Optuna on Windows?

Optuna's multiprocessing backend conflicts with Windows' process spawning model. Setting `n_jobs=-1` causes process pool exhaustion and crashes on Windows. The fix is `n_jobs=1` (sequential Optuna trials), which is stable on all platforms. Linux/Mac users can override this via `ParallelConfig(n_jobs=-1)`.

### The `fit()` Override Pattern

`fit()` accepts optional keyword arguments that temporarily override config values for a single run:

```python
automl.fit(X, y, optuna_trials=5, use_optuna=False)
```

Internally, the original config values are snapshotted, overrides applied, pipeline executed, then originals restored in a `finally` block — so the `AutoML` instance is never permanently mutated by `fit()` kwargs.

---

## 4. Data Profiling (`profiling/data_profiler.py`)

### Purpose

Profile the data **before** any cleaning to capture the "ground truth" state. This raw profile is used for:
- Generating the "Before" side of the Before/After comparison in the report
- Determining which columns are categorical (needed to choose the right encoder)
- Computing the Risk Score

### Output: `DatasetProfile`

A lightweight dataclass (not a copy of the data) containing:
- `n_rows`, `n_columns`, `task_type`
- `numeric_columns`, `categorical_columns`
- `missing_ratio` (per-column dict)
- `duplicate_rows`, `skewed_features`

### Why profile before cleaning?

If we profiled after cleaning, we'd lose the "before" state needed for the transformation journey report. We'd also lose the original column types (encoding changes categorical → numeric).

---

## 5. Data Cleaning (`preprocessing/auto_cleaner.py`)

### The Leakage Prevention Rule

**`fit_transform` on Train only. `transform` on Test.**

This is the most critical rule. If you compute the column mean using the full dataset and use it to fill NaNs, your model indirectly "sees" test data during training (data leakage). OctoLearn enforces this by:

1. Calling `cleaner_.fit_transform(X_train, y_train)` — learns imputation stats from train only
2. Calling `cleaner_.transform(X_test)` — applies the same learned stats to test

### Cleaning Steps (in order)

1. **ID column removal**: Columns with near-unique values (e.g., user IDs) are dropped — they have no predictive value and cause overfitting
2. **Constant column removal**: Zero-variance columns are dropped
3. **Duplicate row removal**: Done on train set only (after split)
4. **Numeric imputation**: `mean` (default) or `median` (more robust to outliers)
5. **Categorical imputation**: `mode` (most frequent value)
6. **Rare category encoding**: Categories appearing < 5% of the time are grouped into "Other" to prevent high-cardinality explosions
7. **One-hot encoding**: Converts categorical columns to numeric (required by most sklearn models)
8. **Scaling**: `StandardScaler` (default) — zero mean, unit variance. Alternatives: `RobustScaler` (outlier-resistant), `MinMaxScaler`

### Why StandardScaler by default?

Most linear models (Logistic Regression, SVM) require scaled features. Tree-based models (Random Forest, XGBoost) don't need scaling but aren't harmed by it. StandardScaler is the safest universal default.

### Imputation Tracking

`auto_cleaner.py` tracks how many columns had missing values imputed and exposes this in `cleaning_log['missing_imputed']`. This feeds into the report's "What OctoLearn Did" section.

---

## 6. Model Training (`models/model_trainer.py`)

### Supported Models

**Classification:**
- `logistic_regression` (sklearn)
- `random_forest` (sklearn)
- `gradient_boosting` (sklearn)
- `xgboost` (xgboost)
- `lightgbm` (lightgbm)
- `svm` (sklearn)

**Regression:**
- `linear_regression` (sklearn)
- `random_forest` (sklearn)
- `gradient_boosting` (sklearn)
- `xgboost` (xgboost)
- `lightgbm` (lightgbm)
- `svr` (sklearn)

### Hyperparameter Optimization: Why Optuna?

| Method | Pros | Cons |
|--------|------|------|
| Grid Search | Exhaustive | Exponential time complexity |
| Random Search | Fast | Misses good regions |
| **Optuna (Bayesian)** | Smart, adaptive | Requires more code |

Optuna uses **Tree-structured Parzen Estimators (TPE)** — a Bayesian method that builds a probabilistic model of which hyperparameter regions give good results, then samples from those regions. This finds better hyperparameters in fewer trials than random search.

### Default Hyperparameter Search Spaces

Defined in `config.py` under `OPTUNA_CONFIG['search_spaces']`. Each model has a search space dict mapping parameter names to ranges. The `_optimize_hyperparameters` method in `ModelTrainer` reads these and passes them to Optuna's `trial.suggest_*` API.

### Why `n_trials=20` default?

20 trials is a practical sweet spot:
- Enough for Optuna to explore the search space meaningfully
- Fast enough for interactive use (< 2 minutes per model on typical hardware)
- Users can increase to 50-100 for production runs via `OptimizationConfig(optuna_trials_per_model=50)` or `fit(X, y, optuna_trials=50)`

### Model Selection

After all models are trained, `ModelTrainer` ranks them by the primary metric (F1 for classification, RMSE for regression) on the held-out test set. The best model is stored as `automl.best_model_`.

---

## 7. Report Generation (`experiments/report_generator.py`)

### Technology: ReportLab

ReportLab is a Python library for programmatic PDF generation. Unlike HTML-to-PDF converters (WeasyPrint, wkhtmltopdf), ReportLab:
- Has no external binary dependencies
- Gives pixel-perfect layout control
- Works identically on Windows, Mac, and Linux
- Supports custom fonts (TTF), vector graphics, and canvas-level drawing

### Font: ShantellSans

ShantellSans is a humanist sans-serif font with a friendly, approachable character — appropriate for a data science tool that aims to be accessible. Four weights are registered:
- `ShantellSans-Regular` → body text
- `ShantellSans-Bold` → table headers, emphasis
- `ShantellSans-ExtraBold` → section titles, cover title
- `ShantellSans-Italic` → narrative callouts

Fallback: Helvetica (built into all PDF viewers) if font files are missing.

### Report Structure

| Section | Content |
|---------|---------|
| Cover Page | Logo watermark, title, metadata table, tagline |
| Recommendations | Priority-grouped actions (Critical/High/Medium/Low) |
| Data Story | Narrative introduction to the dataset |
| Data Health Dashboard | Risk score gauge, metric cards, risk factors table |
| Data Transformation Journey | Before/After comparison table + distribution histograms |
| Feature Intelligence | Importance ranking table + horizontal bar chart |
| Model Arena | Champion card + benchmark table |
| Visual Insights | Correlation heatmap + feature distribution plots |
| Advanced Analysis | Outlier detection results, feature interactions |

### Logo Watermark

The cover page uses `canvas.drawImage()` with `setFillAlpha(0.06)` to draw the logo as a large, nearly-invisible background watermark. This is done in the `onFirstPage` callback so it only appears on the cover — not on content pages.

### Before/After Distribution Plots

`_generate_before_after_plots()` generates side-by-side histograms comparing raw vs. cleaned distributions for the top 4 numeric features. These are saved as temp PNG files, embedded in the PDF, then deleted in the `finally` block of `generate()`.

### Temp File Cleanup

All matplotlib figures are saved to `tempfile.NamedTemporaryFile` paths, tracked in `self._temp_files`, and deleted in `generate()`'s `finally` block — ensuring no temp files are left behind even if PDF generation fails.

---

## 8. Stratification Logic

For classification tasks, train/test splitting should be stratified (same class distribution in both sets). The condition in `_split_data`:

```python
if y.dtype.kind in ('O', 'U', 'S') or y.dtype.name == 'category' or (
    y.dtype.kind in ('i', 'u') and n_unique < 20
):
    stratify = y
```

This correctly handles:
- String labels (`dtype.kind == 'O'`)
- Integer labels like 0/1 (`dtype.kind in ('i', 'u')`) — the previous bug was `isinstance(y.dtype, type)` which always returned False for integer dtypes

---

## 9. How to Build OctoLearn from Source

### Prerequisites

```bash
python >= 3.8
pip install -r requirements.txt
```

Key dependencies:
- `pandas`, `numpy`, `scikit-learn` — core data science stack
- `xgboost`, `lightgbm` — gradient boosting models
- `optuna` — hyperparameter optimization
- `reportlab` — PDF generation
- `matplotlib`, `seaborn` — visualizations
- `Pillow` — image processing for report

### Running Tests

```bash
python -m pytest tests/test_complete_pipeline.py -v
```

All 16 tests should pass. Tests cover:
- Config validation
- Data splitting and stratification
- Cleaning pipeline (imputation, encoding, scaling)
- Model training (classification + regression)
- Report generation
- fit() API override params

### Generating a Report

```bash
python generate_final_report.py
```

---

## 10. How to Extend OctoLearn

### Add a New Model

1. Open `octolearn/models/model_trainer.py`
2. Add the model class to the `_get_model_instance()` method
3. Add its hyperparameter search space to `config.py` under `OPTUNA_CONFIG['search_spaces']`
4. Add it to the `MODEL_REGISTRY` in `config.py`

### Add a New Report Section

1. Open `octolearn/experiments/report_generator.py`
2. Add a new `_add_my_section(self, story)` method
3. Call it from `generate()` in the appropriate order
4. Use `self._add_section_header(story, "My Section")` for consistent styling
5. Use `self.styles['Narrative']` for body text, `self.styles['SubsectionHeading']` for sub-headers

### Add a New Preprocessing Step

1. Open `octolearn/preprocessing/auto_cleaner.py`
2. Add a new sklearn-compatible transformer class (with `fit` and `transform` methods)
3. Add it to the pipeline in `fit_transform()`
4. Track any relevant stats in `self.cleaning_log`

### Add a New Metric

1. Open `octolearn/evaluation/metrics.py`
2. Add the metric function
3. Register it in the `METRIC_REGISTRY` dict
4. Add it as a valid option in `ModelingConfig.evaluation_metric` docstring

---

## 11. Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Dataclass configs | Type-safe, discoverable, clean API |
| fit() override params | Experiment without re-creating AutoML instance |
| n_jobs=1 for Optuna | Windows stability; avoids process pool exhaustion |
| n_trials=20 default | Speed/quality tradeoff for interactive use |
| ReportLab for PDF | No external binaries, pixel-perfect control |
| ShantellSans font | Humanist, approachable, brand-consistent |
| Temp files for plots | No leftover files even on failure |
| fit_transform on train only | Prevents data leakage |
| Stratify integer targets | Correct class balance for 0/1 classification |
| Snapshot/restore in fit() | Config objects never permanently mutated by kwargs |

---

*OctoLearn Architecture v0.8.0 — Updated 2026-02-18*
