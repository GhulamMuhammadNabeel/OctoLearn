# OctoLearn User Guide

Welcome to the comprehensive guide for OctoLearn. This document provides in-depth information on how to configure and extend the AutoML pipeline.


---

## Core Concepts

OctoLearn is built on the principle of **Transparent AutoML**. Unlike other libraries that hide the "magic" inside a black box, OctoLearn exposes the results of every stage—from raw data profiling to final model staging.

### The fit-then-report Pattern
Success with OctoLearn follows a simple pattern:

1. **`fit()`**: Orchestrates the technical pipeline (cleaning, tuning, training).
2. **`generate_report()`**: Decodes the technical results into human-readable business intelligence.

!!! tip "Remember the Report"
    The real power of OctoLearn isn't just the model—it's the PDF report. Always generate a report before deploying!

---

## Configuration Reference

You can pass specific configuration objects to the `AutoML` constructor to control behavior.

### DataConfig
Controls how data is sampled and split.
- `use_full_data`: If True, bypasses sampling.
- `test_size`: Fraction of data held out for validation (default 0.2).
- `random_state`: Integer seed for reproducibility.

### PreprocessingConfig
Controls the automated data cleaning engine.
- `imputer_strategy`: `{'numeric': 'median', 'categorical': 'mode'}`.
- `scaler`: `'standard'`, `'robust'`, or `'minmax'`.

---

## Advanced Features

### Class Imbalance Handling
OctoLearn automatically detects class imbalance during profiling and utilizes stratified splitting to ensure stable evaluation metrics.

### Target Leakage Detection
The `DataProfiler` looks for features that are essentially identical to or highly correlated with the target, flagging them as potential "leakage suspects" in the risk report.

---

## Advanced Tutorial: Deep Configuration
OctoLearn shines when you take fine-grained control over its internal components. Let's look at how to tune the Optuna integration for maximum performance using the `OptimizationConfig` class.

### Tuning the Bayesian Search
By default, OctoLearn employs Tree-structured Parzen Estimator (TPE) via Optuna. You can override the trial count and timeout settings to force a deep search:

```python
from octolearn import AutoML, OptimizationConfig

deep_tuner = OptimizationConfig(
    use_optuna=True,
    optuna_trials_per_model=100,      # Run many more trials
    optuna_timeout_seconds=3600,      # Give each model an hour
    optuna_parallel_jobs=-1,          # Utilize all CPU cores safely
    use_registry=True                 # Remember all checkpoints
)

# Pass the custom tuning behavior to the orchestrator
automl = AutoML(optimization_config=deep_tuner)
automl.fit(X, y)
```

## Methodology

### Stacking Ensembles
When multiple models are selected for training, OctoLearn generates a Stacking Regressor or Classifier by using the top-performing base models and a meta-model to blend their predictions.

---

## Troubleshooting

### PDF Generation Fails
!!! failure "Missing Fonts or Dependencies"
    Ensure you have `reportlab` installed. If your environment lacks specific fonts, OctoLearn will automatically fallback to standard Helvetica, but if `reportlab` itself is missing, the CLI will throw an `ImportError`.

### Optuna is Too Slow
!!! warning "Bayesian Search Latencies"
    Bayesian Optimization takes time to build its probabilistic model of the parameter space. If it's running too slow for your dev cycle, reduce `optuna_trials_per_model` or temporarily disable it by passing `use_optuna=False` to your `fit()` call.

---

## Inside the OctoLearn Engine: Example Flow

To understand what happens to your data at each micro-step, let's trace a standard run with a 1,000-row dataset containing missing values, categorical strings, and ID columns.

### Phase 1: Ingestion & Profiling
**Time: ~0.5s**
- **Input**: 1,000 rows × 12 raw columns.
- **Action**: Inferred types (4 numeric, 5 categorical, 1 date, 1 ID). Detected 10% missingness in 'salary'.
- **Result**: `automl.raw_profile_` object populated.

### Phase 2: Structural Cleaning
**Time: ~0.2s**
- **Action**: Dropped `id_col` (ID-like) and `constant_col` (Zero variance).
- **Transformation**: Data reduced to 10 columns. Missing values highlighted for the next phase.

### Phase 3: Automated Feature Engineering
**Time: ~1.2s**
- **Action**: Extracted `Day`, `Month`, `Year` from `last_login`.
- **Interactions**: Multiplied `age` × `tenure` to create high-signal interactions.
- **Outliers**: Identified 12 extreme outliers via Isolation Forest.

### Phase 4: Preprocessing & Encoding
**Time: ~0.8s**
- **Action**: Median imputation for `salary`, Mode imputation for `gender`.
- **Scaling**: RobustScaler applied to `income` to handle skewness.
- **Encoding**: One-Hot Encoding for `city` (5 columns) and Ordinal for `membership_level`.

### Phase 5: The Model Arena (Bayesian HPO)
**Time: ~45s (Configurable)**
- **Action**: 20 Optuna trials per model (XGBoost, Random Forest).
- **Ensemble**: Stacking the top 3 base models using a Ridge Logistic Meta-Regressor.

### Phase 6: Intelligence Reporting
**Time: ~2s**
- **Action**: Compiled 12 visuals and 15 metrics into a professional PDF.
- **Output**: `octolearn_report_2024.pdf` available in `artifacts/`.

---

## Expected CLI Output Example
When running OctoLearn, you will see a detailed execution log similar to this:

```text
2024-02-24 17:10:00 - octolearn.core - INFO - AutoML initialized (v0.9.0)
2024-02-24 17:10:00 - octolearn.core - INFO - [PHASE 1] Initializing Profiler...
2024-02-24 17:10:01 - octolearn.profiling - INFO - Profiling complete. Quality Score: 84.5
2024-02-24 17:10:01 - octolearn.preprocessing - INFO - Removed ID columns: ['user_id']
2024-02-24 17:10:01 - octolearn.preprocessing - INFO - Imputed 142 missing values in 3 columns.
2024-02-24 17:10:02 - octolearn.models - INFO - Starting Optuna Optimization (20 trials)
2024-02-24 17:10:45 - octolearn.models - INFO - Best Model found: XGBoost (F1: 0.923)
2024-02-24 17:10:47 - octolearn.core - INFO - Pipeline complete! [OK]
```
