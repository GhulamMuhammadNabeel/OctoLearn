<p align="center">
  <img src="octolearn/images/logo.png" alt="OctoLearn Logo" width="220"/>
</p>

<h1 align="center">OctoLearn</h1>

<p align="center">
  <strong>The Enterprise-Grade AutoML Orchestrator for Python.</strong><br>
  Deliver production-ready machine learning from messy data in minutes, not weeks.
</p>

<p align="center">
  <a href="https://github.com/GhulamMuhammadNabeel/OctoLearn/blob/main/guide.md"><strong>Explore the User Guide »</strong></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Key Capabilities</a> •
  <a href="#-installation">Installation</a> •
  <a href="guide.md">Full Documentation</a> •
  <a href="testing.md">Reliability & Testing</a>
</p>

---

OctoLearn is designed for data scientists and engineers who need more than just a "black box" model. It provides **transparent, controllable, and professional** machine learning workflows that automate the tedious parts of data science while giving you full oversight through high-fidelity intelligence reports.

## 🚀 The OctoLearn Advantage

Most AutoML libraries focus solely on leaderboard scores. OctoLearn focuses on the **entire lifecycle**:

1.  **Observability**: Detailed profiling and risk scoring tell you *why* your data might fail.
2.  **Transparency**: The "Data Journey" in our reports shows exactly how every feature was transformed.
3.  **Communication**: Stakeholder-ready PDF reports that look like McKinsey-level analysis.
4.  **Control**: Override any part of the pipeline via simple config objects or per-run parameters.

---

## ✨ Features at a Glance

| Pillar | Capability |
|:---|:---|
| **🔍 Intelligence** | Auto-detects feature types, class imbalance, and **Data Leakage** suspects. |
| **🧹 Resilience** | Industrial-strength cleaners handle outliers, missing values, and high-cardinality features. |
| **📈 Performance** | Bayesian optimization via **Optuna** + Stacking Ensembles for maximum ROI. |
| **📄 Clarity** | Magazine-style PDF reports with SHAP explainability and interactive-ready visuals. |
| **🛡️ Reliability** | Built-in data quality risk scoring (0-100) to flag "garbage-in" scenarios. |

---

## 📦 Installation

```bash
# Recommendation: Use a virtual environment
pip install git+https://github.com/GhulamMuhammadNabeel/OctoLearn.git
```

---

## 🏎️ Quick Start: Production Pipeline in 30 Seconds

### The "Surprise Me" API (Zero Configuration)
Want to see OctoLearn in action instantly without providing your own data? Use the `surprise_me` API to automatically download a benchmark dataset, run the full pipeline, and generate a beautiful intelligence report.

```python
from octolearn import AutoML

# Run the complete pipeline on a sample classification dataset
pdf_path, best_model = AutoML.surprise_me(task='classification')
print(f"Report saved to: {pdf_path}")
```

### Standard Usage with Your Own Data
```python
from octolearn import AutoML
import pandas as pd

# 1. Ingest Data
df = pd.read_csv("telecom_churn.csv")
X, y = df.drop("churn", axis=1), df["churn"]

# 2. Orchestrate (Profile, Clean, Tune, Evaluate)
automl = AutoML()
automl.fit(X, y)

# 3. Deliver Insight
pdf_report = automl.generate_report()
print(f"Analysis complete: {pdf_report}")
```

### Profile Only (No Training)

```python
automl = AutoML(train_models=False)
automl.fit(X, y)

# Access insights
profile = automl.raw_profile_
print(f"Rows: {profile.n_rows}, Columns: {profile.n_columns}")
print(f"Task type: {profile.task_type}")
print(f"Missing values: {profile.missing_ratio}")

# Risk assessment
risk = automl.get_risk_score()
print(f"Risk: {risk['score']}/100 ({risk['category']})")
```

### Generate a PDF Report

```python
automl = AutoML()
automl.fit(X, y)
automl.generate_report()  # Creates a professional PDF report
```

---

## ⚡ Per-Run `fit()` Overrides

You can override key configuration settings for a **single run** without re-creating the `AutoML` instance. This is ideal for rapid experimentation:

```python
automl = AutoML()  # Create once with default config

# Quick exploration run — no Optuna, fast
automl.fit(X, y, use_optuna=False, n_models=2)

# Production run — more trials, longer timeout
automl.fit(X, y, optuna_trials=50, optuna_timeout=600)

# Try a specific metric
automl.fit(X, y, evaluation_metric='roc_auc')

# Override preprocessing
automl.fit(X, y, imputer_strategy={'numeric': 'median'}, scaler='robust')

# Specific models only
automl.fit(X, y, models=['xgboost', 'lightgbm'])
```

### All Override Parameters

| Parameter | Type | Overrides | Description |
|-----------|------|-----------|-------------|
| `optuna_trials` | `int` | `OptimizationConfig.optuna_trials_per_model` | Number of Optuna trials per model |
| `optuna_timeout` | `int` | `OptimizationConfig.optuna_timeout_seconds` | Max seconds per model for Optuna |
| `use_optuna` | `bool` | `OptimizationConfig.use_optuna` | Enable/disable Optuna for this run |
| `test_size` | `float` | `DataConfig.test_size` | Train/test split ratio |
| `random_state` | `int` | `DataConfig.random_state` | Random seed |
| `models` | `List[str]` | `ModelingConfig.models_to_train` | Specific models to train |
| `n_models` | `int` | `ModelingConfig.n_models` | Number of models to train |
| `evaluation_metric` | `str` | `ModelingConfig.evaluation_metric` | Primary metric to optimize |
| `imputer_strategy` | `dict` | `PreprocessingConfig.imputer_strategy` | Imputation strategy |
| `scaler` | `str` | `PreprocessingConfig.scaler` | Scaling method |

> **Note:** Overrides are applied non-destructively. The original config is fully restored after `fit()` completes, so subsequent calls use the original defaults.



## 🔧 Advanced Usage

### 🚀 Deployment: Exporting the Best Pipeline
OctoLearn makes it easy to move from experimentation to production. You can export the entire "Best Pipeline" (Preprocessing + Model) as a standalone scikit-learn object.

```python
# 1. Train the orchestrator
automl.fit(X, y)

# 2. Get the standalone pipeline (Preprocessing + Best Model)
pipeline = automl.get_pipeline()

# 3. Use it like a standard sklearn object (no OctoLearn required for inference!)
predictions = pipeline.predict(X_new)

# 4. Save for deployment
import joblib
joblib.dump(pipeline, "octolearn_prod_pipeline.pkl")
```

---

### 🏋️‍♂️ Training on Full Data
After finding the best model and hyperparameters, you may want to retrain on your **entire** dataset (train + test combined) to maximize performance before deployment.

```python
# 1. Find the best settings
automl.fit(X, y)

# 2. Get the standalone pipeline
pipeline = automl.get_pipeline()

# 3. Retrain the pipeline on the FULL dataset
# This refits the preprocessing and the model on all available data
pipeline.fit(X, y)

# 4. Save your final production model
joblib.dump(pipeline, "final_model_full_data.pkl")
```

---

### 📍 Accessing Data at Specific Points
Need to inspect the data after cleaning but before model training? Or want to see the raw samples? OctoLearn exposes all intermediate states:

```python
# Raw data exactly as provided (or sampled if configured)
X_raw = automl.X_raw_

# The train/test split (pre-cleaning)
X_train_raw = automl.X_train_

# The final cleaned data used for the "Model Arena"
X_final = automl.X_  # Combined cleaned dataset
y_final = automl.y_

# Access the cleaner directly to transform new data manually
clean_data = automl.cleaner_.transform(new_df)
```

---

### Full Configuration Control
Every aspect of OctoLearn is configurable through dataclass objects:

```python
from octolearn import (
    AutoML,
    DataConfig,
    ProfilingConfig,
    PreprocessingConfig,
    ModelingConfig,
    OptimizationConfig,
    ReportingConfig,
    ParallelConfig,
)

automl = AutoML(
    # Data handling
    data_config=DataConfig(
        use_full_data=False,     # Sample large datasets
        sample_size=1000,        # Rows to sample
        test_size=0.2,           # Train/test split ratio
        random_state=42,         # Reproducibility
    ),

    # Profiling behavior
    profiling_config=ProfilingConfig(
        detect_outliers=True,
        analyze_interactions=True,   # Enable interaction analysis
        generate_risk_score=True,
        calculate_feature_importance=True,
    ),

    # Preprocessing strategy
    preprocessing_config=PreprocessingConfig(
        auto_clean=True,
        imputer_strategy={"numeric": "median", "categorical": "mode"},
        scaler="standard",       # "standard", "minmax", "robust", or None
        id_columns=["user_id"],  # Columns to remove
    ),

    # Model training
    modeling_config=ModelingConfig(
        train_models=True,
        n_models=5,
        models_to_train=["random_forest", "xgboost", "logistic_regression"],
    ),

    # Hyperparameter tuning
    optimization_config=OptimizationConfig(
        use_optuna=True,
        optuna_trials_per_model=30,
        optuna_timeout_seconds=600,
    ),

    # Report settings
    reporting_config=ReportingConfig(
        generate_report=True,
        report_detail="detailed",   # "brief" or "detailed"
        include_shap=True,
        plot_mode="simple",         # "simple" or "dashboard"
    ),

    # Parallel processing
    parallel_config=ParallelConfig(
        parallel_processing=True,
        n_jobs=-1,               # -1 = all cores
        backend="threading",
    ),
)

automl.fit(X, y)
```

### Using Individual Components

OctoLearn's components can be used independently:

#### Data Profiling

```python
from octolearn.profiling import DataProfiler

profiler = DataProfiler()
profile = profiler.profile(X, y)

print(f"Shape: {profile.shape}")
print(f"Numeric columns: {profile.numeric_columns}")
print(f"Categorical columns: {profile.categorical_columns}")
print(f"ID-like columns: {profile.id_like_columns}")
print(f"Leakage suspects: {profile.leakage_suspects}")
print(f"Class imbalance ratio: {profile.imbalance_ratio}")
```

#### Auto Cleaning

```python
from octolearn.preprocessing.auto_cleaner import AutoCleaner

cleaner = AutoCleaner(
    imputer_strategy={"numeric": "median"},
    scaler="robust"
)
X_clean, y_clean, cleaning_log = cleaner.fit_transform(X_train, y_train)

# Apply same cleaning to test data
X_test_clean = cleaner.transform(X_test)
```

#### Model Registry

```python
from octolearn.models.registry import ModelRegistry

registry = ModelRegistry(base_dir="./models")
registry.register(model, name="xgboost_v1", metrics={"accuracy": 0.95})

# Load best model later
best = registry.get_best_model(metric="accuracy")
```

### Backward Compatibility

Legacy parameter names are supported via `**kwargs`:

```python
# Both of these work identically:
AutoML(train_models=False)
AutoML(modeling_config=ModelingConfig(train_models=False))
```

---

## 📖 API Reference

### `AutoML` — Main Orchestrator

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Run the complete pipeline |
| `predict(X_new)` | Make predictions using best model |
| `generate_report()` | Generate PDF report |
| `get_risk_score()` | Get data quality risk score (0-100) |
| `get_recommendations()` | Get ML recommendations |
| `get_feature_importance()` | Get feature importance scores |
| `get_preprocessing_suggestions()` | Get preprocessing advice |
| `get_model_benchmarks()` | Get all model metrics |

| Attribute | Description |
|-----------|-------------|
| `raw_profile_` | `DatasetProfile` of raw data |
| `clean_profile_` | `DatasetProfile` of cleaned data |
| `X_`, `y_` | Cleaned feature matrix and target |
| `X_train_`, `X_test_` | Train/test splits |
| `cleaning_log_` | Dictionary of cleaning operations |
| `outlier_results_` | Outlier detection results |
| `trained_models_` | Dictionary of trained models |
| `best_model_` | Best performing model |

### Configuration Dataclasses

<details>
<summary><strong>DataConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_full_data` | `bool` | `False` | Use entire dataset (no sampling) |
| `sample_size` | `int` | `500` | Rows to sample if not using full data |
| `test_size` | `float` | `0.2` | Fraction for test split |
| `random_state` | `int` | `42` | Random seed for reproducibility |
| `stratify_target` | `bool` | `True` | Stratify split on target |
</details>

<details>
<summary><strong>ProfilingConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `detect_outliers` | `bool` | `True` | Run outlier detection |
| `analyze_interactions` | `bool` | `False` | Analyze feature interactions |
| `generate_risk_score` | `bool` | `True` | Calculate risk score |
| `calculate_feature_importance` | `bool` | `True` | Compute importance |
| `generate_recommendations` | `bool` | `True` | Generate ML recommendations |
| `include_duplicates_analysis` | `bool` | `True` | Analyze duplicates |
</details>

<details>
<summary><strong>PreprocessingConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_clean` | `bool` | `True` | Enable auto cleaning |
| `imputer_strategy` | `Dict` | `None` | Imputation methods per type |
| `encoder_strategy` | `Dict` | `None` | Encoding strategy |
| `scaler` | `str` | `"standard"` | Scaling method |
| `id_columns` | `List[str]` | `None` | Columns to remove |
</details>

<details>
<summary><strong>ModelingConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_models` | `bool` | `True` | Whether to train models |
| `models_to_train` | `List[str]` | `None` | Specific models to train |
| `evaluation_metric` | `str` | `None` | Primary evaluation metric |
| `n_models` | `int` | `5` | Number of models to train |
| `test_size` | `float` | `0.2` | Test split ratio |
</details>

<details>
<summary><strong>OptimizationConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_optuna` | `bool` | `True` | Enable Optuna tuning |
| `optuna_trials_per_model` | `int` | `20` | Trials per model |
| `optuna_timeout_seconds` | `int` | `300` | Timeout per model |
| `optuna_parallel_jobs` | `int` | `-1` | Parallel Optuna workers |
| `use_registry` | `bool` | `True` | Save models to registry |
</details>

<details>
<summary><strong>ReportingConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generate_report` | `bool` | `True` | Generate PDF report |
| `report_detail` | `str` | `"detailed"` | `"brief"` or `"detailed"` |
| `include_shap` | `bool` | `True` | Include SHAP analysis |
| `plot_mode` | `str` | `"simple"` | `"simple"` or `"dashboard"` |
| `visuals_limit` | `int` | `10` | Max plots in report |
</details>

<details>
<summary><strong>ParallelConfig</strong></summary>

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `parallel_processing` | `bool` | `True` | Enable parallelism |
| `n_jobs` | `int` | `-1` | Number of cores (-1 = all) |
| `backend` | `str` | `"threading"` | Joblib backend |
| `verbose` | `int` | `0` | Verbosity level |
</details>

---

## 🏗️ Architecture

```
octolearn/
├── __init__.py              # Public API exports
├── config.py                # Centralized configuration constants
├── core.py                  # AutoML orchestrator (main entry point)
│
├── profiling/
│   └── data_profiler.py     # DataProfiler + DatasetProfile
│
├── preprocessing/
│   ├── auto_cleaner.py      # AutoCleaner (impute/encode/scale)
│   └── pipeline_builder.py  # sklearn Pipeline export
│
├── models/
│   ├── model_trainer.py     # ModelTrainer + Optuna integration
│   └── registry.py          # ModelRegistry (versioned storage)
│
├── evaluation/
│   └── metrics.py           # ModelEvaluator (classification/regression)
│
├── experiments/
│   ├── report_generator.py  # PDF report generation
│   ├── plot_generator.py    # Visualization engine
│   ├── recommendation_engine.py  # ML recommendations
│   ├── risk_scorer.py       # Data quality risk scoring
│   ├── outlier_detector.py  # Multi-method outlier detection
│   ├── baseline_importance.py    # Feature importance
│   └── preprocessing_suggester.py # Preprocessing advice
│
├── feature/
│   └── interaction_analyzer.py   # Feature interaction analysis
│
└── utils/
    └── helpers.py           # Logging, decorators, validation
```

### Pipeline Flow

```
Raw Data ──► Profiling ──► Train/Test Split ──► Auto Cleaning ──► Clean Profiling
                                                      │
                                                      ▼
                    PDF Report ◄── Model Training ◄── Feature Engineering
                                       │
                                       ▼
                              Optuna Optimization ──► Model Registry
```

The pipeline executes 6 phases:

1. **Profiling** — Infer types, detect quality issues, estimate task type
2. **Splitting** — Stratified train/test split
3. **Cleaning** — Impute missing values, encode categoricals, scale numerics
4. **Clean Profiling** — Re-profile the cleaned dataset
5. **Feature Engineering** — Outlier detection + interaction analysis
6. **Model Training** — Train multiple models with optional Optuna HPO

---

## 🧪 Running Tests

```bash
# Activate virtual environment first
python test_complete_pipeline.py
```

This exercises all pipeline phases with the Titanic dataset.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ghulam Muhammad Nabeel**

---

<p align="center">
  Built with ❤️ by the OctoLearn team
</p>
