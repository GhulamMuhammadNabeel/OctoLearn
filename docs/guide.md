# OctoLearn User Guide

Welcome to the comprehensive OctoLearn guide. This document covers everything from your first AutoML run to advanced production deployment patterns.

---

## 1. Installation

```bash
pip install octolearn
```

Or clone and install from source:
```bash
git clone https://github.com/GhulamMuhammadNabeel/OctoLearn.git
cd OctoLearn
pip install -e .
```

**Requirements:** Python ≥ 3.9, scikit-learn, pandas, numpy, optuna, reportlab, shap, lightgbm, xgboost

---

## 2. Your First Run (5 minutes)

The fastest way to see what OctoLearn can do — one call:

```python
from octolearn import AutoML

# Automatic demo run (downloads a dataset, runs the full pipeline, saves PDF)
pdf_path, best_model = AutoML.surprise_me(task='classification')
print(f"Report saved to: {pdf_path}")
```

Or bring your own data:

```python
from octolearn import AutoML
import pandas as pd

data = pd.read_csv("your_data.csv")
X = data.drop(columns=["target"])
y = data["target"]

automl = AutoML()
automl.fit(X, y)

# Make predictions
predictions = automl.predict(X_new)

# Generate professional PDF intelligence report
pdf_path = automl.generate_report()
print(f"Report: {pdf_path}")
```

That's it. OctoLearn handles everything in between.

---

## 3. Understanding the Pipeline

When you call `automl.fit(X, y)`, OctoLearn executes a **7-phase pipeline**:

```mermaid
flowchart LR
    A([Raw Data]):::red --> B[Phase 1\nProfiling]:::dark
    B --> C[Phase 2\nOutlier Detection]:::dark
    C --> D[Phase 3\nCleaning]:::dark
    D --> E[Phase 4\nSampling & Split]:::dark
    E --> F[Phase 5\nFeature Optimization]:::dark
    F --> G[Phase 6\nModel Training]:::dark
    G --> H([Champion Model]):::red

    classDef red fill:#E43636,color:#F6EFD2,stroke:#E43636
    classDef dark fill:#1e1e1e,color:#E2DDB4,stroke:#E2DDB4
```

| Phase | What Happens |
|:---|:---|
| **1 · Profiling** | Infers feature types, calculates stats, identifies ID columns, leakage suspects, class imbalance |
| **2 · Outlier Detection** | Runs IQR, Z-Score, and Isolation Forest on all numeric columns |
| **3 · Cleaning** | Drops IDs/constants, imputes missing values, encodes categoricals, scales numerics |
| **4 · Sampling & Split** | Stratified train/test split; applies SMOTE/ADASYN if imbalance detected |
| **5 · Feature Optimization** | Optuna jointly searches synthetic features (interactions, ratios, polynomials) + model type + hyperparams |
| **6 · Model Training** | Trains top models; optionally builds stacking ensemble; saves to Registry |
| **7 · Feature Importance** | SHAP + permutation importance for interpretation |

---

## 4. Configuration Reference

Every aspect of OctoLearn is controlled by **dataclass config objects** passed to the `AutoML(...)` constructor. All parameters have sensible defaults so you can use as much or as little configuration as you need.

### DataConfig

Controls data sampling and train/test splitting.

```python
from octolearn import AutoML, DataConfig

automl = AutoML(data_config=DataConfig(
    use_full_data=False,         # True = use all rows (memory-intensive for large datasets)
    sample_size=2000,            # Rows to sample when use_full_data=False
    test_size=0.2,               # 20% held out for evaluation
    random_state=42,             # Reproducibility seed
    stratify_target=True,        # Stratify split for classification targets
    sampling_strategy='smote',   # Class imbalance strategy (see below)
))
```

**`sampling_strategy` options:**

| Strategy | Description | Best For |
|:---|:---|:---|
| `'none'` | Disable all resampling | Balanced datasets |
| `'auto'` | Auto-select based on imbalance | Most cases |
| `'smote'` | Synthetic Minority Over-sampling | Small imbalanced datasets |
| `'adasyn'` | Adaptive Synthetic Sampling | Boundary-focused oversample |
| `'undersample'` | Random under-sample majority | Large datasets |
| `'combine'` | SMOTE + Tomek-link removal | Noisy imbalanced data |

---

### ProfilingConfig

Controls pre-training dataset analysis.

```python
from octolearn import AutoML, ProfilingConfig

automl = AutoML(profiling_config=ProfilingConfig(
    detect_outliers=True,               # IQR + Isolation Forest + Z-Score
    analyze_interactions=False,         # Pairwise interaction analysis (slow for >50 features)
    generate_risk_score=True,           # 0–100 data quality risk score
    calculate_feature_importance=True,  # Baseline importance before training
    generate_recommendations=True,      # Actionable text recommendations
    include_duplicates_analysis=True,   # Scan for duplicate rows
))
```

---

### PreprocessingConfig

Controls the AutoCleaner behaviour.

```python
from octolearn import AutoML, PreprocessingConfig

automl = AutoML(preprocessing_config=PreprocessingConfig(
    auto_clean=True,                              # Enable full cleaning pipeline
    scaler='standard',                            # 'standard', 'robust', 'minmax', or None
    id_columns=['customer_id', 'row_id'],         # Force-drop these columns
    imputer_strategy={'age': 'median'},           # Per-column imputation override
    encoder_strategy=None,                        # Per-column encoding override
))
```

!!! tip "Which scaler to use?"
    - `'standard'`: Best default — zero mean, unit variance
    - `'robust'`: Use when you have many outliers (median-based)
    - `'minmax'`: Use when you need values in [0,1] range
    - `None`: Skip scaling (for tree models, scaling rarely matters)

---

### ModelingConfig

Controls which algorithms are trained.

```python
from octolearn import AutoML, ModelingConfig

automl = AutoML(modeling_config=ModelingConfig(
    train_models=True,                           # Set False for profiling-only runs
    models_to_train=['xgboost', 'lightgbm'],    # Explicit model list, or None = all defaults
    evaluation_metric='f1',                      # Primary champion selection metric
    n_models=5,                                  # Max models in the leaderboard (1–10)
    use_stacking=True,                           # Build stacking ensemble from top 3
))
```

**Available models:**

| Key | Algorithm |
|:---|:---|
| `'xgboost'` | XGBoost Gradient Boosting |
| `'lightgbm'` | LightGBM Gradient Boosting |
| `'random_forest'` | Random Forest |
| `'gradient_boosting'` | Sklearn GradientBoostingClassifier/Regressor |
| `'extra_trees'` | ExtraTreesClassifier/Regressor |
| `'logistic_regression'` | LogisticRegression (classification only) |
| `'ridge'` | Ridge Regression (regression only) |
| `'lasso'` | Lasso (regression only) |
| `'knn'` | K-Nearest Neighbours |
| `'svm'` | Support Vector Machine |

---

### OptimizationConfig

Controls Bayesian hyperparameter tuning via Optuna.

```python
from octolearn import AutoML, OptimizationConfig

automl = AutoML(optimization_config=OptimizationConfig(
    use_optuna=True,                  # Enable Bayesian HPO
    optuna_trials_per_model=30,       # Trials per algorithm (more = better, slower)
    optuna_timeout_seconds=300,       # Max seconds per algorithm
    optuna_parallel_jobs=-1,          # -1 = use all CPU cores
    use_registry=True,                # Save models to local registry
    early_stopping_rounds=None,       # Early stop for boosting models
    baseline_score=None,              # Target metric; if hit, +25% more trials
    hyperparameter_overrides=None,    # Manual param constraints per model
))
```

---

### FeatureOptimizationConfig

Controls the intelligent feature synthesis and joint optimization engine.

```python
from octolearn import AutoML, FeatureOptimizationConfig

automl = AutoML(feature_optimization_config=FeatureOptimizationConfig(
    enable_feature_optimization=True,  # Enable joint feature+HPO search
    n_trials=40,                       # Optuna trials for feature search
    timeout=300,                       # Max seconds for feature optimization
    cv_folds=3,                        # Cross-validation folds per trial
    max_synthetic_features=30,         # Max number of generated features
    min_features=3,                    # Min features per trial
    generate_interactions=True,        # A × B feature interactions
    generate_ratios=True,              # A / B feature ratios
    generate_polynomials=True,         # A² polynomial terms
    generate_log_transforms=True,      # log(A) log-transforms
))
```

!!! tip "How feature optimization works"
    OctoLearn generates a pool of synthetic features (up to `max_synthetic_features`), then uses Optuna to find the best **subset** of original + synthetic features, combined with the best **model type** and **hyperparameters** — all in a single joint search. This frequently yields 5–15% metric improvement over standard HPO alone.

---

### ReportingConfig

Controls the PDF intelligence report.

```python
from octolearn import AutoML, ReportingConfig

automl = AutoML(reporting_config=ReportingConfig(
    generate_report=True,
    report_title='Q4 Churn Analysis',          # Shown on cover page
    report_detail='detailed',                  # 'brief' or 'detailed'
    include_data_journey=True,                 # Before/after cleaning visuals
    include_model_comparison=True,             # Model Arena leaderboard
    include_recommendations=True,             # Text insights section
    visuals_limit=10,                          # Max feature distribution plots
    include_shap=True,                         # SHAP global importance
    color_scheme='light',                      # 'light', 'dark', or 'neon'
))
```

---

### ParallelConfig

Controls multi-core processing.

```python
from octolearn import AutoML, ParallelConfig

automl = AutoML(parallel_config=ParallelConfig(
    parallel_processing=True,    # Enable parallelism
    n_jobs=-1,                   # -1 = all CPU cores
    backend='threading',         # 'threading', 'loky', 'multiprocessing'
    verbose=0,                   # Logging verbosity for workers
    enable_gpu=False,            # GPU acceleration for XGBoost/LightGBM
))
```

---

## 5. The `fit()` Method — In-Depth

The `fit()` method accepts **per-call overrides** that temporarily replace config values without mutating the stored config object:

```python
automl = AutoML()  # Created with defaults

# Fit with temporary overrides (original config is restored after fit returns)
automl.fit(
    X, y,
    # Optimization overrides
    optuna_trials=50,
    optuna_timeout=600,
    use_optuna=True,
    optuna_baseline_score=0.92,
    # Data overrides
    test_size=0.25,
    random_state=123,
    # Model overrides
    models=['xgboost', 'lightgbm'],
    n_models=3,
    evaluation_metric='f1',
    # Preprocessing overrides
    scaler='robust',
    imputer_strategy={'age': 'median', 'salary': 'mean'},
    # Disable training (profiling-only run)
    train_models=False,
)
```

`fit()` always returns `self`, so you can chain it:

```python
pdf = AutoML().fit(X, y).generate_report()
```

---

## 6. Accessing Post-Fit Insights

After calling `fit()`, OctoLearn exposes a rich set of attributes and methods:

### Instance Attributes

```python
automl.fit(X, y)

# Raw and cleaned data
automl.X_raw_              # Original features before any cleaning
automl.X_train_            # Final cleaned training features
automl.X_test_             # Cleaned test features
automl.y_train_            # Training labels
automl.y_test_             # Test labels

# Dataset profiles
automl.raw_profile_        # DatasetProfile of the original data
automl.clean_profile_      # DatasetProfile after cleaning

# Model results
automl.best_model_         # The champion sklearn estimator
automl.model_benchmarks_   # List of dicts with scores for all models
automl.trained_models_     # Dict of all trained estimators

# Cleaning info
automl.cleaner_            # Fitted AutoCleaner instance
automl.cleaning_log_       # Dict: what was dropped/imputed/encoded

# Analysis results
automl.outlier_results_    # {'col': {'iqr_outliers': N, 'zscore': N, ...}}
automl.interaction_results_ # Pairwise interaction analysis results

# Feature optimization
automl.feature_optimization_result_  # FeatureOptimizationResult object
```

### Helper Methods

```python
# Data quality risk (0 = clean, 100 = high risk)
risk = automl.get_risk_score()
# {'score': 24, 'category': 'Low Risk', 'factors': {...}}

# Preprocessing recommendations
suggestions = automl.get_preprocessing_suggestions()
# [{'column': 'salary', 'suggestion': 'Apply log transform (skewness=3.2)', ...}]

# Feature importance (permutation-based)
importance = automl.get_feature_importance()
# {'age': 0.34, 'salary': 0.28, ...}

# NLP-style recommendations
recs = automl.get_recommendations()
# {'data_quality': ['Consider removing duplicates...'], 'modeling': [...]}

# All model scores
benchmarks = automl.get_model_benchmarks()
# [{'model': 'XGBoost', 'f1': 0.94, 'accuracy': 0.96, ...}, ...]

# Production-ready sklearn pipeline
pipeline = automl.get_pipeline()
# sklearn.pipeline.Pipeline with preprocessing + best model
pipeline.predict(X_new)  # Standard sklearn interface
```

---

## 7. Production Deployment

### Method 1: The OctoLearn Pipeline (Recommended)

```python
automl.fit(X_train, y_train)

# Export a standard sklearn Pipeline — no OctoLearn dependency at inference time
pipeline = automl.get_pipeline()

import joblib
joblib.dump(pipeline, 'model.pkl')

# At inference time (no OctoLearn needed):
import joblib
pipeline = joblib.load('model.pkl')
predictions = pipeline.predict(X_new)
```

### Method 2: Direct Model Access

```python
# Access the raw best model
model = automl.best_model_
cleaner = automl.cleaner_

# Manually apply cleaning + predict
X_clean = cleaner.transform(X_new)
predictions = model.predict(X_clean)
```

### Method 3: Model Registry

```python
from octolearn.models.registry import ModelRegistry

registry = ModelRegistry()
models = registry.list_models()
# ['xgboost_champion_v1', 'stacking_ensemble_v2', ...]

best = registry.get_best_model(metric='f1_score')
predictions = best.predict(X_new)
```

---

## 8. Advanced Use Cases

### Profiling Only (No Training)

```python
automl = AutoML()
automl.fit(X, y, train_models=False)

# Access profiling results
print(automl.raw_profile_.task_type)       # 'classification' or 'regression'
print(automl.raw_profile_.id_like_columns) # ['customer_id', ...]
print(automl.raw_profile_.leakage_suspects)
print(automl.get_risk_score())
```

### Custom Evaluation Metric

```python
automl = AutoML(
    modeling_config=ModelingConfig(evaluation_metric='roc_auc')
)
automl.fit(X, y)
# Champions selected by AUC, not F1
```

### Multi-model Custom Selection

```python
automl = AutoML()
automl.fit(
    X, y,
    models=['xgboost', 'lightgbm', 'random_forest', 'extra_trees'],
    n_models=4,
    evaluation_metric='f1_weighted'
)

# Compare all models
for bench in automl.get_model_benchmarks():
    print(f"{bench['model']}: {bench}")
```

### Feature Interaction Analysis

```python
from octolearn import AutoML, ProfilingConfig

automl = AutoML(profiling_config=ProfilingConfig(
    analyze_interactions=True   # WARNING: O(n²) complexity — slow for >50 features
))
automl.fit(X, y)

# Interaction results embedded in the generated PDF report
pdf = automl.generate_report()
```

### GPU Acceleration

```python
from octolearn import AutoML, ParallelConfig

automl = AutoML(parallel_config=ParallelConfig(
    enable_gpu=True,   # Passes tree_method='gpu_hist' to XGBoost/LightGBM
    n_jobs=-1
))
automl.fit(X, y)
```

### Custom Report Title

```python
from octolearn import AutoML, ReportingConfig

automl = AutoML(reporting_config=ReportingConfig(
    report_title='Monthly Churn Prediction Report',
    color_scheme='dark',
    include_shap=True,
    report_detail='detailed'
))
automl.fit(X, y)
automl.generate_report(filename='churn_report_feb.pdf')
```

---

## 9. Handling Common Scenarios

### Large Datasets (>1M rows)

```python
automl = AutoML(
    data_config=DataConfig(
        use_full_data=False,
        sample_size=10_000          # Profile and train on a 10k sample
    ),
    parallel_config=ParallelConfig(
        n_jobs=-1,
        backend='loky'              # Process-based parallelism for CPU-heavy work
    )
)
automl.fit(X, y)
```

### High-Dimensional Data (>200 features)

```python
automl = AutoML(
    profiling_config=ProfilingConfig(
        analyze_interactions=False  # Skip interaction analysis for speed
    ),
    feature_optimization_config=FeatureOptimizationConfig(
        max_synthetic_features=10,  # Limit synthetic feature pool
        n_trials=20                 # Fewer Optuna trials
    )
)
automl.fit(X, y)
```

### Heavily Imbalanced Data

```python
automl = AutoML(
    data_config=DataConfig(
        sampling_strategy='adasyn'
    ),
    modeling_config=ModelingConfig(
        evaluation_metric='f1_weighted'
    )
)
automl.fit(X, y)
```

### Regression Problems

```python
automl = AutoML(
    modeling_config=ModelingConfig(
        evaluation_metric='r2',     # R² for regression (also: 'rmse', 'mae')
        models_to_train=['xgboost', 'lightgbm', 'ridge', 'lasso']
    )
)
automl.fit(X, y_continuous)
```

---

## 10. Reading the PDF Report

The generated report contains:

| Section | Contents |
|:---|:---|
| **Cover Page** | Title, timestamp, dataset shape, task type |
| **Data Risk Assessment** | Risk score (0–100), breakdown of factors |
| **Dataset Health** | Missing values, duplicates, feature types |
| **Feature Profiles** | Distribution plots for each feature |
| **Outlier Narratives** | Which features have extreme values and their target relationship |
| **Correlation Analysis** | Heatmap + plain-English narrative of high correlations |
| **Data Journey** | Before & after cleaning distributions |
| **Model Arena** | Leaderboard of all trained models |
| **Champion Model** | Deep metrics for the best model (confusion matrix, ROC/PR curves) |
| **SHAP Importance** | Global feature importance via SHAP values |
| **Preprocessing Guide** | Automated suggestions for your specific data |
| **Recommendations** | Actionable ML insights and next steps |

---

## 11. Troubleshooting

### `ValueError: test_size must be between 0.05 and 0.5`
Your `test_size` is outside valid range. Use a value like `0.2`.

### `ValueError: sample_size must be >= 50`
Increase `sample_size` or set `use_full_data=True`.

### `ValueError: n_models must be between 1 and 10`
OctoLearn only supports up to 10 models in the leaderboard.

### PDF generation fails — `ImportError: reportlab`
```bash
pip install reportlab
```

### Feature optimization is slow
```python
# Reduce trials and synthetic features
FeatureOptimizationConfig(
    n_trials=15,
    max_synthetic_features=10,
    timeout=120
)
```

### Optuna is too slow
```python
# Disable or reduce trials
OptimizationConfig(use_optuna=False)
# or
OptimizationConfig(optuna_trials_per_model=10, optuna_timeout_seconds=60)
```
