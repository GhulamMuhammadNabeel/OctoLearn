# API Reference

This page is the complete reference for all public classes, methods, and attributes in OctoLearn.

---

## AutoML

The main entry point. Orchestrates the entire ML pipeline.

```python
from octolearn import AutoML
```

### Constructor

```python
AutoML(
    data_config=None,
    profiling_config=None,
    preprocessing_config=None,
    modeling_config=None,
    optimization_config=None,
    feature_optimization_config=None,
    reporting_config=None,
    parallel_config=None,
    show_progress=True,
    save_artifacts=True,
    artifact_dir='./octolearn_artifacts/',
)
```

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `data_config` | `DataConfig` | `DataConfig()` | Data sampling and split settings |
| `profiling_config` | `ProfilingConfig` | `ProfilingConfig()` | Profiling and outlier analysis settings |
| `preprocessing_config` | `PreprocessingConfig` | `PreprocessingConfig()` | Cleaning, imputation, encoding, scaling |
| `modeling_config` | `ModelingConfig` | `ModelingConfig()` | Algorithm selection and ensemble settings |
| `optimization_config` | `OptimizationConfig` | `OptimizationConfig()` | Optuna HPO settings |
| `feature_optimization_config` | `FeatureOptimizationConfig` | `FeatureOptimizationConfig()` | Feature synthesis + joint HPO |
| `reporting_config` | `ReportingConfig` | `ReportingConfig()` | PDF report settings |
| `parallel_config` | `ParallelConfig` | `ParallelConfig()` | Multi-core and GPU settings |
| `show_progress` | `bool` | `True` | Print pipeline status to console |
| `save_artifacts` | `bool` | `True` | Persist models and logs to disk |
| `artifact_dir` | `str` | `'./octolearn_artifacts/'` | Root directory for all saved files |

---

### Methods

#### `fit(X, y, **overrides) → AutoML`

Execute the complete AutoML pipeline.

```python
automl.fit(
    X,                            # pd.DataFrame — feature matrix
    y,                            # pd.Series — target vector
    # Per-call overrides (temporary; original config is restored after fit)
    optuna_trials=None,           # int — override optuna_trials_per_model
    optuna_timeout=None,          # int — override optuna_timeout_seconds
    use_optuna=None,              # bool — enable/disable Optuna for this run
    optuna_baseline_score=None,   # float — target metric score
    test_size=None,               # float — override test split proportion
    random_state=None,            # int — override random seed
    models=None,                  # list[str] — override models_to_train
    n_models=None,                # int — override leaderboard size
    evaluation_metric=None,       # str — override champion selection metric
    imputer_strategy=None,        # dict — override per-column imputation
    scaler=None,                  # str — override scaler type
    train_models=None,            # bool — False for profiling-only runs
)
```

**Returns:** `self` (AutoML instance) — enables method chaining.

---

#### `predict(X_new) → np.ndarray`

Generate predictions using the fitted champion model.

```python
y_pred = automl.predict(X_new)
```

Automatically applies the same preprocessing pipeline used during training. If string class labels were used during `fit()`, predictions are automatically decoded back to the original string labels via the fitted `target_encoder_`.

**Returns:** `np.ndarray` of predicted values (decoded to original string labels for classification if applicable).

---

#### `generate_report(filename=None) → str`

Produce a professional PDF intelligence report.

```python
pdf_path = automl.generate_report()
pdf_path = automl.generate_report(filename='my_report.pdf')
```

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `filename` | `str` | `None` | Output path. If None, auto-generates timestamped filename |

**Returns:** Absolute file path of the generated PDF.

---

#### `get_pipeline() → sklearn.pipeline.Pipeline`

Export a standalone scikit-learn pipeline (preprocessing + model).

```python
pipeline = automl.get_pipeline()
pipeline.predict(X_new)   # Standard sklearn interface — no OctoLearn needed
```

**Returns:** `sklearn.pipeline.Pipeline` object.

---

#### `get_risk_score() → dict`

Get data quality risk assessment.

```python
result = automl.get_risk_score()
# {'score': 24, 'category': 'Low Risk', 'factors': {'missing_data': 2, ...}}
```

**Returns:** Dict with keys: `score` (int, 0–100), `category` (str), `factors` (dict).

---

#### `get_preprocessing_suggestions() → dict`

Get automated preprocessing recommendations for your dataset.

```python
suggestions = automl.get_preprocessing_suggestions()
# Returns a dict with 6 sections:
# {
#   'missing_value_strategy': ['Impute age with mean/median.', ...],
#   'categorical_encoding':   ['One-Hot Encode sex.', ...],
#   'scaling_strategy':       ['Scaling required for SVM...', ...],
#   'feature_engineering':    ['Explore feature interactions...'],
#   'column_actions':         ['Remove identifier columns: [...]'],
#   'risk_mitigation':        ['Apply class balancing (SMOTE)...']
# }
```

**Returns:** `dict[str, list[str]]` — six categorized sections, each containing a list of actionable suggestion strings.

---

#### `get_feature_importance() → dict`

Get feature importance scores derived from the best trained model.

```python
importance = automl.get_feature_importance()
# {'age': 0.34, 'salary': 0.28, 'tenure': 0.21, ...}
```

For tree-based models (XGBoost, LightGBM, Random Forest, GradientBoosting), uses native `feature_importances_`. For linear models (LogisticRegression, LinearRegression), uses `|coef_|`. For SVM, importance is not available.

**Returns:** `dict[str, float]` mapping feature name to importance score.

---

#### `get_recommendations() → dict`

Get actionable ML recommendations based on data analysis.

```python
recs = automl.get_recommendations()
# {'high': ['Remove highly correlated feature X...'], 'medium': ['...'], 'informational': ['...']}
```

**Returns:** Dict with keys `'high'`, `'medium'`, and `'informational'`, each containing a list of plain-English recommendation strings.

---

#### `get_model_benchmarks() → list[dict]`

Get performance metrics for all trained models.

```python
benchmarks = automl.get_model_benchmarks()
for model in benchmarks:
    print(model['model'], model['score'], model.get('metrics', {}))
```

**Returns:** List of dicts, each with keys `model` (str), `score` (float), `params` (dict), `metrics` (dict), and `training_time` (float).

---

#### `AutoML.surprise_me(task='classification') → tuple`

Class method. Fetch a dataset, run the full pipeline, save a report.

```python
pdf_path, best_model = AutoML.surprise_me(task='classification')
pdf_path, best_model = AutoML.surprise_me(task='regression')
```

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `task` | `str` | `'classification'` | `'classification'` (Breast Cancer) or `'regression'` (California Housing) |

**Returns:** `(pdf_path: str, best_model: estimator)`

---

### Post-Fit Attributes

These are populated after calling `fit()`:

| Attribute | Type | Description |
|:---|:---|:---|
| `raw_profile_` | `DatasetProfile` | Profile of original raw data |
| `clean_profile_` | `DatasetProfile` | Profile of cleaned data |
| `X_raw_` | `pd.DataFrame` | Original features before cleaning |
| `X_train_` | `pd.DataFrame` | Final cleaned training features |
| `X_test_` | `pd.DataFrame` | Cleaned held-out test features |
| `y_train_` | `pd.Series` | Training labels |
| `y_test_` | `pd.Series` | Test labels |
| `best_model_` | `estimator` | Champion sklearn estimator |
| `model_benchmarks_` | `list[dict]` | Scores for all trained models |
| `trained_models_` | `dict` | All fitted estimator objects |
| `cleaner_` | `AutoCleaner` | Fitted cleaning pipeline |
| `cleaning_log_` | `dict` | Record of all cleaning actions |
| `outlier_results_` | `dict` | Outlier detection results. Structure: `{'methods': {'iqr': {col: {count, bounds, indices}}, 'isolation_forest': {overall: {...}}, 'zscore': {col: {...}}}, 'summary': {severity, total_outlier_rows, ...}, 'affected_features': {col: {recommendation, ...}}}` |
| `interaction_results_` | `dict` | Pairwise feature interaction scores |
| `feature_optimization_result_` | `FeatureOptimizationResult` | Joint feature+HPO search results |
| `original_rows_` | `int` | Number of rows in the original dataset |
| `target_encoder_` | `LabelEncoder \| None` | String-label encoder if applicable |

---

## Configuration Classes

All configuration classes are Python dataclasses importable from `octolearn`.

```python
from octolearn import (
    DataConfig,
    ProfilingConfig,
    PreprocessingConfig,
    ModelingConfig,
    OptimizationConfig,
    FeatureOptimizationConfig,
    ReportingConfig,
    ParallelConfig,
)
```

### DataConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `use_full_data` | `bool` | `False` | Use entire dataset instead of sampling |
| `sample_size` | `int` | `500` | Number of rows to sample when `use_full_data=False` |
| `test_size` | `float` | `0.2` | Fraction of data for test split (0.05–0.50) |
| `random_state` | `int` | `42` | Reproducibility seed |
| `stratify_target` | `bool` | `True` | Stratified train/test split for classification |
| `sampling_strategy` | `str` | `'auto'` | Class imbalance strategy: `'none'`, `'auto'`, `'smote'`, `'adasyn'`, `'undersample'`, `'combine'` |

### ProfilingConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `detect_outliers` | `bool` | `True` | Run IQR + Isolation Forest + Z-Score |
| `analyze_interactions` | `bool` | `False` | Pairwise feature interaction analysis (O(n²)) |
| `generate_risk_score` | `bool` | `True` | Calculate 0–100 data quality score |
| `calculate_feature_importance` | `bool` | `True` | Baseline permutation importance |
| `generate_recommendations` | `bool` | `True` | Actionable text recommendations |
| `include_duplicates_analysis` | `bool` | `True` | Scan for duplicate rows |

### PreprocessingConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `auto_clean` | `bool` | `True` | Enable full cleaning pipeline |
| `imputer_strategy` | `dict` | `None` | Per-column imputation override e.g. `{'age': 'median'}` |
| `encoder_strategy` | `dict` | `None` | Per-column encoding override |
| `scaler` | `str` | `'standard'` | `'standard'`, `'robust'`, `'minmax'`, or `None` |
| `id_columns` | `list[str]` | `None` | Columns to force-drop as identifiers |

### ModelingConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `train_models` | `bool` | `True` | Enable model training phase |
| `models_to_train` | `list[str]` | `None` | Specific algorithms; `None` = all defaults |
| `evaluation_metric` | `str` | `None` | Champion selection metric; `None` = auto |
| `n_models` | `int` | `5` | Leaderboard size (1–10) |
| `test_size` | `float` | `0.2` | Test proportion |
| `use_stacking` | `bool` | `True` | Build stacking ensemble from top models |

### OptimizationConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `use_optuna` | `bool` | `True` | Enable Bayesian HPO via Optuna |
| `optuna_trials_per_model` | `int` | `20` | Trials per algorithm |
| `optuna_timeout_seconds` | `int` | `300` | Max seconds per model |
| `optuna_parallel_jobs` | `int` | `-1` | Parallel trials (-1 = all cores) |
| `use_registry` | `bool` | `True` | Save models to local Model Registry |
| `early_stopping_rounds` | `int` | `None` | Early stop for boosting models |
| `baseline_score` | `float` | `None` | Target metric; if hit, +25% more trials |
| `hyperparameter_overrides` | `dict` | `None` | Manual constraints per model |

### FeatureOptimizationConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `enable_feature_optimization` | `bool` | `True` | Enable joint feature+HPO search |
| `n_trials` | `int` | `40` | Optuna trials for feature optimization |
| `timeout` | `int` | `300` | Max seconds for feature search |
| `cv_folds` | `int` | `3` | Cross-validation folds per trial |
| `max_synthetic_features` | `int` | `30` | Maximum synthetic features considered |
| `min_features` | `int` | `3` | Minimum features per trial |
| `generate_interactions` | `bool` | `True` | Generate A×B interaction features |
| `generate_ratios` | `bool` | `True` | Generate A/B ratio features |
| `generate_polynomials` | `bool` | `True` | Generate A² polynomial features |
| `generate_log_transforms` | `bool` | `True` | Generate log(A) transforms |

### ReportingConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `generate_report` | `bool` | `True` | Produce PDF report |
| `report_title` | `str` | `'OctoLearn Intelligence Report'` | Cover page title |
| `report_detail` | `str` | `'detailed'` | `'brief'` or `'detailed'` |
| `include_data_journey` | `bool` | `True` | Before/after cleaning distribution plots |
| `include_model_comparison` | `bool` | `True` | Model Arena leaderboard |
| `include_recommendations` | `bool` | `True` | Text insights section |
| `visuals_limit` | `int` | `10` | Max feature distribution plots |
| `plot_mode` | `str` | `'simple'` | Visual complexity |
| `include_shap` | `bool` | `True` | SHAP global importance |
| `color_scheme` | `str` | `'light'` | `'light'`, `'dark'`, or `'neon'` |

### ParallelConfig

| Field | Type | Default | Description |
|:---|:---|:---|:---|
| `parallel_processing` | `bool` | `True` | Enable parallelism |
| `n_jobs` | `int` | `-1` | Worker count (-1 = all CPU cores) |
| `backend` | `str` | `'threading'` | `'threading'`, `'loky'`, `'multiprocessing'` |
| `verbose` | `int` | `0` | Worker logging verbosity |
| `enable_gpu` | `bool` | `False` | GPU acceleration for XGBoost/LightGBM |

---

## DatasetProfile

The output of the `DataProfiler.profile()` method. Also accessible via `automl.raw_profile_` and `automl.clean_profile_`.

| Attribute | Type | Description |
|:---|:---|:---|
| `shape` | `tuple[int, int]` | Dataset dimensions `(n_rows, n_cols)` |
| `columns` | `list[str]` | Column names |
| `feature_types` | `dict[str, str]` | Inferred type per column: `'numeric'`, `'categorical'`, `'date'`, `'text'`, `'id'` |
| `stats` | `dict[str, dict]` | Summary statistics per column |
| `missing_ratio` | `dict[str, float]` | Proportion of missing values per column |
| `unique_counts` | `dict[str, int]` | Unique value count per column |
| `task_type` | `str` | `'classification'` or `'regression'` |
| `target_col` | `str` | Target column name |
| `id_like_columns` | `list[str]` | Auto-detected ID columns |
| `constant_columns` | `list[str]` | Columns with zero variance |
| `low_variance_columns` | `list[str]` | Numeric columns with very low variance |
| `numeric_columns` | `list[str]` | All numeric columns |
| `categorical_columns` | `list[str]` | All categorical columns |
| `date_columns` | `list[str]` | All datetime columns |
| `text_columns` | `list[str]` | Free text columns |
| `leakage_suspects` | `list[str]` | Columns correlated >0.95 with target (potential leakage) |
| `high_cardinality_cols` | `list[str]` | Categoricals with >50% unique values |
| `imbalance_ratio` | `float` | min_class/max_class count ratio |
| `duplicate_rows` | `int` | Count of identical rows |
| `data_quality_score` | `float` | 0–100 health score |
| `n_rows` | `int` | Property: `shape[0]` |
| `n_columns` | `int` | Property: `shape[1]` |

```python
profile = automl.raw_profile_

print(profile.task_type)           # 'classification'
print(profile.leakage_suspects)    # ['target_proxy', ...]
print(profile.data_quality_score)  # 84.3
print(profile.imbalance_ratio)     # 0.12 (heavily imbalanced)
```

---

## Exceptions

OctoLearn raises standard Python exceptions with descriptive messages:

| Exception | Trigger |
|:---|:---|
| `TypeError` | Wrong config type passed to `AutoML()` |
| `ValueError` | Parameter out of valid range (e.g., `test_size > 0.5`) |
| `ValueError` | Calling `predict()` or `generate_report()` before `fit()` |
| `ValueError` | Data cleaning failed (wraps underlying exception) |
