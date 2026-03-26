# Models Module

> `octolearn/models/`

## Purpose

The models module handles automated model training, hyperparameter optimization, and model lifecycle management. It contains two components: the `ModelTrainer` (the competitive "Model Arena") and the `ModelRegistry` (versioned model persistence).

## Files

### `model_trainer.py`

#### `ModelTrainer`
Orchestrates training of multiple ML algorithms with Optuna-based Bayesian hyperparameter optimization.

**Supported algorithms:**

| Key | Classification | Regression |
|:---|:---|:---|
| `logistic_regression` | LogisticRegression | — |
| `linear_regression` | — | LinearRegression |
| `random_forest` | RandomForestClassifier | RandomForestRegressor |
| `gradient_boosting` | GradientBoostingClassifier | GradientBoostingRegressor |
| `xgboost` | XGBClassifier | XGBRegressor |
| `lightgbm` | LGBMClassifier | LGBMRegressor |
| `svm` | SVC | — |
| `svr` | — | SVR |

**Training modes:**

1. **Joint Bayesian Optimization (default)** — Uses Optuna's TPE sampler to jointly search over model type AND hyperparameters in a single study. The champion is then retrained on the full training set.
2. **Default Parameters Fallback** — When `use_optuna=False`, trains all models with scikit-learn defaults and picks the best.

**Stacking Ensembles:**
When `use_stacking=True` and ≥2 models are trained, the trainer builds a `StackingClassifier`/`StackingRegressor` using the top 3 models as base estimators and Logistic Regression / Ridge as the meta-learner.

**Key design decisions:**
- **Thread safety**: During Optuna trials, `n_jobs=1` and `nthread=1` are forced on all models to prevent CPU oversubscription. Environment variables (`OMP_NUM_THREADS`, etc.) are also set.
- **Baseline stopping**: If a `baseline_score` is provided, Optuna stops early once the target is reached (plus 25% extra exploration trials).
- **External splits**: The trainer accepts pre-split `(X_train, X_test, y_train, y_test)` from the orchestrator to prevent the "double split" anti-pattern.

### `registry.py`

#### `ModelRegistry`
Versioned persistence layer for trained models.

**Storage structure:**
```
octolearn_artifacts/
├── model_registry.json     ← Metadata database
└── trained_models/
    ├── xgboost_v1.pkl
    ├── xgboost_v2.pkl
    └── lightgbm_v1.pkl
```

**Key methods:**
| Method | Description |
|:---|:---|
| `register_model(name, model, task_type, metrics)` | Save model + metadata, returns version number |
| `get_best_model(metric, mode)` | Load the best model across all versions |
| `load_model(name, version)` | Load a specific version |
| `list_models()` | List all registered models with metadata |

**Design rationale:**
- Uses JSON for metadata (human-readable, no DB dependency) and `joblib` for model serialization.
- `NumpyEncoder` handles NumPy/Pandas types during JSON serialization.
- `list_models()` returns a deep copy to prevent accidental mutation.

## Data Flow

```
X_train_clean, X_test_clean, y_train, y_test
    │
    ▼
ModelTrainer.__init__(...)
    │
    ▼
train_all_models()
    ├── Joint Optuna HPO (if enabled)
    │   └── TPE sampler searches model × hyperparameter space
    ├── Train champion with best params
    ├── Train remaining models with defaults (for leaderboard)
    └── Optional: Train stacking ensemble
    │
    ▼
ModelRegistry.register_model(...)  ← persists each model
    │
    ▼
best_model_, model_benchmarks_, trained_models_
```

## Dependencies

- `sklearn` — base models, cross-validation, stacking
- `xgboost`, `lightgbm` — gradient boosting models
- `optuna` — Bayesian optimization (TPE sampler, MedianPruner)
- `joblib` — model serialization
