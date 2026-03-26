# Optimization Module

> `octolearn/optimization/`

## Purpose

The optimization module implements the **Feature Optimization Engine** — an Optuna-driven system that jointly optimizes feature selection, synthetic feature generation, and model+hyperparameter selection in a single Bayesian search.

## Files

### `feature_optimizer.py`

Contains three components:

#### `FeatureOptimizationResult`
A `@dataclass` container for optimization results.

**Key attributes:**
| Attribute | Type | Description |
|:---|:---|:---|
| `best_features` | `list[str]` | Optimal feature subset |
| `best_model_name` | `str` | Optimal algorithm (e.g., `'xgboost'`) |
| `best_params` | `dict` | Optimal hyperparameters |
| `best_score` | `float` | Best cross-validated score |
| `baseline_score` | `float` | Score with all original features |
| `n_original_features` | `int` | Original features in optimal set |
| `n_synthetic_features` | `int` | Synthetic features in optimal set |
| `feature_generation_recipe` | `dict` | Serializable recipe for recreating synthetic features |

#### `_FeaturePoolBuilder`
Internal class that generates the candidate feature pool from training data.

**Synthetic feature types (generated in order, budget-limited):**
1. **Interactions** (`int_{A}_x_{B}`) — Top-K pairwise multiplications ranked by |correlation| with target
2. **Ratios** (`rat_{A}_div_{B}`) — Top-K pairwise divisions ranked by |correlation| with target
3. **Polynomials** (`poly_{A}_sq`) — Squared features ranked by |correlation| with target
4. **Log transforms** (`log_{A}`) — Applied to features with skewness > 1.0

**Key design: The Recipe Pattern**
The builder stores a `recipe_` dict that records exactly which operations were performed. The `transform()` method replays this recipe on new data (e.g., test set), ensuring train-test consistency without recomputing correlations.

#### `FeatureOptimizer`
The main optimizer class that runs the Optuna study.

**How the joint search works:**
Each Optuna trial simultaneously decides:
1. How many features to select (`n_features`)
2. Which features to include (binary toggle per feature, with bias toward original)
3. Which model to use (`model_type`)
4. What hyperparameters to use (from `OPTUNA_CONFIG`)

The trial's score is the cross-validated performance of the selected model on the selected feature subset.

```python
optimizer = FeatureOptimizer(X_train, y_train, X_test, y_test, profile)
result = optimizer.optimize()         # Returns FeatureOptimizationResult
X_train_opt, X_test_opt = optimizer.get_optimized_data()
```

**Why joint optimization?**
Traditional pipelines run feature selection and HPO separately. This misses feature-model interactions: some features are only useful with certain models. Joint search explores the full `(features × model × params)` space, often yielding 5–15% improvement.

## Data Flow

```
X_train_clean, y_train
    │
    ▼
_FeaturePoolBuilder.build()
    ├── Original features
    ├── Top-K interaction features (A × B)
    ├── Top-K ratio features (A / B)
    ├── Polynomial features (A²)
    └── Log transforms
    │
    ▼
Feature Pool (original + synthetic columns)
    │
    ▼
Optuna Study (n_trials, timeout)
    ├── Trial: select features + model + params
    ├── Cross-validate
    └── Track best
    │
    ▼
FeatureOptimizationResult
```

## Dependencies

- `optuna` — Bayesian optimization (TPESampler, MedianPruner)
- `sklearn` — cross-validation, models
- `xgboost`, `lightgbm` — gradient boosting models
