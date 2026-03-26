# Evaluation Module

> `octolearn/evaluation/`

## Purpose

The evaluation module provides standardized model scoring across classification and regression tasks. It is used by both the `ModelTrainer` (for model selection) and the `ReportGenerator` (for displaying metrics).

## Files

### `metrics.py`

#### `ModelEvaluator`
Comprehensive evaluation engine that computes task-appropriate metrics.

**Constructor:**
```python
evaluator = ModelEvaluator(model, X_test, y_test, task_type)
```

**Key methods:**

| Method | Description |
|:---|:---|
| `evaluate()` | Compute all configured metrics, returns dict |
| `cross_validate(X_train, y_train, cv)` | K-fold cross-validation with multiple scorers |
| `feature_importance()` | Extract built-in importance or coefficients |
| `get_evaluation_report()` | Human-readable text report |

**Classification metrics** (`EVALUATION_CONFIG['classification_metrics']`):
- Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
- ROC-AUC — handles both binary (probability of class 1) and multiclass (OVR weighted)
- Confusion Matrix, Classification Report

**Regression metrics** (`EVALUATION_CONFIG['regression_metrics']`):
- MSE, RMSE, MAE, R², MAPE

**Key design decisions:**
- `zero_division=0` is used for precision/recall/F1 to handle edge cases with missing classes in predictions.
- ROC-AUC automatically detects binary vs multiclass and adjusts the calculation method.
- The `evaluate()` return dict includes `predictions` and `probabilities` arrays — these are stored by `ModelTrainer` for use in performance plots (ROC curves, calibration).
- Feature importance supports both `feature_importances_` (tree models) and `|coef_|` (linear models).

## Data Flow

```
fitted_model, X_test, y_test
    │
    ▼
ModelEvaluator.evaluate()
    ├── model.predict(X_test)
    ├── model.predict_proba(X_test)  (if available)
    └── Compute all metrics
    │
    ▼
{
    'task': 'classification',
    'metrics': {'accuracy': 0.95, 'f1': 0.93, ...},
    'predictions': [...],
    'probabilities': [...],
    'confusion_matrix': [[...]],
}
```

## Dependencies

- `sklearn.metrics` — all scoring functions
- `sklearn.model_selection` — `cross_validate`
- `numpy` — array operations
