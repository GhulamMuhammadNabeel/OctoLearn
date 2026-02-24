# OctoLearn Testing and Benchmarking

Quality assurance is at the core of OctoLearn. This document outlines our testing strategies, benchmarking methodology, and how to verify the library's performance.

## Testing Strategy

OctoLearn uses a multi-layered, exhaustive testing approach managed via `pytest`:
1.  **Unit Tests**: Verify individual components (imputers, encoders, scorers).
2.  **Integration Tests**: Ensure the `AutoML` orchestrator correctly passes data between modules.
3.  **Grand Stress Tests**: We synthetically generate edge-case scenarios including:
    - 100% missing data columns
    - Emoticon/Unicode column names
    - Single-class targets in classification
    - Cardinality explosion (1,000+ unique categories)

!!! example "Running the Test Suite"
    To execute the standard pipeline tests and verify the Grand Stress Tests locally, navigate to the root directory and run the following command:
    
    ```bash
    python -m pytest tests/ -v
    ```

## The Model Arena: Champion Search

OctoLearn's internal `ModelArena` is where hyperparameter optimization and cross-validation meet. We use **Bayesian Search** (via Optuna) to navigate complex parameter spaces for:

- **XGBoost & LightGBM**: Fine-tuning learning rates, depths, and tree counts.
- **Random Forest**: Optimizing split criteria and ensemble size.
- **Stacking Ensemble**: Using the top 3 performers as base learners with a meta-regressor/classifier.

### Benchmarking Protocols
We bench OctoLearn against standard industry datasets to ensures zero-regression in performance:
- **Tabular Benchmarks**: Titanic (Classification), Housing (Regression).
- **Stress Datasets**: High-cardinality churn datasets and sparse clinical data.

---

!!! success "Performance Highlights"
    On our baseline benchmark datasets (e.g., Titanic, Breast Cancer), the OctoLearn automated pipeline consistently achieves:
    
    *   **Primary Metric**: > 0.95+ F1 / ROC-AUC (without manual tuning).
    *   **Execution Time**: < 30 seconds average (including the full Data Journey and Bayesian optimization).

---

## The Data Journey: Full Depth Trace

Based on our intensive architectural audit (last run: 2026-02-24), here is how OctoLearn transforms data through various configuration permutations.

### Stage 1: Structural Ingestion
**Input**: `['age', 'salary', 'gender', 'id_col', 'constant', 'missing']`

| Configuration | Action | Transformation Result |
|---------------|--------|-----------------------|
| **Default** | Auto-detect ID & Constant | Dropped `id_col`, `constant` |
| **PreprocessingConfig** | `id_columns=['age']` | Dropped `age`, kept `id_col` |

### Stage 2: Preprocessing & Imputation
**Trace Log from Audit Output:**

```json
{
  "Default": {
    "numeric_imputation": "mean",
    "scaling": "standard",
    "result": "4 columns final"
  },
  "Override": {
    "numeric_imputation": "median",
    "scaling": "robust",
    "test_size": 0.3
  },
  "Per-Call": {
    "numeric_imputation": "constant",
    "reason": "fit() override"
  }
}
```

### Stage 3: Optimization Trace
**Config: OptimizationConfig(use_optuna=True, trials=5)**

1. **Trial 0**: Baseline model training (Random Forest).
2. **Trial 1-4**: Bayesian sampling of `max_depth` and `n_estimators`.
3. **Registry**: Winning parameters saved to `octolearn_artifacts/registry.v1`.

---

## Verification Checklist

When updates are made to the core logic, we verify the following:
- [x] **Preprocessing Consistency**: `transform()` matches `fit()`.
- [x] **Risk Score Accuracy**: Validated via high-missingness synthetic data.
- [x] **Report Fidelity**: Verified across 10+ PDF compositions.
- [x] **UI Responsiveness**: Verified on mobile, tablet, and desktop viewports.
