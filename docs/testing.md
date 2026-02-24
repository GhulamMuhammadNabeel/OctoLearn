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

## Verification Checklist

When updates are made to the core logic, we verify the following:
- [ ] **Preprocessing Consistency**: Ensure `transform()` applies the exact same logic as `fit_transform()`.
- [ ] **Risk Score Accuracy**: Validate that injected leakage or missingness is correctly identified.
- [ ] **Report Fidelity**: Verify that all charts and tables in the PDF correctly reflect the model benchmarks.
