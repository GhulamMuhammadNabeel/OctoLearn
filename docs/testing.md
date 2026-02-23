# OctoLearn Testing & Benchmarking

Quality assurance is at the core of OctoLearn. This document outlines our testing strategies, benchmarking methodology, and how to verify the library's performance.

## 🧪 Testing Strategy

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

---

## 📊 Benchmarking Methodology

We evaluate OctoLearn based on three pillars:
1.  **Predictive Accuracy**: Comparison against vanilla scikit-learn baselines.
2.  **Orchestration Time**: Efficiency of the profiling and cleaning stages.
3.  **Stability**: Performance on datasets with high missingness, extreme counts, and high cardinality.

!!! success "Performance Highlights"
    On our baseline benchmark datasets (e.g., Titanic, Breast Cancer), the OctoLearn automated pipeline consistently achieves:
    
    *   **Primary Metric**: > 0.95+ F1 / ROC-AUC (without manual tuning).
    *   **Execution Time**: < 30 seconds average (including the full Data Journey and Bayesian optimization).

---

## 🛠️ Verification Checklist

When updates are made to the core logic, we verify the following:
- [ ] **Preprocessing Consistency**: Ensure `transform()` applies the exact same logic as `fit_transform()`.
- [ ] **Risk Score Accuracy**: Validate that injected leakage or missingness is correctly identified.
- [ ] **Report Fidelity**: Verify that all charts and tables in the PDF correctly reflect the model benchmarks.
