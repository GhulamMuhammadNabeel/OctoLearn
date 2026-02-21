# OctoLearn Testing & Benchmarking

Quality assurance is at the core of OctoLearn. This document outlines our testing strategies, benchmarking methodology, and how to verify the library's performance.

## 🧪 Testing Strategy

OctoLearn uses a multi-layered testing approach:
1.  **Unit Tests**: Verify individual components (imputers, encoders, scorers).
2.  **Integration Tests**: Ensure the `AutoML` orchestrator correctly passes data between modules.
3.  **Pipeline Tests**: End-to-end execution on real-world datasets (Titanic, Iris, Housing).

### Running the Test Suite
To run the standard pipeline test:
```bash
python test_complete_pipeline.py
```

---

## 📊 Benchmarking Methodology

We evaluate OctoLearn based on three pillars:
1.  **Predictive Accuracy**: Comparison against vanilla scikit-learn baselines.
2.  **Orchestration Time**: Efficiency of the profiling and cleaning stages.
3.  **Stability**: Performance on datasets with high missingness, extreme counts, and high cardinality.

### Performance Highlights
On the Titanic dataset, the OctoLearn automated pipeline consistently achieves:
- **ROC-AUC**: 0.84+ (without manual tuning).
- **Execution Time**: < 30 seconds (including Optuna optimization).

---

## 🛠️ Verification Checklist

When updates are made to the core logic, we verify the following:
- [ ] **Preprocessing Consistency**: Ensure `transform()` applies the exact same logic as `fit_transform()`.
- [ ] **Risk Score Accuracy**: Validate that injected leakage or missingness is correctly identified.
- [ ] **Report Fidelity**: Verify that all charts and tables in the PDF correctly reflect the model benchmarks.
