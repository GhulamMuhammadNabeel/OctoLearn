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

## The Lifecycle of an AutoML Run: A Technical Deep Dive

OctoLearn is designed for high observability. Below is a step-by-step trace of how the core orchestrator processes a dataset, comparing **Default behavior** against **Overridden configurations**.

### Phase 1: Ingestion & Smart Sampling
The `AutoML` constructor sets the stage. Sampling ensures performance on large datasets without losing statistical significance.

???+ example "Chunk 1: Initialization"
    ```python
    # Path A: Default (Safe Sampling)
    automl = AutoML() 
    
    # Path B: Full Depth (No Sampling)
    automl = AutoML(data_config=DataConfig(use_full_data=True))
    ```
    
    | Parameter | Default (`False`) | Override (`True`) |
    |-----------|-------------------|-------------------|
    | `use_full_data` | Samples 500 rows for speed. | Processes every single row. |
    | `stratify_target` | Ensures class balance in split. | (Same by default) |

---

### Phase 2: Structural Foundation (AutoCleaner)
Before ML begins, OctoLearn cleans the "noise".

???+ observation "Chunk 2: Structural Logs"
    ```text
    INFO - [AutoCleaner] Starting structural transformation...
    INFO - Dropped ID columns: ['user_id', 'uuid']
    INFO - Dropped Constant columns: ['is_human', 'version']
    INFO - Removed 14 duplicates rows.
    ```

    **Parameter Impact**:
    - `auto_clean=True`: Automatically handles the drops above.
    - `auto_clean=False` (NOT RECOMMENDED): Passes IDs to the model, likely causing overfitting or target leakage.

---

### Phase 3: Intelligent Preprocessing
OctoLearn chooses imputation and scaling strategies based on distribution patterns.

???+ bug "What if we override strategies?"
    ```python
    # Overriding Imputation Logic
    config = PreprocessingConfig(
        imputer_strategy={'numeric': 'median', 'categorical': 'mode'},
        scaler='robust'
    )
    automl = AutoML(preprocessing_config=config)
    ```

    | Component | Default | Override Result |
    |-----------|---------|-----------------|
    | **Numeric Imputer** | `mean` (standard) | `median` (handled skewed outliers) |
    | **Scaler** | `standard` | `robust` (used Interquartile Range) |
    | **Final State** | Matrix: 160x12 | Matrix: 160x12 (different distribution) |

---

### Phase 4: The Model Arena (Bayesian HPO)
This is where the magic happens. OctoLearn doesn't just train; it evolves.

???+ success "Expected Output: Optuna Trace"
    ```text
    INFO - Starting Bayesian Optimization for XGBoost...
    [Trial 1] Value: 0.842 (Params: max_depth=3, lr=0.1)
    [Trial 2] Value: 0.875 (Params: max_depth=6, lr=0.01) [New Best]
    INFO - Best Model: XGBoost with 0.912 F1-Score
    ```

    | Parameter | Default (`True`) | Override (`False`) |
    |-----------|-------------------|-------------------|
    | `use_optuna` | Runs 20 trials of HPO. | Trains model with default params. |
    | `use_stacking` | Blends top 3 into an ensemble. | Returns a single best model. |

---

### Phase 5: Delivery & Insight
The final artifacts are generated based on the `ReportingConfig`.

???+ info "Chunk 5: Artifact Registry"
    - **`trained_models_`**: Dictionary of all fitted sklearn/XGB objects.
    - **`best_model_`**: The winning serialized model.
    - **`octolearn_report.pdf`**: The full visual trace of the journey above.

---

## Verification Checklist

When updates are made to the core logic, we verify the following:
- [x] **Preprocessing Consistency**: `transform()` matches `fit()`.
- [x] **Risk Score Accuracy**: Validated via high-missingness synthetic data.
- [x] **Report Fidelity**: Verified across 10+ PDF compositions.
- [x] **UI Responsiveness**: Verified header autohide and mobile logo scaling.
