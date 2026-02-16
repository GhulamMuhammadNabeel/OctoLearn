# How to Use Octolearn: Full Guide

## 🚀 Quick Start

```python
import pandas as pd
from octolearn import AutoML

df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Minimal usage (all defaults)
automl = AutoML()
automl.fit(X, y)
automl.generate_report()
```

---

## 🛠️ Full Control: User-Param Driven Pipeline


### 1. Preprocessing (Imputer, Encoder, Scaler, ID Columns)

```python
automl = AutoML(
    imputer_strategy={"numeric": "median", "categorical": "mode"},
    encoder_strategy={"ordinal_cols": ["grade"], "default": "ohe"},
    scaler="minmax",
    id_columns=["id", "customer_id"]  # Specify columns to treat as IDs
)
automl.fit(X, y)
```

**What happens if you pass ID columns?**
- Any columns listed in `id_columns` will be automatically excluded from all modeling, training, and predictions.
- These columns will NOT be used for feature engineering, model fitting, or predictions, but will be preserved in the cleaned data for reference or merging.
- This prevents data leakage and ensures IDs do not influence model results.
- You can always access the cleaned data (with IDs preserved) after fitting:
    - `automl.X_` — Cleaned features (IDs dropped for modeling, but available if you want to merge back)
    - `automl.y_` — Cleaned target
    - `automl.cleaning_log_` — Log of all cleaning steps, including which columns were dropped as IDs.

**Advanced:** If you want to keep IDs for merging predictions back to the original data, simply join on the index or the ID column after prediction.

---

### 1.1 Data Access at Any Point

- `automl.X_` — Cleaned features after all preprocessing (IDs dropped for modeling)
- `automl.y_` — Cleaned target
- `automl.profile_` — Full dataset profile (metrics, types, etc.)
- `automl.outlier_results_` — Outlier analysis
- `automl.interaction_results_` — Feature interaction analysis
- `automl.cleaning_log_` — Step-by-step cleaning log
- `automl.trained_models_` — All trained model objects
- `automl.best_model_` — Best model object
- `automl.model_benchmarks_` — List of all model results/benchmarks

You can access these attributes at any time after calling `fit()`.

---

### 1.2 Use Cases

- **Basic AutoML:** Just call `fit()` and `generate_report()` for a full pipeline.
- **Custom Preprocessing:** Pass your own imputer/encoder/scaler strategies.
- **Exclude IDs:** Prevent data leakage by specifying ID columns.
- **Model Selection:** Choose the best model by your preferred metric (e.g., F1, RMSE).
- **Registry:** Save all models for later use or deployment.
- **Data Access:** Extract cleaned data, logs, and all intermediate results for further analysis or custom workflows.

---

### 1.3 Advanced Tips

- You can run only part of the pipeline (e.g., just profiling or cleaning) by disabling later phases.
- All steps are user-param driven and can be overridden at any time.
- All outputs (plots, reports, models) are saved with unique hashes for reproducibility.
- The PDF report is fully professional, with a model benchmarks table, all metrics, and modern visuals.

---

### 2. Model Training & Registry

```python
automl = AutoML(
    train_models=True,           # Enable/disable model training
    use_registry=True,           # Save all models in trained_models/
    n_models=3,                  # Train only 3 models (faster)
    parallel_processing=True     # Use all CPU cores
)
automl.fit(X, y)
results = automl.train_auto_models()
```

### 3. User-Param Driven Evaluation Metric

```python
# Select best model by F1, RMSE, etc.
automl = AutoML(evaluation_metric="f1")
automl.fit(X, y)
results = automl.train_auto_models(evaluation_metric="f1")
```

---

## 📊 PDF Report: Model Benchmarks Table
- The PDF report includes a table of all trained models, their parameters, and all metrics.
- The best model (by your chosen metric) is always at the top.

---

## 🔍 API Reference

- `automl.get_risk_score()` — Get risk score and factors
- `automl.get_preprocessing_suggestions()` — Get preprocessing plan
- `automl.get_feature_importance()` — Get feature ranking
- `automl.get_trained_models()` — Dict of all trained models
- `automl.get_best_model()` — Best model object
- `automl.generate_report()` — Generate PDF report

---

## 🧑‍💻 Example: Full Customization

```python
automl = AutoML(
    imputer_strategy={"numeric": "knn"},
    encoder_strategy={"default": "ordinal"},
    scaler="standard",
    id_columns=["id"],
    train_models=True,
    n_models=4,
    evaluation_metric="rmse",
    use_registry=True
)
automl.fit(X, y)
results = automl.train_auto_models()
pdf_path = automl.generate_report()
```

---

## 📝 Notes
- All features are user-param driven and can be set at any point.
- All trained models are saved in `trained_models/` and tracked in the registry.
- The PDF report uses a black background, red accents, and modern font.
- See `README.md` and `ARCHITECTURE.md` for more details.

---

## 📚 More Examples
- See `notebooks/octolearn_demo.ipynb` and `notebooks/test_octolearn_full.ipynb` for end-to-end demos.
- Benchmarks: `benchmarks/README.md`

---

**Made with ☕ and 🐙 Logic.**
