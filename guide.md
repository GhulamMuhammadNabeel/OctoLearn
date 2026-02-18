<p align="center">
  <img src="octolearn/images/logo.png" alt="OctoLearn Logo" width="200"/>
</p>
# OctoLearn User Guide

Welcome to the comprehensive guide for **OctoLearn**, the enterprise-grade AutoML library designed for robustness, transparency, and professional reporting.

## 📚 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Core Concepts](#-core-concepts)
4. [Cookbook & Examples](#-cookbook--examples)
5. [Configuration Reference](#-configuration-reference)
6. [Reporting & Visualization](#-reporting--visualization)
7. [Advanced Features](#-advanced-features)
8. [Troubleshooting](#-troubleshooting)

---

## 📦 Installation

OctoLearn requires Python 3.8 or higher.

```bash
# Clone the repository
git clone https://github.com/GhulamMuhammadNabeel/OctoLearn.git
cd OctoLearn

# Create a virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

### Verification
To verify your installation, run the test suite:
```bash
python test_complete_pipeline.py
```

---

## 🚀 Quick Start

The fastest way to get value from OctoLearn is the one-line pipeline.

### Basic Classification

```python
import pandas as pd
from octolearn import AutoML

# 1. Load Data
df = pd.read_csv("titanic.csv")
X = df.drop("survived", axis=1)
y = df["survived"]

# 2. Initialize & Run
automl = AutoML()
automl.fit(X, y)

# 3. Generate Insights
automl.generate_report("titanic_report.pdf")
print("Best model:", automl.best_model_)
```

### Basic Regression

OctoLearn automatically detects regression tasks based on the target variable.

```python
# Assuming 'price' is a continuous variable
y_reg = df["price"]
X_reg = df.drop("price", axis=1)

automl_reg = AutoML()
automl_reg.fit(X_reg, y_reg)
```

---

## 🧠 Core Concepts

### The Pipeline
OctoLearn is built as a sequential pipeline:
1.  **Profiling**: Understands your data before touching it.
2.  **Cleaning**: Imputes missing values, encodes categoricals, removes outliers.
3.  **Engineering**: Creates new features (interactions) and detects deeper issues.
4.  **Training**: Trains multiple models with Optuna hyperparameter optimization.
5.  **Reporting**: Generates a PDF report summarizing the entire journey.

### Configuration Objects vs. Overrides
You can configure OctoLearn in two ways:

1.  **Global Configuration** (via `AutoML` constructor): Best for production pipelines where settings are fixed.
2.  **Runtime Overrides** (via `fit` method): Best for experimentation.

**Example: Override for Quick Experiment**
```python
# Run only 2 models, no Optuna, limit iterations
automl.fit(X, y, 
    n_models=2, 
    use_optuna=False, 
    optuna_trials=0
)
```

---

## 🍲 Cookbook & Examples

### 1. Data Profiling Only (No Training)
Useful for initial data exploration.

```python
automl = AutoML(train_models=False)
automl.fit(X, y)

print(automl.raw_profile_.missing_ratio)
print(automl.get_risk_score())
```

### 2. Customizing Data Cleaning
Override default imputation or scaling.

```python
from octolearn import AutoML, PreprocessingConfig

config = PreprocessingConfig(
    imputer_strategy={"numeric": "mean", "categorical": "constant"},
    scaler="minmax",  # Default is 'standard'
    id_columns=["passenger_id", "ticket"] # Force drop these
)
automl = AutoML(preprocessing_config=config)
automl.fit(X, y)
```

### 3. Production Training (High Accuracy)
Increase trials and timeout for better models.

```python
from octolearn import AutoML, OptimizationConfig, ModelingConfig

automl = AutoML(
    optimization_config=OptimizationConfig(
        optuna_trials_per_model=100,
        optuna_timeout_seconds=3600
    ),
    modeling_config=ModelingConfig(
        metrics="f1_macro" # Optimize for F1 Macro
    )
)
automl.fit(X, y)
```

---

## ⚙️ Configuration Reference

### `DataConfig`
Controls data loading and splitting.
- `sample_size` (int): Rows to sample for speed (default: 500).
- `test_size` (float): Test split ratio (default: 0.2).
- `stratify_target` (bool): Stratify split for classification (default: True).

### `ProfilingConfig`
Controls analysis depth.
- `detect_outliers` (bool): Run outlier detection (default: True).
- `analyze_interactions` (bool): Run interaction analysis (expensive) (default: False).

### `ReportingConfig`
Controls PDF output.
- `report_detail` (str): 'brief' or 'detailed'.
- `color_scheme` (str): 'light' (default) or 'dark'.
- `visuals_limit` (int): Max plots in report.

---

## 📊 Reporting & Visualization

### Professional PDF Reports
OctoLearn generates magazine-quality PDF reports.

```python
automl.generate_report("my_analysis.pdf")
```

**New in v0.8.0**:
- **Light Theme**: Reports now use a clean, white-background theme for all plots, ensuring they look great on paper/PDF.
- **Margins**: Professional 0.75" margins are standard.
- **Enhanced Plots**: Correlation heatmaps and feature importance charts are styled to match the report palette (Navy/Red/Grey).

### Standalone Visuals
You can also use the plotting engine directly:

```python
from octolearn.experiments.plot_generator import PlotGenerator

plotter = PlotGenerator(X, y, profile, theme='light')
plotter.generate_correlation_heatmap()
```

---

## ⚠️ Troubleshooting

**Q: Optuna hangs on Windows?**
A: OctoLearn automatically sets `n_jobs=1` for Optuna on Windows to prevent multiprocessing crashes. Do not override this unless you are sure using `backend='threading'`.

**Q: "Input contains infinity or a value too large for dtype('float64')"**
A: Check if your dataset has very large numbers or unhandled nulls. OctoLearn's `AutoCleaner` handles most, but extreme outliers might need manual cleanup.

**Q: Report generation fails with "Permission denied"**
A: Ensure the PDF file is not open in another application (e.g., Acrobat, Browser) when running `generate_report()`.

---
*Generated for OctoLearn v0.8.0*
