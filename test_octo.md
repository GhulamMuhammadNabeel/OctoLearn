# OctoLearn - Complete Usage Guide 📚

**Version**: 0.4.0  
**Purpose**: Automated ML Pipeline Library for Python  
**Use Case**: AutoML with full user control over each pipeline phase

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Phase 1: Dataset Profiling](#phase-1-dataset-profiling)
3. [Phase 2: Exploratory Data Analysis](#phase-2-exploratory-data-analysis)
4. [Phase 3: Feature Engineering & Auto-Cleaning](#phase-3-feature-engineering--auto-cleaning)
5. [Phase 4: Model Training & Optimization](#phase-4-model-training--optimization)
6. [Complete End-to-End Pipeline](#complete-end-to-end-pipeline)
7. [Control Points & Parameters](#control-points--parameters)
8. [Advanced Examples](#advanced-examples)
9. [API Reference](#api-reference)

---

## Quick Start

### Installation

```python
# OctoLearn is a library - import and use in your code
from octolearn import AutoML
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
```

### Minimal Example (3 lines)

```python
from octolearn import AutoML

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').iloc[:, 0]

# Run full pipeline (Phase 1-4)
automl = AutoML(train_models=True, use_registry=True)
automl.fit(X, y)
pdf_report = automl.generate_report()
best_model = automl.get_best_model()
predictions = best_model.predict(X_new)
```

**That's it!** ✅ AutoML pipeline complete with:
- ✅ Dataset profiling
- ✅ Risk assessment
- ✅ Auto-cleaning
- ✅ Multiple model training
- ✅ Hyperparameter optimization
- ✅ Model registry
- ✅ PDF report

---

## Phase 1: Dataset Profiling

### What It Does (Automatic)

Analyzes your dataset and extracts 16 intelligent metrics:

```python
from octolearn import AutoML

automl = AutoML()
automl.fit(X, y)

# Access the profile
profile = automl.report()

# ✅ Now you have:
# - Feature types (numeric, categorical, datetime, id-like)
# - Missing value analysis
# - Duplicate detection
# - Skewness analysis
# - Low variance columns
# - Leakage suspects (potential target leakage)
# - Task type detection (classification vs regression)
# - Class imbalance ratio
# - And 8 more metrics...
```

### Control Points in Phase 1

```python
# Control Point 1: Data Sampling
automl = AutoML(
    use_full_data=False,    # Sample for speed
    sample_size=500         # Rows to sample (adjust as needed)
)

# Control Point 2: Progress Visibility
automl = AutoML(
    show_progress=True      # See what's happening at each step
)
```

### Example: Quick Dataset Scan

```python
from octolearn import AutoML
import pandas as pd

# Your data
X = pd.read_csv('data.csv')
y = pd.read_csv('target.csv').iloc[:, 0]

# Scan dataset (Phase 1 only)
automl = AutoML(
    use_full_data=True,              # Analyze everything
    train_models=False,              # Skip model training for now
    generate_shap=False              # Skip expensive SHAP plots
)

automl.fit(X, y)
profile = automl.report()

print(f"✅ Dataset Profile:")
print(f"  Shape: {profile.n_rows} rows × {profile.n_columns} columns")
print(f"  Task: {profile.task_type}")
print(f"  Numeric features: {len(profile.numeric_features)}")
print(f"  Categorical features: {len(profile.categorical_features)}")
print(f"  Missing values: {sum(profile.missing_report.values()):.1%}")
print(f"  Duplicates: {profile.duplicate_rows}")
print(f"  Potential leakage: {profile.leakage_suspects}")
```

---

## Phase 2: Exploratory Data Analysis

### What It Does (Automatic + Reports)

Generates comprehensive visualization and analysis:

```python
from octolearn import AutoML

automl = AutoML()
automl.fit(X, y)

# Generate professional PDF report
pdf_path = automl.generate_report()
# 📄 Creates: octolearn_report_[HASH].pdf with:
#   - Feature distributions
#   - Correlation heatmap
#   - SHAP explanations
#   - Feature importance ranking
#   - Risk score (0-100)
#   - Preprocessing recommendations
#   - Strategic recommendations
```

### Control Points in Phase 2

```python
# Control Point 1: SHAP Explanations
automl = AutoML(
    generate_shap=True      # CPU-intensive, can disable
)

# Control Point 2: Feature Importance
automl = AutoML(
    calculate_feature_importance=True   # Include top features
)

# Control Point 3: Recommendations
automl = AutoML(
    generate_recommendations=True       # Get strategic advice
)

# Control Point 4: Parallel Processing
automl = AutoML(
    parallel_workers=7              # Threads for generation
    parallel_processing=True        # Use threads (vs sequential)
)
```

### Example: Risk Assessment Only

```python
from octolearn import AutoML

automl = AutoML(
    train_models=False,         # No model training
    generate_shap=False,        # No expensive plots
    parallel_workers=1          # Sequential processing
)

automl.fit(X, y)

# Get risk score
risk_data = automl.get_risk_score()
print(f"Risk Score: {risk_data['score']}/100")
print(f"Category: {risk_data['category']}")
print(f"Risk Factors: {risk_data['factors']}")

# Get preprocessing advice
suggestions = automl.get_preprocessing_suggestions()
for category, items in suggestions.items():
    print(f"\n{category.upper()}:")
    for item in items:
        print(f"  - {item}")
```

### Example: Feature Importance Analysis

```python
from octolearn import AutoML

automl = AutoML(
    train_models=False,              # Focus on EDA only
    calculate_feature_importance=True
)

automl.fit(X, y)

# Get feature importance
importance = automl.get_feature_importance()

print("Top 10 Most Important Features:")
for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
    print(f"{i:2d}. {feature:20s} : {score:.4f}")
```

---

## Phase 3: Feature Engineering & Auto-Cleaning

### What It Does (Automatic Preprocessing)

Three advanced preprocessing steps:

#### 3.1 Outlier Detection

```python
from octolearn import AutoML

automl = AutoML(
    detect_outliers=True    # Enable (default)
)

automl.fit(X, y)

# Get outlier analysis
outliers = automl.get_outlier_analysis()

print(f"Outliers found by:")
print(f"  - IQR method: {outliers['methods']['iqr']['n_outliers']}")
print(f"  - Isolation Forest: {outliers['methods']['isolation_forest']['n_outliers']}")
print(f"  - Z-score: {outliers['methods']['zscore']['n_outliers']}")
print(f"Severity: {outliers['summary']['severity']}")
```

#### 3.2 Feature Interactions

```python
from octolearn import AutoML

automl = AutoML(
    analyze_interactions=True   # Enable (default)
)

automl.fit(X, y)

# Get interaction analysis
interactions = automl.get_interaction_analysis()

print(f"Feature Interactions Found:")
print(f"  - Polynomial (degree-2): {interactions['polynomial_interactions']['n_interactions']}")
print(f"  - Pairwise (x1*x2): {interactions['pairwise_interactions']['n_interactions']}")
print(f"  - Ratio (x1/x2): {interactions['ratio_interactions']['n_interactions']}")

# Top interactions
for inter in interactions['polynomial_interactions']['top_interactions'][:5]:
    print(f"    {inter['interaction']}: correlation {inter['correlation']:.4f}")
```

#### 3.3 Automatic Data Cleaning

```python
from octolearn import AutoML

automl = AutoML(
    auto_clean=True     # Enable (default)
)

automl.fit(X, y)

# Get cleaning report
cleaning = automl.get_cleaning_log()

print("Data Cleaning Report:")
print(f"  Duplicates removed: {cleaning.get('duplicates_removed', 0)}")
print(f"  ID columns removed: {cleaning.get('id_columns_removed', [])}")
print(f"  Constant columns removed: {cleaning.get('constant_columns_removed', [])}")
print(f"  Low variance columns: {cleaning.get('low_variance_columns_removed', [])}")
print(f"  Missing values imputed: {cleaning.get('imputed_columns', {})}")
```

### Control Points in Phase 3

```python
# Control Point 1: Outlier Detection
automl = AutoML(
    detect_outliers=True    # True/False
)

# Control Point 2: Feature Interactions
automl = AutoML(
    analyze_interactions=True   # True/False
)

# Control Point 3: Auto-Cleaning
automl = AutoML(
    auto_clean=True     # True/False
)

# Example: Skip auto-cleaning and do it manually
automl = AutoML(
    auto_clean=False    # User will clean manually
)

automl.fit(X, y)

# User can now get the raw analysis
outliers = automl.get_outlier_analysis()
interactions = automl.get_interaction_analysis()

# User decides what to do with this information
# and can manually clean their data
```

---

## Phase 4: Model Training & Optimization

### What It Does (Automatic Model Training)

Trains multiple models with intelligent hyperparameter optimization:

```python
from octolearn import AutoML

automl = AutoML(
    train_models=True,      # Enable Phase 4
    use_optuna=True,        # Use Optuna for HPO
    use_registry=True,      # Save models with versioning
    parallel_processing=True
)

automl.fit(X, y)

# Train models automatically
results = automl.train_auto_models()

print(f"✅ Model Training Complete:")
print(f"  Best Model: {results['best_model']}")
print(f"  Best Score: {results['best_score']:.4f}")
print(f"Models trained:")
for name, score in results['model_scores'].items():
    print(f"  - {name}: train={score['train']:.4f}, test={score['test']:.4f}")
```

### Models Trained (Automatic Selection)

**For Classification**:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- SVM

**For Regression**:
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- SVR

### Hyperparameter Optimization

Each model gets 50 trials with:
- ✅ **TPE Sampler** (Tree-structured Parzen Estimator)
- ✅ **Early Stopping** (MedianPruner)
- ✅ **5-Fold Cross-Validation**
- ✅ **Smart Search Spaces** (configured per model)

### Control Points in Phase 4

```python
# Control Point 1: Model Training
automl = AutoML(
    train_models=True       # Enable (default)
)

# Control Point 2: Hyperparameter Optimization
automl = AutoML(
    use_optuna=True         # Use Optuna (vs simple training)
)

# Control Point 3: Model Registry
automl = AutoML(
    use_registry=True       # Save models (default JSON, optional SQLite)
)

# Control Point 4: Number of Models
automl = AutoML(
    n_models=5              # How many models to train (0-6)
)

# Control Point 5: Parallel Processing
automl = AutoML(
    parallel_processing=True    # Train models in parallel
)
```

### Example: Train Fewer Models (Faster)

```python
from octolearn import AutoML

automl = AutoML(
    train_models=True,
    n_models=3,             # Train only 3 models (faster)
    use_optuna=True,        # Still optimize them well
    parallel_processing=True
)

automl.fit(X, y)
results = automl.train_auto_models()
best = automl.get_best_model()
```

### Example: Get All Trained Models

```python
from octolearn import AutoML

automl = AutoML(train_models=True)
automl.fit(X, y)
automl.train_auto_models()

# Get all models
models = automl.get_trained_models()

print("Trained Models:")
for model_name, model_obj in models['models'].items():
    score = models['scores'].get(model_name, {}).get('test', 0)
    print(f"  {model_name}: {score:.4f}")

# Get best model
best_model = automl.get_best_model()
predictions = best_model.predict(X_new)
```

### Example: Comprehensive Model Evaluation

```python
from octolearn import AutoML

automl = AutoML(train_models=True)
automl.fit(X, y)
automl.train_auto_models()

# Evaluate best model comprehensively
evaluation = automl.evaluate_best_model()

if 'test_metrics' in evaluation:
    metrics = evaluation['test_metrics']
    print("Model Performance:")
    if 'accuracy' in metrics:  # Classification
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
    else:  # Regression
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
```

---

## Complete End-to-End Pipeline

### Scenario 1: Default Full Pipeline (Recommended)

**Use Case**: You want everything automatic

```python
from octolearn import AutoML
import pandas as pd

# Load data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv').iloc[:, 0]

# Create AutoML with defaults
automl = AutoML()

# Phase 1-3: Fit and prepare
automl.fit(X, y)

# Phase 2: Generate report
pdf_path = automl.generate_report()
print(f"📊 Report saved: {pdf_path}")

# Phase 4: Train models
results = automl.train_auto_models()
print(f"✅ Best model: {results['best_model']}")

# Evaluate
eval_results = automl.evaluate_best_model()
print(f"📈 Best score: {results['best_score']:.4f}")

# Use model
best_model = automl.get_best_model()
predictions = best_model.predict(X_test)

# Access everything
profile = automl.report()
risk = automl.get_risk_score()
importance = automl.get_feature_importance()
outliers = automl.get_outlier_analysis()
interactions = automl.get_interaction_analysis()
cleaning = automl.get_cleaning_log()
trained_models = automl.get_trained_models()
```

### Scenario 2: EDA Only (No Model Training)

**Use Case**: Explore data first, decide on modeling later

```python
from octolearn import AutoML

automl = AutoML(
    train_models=False,         # Skip Phase 4
    generate_shap=False,        # Skip expensive plots
    sample_size=1000            # Fast analysis
)

automl.fit(X, y)

# Analyze data
pdf = automl.generate_report()

# Extract insights
risk = automl.get_risk_score()
suggestions = automl.get_preprocessing_suggestions()
importance = automl.get_feature_importance()

print(f"Risk Score: {risk['score']}/100")
print(f"Risk Category: {risk['category']}")
print(f"\nTop Preprocessing Actions:")
for action in suggestions['column_actions'][:5]:
    print(f"  - {action}")

# Now user decides what to do:
# 1. Apply suggestions manually
# 2. Clean data themselves
# 3. Then train models with Phase 4
```

### Scenario 3: Custom Feature Engineering

**Use Case**: You want to apply custom transformations

```python
from octolearn import AutoML
from sklearn.preprocessing import StandardScaler

automl = AutoML(
    auto_clean=False,           # Skip auto-cleaning
    analyze_interactions=True   # But get interaction suggestions
)

automl.fit(X, y)

# Get analysis
interactions = automl.get_interaction_analysis()
outliers = automl.get_outlier_analysis()

# User applies own transformations
print("Suggested interactions:", interactions['polynomial_interactions']['top_interactions'])
print("Outliers detected:", outliers['summary'])

# User manually transforms data
X_transformed = X.copy()

# Apply custom scaling
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_transformed[automl.profile_.numeric_features])

# Train models on transformed data
from octolearn.models import ModelTrainer
trainer = ModelTrainer(X_transformed, y, automl.profile_)
trainer.train_all_models()
```

### Scenario 4: Production Workflow

**Use Case**: Automated pipeline for production

```python
from octolearn import AutoML
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Automated pipeline
def automated_ml_pipeline(X, y, model_path='./models'):
    """Production-ready ML pipeline."""
    
    # Initialize
    automl = AutoML(
        use_full_data=True,
        train_models=True,
        use_optuna=True,
        use_registry=True,
        parallel_processing=True,
        show_progress=True
    )
    
    # Execute
    try:
        # Phase 1-3
        automl.fit(X, y)
        print("✅ Data preprocessing complete")
        
        # Phase 2
        report = automl.generate_report()
        print(f"✅ Report generated: {report}")
        
        # Phase 4
        results = automl.train_auto_models()
        print(f"✅ Models trained. Best: {results['best_model']}")
        
        # Evaluate
        eval_results = automl.evaluate_best_model()
        print(f"✅ Evaluation complete. Score: {results['best_score']:.4f}")
        
        # Return best model
        return automl.get_best_model()
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None

# Use it
best_model = automated_ml_pipeline(X_train, y_train)
if best_model:
    predictions = best_model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Control Points & Parameters

## Quick Reference Table

| Parameter | Phase | Default | Purpose | Checkpoint |
|-----------|-------|---------|---------|------------|
| `use_full_data` | 1 | False | Use full dataset or sample | ✅ Speed vs accuracy |
| `sample_size` | 1 | 500 | Rows to sample | ✅ Dataset size |
| `show_progress` | All | True | Print status messages | ✅ Visibility |
| `parallel_workers` | 2 | 7 | Report generation threads | ✅ Speed vs memory |
| `generate_shap` | 2 | True | SHAP explanations | ✅ Time vs insights |
| `calculate_feature_importance` | 2 | True | Feature importance | ✅ Include in report |
| `generate_recommendations` | 2 | True | Strategic recommendations | ✅ Include in report |
| `detect_outliers` | 3 | True | Outlier detection | ✅ Data quality |
| `analyze_interactions` | 3 | True | Feature interactions | ✅ Feature engineering |
| `auto_clean` | 3 | True | Auto data cleaning | ✅ Data preprocessing |
| `train_models` | 4 | True | Train models | ✅ Model phase |
| `use_optuna` | 4 | True | Optuna HPO | ✅ Optimization depth |
| `use_registry` | 4 | True | Model registry | ✅ Model versioning |
| `parallel_processing` | 4 | True | Parallel training | ✅ Training speed |
| `n_models` | 4 | 5 | Number of models | ✅ Model diversity |

---

## Advanced Examples

### Example 1: Fast Prototyping (2 seconds)

```python
from octolearn import AutoML

automl = AutoML(
    use_full_data=False,
    sample_size=100,
    generate_shap=False,
    train_models=False,
    parallel_processing=False,
    show_progress=False
)

automl.fit(X, y)
profile = automl.report()
risk = automl.get_risk_score()

print(f"Quick Analysis: Risk={risk['score']}, Rows={profile.n_rows}")
```

### Example 2: Detailed Analysis (Manual Control)

```python
from octolearn import AutoML

# Phase 1: Profile
print("Phase 1: Profiling...")
automl = AutoML(
    use_full_data=True,
    train_models=False
)
automl.fit(X, y)
print(f"✅ Dataset profiled")

# CHECKPOINT 1: User reviews profile
profile = automl.report()
print(f"Found {len(profile.id_like_columns)} ID columns")
print(f"Found {len(profile.leakage_suspects)} leakage suspects")

# CHECKPOINT 2: User decides next steps
user_input = input("Continue to Phase 2? (y/n): ")
if user_input == 'y':
    # Phase 2: EDA
    print("Phase 2: EDA...")
    report = automl.generate_report()
    print(f"✅ Report generated: {report}")

# CHECKPOINT 3: Review risk assessment
risk = automl.get_risk_score()
if risk['score'] > 50:
    print(f"⚠️ High risk data detected! Score={risk['score']}")
    actions = input("Auto-clean data? (y/n): ")
    if actions == 'y':
        # Phase 3: Auto-clean
        automl.auto_clean = True
        automl.fit(X, y)
        cleaning = automl.get_cleaning_log()
        print(f"✅ Data cleaned. Removed {cleaning.get('rows_removed', 0)} rows")
else:
    print(f"✅ Low risk data. Score={risk['score']}")

# CHECKPOINT 4: Model training
user_input = input("Train models? (y/n): ")
if user_input == 'y':
    # Phase 4: Model training
    print("Phase 4: Training...")
    results = automl.train_auto_models()
    print(f"✅ Best model: {results['best_model']} ({results['best_score']:.4f})")
```

### Example 3: Conditional Pipelines

```python
from octolearn import AutoML

def smart_automl_pipeline(X, y, task_complexity='auto'):
    """Auto-adjust complexity based on dataset size and characteristics."""
    
    if task_complexity == 'auto':
        # Determine complexity
        if X.shape[0] > 100000:
            task_complexity = 'light'
        elif X.shape[0] > 10000:
            task_complexity = 'medium'
        else:
            task_complexity = 'heavy'
    
    # Configure based on complexity
    if task_complexity == 'light':
        automl = AutoML(
            use_full_data=False,
            sample_size=5000,
            generate_shap=False,
            parallel_processing=True,
            n_models=3
        )
    
    elif task_complexity == 'medium':
        automl = AutoML(
            use_full_data=True,
            generate_shap=True,
            n_models=5
        )
    
    else:  # heavy
        automl = AutoML(
            use_full_data=True,
            generate_shap=True,
            use_optuna=True,
            n_models=6
        )
    
    # Run pipeline
    automl.fit(X, y)
    automl.generate_report()
    results = automl.train_auto_models()
    
    return automl.get_best_model()

# Use it
best_model = smart_automl_pipeline(X, y)
```

### Example 4: Model Comparison & Selection

```python
from octolearn import AutoML

automl = AutoML(train_models=True, n_models=6)
automl.fit(X, y)
automl.train_auto_models()

# Get all models
models = automl.get_trained_models()

# Compare
print("Model Comparison:")
print("-" * 60)
print(f"{'Model':<20} {'Train Score':<15} {'Test Score':<15}")
print("-" * 60)
for model_name in models['models'].keys():
    train_score = models['scores'][model_name]['train']
    test_score = models['scores'][model_name]['test']
    gap = train_score - test_score
    overfitting = "⚠️ OVERFIT" if gap > 0.1 else "✅ OK"
    print(f"{model_name:<20} {train_score:<15.4f} {test_score:<15.4f} {overfitting}")

# Select best
best = automl.get_best_model()
print(f"\nSelected: {models['best_model']}")
```

---

## API Reference

### AutoML Class

#### Constructor: `AutoML(...)`

```python
automl = AutoML(
    # Phase 1 Control
    use_full_data=False,
    sample_size=500,
    show_progress=True,
    
    # Phase 2 Control
    generate_shap=True,
    calculate_feature_importance=True,
    generate_recommendations=True,
    parallel_workers=7,
    parallel_processing=True,
    
    # Phase 3 Control
    detect_outliers=True,
    analyze_interactions=True,
    auto_clean=True,
    
    # Phase 4 Control
    train_models=True,
    use_optuna=True,
    use_registry=True,
    n_models=5
)
```

#### Phase 1: `fit(X, y)`

```python
automl.fit(X_train, y_train)
# Returns: self (for chaining)
# Executes: Phases 1-3 of pipeline
```

#### Phase 2: `generate_report()`

```python
pdf_path = automl.generate_report()
# Returns: str (path to PDF file)
# Creates: Comprehensive report with visualizations
```

#### Phase 4: `train_auto_models()`

```python
results = automl.train_auto_models()
# Returns: dict with model info
# Trains: All models with Optuna HPO
```

#### Phase 4: `evaluate_best_model()`

```python
eval_results = automl.evaluate_best_model()
# Returns: dict with evaluation metrics
# Evaluates: Best model comprehensively
```

#### Getters (Access Results)

```python
profile = automl.report()                           # Phase 1 result
risk = automl.get_risk_score()                      # Phase 2 result
importance = automl.get_feature_importance()        # Phase 2 result
outliers = automl.get_outlier_analysis()            # Phase 3 result
interactions = automl.get_interaction_analysis()    # Phase 3 result
cleaning = automl.get_cleaning_log()                # Phase 3 result
models = automl.get_trained_models()                # Phase 4 result
best_model = automl.get_best_model()                # Phase 4 result
```

---

## Summary: When to Use Each Control Point

| Use Case | Configuration |
|----------|---------------|
| **Fast prototype** | `sample_size=100, generate_shap=False, train_models=False` |
| **Detailed EDA** | `use_full_data=True, generate_shap=True, train_models=False` |
| **Production** | `use_full_data=True, use_optuna=True, use_registry=True` |
| **Custom preprocessing** | `auto_clean=False` (then process manually) |
| **Low latency** | `parallel_processing=True, n_models=3` |
| **High accuracy** | `use_full_data=True, n_models=6, use_optuna=True` |

---

## Benefits Summary

### ✅ What OctoLearn Gives You

1. **Automated Intelligence**
   - 16 dataset metrics automatically extracted
   - Risk scoring (0-100)
   - Preprocessing recommendations
   - Feature importance analysis

2. **Full Control via Parameters**
   - 13+ parameters to fine-tune pipeline
   - Skip/enable any phase
   - Adjust sample size, parallelization, etc.
   - Checkpoints between phases

3. **Intelligent Preprocessing (Phase 3)**
   - Multi-method outlier detection
   - Automatic feature interactions
   - Smart data cleaning pipeline
   - All configurable

4. **State-of-the-Art Model Training (Phase 4)**
   - 6 classification / 6 regression models
   - Optuna hyperparameter optimization
   - Model registry with versioning
   - Cross-validation evaluation

5. **Professional Reports**
   - PDF with visualizations
   - Risk assessment
   - Preprocessing strategy
   - Feature importance rankings

---

## Conclusion

**OctoLearn is your ML pipeline automation library:**

- 🎯 **Automatic**: Default behavior handles everything
- 🎮 **Controllable**: 13+ parameters for fine-tuning each phase
- 📊 **Intelligent**: Advanced profiling, outlier detection, HPO
- 📧 **Professional**: PDF reports, model registry, evaluation
- 🚀 **Production-ready**: Error handling, logging, validation

---

**Start using OctoLearn:**

```python
from octolearn import AutoML

automl = AutoML()
automl.fit(X, y)
best_model = automl.get_best_model()
predictions = best_model.predict(X_new)
```

**That's it!** 🐙
