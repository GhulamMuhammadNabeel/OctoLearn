"""
Comprehensive Master Debugging Test
This script runs the pipeline across multiple synthetic tasks (binary, multiclass, regression)
with various combinations of configurations to catch edge cases, errors, and warnings.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import make_classification, make_regression
from octolearn import (
    AutoML, DataConfig, ProfilingConfig, PreprocessingConfig, 
    ModelingConfig, FeatureOptimizationConfig, OptimizationConfig, ReportingConfig
)

warnings.filterwarnings('ignore')

print("="*60)
print("STARTING COMPREHENSIVE DEBUGGING RUN")
print("="*60)

# ---------------------------------------------------------
# Dataset 1: Imbalanced Binary Classification (Hard)
# ---------------------------------------------------------
print("\n[DATASET 1] Highly Imbalanced Binary Classification")
X_bin_np, y_bin_np = make_classification(
    n_samples=5000, n_features=15, n_informative=8, weights=[0.95, 0.05], random_state=42
)
X_bin = pd.DataFrame(X_bin_np, columns=[f'feat_{i}' for i in range(15)])
X_bin['cat_feat'] = np.random.choice(['A', 'B', 'C', np.nan], size=len(X_bin))
X_bin.loc[np.random.choice(X_bin.index, 100), 'feat_0'] = np.nan
y_bin = pd.Series(y_bin_np, name='target')

# Config: Use SMOTE, Enable Feature Optimization, Fast Optuna
config_bin = {
    'data_config': DataConfig(use_full_data=False, sample_size=1000, sampling_strategy='smote'),
    'feature_optimization_config': FeatureOptimizationConfig(enable_feature_optimization=True, n_trials=3, timeout=30),
    'optimization_config': OptimizationConfig(optuna_trials_per_model=3),
    'reporting_config': ReportingConfig(generate_report=False)
}

try:
    automl_bin = AutoML(**config_bin)
    automl_bin.fit(X_bin, y_bin)
    print("-> DATASET 1: SUCCESS [OK]")
except Exception as e:
    print(f"-> DATASET 1: FAILED [ERROR: {e}]")
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------
# Dataset 2: Regression with missing Data & No Feature Optimization
# ---------------------------------------------------------
print("\n[DATASET 2] Regression (No Feature Optimization)")
X_reg_np, y_reg_np = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_reg = pd.DataFrame(X_reg_np, columns=[f'rn_{i}' for i in range(20)])
y_reg = pd.Series(y_reg_np, name='price')

config_reg = {
    'data_config': DataConfig(use_full_data=True),
    'feature_optimization_config': FeatureOptimizationConfig(enable_feature_optimization=False),
    'optimization_config': OptimizationConfig(optuna_trials_per_model=2, use_optuna=False), # Fast fallback
    'reporting_config': ReportingConfig(generate_report=False)
}

try:
    automl_reg = AutoML(**config_reg)
    automl_reg.fit(X_reg, y_reg)
    print("-> DATASET 2: SUCCESS [OK]")
except Exception as e:
    print(f"-> DATASET 2: FAILED [ERROR: {e}]")
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------
# Dataset 3: Multiclass with Undersampling
# ---------------------------------------------------------
print("\n[DATASET 3] Multiclass with Undersampling")
X_mc_np, y_mc_np = make_classification(
    n_samples=2000, n_features=10, n_informative=5, n_classes=3, weights=[0.8, 0.1, 0.1], random_state=42
)
X_mc = pd.DataFrame(X_mc_np, columns=[f'v_{i}' for i in range(10)])
y_mc = pd.Series(y_mc_np, name='category')

config_mc = {
    'data_config': DataConfig(use_full_data=True, sampling_strategy='undersample'),
    'feature_optimization_config': FeatureOptimizationConfig(enable_feature_optimization=True, n_trials=2, timeout=20),
    'optimization_config': OptimizationConfig(optuna_trials_per_model=2),
    'reporting_config': ReportingConfig(generate_report=False)
}

try:
    automl_mc = AutoML(**config_mc)
    automl_mc.fit(X_mc, y_mc)
    print("-> DATASET 3: SUCCESS [OK]")
except Exception as e:
    print(f"-> DATASET 3: FAILED [ERROR: {e}]")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DEBUGGING RUN COMPLETE")
print("="*60)
