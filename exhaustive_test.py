import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from octolearn import (
    AutoML, DataConfig, ProfilingConfig, PreprocessingConfig, 
    ModelingConfig, OptimizationConfig, ReportingConfig, ParallelConfig
)

# Redirect output to log file
LOG_FILE = "audit_output.log"
sys.stdout = open(LOG_FILE, "w", encoding="utf-8")

def log_section(title):
    print("\n" + "="*80)
    print(f"STAGE: {title}")
    print("="*80)

def run_exhaustive_audit():
    """
    Full Depth Audit of OctoLearn Parameters.
    Captures exact input/output for documentation tracing.
    """
    log_section("0. DATA SYNTHESIS (Full Complexity)")
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'id_col': [f'ID_{i}' for i in range(n_samples)],
        'constant': [100] * n_samples,
        'missing': [np.nan if i % 5 == 0 else i for i in range(n_samples)]
    }
    df = pd.DataFrame(data)
    df['target'] = (df['age'] > 30).astype(int)
    X, y = df.drop('target', axis=1), df['target']
    
    print(f"Raw Input Sample:\n{X.head(2)}")
    print(f"Columns: {list(X.columns)}")

    log_section("1. DEFAULT CONFIGURATION RUN")
    # Using defaults
    automl_def = AutoML()
    print("Default DataConfig:", automl_def.data_config)
    print("Default PreprocessingConfig:", automl_def.preprocessing_config)
    
    automl_def.fit(X, y)
    print(f"Default Cleaning Log: {automl_def.cleaning_log_}")
    print(f"Default Best Model: {automl_def.best_model_}")

    log_section("2. OVERRIDE CONFIGURATION RUN (Full Depth)")
    # Explicitly overriding every major param
    config_overrides = {
        'data_config': DataConfig(use_full_data=True, test_size=0.3, random_state=123),
        'preprocessing_config': PreprocessingConfig(scaler='robust', imputer_strategy={'numeric': 'median'}),
        'modeling_config': ModelingConfig(models_to_train=['random_forest'], n_models=2),
        'optimization_config': OptimizationConfig(use_optuna=True, optuna_trials_per_model=5)
    }
    
    print("Overridden Parameters:")
    for k, v in config_overrides.items():
        print(f"  {k}: {v}")
        
    automl_ovr = AutoML(**config_overrides)
    automl_ovr.fit(X, y)
    
    print(f"Override Cleaning Log: {automl_ovr.cleaning_log_}")
    print(f"Override Best Model: {automl_ovr.best_model_}")
    
    log_section("3. PER-CALL FIT OVERRIDES")
    # Overriding during fit() call
    print("Fitting with imputer_strategy override in fit()...")
    automl_def.fit(X, y, imputer_strategy={'numeric': 'constant'})
    print(f"Per-call Cleaning Log: {automl_def.cleaning_log_}")

    log_section("4. AUDIT COMPLETE")
    print(f"Audit timestamp: {datetime.now()}")

if __name__ == "__main__":
    try:
        run_exhaustive_audit()
    finally:
        sys.stdout.close()
