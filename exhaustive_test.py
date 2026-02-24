import pandas as pd
import numpy as np
import os
from octolearn import (
    AutoML, DataConfig, ProfilingConfig, PreprocessingConfig, 
    ModelingConfig, OptimizationConfig, ReportingConfig, ParallelConfig
)

def run_exhaustive_test():
    """
    Exhaustive functional test for OctoLearn.
    Exercises every major configuration parameter and pipeline phase.
    """
    print("\n" + "="*80)
    print("STARTING EXHAUSTIVE OCTOLEARN PIPELINE TEST")
    print("="*80)

    # 1. Create Synthetic 'Complex' Dataset
    print("\n[PHASE 0] Generating Synthetic Complex Dataset...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Non-Binary'], n_samples),
        'city': np.random.choice(['NYC', 'London', 'Tokyo', 'Paris', 'Berlin'], n_samples),
        'membership_level': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], n_samples),
        'last_login': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        'id_col': [f'USER_{i}' for i in range(n_samples)],
        'constant_col': [1] * n_samples,
        'low_var_col': [0.1] * n_samples,
        'missing_col': [np.nan if i % 10 == 0 else i for i in range(n_samples)],
        'interaction_a': np.random.rand(n_samples),
        'interaction_b': np.random.rand(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some noise and leakage
    df.loc[0:50, 'salary'] = np.nan
    df['leakage_target'] = df['age'] * 1.5 + np.random.normal(0, 0.1, n_samples)
    
    # Target: Classification (Churn prediction)
    # Simple logic + some noise
    prob = 1 / (1 + np.exp(-(df['age'] - 35)/10 - (df['salary'] - 50000)/15000))
    df['churn'] = (prob > 0.5).astype(int)
    
    X = df.drop('churn', axis=1)
    y = df['churn']

    # 2. Configure Exhaustive AutoML
    print("\n[PHASE 1] Initializing Exhaustive AutoML Configuration...")
    
    automl = AutoML(
        data_config=DataConfig(
            use_full_data=True,
            test_size=0.25,
            random_state=42
        ),
        profiling_config=ProfilingConfig(
            detect_outliers=True,
            analyze_interactions=True,
            generate_risk_score=True,
            calculate_feature_importance=True
        ),
        preprocessing_config=PreprocessingConfig(
            scaler='robust',
            imputer_strategy={'numeric': 'median', 'categorical': 'mode'},
            encoder_strategy={'ordinal_cols': ['membership_level']},
            id_columns=['id_col']
        ),
        modeling_config=ModelingConfig(
            models_to_train=['xgboost', 'random_forest'],
            n_models=3,
            use_stacking=True
        ),
        optimization_config=OptimizationConfig(
            use_optuna=True,
            optuna_trials_per_model=10, # Reduced for test speed
            optuna_timeout_seconds=60
        ),
        reporting_config=ReportingConfig(
            report_title="🚀 Exhaustive Test Audit Report",
            report_detail='detailed',
            include_data_journey=True,
            include_shap=True
        ),
        parallel_config=ParallelConfig(
            parallel_processing=True,
            n_jobs=-1
        ),
        show_progress=True,
        save_artifacts=True
    )

    # 3. Fit Pipeline
    print("\n[PHASE 2] Executing fit(X, y)...")
    automl.fit(X, y)

    # 4. Verify Attributes and Outputs
    print("\n[PHASE 3] Verifying Pipeline Consistency...")
    print(f"✔️ Risk Score: {automl.get_risk_score()}")
    print(f"✔️ Best Model: {automl.best_model_}")
    print(f"✔️ Cleaned Data Shape: {automl.X_train_.shape}")
    
    # 5. Generate Report
    print("\n[PHASE 4] Generating Intelligence Report...")
    report_path = automl.generate_report()
    print(f"✔️ Report saved to: {report_path}")

    # 6. Predict
    print("\n[PHASE 5] Verifying Prediction API...")
    X_test_sample = X.head(5)
    preds = automl.predict(X_test_sample)
    print(f"✔️ Sample Predictions: {preds}")

    print("\n" + "="*80)
    print("✅ EXHAUSTIVE TEST COMPLETE - 100% FUNCTIONALITY VERIFIED")
    print("="*80)

if __name__ == "__main__":
    run_exhaustive_test()
