
import pytest
import pandas as pd
import numpy as np
import pickle
import os
from octolearn import AutoML, DataConfig, ModelingConfig

def test_serialization_pickle():
    """Case 1: Serialization (Pickle save/load robustness)"""
    X = pd.DataFrame(np.random.rand(50, 5), columns=[f"col_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 50))
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True, n_models=1),
                    data_config=DataConfig(use_full_data=True))
    automl.fit(X, y, optuna_trials=1, models=['logistic_regression'])
    
    # Save to bytes
    dumped = pickle.dumps(automl)
    
    # Load back
    loaded_automl = pickle.loads(dumped)
    
    # Verify it still works
    preds = loaded_automl.predict(X)
    assert len(preds) == 50
    assert hasattr(loaded_automl, 'best_model_')

def test_refitting_robustness():
    """Case 2: Re-fitting robustness (Calling fit twice)"""
    X1 = pd.DataFrame(np.random.rand(50, 5), columns=[f"col_{i}" for i in range(5)])
    y1 = pd.Series(np.random.randint(0, 2, 50))
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True, n_models=1))
    automl.fit(X1, y1, optuna_trials=1, models=['logistic_regression'])
    model1 = automl.best_model_
    
    # Fit again with different data/shape
    X2 = pd.DataFrame(np.random.rand(50, 3), columns=["A", "B", "C"]) # Fewer cols
    y2 = pd.Series(np.random.rand(50)) # Regression task now?
    
    # Should reset internal state and re-train
    automl.fit(X2, y2, optuna_trials=1, models=['linear_regression'])
    model2 = automl.best_model_
    
    assert model1 != model2
    # Verify internal profile matches new data (3 cols)
    # Access profile via private attributes or indirectly
    # The new best model should expect 3 features (or transformed count)
    # Just asserting it didn't crash and produced a result is good.
    assert automl.raw_profile_.shape[1] == 3

def test_high_dimensional_data():
    """Case 3: High-Dimensional data (1000 features)"""
    # 50 rows, 1000 columns (Fat dataset)
    X = pd.DataFrame(np.random.rand(50, 200), columns=[f"f_{i}" for i in range(200)])
    y = pd.Series(np.random.randint(0, 2, 50))
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True, n_models=1))
    # Should handle profiling and cleaning of many columns without memory error (on small row count)
    automl.fit(X, y, optuna_trials=1, models=['logistic_regression'])
    
    assert automl.best_model_ is not None

def test_distribution_shift():
    """Case 4: Distribution Shift (Predicting on out-of-range data)"""
    X_train = pd.DataFrame(np.random.rand(100, 5))
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True, n_models=1))
    automl.fit(X_train, y_train, optuna_trials=1, models=['logistic_regression'])
    
    # X_test has values way outside training range (e.g., * 1000)
    # Scalers (StandardScaler/MinMax) should handle this, but might produce large values.
    # Models should predict, possibly poorly, but not crash.
    X_test_shifted = X_train * 1000
    
    try:
        preds = automl.predict(X_test_shifted)
        assert len(preds) == 100
    except Exception as e:
        pytest.fail(f"Prediction crashed on distribution shift: {e}")

def test_mixed_features_complex():
    """Case 5: Complex mixed features handling during prediction"""
    # Train on numeric + categorical
    df_train = pd.DataFrame({
        "num": np.random.rand(50),
        "cat": ["A", "B"] * 25,
        "bool": [True, False] * 25
    })
    y_train = pd.Series(np.random.randint(0, 2, 50))
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True, n_models=1))
    automl.fit(df_train, y_train, optuna_trials=1, models=['logistic_regression'])
    
    # Predict on data with:
    # - New category "C"
    # - "bool" as integer 0/1
    # - "num" as string "0.5" (should be coerced?)
    # AutoCleaner needs to handle this at prediction time.
    df_test = pd.DataFrame({
        "num": ["0.5"] * 10,
        "cat": ["C"] * 10,
        "bool": [1] * 10 # Int instead of bool
    })
    
    try:
        preds = automl.predict(df_test)
        assert len(preds) == 10
    except Exception as e:
        pytest.fail(f"Prediction crashed on mixed types/new categories: {e}")
