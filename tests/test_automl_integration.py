
import pytest
import pandas as pd
import numpy as np
from octolearn import AutoML, DataConfig, ModelingConfig

def test_e2e_garbage_data():
    """Case 13: End-to-end run with 'Garbage' data"""
    # Random noise, mixed types, nans
    df = pd.DataFrame(np.random.rand(50, 5), columns=[f"col_{i}" for i in range(5)])
    df["col_0"] = np.nan # All NaN
    df["col_1"] = "string_noise" # Constant string
    df["target"] = np.random.randint(0, 2, 50)
    
    automl = AutoML(
        modeling_config=ModelingConfig(train_models=True),
        data_config=DataConfig(use_full_data=True)
    )
    # Should run without crashing, dropping bad cols
    automl.fit(df.drop(columns=['target']), df['target'])
    assert automl.best_model_ is not None

def test_config_override_check():
    """Case 14: Config override check (train_models=False)"""
    X = pd.DataFrame(np.random.rand(20, 2))
    y = pd.Series([0, 1] * 10)
    
    automl = AutoML(
        modeling_config=ModelingConfig(train_models=False)
    )
    automl.fit(X, y)
    
    assert automl.best_model_ is None, "Should not train models if train_models=False"

def test_binary_class_string_targets():
    """Case 17: Binary classification with string targets"""
    X = pd.DataFrame(np.random.rand(20, 2))
    y = pd.Series(["Yes", "No"] * 10)
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True))
    automl.fit(X, y)
    
    # Should automatically encode target
    assert automl.best_model_ is not None

def test_empty_dataset_error():
    """Case 20: Empty dataset passed to AutoML"""
    df = pd.DataFrame()
    y = pd.Series([])
    
    automl = AutoML()
    with pytest.raises(Exception): # Should match specific error ideally
        automl.fit(df, y)

def test_emoji_columns():
    """Case 19: Emoji/Unicode column names"""
    X = pd.DataFrame({
        "👍": np.random.rand(20),
        "🐍": np.random.randint(0, 100, 20)
    })
    y = pd.Series([0, 1] * 10)
    
    automl = AutoML(modeling_config=ModelingConfig(train_models=True))
    automl.fit(X, y)
    
    assert automl.best_model_ is not None, "Failed on Emoji columns"
