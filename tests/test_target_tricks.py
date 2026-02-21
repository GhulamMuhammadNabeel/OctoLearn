
import pytest
import pandas as pd
import numpy as np
from octolearn.profiling.data_profiler import DataProfiler

@pytest.fixture
def profiler():
    return DataProfiler()

def test_regression_target_few_values(profiler):
    """Case 5: Regression target but few unique values (Floats)"""
    y = pd.Series([10.1, 10.2, 10.1, 10.3, 10.2, 10.1], name="target")
    # Should be regression because floats have decimals
    task_type = profiler._detect_task_type(y)
    assert task_type == "regression", "Floats with decimals should be regression"

def test_classification_target_numeric_encoded(profiler):
    """Case 6: Classification target but numeric encoded (0, 1, 2)"""
    y = pd.Series([0, 1, 2, 1, 0] * 10, name="target")
    # Low cardinality integers -> Classification
    task_type = profiler._detect_task_type(y)
    assert task_type == "classification", "Low cardinality integers should be classification"

def test_highly_imbalanced_target(profiler):
    """Case 7: Highly imbalanced"""
    # 95% class 0, 5% class 1
    y = pd.Series([0]*95 + [1]*5, name="target")
    df = pd.DataFrame({"feature": range(100), "target": y})
    
    profile = profiler.profile(df, target=y)
    
    assert profile.imbalance_ratio is not None
    assert profile.imbalance_ratio <= 0.06, f"Imbalance ratio calculation incorrect: {profile.imbalance_ratio}"

def test_target_leakage_column(profiler):
    """Case 8: Target leakage column"""
    y = pd.Series(np.random.rand(100), name="target")
    df = pd.DataFrame({
        "feature_ok": np.random.rand(100),
        "feature_leak": y, # Identical to target
        "target": y
    })
    
    profile = profiler.profile(df, target=y)
    
    assert "feature_leak" in profile.leakage_suspects, "Failed to detect exact match leakage"
