
import pytest
import pandas as pd
import numpy as np
from octolearn.profiling.data_profiler import DataProfiler

@pytest.fixture
def profiler():
    return DataProfiler()

def test_extreme_scale_small_sample(profiler):
    """Case 21: 10 rows, 2000 columns"""
    data = np.random.rand(10, 2000)
    cols = [f"col_{i}" for i in range(2000)]
    df = pd.DataFrame(data, columns=cols)
    
    profile = profiler.profile(df)
    
    assert profile.shape == (10, 2000)
    assert len(profile.columns) == 2000

def test_edge_column_names(profiler):
    """Edge Case: Column names with spaces, special chars, unicode"""
    df = pd.DataFrame({
        "col with space": [1, 2, 3],
        "col$special": [4, 5, 6],
        "col_unicode_👍": [7, 8, 9]
    })
    
    profile = profiler.profile(df)
    
    assert "col with space" in profile.columns
    assert "col$special" in profile.columns
    assert "col_unicode_👍" in profile.columns

def test_empty_dataframe(profiler):
    """Edge Case: Empty DataFrame"""
    df = pd.DataFrame()
    profile = profiler.profile(df)
    
    assert profile.shape == (0, 0)

def test_single_value_nan_target(profiler):
    """Edge Case: Target with NaN values"""
    y = pd.Series([1, 0, np.nan, 1, 0], name="target")
    # Should handle or ignore NaNs in detection
    task_type = profiler._detect_task_type(y)
    assert task_type in ["classification", "regression"]
