
import pytest
import pandas as pd
import numpy as np
from octolearn.profiling.data_profiler import DataProfiler

@pytest.fixture
def profiler():
    return DataProfiler()

def test_missing_values_storm(profiler):
    """Case 11 & 12: Missing values"""
    df = pd.DataFrame({
        "col_80_missing": [1.0]*20 + [np.nan]*80,
        "col_100_missing": [np.nan]*100,
        "col_clean": np.random.rand(100)
    })
    
    profile = profiler.profile(df)
    
    assert profile.missing_ratio["col_80_missing"] == 0.8
    assert profile.missing_ratio["col_100_missing"] == 1.0

def test_perfect_correlation(profiler):
    """Case 14: Perfectly correlated features (Leakage check usually handles correlation to target, but maybe feature-feature?)"""
    # Note: Profiler currently focuses on Target Leakage.
    # Feature-feature correlation is usually calculated later or in a separate step.
    # We will check if it calculates leakage if provided a target.
    
    x1 = np.random.rand(100)
    x2 = x1 * 2
    y = x1 + np.random.normal(0, 0.1, 100)
    
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    
    profile = profiler.profile(df, target=df["target"])
    
    # x1 and x2 are highly correlated to target. Both should be leakage suspects.
    assert "x1" in profile.leakage_suspects
    assert "x2" in profile.leakage_suspects

def test_constant_column_and_duplicates(profiler):
    """Case 16 & 17: Constant column and Duplicates"""
    df = pd.DataFrame({
        "constant": [1]*100,
        "varied": range(100)
    })
    # Add duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    profile = profiler.profile(df)
    
    assert "constant" in profile.constant_columns
    assert profile.duplicate_rows == 10
