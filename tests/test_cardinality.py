
import pytest
import pandas as pd
import numpy as np
from octolearn.profiling.data_profiler import DataProfiler

@pytest.fixture
def profiler():
    return DataProfiler()

def test_high_cardinality_categorical(profiler):
    """Case 9: High cardinal categorical"""
    # 1000 rows, 950 unique strings
    df = pd.DataFrame({
        "cat_high": [f"val_{i % 600}" for i in range(1000)],
    "target": np.random.randint(0, 2, 1000)
    })
    # This creates 600 unique values. Ratio 0.6.
    # ID threshold is usually 0.9. High Card threshold is 0.5.
    # So this should be detected as High Card Categorical, not ID.
    
    profile = profiler.profile(df)
    
    assert "cat_high" in profile.high_cardinality_cols, "Failed to flag high cardinality categorical"
    assert profile.feature_types["cat_high"] in ["categorical", "id", "text"], "Wrong type detection for high card string"

def test_id_column(profiler):
    """Case 10: ID column"""
    df = pd.DataFrame({
        "user_id": range(1000), # Unique integers
        "sku_code": [f"SKU_{i}" for i in range(1000)], # Unique strings
        "target": np.random.randint(0, 2, 1000)
    })
    
    profile = profiler.profile(df)
    
    # Should detect as id or at least trigger leakage/unique checks
    # Note: Logic for 'id' type usually looks for 'id' in name OR high uniqueness
    assert profile.feature_types["user_id"] == "id" or "user_id" in profile.id_like_columns
    assert profile.feature_types["sku_code"] in ["id", "categorical", "text"]
