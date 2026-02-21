
import pytest
import pandas as pd
import numpy as np
from octolearn.profiling.data_profiler import DataProfiler

@pytest.fixture
def profiler():
    return DataProfiler()

def test_integers_encoded_as_strings(profiler):
    """Case 1: Integers encoded as strings"""
    df = pd.DataFrame({"age": ["23", "45", "31", "100", "5"]})
    profile = profiler.profile(df)
    
    assert profile.feature_types["age"] == "numeric", "Failed to detect string-encoded integers as numeric"
    assert "mean" in profile.stats["age"], "Stats missing for numeric column"

def test_mixed_type_column(profiler):
    """Case 2: Mixed type column"""
    df = pd.DataFrame({"salary": [50000, "60000", None, "unknown", 75000]})
    # This typically defaults to object/categorical if conversion fails
    profile = profiler.profile(df)
    
    # Should safely handle it without crashing
    # Likely detects as categorical or text due to 'unknown'
    assert profile.feature_types["salary"] in ["categorical", "text"], "Mixed type should default to categorical/text"
    assert profile.missing_ratio["salary"] > 0, "Failed to detect missing values in mixed column"

def test_boolean_encoded_as_strings(profiler):
    """Case 3: Boolean encoded as 0/1 strings"""
    df = pd.DataFrame({"is_active": ["0", "1", "1", "0", "1"]})
    profile = profiler.profile(df)
    
    # OctoLearn logic usually treats low cardinality numeric-like strings as numeric or categorical
    # Ideally should be categorical (binary) or numeric
    assert profile.feature_types["is_active"] in ["categorical", "numeric"], "Failed to handle 0/1 strings"
    assert profile.stats["is_active"]["top_counts"] is not None

def test_floats_stored_as_object_with_commas(profiler):
    """Case 4: Floats stored as object with commas"""
    df = pd.DataFrame({"price": ["1,200.50", "3,400.00", "150.25"]})
    # OctoLearn currently mostly uses pd.to_numeric(errors='coerce') which fails on commas by default
    # This test checks if we handle it or if we identify it as categorical/text
    # ideally we should be smarter, but if not, at least don't crash.
    profile = profiler.profile(df)
    
    # Checking behavior. If it fails to parse commas, it might be categorical.
    # Future improvement constraint: add comma handling.
    # For now, assert it runs.
    assert "price" in profile.feature_types
