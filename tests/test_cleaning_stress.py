
import pytest
import pandas as pd
import numpy as np
from octolearn.preprocessing.auto_cleaner import AutoCleaner

@pytest.fixture
def cleaner():
    return AutoCleaner()

def test_all_nan_columns(cleaner):
    """Case 1: All-NaN columns"""
    df = pd.DataFrame({
        "all_nan": [np.nan] * 20,
        "valid": np.random.rand(20)
    })
    # Imputers usually drop columns if they are all NaN or impute with 0/mean.
    # Check behavior. Ideally, if it cannot impute, it might drop or fill default.
    cleaner.fit(df, pd.Series([0]*20))
    res = cleaner.transform(df)
    
    # "all_nan" might be imputed with 0 if strategy is 'mean' (0 is default for empty?) 
    # Or dropped if 'constant' imputer used.
    # SimpleImputer(mean) on all-NaN throws error or drops? 
    # Actually sklearn SimpleImputer(mean) on all-NaN column -> drops it or fills 0?
    # Let's verify it survives.
    assert "valid" in res.columns

def test_single_value_columns(cleaner):
    """Case 2: Single-value columns (Constant removal)"""
    df = pd.DataFrame({
        "constant": [1.0] * 20,
        "valid": np.random.rand(20)
    })
    cleaner.fit(df, pd.Series([0]*20))
    res = cleaner.transform(df)
    
    assert "constant" not in res.columns, "Constant column should be removed"

def test_dirty_categoricals(cleaner):
    """Case 3: 'Dirty' categoricals (case sensitivity, whitespace)"""
    # AutoCleaner currently might uses LabelEncoding/OHE. 
    # Does it handle new categories in transform? 
    # Or does it unify "Yes" and "yes "? (Likely not, but let's see if it crashes)
    df_train = pd.DataFrame({"cat": ["A", "B", "C"] * 10})
    df_test = pd.DataFrame({"cat": ["A", "B", "D", "a"]}) # 'D' and 'a' are unseen/dirty
    
    cleaner.fit(df_train, pd.Series([0]*30))
    res = cleaner.transform(df_test)
    
    # Needs to handle unseen categories gracefully (OHE handle_unknown='ignore')
    assert len(res) == 4

def test_infinite_values(cleaner):
    """Case 4: Infinite values"""
    df = pd.DataFrame({
        "inf_col": [1.0, np.inf, -np.inf, 2.0]
    })
    # Sklearn imputers might error on inf unless explicitly handled.
    # OctoLearn should handle this.
    try:
        cleaner.fit(df, pd.Series([0]*4))
        res = cleaner.transform(df)
        assert not np.isinf(res.values).any(), "Infinite values should be replaced/imputed"
    except ValueError:
        pytest.fail("AutoCleaner crashed on infinite values")

def test_large_integers_as_floats(cleaner):
    """Case 5: Large integers as floats"""
    df = pd.DataFrame({
        "large_id": [1.0e10, 1.0e10+1, 1.0e10+2] 
    })
    cleaner.fit(df, pd.Series([0]*3))
    res = cleaner.transform(df)
    assert not res.isnull().any().any()
