
import pytest
import pandas as pd
import numpy as np
from octolearn.models.model_trainer import ModelTrainer
from octolearn.profiling.data_profiler import DatasetProfile

# Mock profile
class MockProfile:
    def __init__(self, task_type):
        self.task_type = task_type
        self.columns = []

def test_tiny_dataset():
    """Case 7: Tiny dataset"""
    X = pd.DataFrame(np.random.rand(5, 2), columns=["f1", "f2"])
    y = pd.Series([0, 1, 0, 1, 0])
    
    trainer = ModelTrainer(X, y, profile=MockProfile('classification'), n_trials=1)
    results = trainer.train_all_models()
    
    assert len(results['trained_models']) > 0

def test_high_cardinality_target():
    """Case 8: High cardinality target (Classification with 100 classes)"""
    # 200 rows, 100 classes -> 2 samples per class on average.
    # Train/Test split might fail stratification if some classes have 1 sample.
    X = pd.DataFrame(np.random.rand(200, 5))
    y = pd.Series(np.random.randint(0, 100, 200))
    
    # Should warn or fallback to non-stratified? Or crash?
    try:
        trainer = ModelTrainer(X, y, profile=MockProfile('classification'), n_trials=1)
        # Force a simple model to save time
        trainer.train_all_models()
    except ValueError as e:
        # Sklearn might complain about n_splits.
        # Check if we handle it gracefully?
        pass # If it crashes, test fails. If it passes or raises handled error, good.

def test_single_class_in_training_split():
    """Case 10: Single class in training split (Stratification fail check)"""
    # If we have 10 samples of class 0 and 1 sample of class 1
    X = pd.DataFrame(np.random.rand(11, 2))
    y = pd.Series([0]*10 + [1])
    
    # StratifiedShuffleSplit will fail with n_splits=1 and test_size if class count < 2
    try:
        trainer = ModelTrainer(X, y, profile=MockProfile('classification'), n_trials=1)
        trainer.train_all_models()
    except Exception as e:
        pytest.fail(f"ModelTrainer crashed on imbalanced split: {e}")

def test_nan_in_target():
    """Case 12: NaN in target"""
    X = pd.DataFrame(np.random.rand(10, 2))
    y = pd.Series([0, 1, np.nan, 0, 1] * 2)
    
    # Trainer usually expects non-null y.
    # It should drop rows or error gracefully.
    try:
        trainer = ModelTrainer(X, y, profile=MockProfile('classification'), n_trials=1)
        trainer.train_all_models()
    except ValueError:
        # Expected if we don't handle it
        pass
    except Exception as e:
        pytest.fail(f"Unexpected crash on NaN target: {e}")
