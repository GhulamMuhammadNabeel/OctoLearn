"""
OctoLearn Library Debug Test
Tests all public APIs against real sklearn datasets.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes

import sys
sys.path.insert(0, r'c:\Users\Nabeel\Desktop\OctoLearn')

from octolearn import (
    AutoML, DataConfig, ModelingConfig, OptimizationConfig,
    FeatureOptimizationConfig, ProfilingConfig
)

PASS = "[PASS]"
FAIL = "[FAIL]"

def test_classification():
    print("\n=== TEST 1: Classification (Breast Cancer) ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    
    automl = AutoML(
        data_config=DataConfig(use_full_data=True, test_size=0.2),
        modeling_config=ModelingConfig(
            train_models=True, n_models=3,
            models_to_train=['xgboost', 'random_forest']
        ),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        show_progress=False
    )
    automl.fit(X, y)
    
    best_name = automl.best_model_.__class__.__name__
    print(f"  {PASS} fit() completed. Best model: {best_name}")
    
    n_benchmarks = len(automl.model_benchmarks_)
    print(f"  {PASS} {n_benchmarks} models in benchmarks")
    
    risk = automl.get_risk_score()
    score = risk['score']
    cat = risk['category']
    print(f"  {PASS} Risk score: {score} ({cat})")
    
    preds = automl.predict(X.head(5))
    print(f"  {PASS} predict() -> {preds}")
    
    imp = automl.get_feature_importance()
    print(f"  {PASS} get_feature_importance() -> {len(imp)} features")
    
    recs = automl.get_recommendations()
    print(f"  {PASS} get_recommendations() -> categories: {list(recs.keys())}")
    
    benchmarks = automl.get_model_benchmarks()
    print(f"  {PASS} get_model_benchmarks() -> {len(benchmarks)} entries")
    
    pipeline = automl.get_pipeline()
    pipeline_preds = pipeline.predict(X.head(3))
    print(f"  {PASS} get_pipeline() -> {pipeline_preds}")
    
    print(f"  {PASS} raw_profile_.task_type = {automl.raw_profile_.task_type}")
    print(f"  {PASS} raw_profile_.leakage_suspects = {automl.raw_profile_.leakage_suspects}")
    print(f"  {PASS} raw_profile_.data_quality_score = {automl.raw_profile_.data_quality_score}")
    
    return automl


def test_regression():
    print("\n=== TEST 2: Regression (Diabetes) ===")
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    
    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        modeling_config=ModelingConfig(
            n_models=2,
            models_to_train=['lightgbm', 'ridge'],
            evaluation_metric='r2'
        ),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        show_progress=False
    )
    automl.fit(X, y)
    best_name = automl.best_model_.__class__.__name__
    print(f"  {PASS} fit() completed. Best model: {best_name}")
    
    preds = automl.predict(X.head(3))
    print(f"  {PASS} predict() -> {preds}")
    return automl


def test_profiling_only():
    print("\n=== TEST 3: Profiling-Only Run ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    
    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        show_progress=False
    )
    automl.fit(X, y, train_models=False)
    
    print(f"  {PASS} task_type: {automl.raw_profile_.task_type}")
    print(f"  {PASS} quality_score: {automl.raw_profile_.data_quality_score}")
    print(f"  {PASS} numeric_columns: {len(automl.raw_profile_.numeric_columns)}")
    
    sug = automl.get_preprocessing_suggestions()
    sug_count = len(sug) if hasattr(sug, '__len__') else 'N/A'
    print(f"  {PASS} preprocessing suggestions: {sug_count}")
    return automl


def test_string_labels():
    print("\n=== TEST 4: String Class Labels ===")
    X, y_raw = load_breast_cancer(return_X_y=True, as_frame=True)
    y = y_raw.map({0: 'malignant', 1: 'benign'})
    
    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        modeling_config=ModelingConfig(
            n_models=2, models_to_train=['random_forest'],
        ),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        show_progress=False
    )
    automl.fit(X, y)
    print(f"  {PASS} String labels handled. Best: {automl.best_model_.__class__.__name__}")
    return automl


def test_outlier_detection():
    print("\n=== TEST 5: Outlier Detection ===")
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    
    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        profiling_config=ProfilingConfig(detect_outliers=True),
        modeling_config=ModelingConfig(train_models=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        show_progress=False
    )
    automl.fit(X, y, train_models=False)
    outliers = getattr(automl, 'outlier_results_', {})
    print(f"  {PASS} Outlier results available: {bool(outliers)} ({type(outliers).__name__})")


def test_imbalanced_sampling():
    print("\n=== TEST 6: Imbalanced Data + SMOTE ===")
    # Create severely imbalanced dataset
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    # Make it 9:1 imbalance
    mask_minority = y == 0
    X_imb = pd.concat([X[y == 1], X[mask_minority].head(50)], ignore_index=True)
    y_imb = pd.concat([y[y == 1], y[mask_minority].head(50)], ignore_index=True)
    
    automl = AutoML(
        data_config=DataConfig(use_full_data=True, sampling_strategy='smote'),
        modeling_config=ModelingConfig(
            n_models=2, models_to_train=['random_forest'],
        ),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        show_progress=False
    )
    automl.fit(X_imb, y_imb)
    print(f"  {PASS} SMOTE completed. Best: {automl.best_model_.__class__.__name__}")


if __name__ == '__main__':
    print("=" * 60)
    print("  OctoLearn Library Debug & Validation Test Suite")
    print("=" * 60)
    
    tests = [
        test_classification,
        test_regression,
        test_profiling_only,
        test_string_labels,
        test_outlier_detection,
        test_imbalanced_sampling,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {FAIL} {test.__name__} raised: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} tests PASSED, {failed} FAILED")
    print(f"{'='*60}")
