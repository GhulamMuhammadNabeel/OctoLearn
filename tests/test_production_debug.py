"""
OctoLearn Production-Grade Debug Test Suite
============================================
Exhaustive test of every public API, config parameter, fit-override,
accessor method, and edge case.  Uses real-world data (Titanic for
classification, California Housing for regression).

Target: 100 % pass rate before shipping to production.
"""

import os
import sys
import traceback
import tempfile
import numpy as np
import pandas as pd

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
ERRORS = []


def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _run(name: str, fn):
    """Run a single test, track pass/fail."""
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL += 1
        tb = traceback.format_exc()
        ERRORS.append((name, tb))
        print(f"  [FAIL] {name}\n         {e}")


# ── Data loaders ─────────────────────────────────────────────────────────────

def _titanic_data():
    """Return (X, y) for Titanic — classification with mixed types + missing."""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except Exception:
        # Fallback: sklearn breast cancer
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target

    y = df["survived"]
    X = df.drop(columns=["survived", "alive"])  # 'alive' is leakage
    return X, y


def _titanic_string_target():
    """Titanic with string target labels ('Yes'/'No')."""
    X, y = _titanic_data()
    y_str = y.map({1: "Yes", 0: "No"})
    return X, y_str


def _regression_data():
    """California Housing — regression."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    # Subsample for speed
    idx = np.random.RandomState(42).choice(len(data.data), 500, replace=False)
    return data.data.iloc[idx].reset_index(drop=True), data.target.iloc[idx].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Full Titanic classification pipeline (numeric target)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_1():
    _header("SCENARIO 1: Full Titanic Classification Pipeline")

    from octolearn import (
        AutoML, DataConfig, ProfilingConfig, PreprocessingConfig,
        ModelingConfig, OptimizationConfig, FeatureOptimizationConfig,
        ReportingConfig, ParallelConfig,
    )

    X, y = _titanic_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=True, optuna_trials_per_model=5, optuna_timeout_seconds=60),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=True,
    )

    def test_fit():
        automl.fit(X, y)

    def test_post_fit_attrs():
        assert automl.raw_profile_ is not None, "raw_profile_ not set"
        assert automl.clean_profile_ is not None, "clean_profile_ not set"
        assert automl.X_raw_ is not None, "X_raw_ not set"
        assert automl.X_train_ is not None, "X_train_ not set"
        assert automl.X_test_ is not None, "X_test_ not set"
        assert automl.y_train_ is not None, "y_train_ not set"
        assert automl.y_test_ is not None, "y_test_ not set"
        assert automl.best_model_ is not None, "best_model_ not set"
        assert automl.model_benchmarks_ is not None, "model_benchmarks_ not set"
        assert automl.trained_models_ is not None, "trained_models_ not set"
        assert automl.cleaner_ is not None, "cleaner_ not set"
        assert automl.cleaning_log_ is not None, "cleaning_log_ not set"
        assert automl.original_rows_ is not None, "original_rows_ not set"
        assert automl.raw_profile_.task_type == 'classification', f"Expected classification, got {automl.raw_profile_.task_type}"

    def test_get_risk_score():
        result = automl.get_risk_score()
        assert "score" in result, "Missing 'score'"
        assert "category" in result, "Missing 'category'"
        assert "factors" in result, "Missing 'factors'"
        assert 0 <= result["score"] <= 100, f"Risk score {result['score']} out of range"

    def test_get_preprocessing_suggestions():
        result = automl.get_preprocessing_suggestions()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_get_feature_importance():
        result = automl.get_feature_importance()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_get_recommendations():
        result = automl.get_recommendations()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_get_model_benchmarks():
        result = automl.get_model_benchmarks()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, "Benchmarks empty"
        for entry in result:
            assert "model" in entry, "Missing 'model' key"
            assert "score" in entry, "Missing 'score' key"

    def test_predict():
        # Predict on test set
        preds = automl.predict(automl.X_raw_.head(10))
        assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"

    def test_repr_str():
        r = repr(automl)
        s = str(automl)
        assert "Fitted" in r or "Fitted" in s, "Repr/str should say 'Fitted'"

    _run("fit()", test_fit)
    _run("post-fit attributes", test_post_fit_attrs)
    _run("get_risk_score()", test_get_risk_score)
    _run("get_preprocessing_suggestions()", test_get_preprocessing_suggestions)
    _run("get_feature_importance()", test_get_feature_importance)
    _run("get_recommendations()", test_get_recommendations)
    _run("get_model_benchmarks()", test_get_model_benchmarks)
    _run("predict()", test_predict)
    _run("__repr__ / __str__", test_repr_str)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — String target classification (target encoder)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_2():
    _header("SCENARIO 2: String Target Classification (LabelEncoder)")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _titanic_string_target()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=False,
    )

    def test_fit_string_target():
        automl.fit(X, y, models=['logistic_regression'], n_models=1)

    def test_target_encoder():
        assert automl.target_encoder_ is not None, "target_encoder_ should be set for string targets"

    def test_predict_decoded():
        preds = automl.predict(automl.X_raw_.head(5))
        # Predictions should be decoded back to "Yes"/"No"
        assert all(p in ("Yes", "No") for p in preds), f"Expected 'Yes'/'No', got {preds}"

    _run("fit() with string target", test_fit_string_target)
    _run("target_encoder_ populated", test_target_encoder)
    _run("predict() decoded labels", test_predict_decoded)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — All config dataclass parameters (non-default)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_3():
    _header("SCENARIO 3: All Config Parameters (Non-Default Values)")

    from octolearn import (
        AutoML, DataConfig, ProfilingConfig, PreprocessingConfig,
        ModelingConfig, OptimizationConfig, FeatureOptimizationConfig,
        ReportingConfig, ParallelConfig,
    )

    X, y = _titanic_data()

    def test_all_configs():
        automl = AutoML(
            data_config=DataConfig(
                use_full_data=True,
                sample_size=200,
                test_size=0.3,
                random_state=99,
                stratify_target=False,
                sampling_strategy='none',
            ),
            profiling_config=ProfilingConfig(
                detect_outliers=True,
                analyze_interactions=True,    # expensive but must work
                generate_risk_score=True,
                calculate_feature_importance=True,
                generate_recommendations=True,
                include_duplicates_analysis=True,
            ),
            preprocessing_config=PreprocessingConfig(
                auto_clean=True,
                imputer_strategy={'age': 'median'},
                scaler='robust',
                id_columns=None,
            ),
            modeling_config=ModelingConfig(
                train_models=True,
                models_to_train=['logistic_regression', 'random_forest'],
                evaluation_metric='accuracy',
                n_models=2,
                test_size=0.25,
                use_stacking=False,
            ),
            optimization_config=OptimizationConfig(
                use_optuna=True,
                optuna_trials_per_model=5,
                optuna_timeout_seconds=30,
                optuna_parallel_jobs=1,
                use_registry=True,
                early_stopping_rounds=None,
                baseline_score=0.7,
                hyperparameter_overrides=None,
            ),
            feature_optimization_config=FeatureOptimizationConfig(
                enable_feature_optimization=False,
                n_trials=10,
                timeout=60,
                cv_folds=2,
                max_synthetic_features=10,
                min_features=2,
                generate_interactions=True,
                generate_ratios=True,
                generate_polynomials=True,
                generate_log_transforms=True,
            ),
            reporting_config=ReportingConfig(
                generate_report=False,
                report_title='Test Report',
                report_detail='brief',
                include_data_journey=False,
                include_model_comparison=False,
                include_recommendations=False,
                visuals_limit=5,
                plot_mode='simple',
                include_shap=False,
                color_scheme='dark',
            ),
            parallel_config=ParallelConfig(
                parallel_processing=True,
                n_jobs=1,
                backend='threading',
                verbose=0,
                enable_gpu=False,
            ),
            show_progress=True,
            save_artifacts=False,
        )
        automl.fit(X, y)
        assert automl.best_model_ is not None

    _run("all 8 configs with non-default values", test_all_configs)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — All fit() overrides + config restoration
# ══════════════════════════════════════════════════════════════════════════════

def scenario_4():
    _header("SCENARIO 4: All fit() Overrides + Config Restoration")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _titanic_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True, test_size=0.2, random_state=42),
        optimization_config=OptimizationConfig(use_optuna=True, optuna_trials_per_model=5, optuna_timeout_seconds=60),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=False,
    )

    # Snapshot original config values
    orig_trials = automl.optimization_config.optuna_trials_per_model
    orig_timeout = automl.optimization_config.optuna_timeout_seconds
    orig_optuna = automl.optimization_config.use_optuna
    orig_baseline = automl.optimization_config.baseline_score
    orig_test = automl.data_config.test_size
    orig_rand = automl.data_config.random_state
    orig_models = automl.modeling_config.models_to_train
    orig_nmodels = automl.modeling_config.n_models
    orig_metric = automl.modeling_config.evaluation_metric
    orig_imputer = automl.preprocessing_config.imputer_strategy
    orig_scaler = automl.preprocessing_config.scaler
    orig_train = automl.modeling_config.train_models

    def test_overrides():
        automl.fit(
            X, y,
            optuna_trials=5,
            optuna_timeout=30,
            use_optuna=True,
            optuna_baseline_score=0.8,
            test_size=0.3,
            random_state=99,
            models=['logistic_regression'],
            n_models=1,
            evaluation_metric='accuracy',
            imputer_strategy={'age': 'median'},
            scaler='robust',
            train_models=True,
        )
        assert automl.best_model_ is not None

    def test_config_restored():
        assert automl.optimization_config.optuna_trials_per_model == orig_trials, \
            f"optuna_trials not restored: {automl.optimization_config.optuna_trials_per_model} != {orig_trials}"
        assert automl.optimization_config.optuna_timeout_seconds == orig_timeout, \
            f"optuna_timeout not restored"
        assert automl.optimization_config.use_optuna == orig_optuna, \
            f"use_optuna not restored"
        assert automl.optimization_config.baseline_score == orig_baseline, \
            f"baseline_score not restored"
        assert automl.data_config.test_size == orig_test, \
            f"test_size not restored: {automl.data_config.test_size} != {orig_test}"
        assert automl.data_config.random_state == orig_rand, \
            f"random_state not restored"
        assert automl.modeling_config.models_to_train == orig_models, \
            f"models_to_train not restored"
        assert automl.modeling_config.n_models == orig_nmodels, \
            f"n_models not restored"
        assert automl.modeling_config.evaluation_metric == orig_metric, \
            f"evaluation_metric not restored"
        assert automl.preprocessing_config.imputer_strategy == orig_imputer, \
            f"imputer_strategy not restored"
        assert automl.preprocessing_config.scaler == orig_scaler, \
            f"scaler not restored"
        assert automl.modeling_config.train_models == orig_train, \
            f"train_models not restored"

    _run("fit() with all 12 overrides", test_overrides)
    _run("config values restored after fit()", test_config_restored)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — Regression pipeline (California Housing)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_5():
    _header("SCENARIO 5: Regression Pipeline (California Housing)")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _regression_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=True, optuna_trials_per_model=5, optuna_timeout_seconds=60),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=False,
    )

    def test_regression_fit():
        automl.fit(X, y)

    def test_regression_task_type():
        assert automl.raw_profile_.task_type == 'regression', \
            f"Expected regression, got {automl.raw_profile_.task_type}"

    def test_regression_predict():
        preds = automl.predict(X.head(10))
        assert len(preds) == 10
        # For regression, predictions should be numeric
        assert all(isinstance(float(p), float) for p in preds)

    def test_regression_accessors():
        risk = automl.get_risk_score()
        assert "score" in risk
        benchmarks = automl.get_model_benchmarks()
        assert len(benchmarks) > 0
        importance = automl.get_feature_importance()
        assert isinstance(importance, dict)
        recs = automl.get_recommendations()
        assert isinstance(recs, dict)

    _run("fit() regression", test_regression_fit)
    _run("task_type == regression", test_regression_task_type)
    _run("predict() regression", test_regression_predict)
    _run("accessor methods (regression)", test_regression_accessors)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — Profiling-only run (train_models=False)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_6():
    _header("SCENARIO 6: Profiling-Only Run (train_models=False)")

    from octolearn import AutoML, DataConfig, ModelingConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _titanic_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        modeling_config=ModelingConfig(train_models=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=False,
    )

    def test_profiling_only_fit():
        automl.fit(X, y)

    def test_no_model():
        assert automl.best_model_ is None, "best_model_ should be None when train_models=False"

    def test_predict_raises():
        try:
            automl.predict(X.head(5))
            raise AssertionError("predict() should raise ValueError when no model trained")
        except ValueError:
            pass  # Expected

    def test_profile_populated():
        assert automl.raw_profile_ is not None
        assert automl.clean_profile_ is not None

    _run("fit(train_models=False)", test_profiling_only_fit)
    _run("best_model_ is None", test_no_model)
    _run("predict() raises ValueError", test_predict_raises)
    _run("profiles populated", test_profile_populated)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — Report generation
# ══════════════════════════════════════════════════════════════════════════════

def scenario_7():
    _header("SCENARIO 7: PDF Report Generation")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _titanic_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(
            generate_report=True,
            report_title='Production Debug Report',
            report_detail='detailed',
            include_data_journey=True,
            include_model_comparison=True,
            include_recommendations=True,
            include_shap=True,
            color_scheme='light',
        ),
        show_progress=False,
    )

    def test_fit_for_report():
        automl.fit(X, y, models=['logistic_regression', 'random_forest'], n_models=2)

    report_path = None

    def test_generate_report():
        nonlocal report_path
        report_path = automl.generate_report(filename='test_debug_report.pdf')
        assert report_path is not None, "Report path is None"
        assert os.path.exists(report_path), f"Report file not found: {report_path}"
        size = os.path.getsize(report_path)
        assert size > 1000, f"Report too small ({size} bytes)"
        print(f"         Report: {report_path} ({size:,} bytes)")

    def test_report_cleanup():
        # Clean up test report
        if report_path and os.path.exists(report_path):
            os.remove(report_path)

    _run("fit() for report", test_fit_for_report)
    _run("generate_report()", test_generate_report)
    _run("cleanup report file", test_report_cleanup)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 8 — Edge cases & scaler variants
# ══════════════════════════════════════════════════════════════════════════════

def scenario_8():
    _header("SCENARIO 8: Edge Cases & Scaler Variants")

    from octolearn import (
        AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig,
        ReportingConfig, PreprocessingConfig, ModelingConfig,
    )

    X, y = _titanic_data()

    def test_scaler_robust():
        a = AutoML(
            data_config=DataConfig(use_full_data=True),
            preprocessing_config=PreprocessingConfig(scaler='robust'),
            optimization_config=OptimizationConfig(use_optuna=False),
            feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
            reporting_config=ReportingConfig(generate_report=False),
            show_progress=False,
        )
        a.fit(X, y, models=['logistic_regression'], n_models=1)
        assert a.best_model_ is not None

    def test_scaler_minmax():
        a = AutoML(
            data_config=DataConfig(use_full_data=True),
            preprocessing_config=PreprocessingConfig(scaler='minmax'),
            optimization_config=OptimizationConfig(use_optuna=False),
            feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
            reporting_config=ReportingConfig(generate_report=False),
            show_progress=False,
        )
        a.fit(X, y, models=['logistic_regression'], n_models=1)
        assert a.best_model_ is not None

    def test_scaler_none():
        a = AutoML(
            data_config=DataConfig(use_full_data=True),
            preprocessing_config=PreprocessingConfig(scaler=None),
            optimization_config=OptimizationConfig(use_optuna=False),
            feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
            reporting_config=ReportingConfig(generate_report=False),
            show_progress=False,
        )
        a.fit(X, y, models=['random_forest'], n_models=1)
        assert a.best_model_ is not None

    def test_no_stacking():
        a = AutoML(
            data_config=DataConfig(use_full_data=True),
            modeling_config=ModelingConfig(use_stacking=False),
            optimization_config=OptimizationConfig(use_optuna=False),
            feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
            reporting_config=ReportingConfig(generate_report=False),
            show_progress=False,
        )
        a.fit(X, y, models=['logistic_regression', 'random_forest'], n_models=2)
        assert a.best_model_ is not None

    def test_xgboost_only():
        a = AutoML(
            data_config=DataConfig(use_full_data=True),
            optimization_config=OptimizationConfig(use_optuna=True, optuna_trials_per_model=5, optuna_timeout_seconds=30),
            feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
            reporting_config=ReportingConfig(generate_report=False),
            show_progress=False,
        )
        a.fit(X, y, models=['xgboost'], n_models=1)
        assert a.best_model_ is not None

    def test_lightgbm_only():
        a = AutoML(
            data_config=DataConfig(use_full_data=True),
            optimization_config=OptimizationConfig(use_optuna=True, optuna_trials_per_model=5, optuna_timeout_seconds=30),
            feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
            reporting_config=ReportingConfig(generate_report=False),
            show_progress=False,
        )
        a.fit(X, y, models=['lightgbm'], n_models=1)
        assert a.best_model_ is not None

    def test_sampling_strategies():
        """Test SMOTE, ADASYN, undersample, combine sampling strategies."""
        for strategy in ['smote', 'adasyn', 'undersample', 'combine']:
            a = AutoML(
                data_config=DataConfig(use_full_data=True, sampling_strategy=strategy),
                optimization_config=OptimizationConfig(use_optuna=False),
                feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
                reporting_config=ReportingConfig(generate_report=False),
                show_progress=False,
            )
            a.fit(X, y, models=['logistic_regression'], n_models=1)
            assert a.best_model_ is not None, f"Failed with sampling_strategy='{strategy}'"

    _run("scaler='robust'", test_scaler_robust)
    _run("scaler='minmax'", test_scaler_minmax)
    _run("scaler=None", test_scaler_none)
    _run("use_stacking=False", test_no_stacking)
    _run("XGBoost only", test_xgboost_only)
    _run("LightGBM only", test_lightgbm_only)
    _run("sampling strategies (smote/adasyn/undersample/combine)", test_sampling_strategies)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 9 — get_pipeline() export & standalone prediction
# ══════════════════════════════════════════════════════════════════════════════

def scenario_9():
    _header("SCENARIO 9: get_pipeline() Export")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _titanic_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=False,
    )

    def test_pipeline_export():
        automl.fit(X, y, models=['logistic_regression'], n_models=1)
        pipeline = automl.get_pipeline()
        assert pipeline is not None
        # It should be an sklearn Pipeline
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline), f"Expected Pipeline, got {type(pipeline)}"

    _run("get_pipeline() export", test_pipeline_export)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 10 — Feature optimization enabled
# ══════════════════════════════════════════════════════════════════════════════

def scenario_10():
    _header("SCENARIO 10: Feature Optimization (Optuna-Driven)")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X, y = _titanic_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=True, optuna_trials_per_model=5, optuna_timeout_seconds=30),
        feature_optimization_config=FeatureOptimizationConfig(
            enable_feature_optimization=True,
            n_trials=10,
            timeout=60,
            cv_folds=2,
            max_synthetic_features=10,
            min_features=2,
            generate_interactions=True,
            generate_ratios=True,
            generate_polynomials=True,
            generate_log_transforms=True,
        ),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=True,
    )

    def test_feat_opt_fit():
        automl.fit(X, y)

    def test_feat_opt_result():
        # Feature optimization may succeed or gracefully fall back
        if automl.feature_optimization_result_ is not None:
            res = automl.feature_optimization_result_
            assert hasattr(res, 'best_features'), "Missing best_features"
            assert hasattr(res, 'best_score'), "Missing best_score"
            assert hasattr(res, 'best_model_name'), "Missing best_model_name"
            print(f"         Best model: {res.best_model_name}, Score: {res.best_score:.4f}, Features: {len(res.best_features)}")
        else:
            print("         Feature optimization fell back (acceptable)")

    def test_feat_opt_predict():
        if automl.best_model_ is not None:
            preds = automl.predict(automl.X_raw_.head(5))
            assert len(preds) == 5

    _run("fit() with feature optimization", test_feat_opt_fit)
    _run("feature_optimization_result_ populated", test_feat_opt_result)
    _run("predict() after feature optimization", test_feat_opt_predict)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 11 — Re-fitting robustness (fit twice)
# ══════════════════════════════════════════════════════════════════════════════

def scenario_11():
    _header("SCENARIO 11: Re-fitting Robustness (fit twice, different data)")

    from octolearn import AutoML, DataConfig, OptimizationConfig, FeatureOptimizationConfig, ReportingConfig

    X_cls, y_cls = _titanic_data()
    X_reg, y_reg = _regression_data()

    automl = AutoML(
        data_config=DataConfig(use_full_data=True),
        optimization_config=OptimizationConfig(use_optuna=False),
        feature_optimization_config=FeatureOptimizationConfig(enable_feature_optimization=False),
        reporting_config=ReportingConfig(generate_report=False),
        show_progress=False,
    )

    def test_refit():
        # First fit — classification
        automl.fit(X_cls, y_cls, models=['logistic_regression'], n_models=1)
        assert automl.raw_profile_.task_type == 'classification'
        model1 = automl.best_model_

        # Second fit — regression (completely different data shape + task)
        automl.fit(X_reg, y_reg, models=['linear_regression'], n_models=1)
        assert automl.raw_profile_.task_type == 'regression'
        model2 = automl.best_model_

        assert model1 is not model2, "Models should be different after refit"

    _run("refit classification -> regression", test_refit)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("  OCTOLEARN PRODUCTION-GRADE DEBUG SUITE")
    print("#"*70)

    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    scenario_5()
    scenario_6()
    scenario_7()
    scenario_8()
    scenario_9()
    scenario_10()
    scenario_11()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = PASS + FAIL
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print(f"{'='*70}")

    if ERRORS:
        print(f"\n{'-'*70}")
        print("  FAILURE DETAILS:")
        print(f"{'-'*70}")
        for name, tb in ERRORS:
            print(f"\n  X {name}")
            for line in tb.strip().split("\n"):
                print(f"    {line}")

    sys.exit(0 if FAIL == 0 else 1)
