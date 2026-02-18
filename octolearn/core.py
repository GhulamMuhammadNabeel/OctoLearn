"""
OctoLearn Core Module: Enterprise-Grade AutoML Orchestrator

This module contains the central AutoML class that orchestrates the entire machine learning
pipeline, from raw data ingestion through model deployment. It handles data validation,
profiling, cleaning, feature engineering, model training with hyperparameter optimization,
and comprehensive reporting.

The design philosophy emphasizes:
- **Transparency**: Users can inspect data at every stage
- **Control**: Every aspect is configurable with sensible defaults
- **Performance**: Optimized for both small and large datasets
- **Production-Readiness**: Handles edge cases and provides clear error messages

Module Organization:
    AutoML: Main orchestrator class (primary entry point)

Example:
    >>> from octolearn import AutoML
    >>> import pandas as pd
    >>> 
    >>> # Load your data
    >>> data = pd.read_csv('data.csv')
    >>> X = data.drop('target', axis=1)
    >>> y = data['target']
    >>> 
    >>> # Create AutoML instance with your preferences
    >>> automl = AutoML(
    ...     optimization_config={'use_optuna': True, 'optuna_trials': 20},
    ...     reporting_config={'report_detail': 'detailed', 'include_data_journey': True}
    ... )
    >>> 
    >>> # Fit the pipeline
    >>> automl.fit(X, y)
    >>> 
    >>> # Generate professional report
    >>> pdf_path = automl.generate_report()
    >>> 
    >>> # Make predictions on new data
    >>> y_pred = automl.predict(X_new)
    >>> 
    >>> # Access insights at any point
    >>> risk_score = automl.get_risk_score()
    >>> recommendations = automl.get_recommendations()
    >>> feature_importance = automl.get_feature_importance()
    >>> 
    >>> # Get all intermediate results
    >>> automl.X_train_  # Cleaned training data
    >>> automl.best_model_  # The best trained model
    >>> automl.model_benchmarks_  # Performance of all trained models

Author:
    OctoLearn Development Team

License:
    MIT

Version:
    0.8.0
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List, Any, Union
import warnings
import os
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=UserWarning)

# Phase 1 modules
from .profiling.data_profiler import DataProfiler
from .experiments.report_generator import ReportGenerator
from .experiments.plot_generator import PlotGenerator
from .experiments.recommendation_engine import RecommendationEngine
from .experiments.risk_scorer import RiskScorer
from .experiments.preprocessing_suggester import PreprocessingSuggester
from .experiments.baseline_importance import BaselineImportance

# Phase 2 modules
from .experiments.outlier_detector import OutlierDetector

# Phase 3 modules
from .feature.interaction_analyzer import FeatureInteractionAnalyzer
from .feature.generator import FeatureGenerator
from .preprocessing.auto_cleaner import AutoCleaner



# Phase 4 modules
from .models.model_trainer import ModelTrainer
from .models.registry import ModelRegistry
from .evaluation.metrics import ModelEvaluator

# Utilities
from .utils.helpers import setup_logger, validate_dataframe, validate_series
from .config import PARALLEL_CONFIG
from sklearn.model_selection import train_test_split

logger = setup_logger(__name__)


@dataclass
class DataConfig:
    """
    Configuration for data handling and validation.
    
    This dataclass encapsulates all data-related configuration options, including
    sampling strategy, size limits, and data-specific parameters.
    
    Attributes:
        use_full_data (bool): If True, uses entire dataset for all operations.
            If False, samples data for faster profiling while using full data for training.
            Default: False (recommended for datasets > 10,000 rows).
        
        sample_size (int): Number of rows to sample for profiling when use_full_data=False.
            Larger values give more accurate profiles but slower execution.
            Recommended: 500-2000. Default: 500.
        
        test_size (float): Proportion of data to reserve for testing (0.0-1.0).
            Must be between 0.1 and 0.5. Default: 0.2 (20% test, 80% train).
        
        random_state (int): Random seed for reproducibility. All random operations use this seed.
            Set to None for non-deterministic results. Default: 42.
        
        stratify_target (bool): If True, stratifies train/test split by target variable.
            Recommended for classification with imbalanced classes. Default: True.
    
    Example:
        >>> config = DataConfig(sample_size=1000, test_size=0.2, random_state=42)
        >>> automl = AutoML(data_config=config)
    """
    use_full_data: bool = False
    sample_size: int = 500
    test_size: float = 0.2
    random_state: int = 42
    stratify_target: bool = True


@dataclass
class ProfilingConfig:
    """
    Configuration for dataset profiling and analysis.
    
    Controls how the data is analyzed to understand its characteristics, quality,
    and potential issues.
    
    Attributes:
        detect_outliers (bool): Whether to perform outlier detection using IQR and Isolation Forest.
            Helpful for understanding data quality. Default: True.
        
        analyze_interactions (bool): Whether to analyze feature interactions and correlations.
            Can be slow for high-dimensional data (>100 features). Default: True.
        
        generate_risk_score (bool): Whether to calculate data quality risk score (0-100).
            Default: True.
        
        calculate_feature_importance (bool): Whether to calculate feature importance scores.
            Uses Random Forest baseline + SHAP analysis. Default: True.
        
        generate_recommendations (bool): Whether to generate actionable recommendations.
            Based on detected data characteristics. Default: True.
        
        include_duplicates_analysis (bool): Whether to detect and analyze duplicate rows.
            Default: True.
    
    Example:
        >>> config = ProfilingConfig(
        ...     detect_outliers=True,
        ...     analyze_interactions=False  # Skip for speed with high-dimensional data
        ... )
        >>> automl = AutoML(profiling_config=config)
    """
    detect_outliers: bool = True
    analyze_interactions: bool = False  # Disabled by default for performance
    generate_risk_score: bool = True
    calculate_feature_importance: bool = True
    generate_recommendations: bool = True
    include_duplicates_analysis: bool = True


@dataclass
class PreprocessingConfig:
    """
    Configuration for data cleaning and preprocessing.
    
    Attributes:
        auto_clean (bool): Whether to automatically impute, encode, and scale data.
            Default: True. Set to False only if you provide pre-cleaned data.
        
        imputer_strategy (Dict): Imputation strategy {'numeric': 'mean'|'median'|'knn',
            'categorical': 'mode'|'constant'}. Default: {'numeric': 'median', 'categorical': 'mode'}.
        
        encoder_strategy (Dict): Encoding strategy with keys:
            - 'ordinal_cols': List of columns for ordinal encoding
            - 'bool_cols': List of columns for binary encoding (0/1)
            - 'ohe_cols': List of columns for one-hot encoding
            Default: Auto-detects based on cardinality.
        
        scaler (str): Feature scaling method: 'standard', 'robust', 'minmax', or None.
            'standard': Zero mean, unit variance (good for normally distributed features)
            'robust': Uses median and IQR (robust to outliers)
            'minmax': Scales to [0, 1] range
            Default: 'standard'.
        
        id_columns (List[str]): Columns to drop as non-predictive identifiers.
            Detected automatically but can be overridden. Default: None (auto-detect).
    
    Example:
        >>> config = PreprocessingConfig(
        ...     imputer_strategy={'numeric': 'median', 'categorical': 'mode'},
        ...     scaler='robust'  # Robust to outliers
        ... )
        >>> automl = AutoML(preprocessing_config=config)
    """
    auto_clean: bool = True
    imputer_strategy: Dict[str, str] = None
    encoder_strategy: Dict[str, List[str]] = None
    scaler: Optional[str] = 'standard'
    id_columns: Optional[List[str]] = None


@dataclass
class ModelingConfig:
    """
    Configuration for model selection and training.
    
    Attributes:
        train_models (bool): Whether to train machine learning models.
            Default: True. Set to False to only profile data.
        
        models_to_train (List[str]): Specific models to train. Options:
            Classification: 'logistic_regression', 'random_forest', 'gradient_boosting',
                           'xgboost', 'lightgbm', 'svm'
            Regression: 'linear_regression', 'random_forest', 'gradient_boosting',
                       'xgboost', 'lightgbm', 'svr'
            Default: All available models for detected task.
        
        evaluation_metric (str): Metric to optimize during training.
            Classification: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
            Regression: 'mse', 'rmse', 'mae', 'r2', 'mape'
            Default: 'f1' for classification, 'rmse' for regression.
        
        n_models (int): Number of models to train (if using auto-selection).
            Default: 5. Range: 1-10.
        
        test_size (float): Proportion of data for testing. Default: 0.2.
    
    Example:
        >>> config = ModelingConfig(
        ...     models_to_train=['xgboost', 'lightgbm'],  # Only the best two
        ...     evaluation_metric='f1'
        ... )
        >>> automl = AutoML(modeling_config=config)
    """
    train_models: bool = True
    models_to_train: Optional[List[str]] = None
    evaluation_metric: Optional[str] = None
    n_models: int = 5
    test_size: float = 0.2


@dataclass
class OptimizationConfig:
    """
    Configuration for hyperparameter optimization using Optuna.
    
    Attributes:
        use_optuna (bool): Whether to use Optuna for hyperparameter tuning.
            Default: True. Adds ~5-10 minutes to training time but improves performance 3-8%.
        
        optuna_trials_per_model (int): Number of trials per model for optimization.
            More trials = better hyperparameters but slower. Default: 20.
            Recommended: 10-50 depending on time budget.
        
        optuna_timeout_seconds (int): Maximum time (seconds) to spend optimizing each model.
            Default: 300 (5 minutes). Set to None for no timeout.
        
        optuna_parallel_jobs (int): Number of parallel workers for Optuna.
            -1: Use all CPU cores. 1: Sequential. Default: -1.
        
        use_registry (bool): Whether to save trained models to registry.
            Default: True. Models saved to './trained_models/' directory.
    
    Example:
        >>> config = OptimizationConfig(
        ...     use_optuna=True,
        ...     optuna_trials_per_model=20,
        ...     optuna_timeout_seconds=300
        ... )
        >>> automl = AutoML(optimization_config=config)
    """
    use_optuna: bool = True
    optuna_trials_per_model: int = 20
    optuna_timeout_seconds: int = 300
    optuna_parallel_jobs: int = -1
    use_registry: bool = True
    early_stopping_rounds: int = None  # New: Early stopping for XGB/LGBM/CatBoost
    hyperparameter_overrides: Dict[str, Dict] = None  # New: Custom search spaces



@dataclass
class ReportingConfig:
    """
    Configuration for report generation and output.
    
    Attributes:
        generate_report (bool): Whether to generate PDF report after training.
            Default: True.
        
        report_detail (str): Detail level of PDF report.
            'brief': Summary insights only (~5 pages)
            'detailed': Complete analysis (~20+ pages)
            Default: 'detailed'.
        
        include_data_journey (bool): Whether to show complete data transformation journey.
            Shows original → cleaned → after each step. Default: True.
        
        include_model_comparison (bool): Whether to show detailed model benchmarks.
            Default: True.
        
        include_recommendations (bool): Whether to include ML recommendations.
            Default: True.
        
        visuals_limit (int): Maximum number of feature visualizations to include.
            More = larger PDF but more insights. Default: 10.
        
        plot_mode (str): Style of plots. Options: 'simple', 'dashboard', 'publication'.
            Default: 'simple'.
        
        include_shap (bool): Whether to include SHAP feature importance plots.
            Default: True. Adds ~100ms to report generation.
        
        color_scheme (str): Color scheme for visualizations.
            'dark' (black background, red accents) or 'light'. Default: 'dark'.
    
    Example:
        >>> config = ReportingConfig(
        ...     report_detail='detailed',
        ...     include_data_journey=True,
        ...     visuals_limit=15
        ... )
        >>> automl = AutoML(reporting_config=config)
    """
    generate_report: bool = True
    report_detail: str = 'detailed'
    include_data_journey: bool = True
    include_model_comparison: bool = True
    include_recommendations: bool = True
    visuals_limit: int = 10
    plot_mode: str = 'simple'
    include_shap: bool = True
    include_shap: bool = True
    color_scheme: str = 'light'


@dataclass
class ParallelConfig:
    """
    Configuration for parallel processing.
    
    Attributes:
        parallel_processing (bool): Whether to use parallel processing.
            Default: True. Provides 2-8x speedup on multi-core systems.
        
        n_jobs (int): Number of parallel jobs.
            -1: Use all CPU cores
            1: Sequential (single core)
            2+: Use specific number of cores
            Default: -1.
        
        backend (str): Joblib backend: 'threading', 'loky', 'dask'.
            'threading': Good for I/O bound, light computation
            'loky': Good for heavy computation
            'dask': For very large datasets
            Default: 'threading'.
        
        verbose (int): Verbosity level (0-2). Default: 0 (silent).
    
    Example:
        >>> config = ParallelConfig(n_jobs=4, enable_gpu=True)
        >>> automl = AutoML(parallel_config=config)
    """
    parallel_processing: bool = True
    n_jobs: int = -1
    backend: str = 'threading'
    verbose: int = 0
    enable_gpu: bool = False  # New: Enable GPU acceleration



class AutoML:
    """
    Enterprise-Grade AutoML Pipeline Orchestrator.
    
    This is the main class that orchestrates the entire machine learning workflow.
    It handles data validation, profiling, cleaning, feature engineering, model
    training with hyperparameter optimization, and comprehensive reporting.
    
    The design emphasizes transparency (users can see data at every stage), control
    (every aspect is configurable), and production-readiness (handles edge cases gracefully).
    
    Attributes:
        data_config (DataConfig): Configuration for data handling
        profiling_config (ProfilingConfig): Configuration for analysis
        preprocessing_config (PreprocessingConfig): Configuration for cleaning
        modeling_config (ModelingConfig): Configuration for model training
        optimization_config (OptimizationConfig): Configuration for hyperparameter tuning
        reporting_config (ReportingConfig): Configuration for report generation
        parallel_config (ParallelConfig): Configuration for parallel processing
        
        # State attributes (populated after fit())
        raw_profile_ (DatasetProfile): Profile of raw data before cleaning
        clean_profile_ (DatasetProfile): Profile of cleaned data
        X_ (DataFrame): Cleaned feature data (combined train + test)
        y_ (Series): Target variable (combined train + test)
        X_raw_ (DataFrame): Original raw data (for risk scoring)
        X_train_ (DataFrame): Cleaned training features
        X_test_ (DataFrame): Cleaned test features
        y_train_ (Series): Training target
        y_test_ (Series): Test target
        best_model_ (Model): The best trained model
        trained_models_ (Dict): All trained models
        model_benchmarks_ (List): Performance metrics for all models
        registry_ (ModelRegistry): Registry of trained models
        cleaning_log_ (Dict): Log of all cleaning operations performed
    
    Parameters:
        data_config (DataConfig, optional): Custom data configuration. If None, uses defaults.
        profiling_config (ProfilingConfig, optional): Custom profiling configuration.
        preprocessing_config (PreprocessingConfig, optional): Custom preprocessing configuration.
        modeling_config (ModelingConfig, optional): Custom modeling configuration.
        optimization_config (OptimizationConfig, optional): Custom optimization configuration.
        reporting_config (ReportingConfig, optional): Custom reporting configuration.
        parallel_config (ParallelConfig, optional): Custom parallel processing configuration.
        show_progress (bool): Whether to show progress messages. Default: True.
        save_artifacts (bool): Whether to save artifacts (models, plots, etc). Default: True.
        artifact_dir (str): Directory to save artifacts. Default: './octolearn_artifacts/'.
        **kwargs: For backward compatibility with previous parameter names.
    
    Example:
        >>> from octolearn import AutoML
        >>> import pandas as pd
        >>> 
        >>> # Basic usage with defaults
        >>> data = pd.read_csv('data.csv')
        >>> X = data.drop('target', axis=1)
        >>> y = data['target']
        >>> 
        >>> automl = AutoML()
        >>> automl.fit(X, y)
        >>> pdf = automl.generate_report()
        >>> predictions = automl.predict(X_new)
        >>> 
        >>> # Advanced usage with custom configuration
        >>> automl = AutoML(
        ...     data_config=DataConfig(sample_size=1000, test_size=0.25),
        ...     profiling_config=ProfilingConfig(analyze_interactions=False),
        ...     optimization_config=OptimizationConfig(optuna_trials_per_model=50),
        ...     reporting_config=ReportingConfig(report_detail='detailed')
        ... )
        >>> automl.fit(X, y)
        >>> 
        >>> # Access insights
        >>> print(f"Risk Score: {automl.get_risk_score()}")
        >>> print(f"Best Model: {automl.best_model_.__class__.__name__}")
        >>> print(f"Feature Importance: {automl.get_feature_importance()}")
        >>> 
        >>> # Access intermediate results
        >>> print(f"Training shape: {automl.X_train_.shape}")
        >>> print(f"Cleaned columns: {automl.X_.columns.tolist()}")
        >>> print(f"Outliers detected: {automl.outlier_results_}")
    
    Raises:
        ValueError: If input data is invalid or incompatible
        TypeError: If parameters are of wrong type
        RuntimeError: If pipeline execution fails
    
    Note:
        All configuration objects have sensible defaults, so you can start with
        AutoML() and customize only what you need.
    
    Version:
        0.8.0
    """
    
    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        profiling_config: Optional[ProfilingConfig] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        modeling_config: Optional[ModelingConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        reporting_config: Optional[ReportingConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
        show_progress: bool = True,
        save_artifacts: bool = True,
        artifact_dir: str = './octolearn_artifacts/',
        # Backward compatibility parameters
        use_full_data: bool = None,
        sample_size: int = None,
        test_size: float = None,
        random_state: int = None,
        **kwargs
    ):
        """
        Initialize the AutoML pipeline.
        
        Parameters are organized into logical configuration groups for clarity and
        maintainability. Each group can be customized independently.
        
        Args:
            data_config: DataConfig object for data handling. Default: DataConfig()
            profiling_config: ProfilingConfig object for analysis. Default: ProfilingConfig()
            preprocessing_config: PreprocessingConfig object for cleaning. Default: PreprocessingConfig()
            modeling_config: ModelingConfig object for model training. Default: ModelingConfig()
            optimization_config: OptimizationConfig for hyperparameter tuning. Default: OptimizationConfig()
            reporting_config: ReportingConfig for report generation. Default: ReportingConfig()
            parallel_config: ParallelConfig for parallel processing. Default: ParallelConfig()
            show_progress: Whether to print progress messages. Default: True.
            save_artifacts: Whether to save models and plots to disk. Default: True.
            artifact_dir: Directory for artifacts. Default: './octolearn_artifacts/'.
            **kwargs: Backward compatibility for old parameter names (deprecated).
        
        Raises:
            TypeError: If config objects are not of correct type
            ValueError: If parameter values are out of valid range
        """
        # Initialize config objects with defaults or provided values
        self.data_config = data_config or DataConfig()
        self.profiling_config = profiling_config or ProfilingConfig()
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.modeling_config = modeling_config or ModelingConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        self.reporting_config = reporting_config or ReportingConfig()
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Backward compatibility: extract legacy kwargs
        if 'train_models' in kwargs:
            self.modeling_config.train_models = kwargs.pop('train_models')
        if 'generate_shap' in kwargs:
            self.reporting_config.include_shap = kwargs.pop('generate_shap')
        
        # Validate all config objects
        self._validate_configs()
        
        # General settings
        self.show_progress = show_progress
        self.save_artifacts = save_artifacts
        self.artifact_dir = artifact_dir
        
        # Create artifact directory if needed
        if save_artifacts:
            os.makedirs(artifact_dir, exist_ok=True)
        
        # Initialize profiler
        self.profiler = DataProfiler()
        
        # State attributes - initialized as None, populated during fit()
        self.raw_profile_ = None
        self.clean_profile_ = None
        self.X_ = None
        self.y_ = None
        self.X_raw_ = None
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        
        # Components
        self.cleaner_ = None
        self.outlier_results_ = None
        self.interaction_results_ = None
        self.cleaning_log_ = None
        
        # Model artifacts
        self.trained_models_ = None
        self.best_model_ = None
        self.model_benchmarks_ = None
        self.registry_ = None
        
        if self.show_progress:
            logger.info(f"AutoML initialized (v0.8.0)")
            self._log_configuration()
    
    def _validate_configs(self):
        """
        Validate all configuration objects and their values.
        
        Checks that:
        - Config objects are correct types
        - Numeric values are in valid ranges
        - Mutually exclusive options are not both set
        
        Raises:
            TypeError: If config is wrong type
            ValueError: If parameter out of valid range
        """
        # Validate data config
        if not isinstance(self.data_config, DataConfig):
            raise TypeError(f"data_config must be DataConfig, got {type(self.data_config)}")
        
        if not 0.05 <= self.data_config.test_size <= 0.5:
            raise ValueError(f"test_size must be between 0.05 and 0.5, got {self.data_config.test_size}")
        
        if self.data_config.sample_size < 50:
            raise ValueError(f"sample_size must be >= 50, got {self.data_config.sample_size}")
        
        # Validate modeling config
        if not isinstance(self.modeling_config, ModelingConfig):
            raise TypeError(f"modeling_config must be ModelingConfig, got {type(self.modeling_config)}")
        
        if not 1 <= self.modeling_config.n_models <= 10:
            raise ValueError(f"n_models must be between 1 and 10, got {self.modeling_config.n_models}")
        
        # Validate optimization config
        if not isinstance(self.optimization_config, OptimizationConfig):
            raise TypeError(f"optimization_config must be OptimizationConfig, got {type(self.optimization_config)}")
        
        if self.optimization_config.optuna_trials_per_model < 5:
            logger.warning(f"optuna_trials_per_model < 5 may result in poor hyperparameter optimization")
        
        # Validate reporting config
        if self.reporting_config.report_detail not in ['brief', 'detailed']:
            raise ValueError(f"report_detail must be 'brief' or 'detailed', got '{self.reporting_config.report_detail}'")
        
        if self.reporting_config.visuals_limit < 5:
            logger.warning(f"visuals_limit < 5 may result in sparse visualizations")
    
    def _log_configuration(self):
        """
        Log all configuration settings at startup for transparency.
        
        This helps users understand what parameters are being used and makes
        debugging and reproducibility easier.
        """
        logger.info("\n" + "="*70)
        logger.info("CONFIGURATION SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nData Configuration:")
        logger.info(f"  - Use full data: {self.data_config.use_full_data}")
        logger.info(f"  - Sample size: {self.data_config.sample_size}")
        logger.info(f"  - Test size: {self.data_config.test_size}")
        logger.info(f"  - Random state: {self.data_config.random_state}")
        
        logger.info(f"\nProfiling Configuration:")
        logger.info(f"  - Detect outliers: {self.profiling_config.detect_outliers}")
        logger.info(f"  - Analyze interactions: {self.profiling_config.analyze_interactions}")
        logger.info(f"  - Generate risk score: {self.profiling_config.generate_risk_score}")
        logger.info(f"  - Calculate feature importance: {self.profiling_config.calculate_feature_importance}")
        
        logger.info(f"\nModeling Configuration:")
        logger.info(f"  - Train models: {self.modeling_config.train_models}")
        logger.info(f"  - Number of models: {self.modeling_config.n_models}")
        logger.info(f"  - Use Optuna: {self.optimization_config.use_optuna}")
        logger.info(f"  - Optuna trials per model: {self.optimization_config.optuna_trials_per_model}")
        
        logger.info(f"\nReporting Configuration:")
        logger.info(f"  - Report detail: {self.reporting_config.report_detail}")
        logger.info(f"  - Include data journey: {self.reporting_config.include_data_journey}")
        logger.info(f"  - Visuals limit: {self.reporting_config.visuals_limit}")
        
        logger.info(f"\nParallel Processing:")
        logger.info(f"  - Enabled: {self.parallel_config.parallel_processing}")
        logger.info(f"  - N jobs: {self.parallel_config.n_jobs}")
        logger.info(f"  - Backend: {self.parallel_config.backend}")
        logger.info("="*70 + "\n")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        # ── Optuna / hyperparameter tuning overrides ──────────────────────
        optuna_trials: Optional[int] = None,
        optuna_timeout: Optional[int] = None,
        use_optuna: Optional[bool] = None,
        # ── Data split overrides ──────────────────────────────────────────
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        # ── Model selection overrides ─────────────────────────────────────
        models: Optional[List[str]] = None,
        n_models: Optional[int] = None,
        evaluation_metric: Optional[str] = None,
        # ── Preprocessing overrides ───────────────────────────────────────
        imputer_strategy: Optional[Dict[str, str]] = None,
        scaler: Optional[str] = None,
    ) -> 'AutoML':
        """
        Execute the complete AutoML pipeline on your data.

        By default OctoLearn handles everything automatically with sensible
        defaults. The optional keyword arguments let you override specific
        settings **for this run only** without touching the stored config
        objects — so you can experiment freely without re-creating the
        AutoML instance.

        Pipeline stages:
            1. Input validation
            2. Raw data profiling
            3. Sampling (if configured)
            4. Train/test split with stratification
            5. Automatic data cleaning (imputation, encoding, scaling)
            6. Cleaned data profiling
            7. Feature engineering (outlier detection, interactions)
            8. Model training with hyperparameter optimisation
            9. Artifact saving

        Args:
            X (pd.DataFrame): Feature matrix (n_samples × n_features).
                Column names must be strings (auto-converted if not).
            y (pd.Series): Target variable (n_samples,).
                Classification → labels; Regression → continuous values.

            optuna_trials (int, optional): Override ``OptimizationConfig.
                optuna_trials_per_model`` for this run.
                Example: ``automl.fit(X, y, optuna_trials=5)`` for a quick run.
            optuna_timeout (int, optional): Override per-model Optuna timeout
                in seconds. ``None`` means no timeout.
            use_optuna (bool, optional): Override whether Optuna is used.
                ``False`` → use default hyperparameters (much faster).
            test_size (float, optional): Override ``DataConfig.test_size``.
                Must be between 0.05 and 0.5.
            random_state (int, optional): Override ``DataConfig.random_state``
                for reproducibility.
            models (list[str], optional): Override ``ModelingConfig.
                models_to_train``. E.g. ``['xgboost', 'lightgbm']``.
            n_models (int, optional): Override ``ModelingConfig.n_models``.
            evaluation_metric (str, optional): Override the primary metric
                used to rank models (e.g. ``'roc_auc'``, ``'rmse'``).
            imputer_strategy (dict, optional): Override imputation strategy,
                e.g. ``{'numeric': 'median', 'categorical': 'mode'}``.
            scaler (str, optional): Override feature scaler:
                ``'standard'``, ``'robust'``, ``'minmax'``, or ``None``.

        Returns:
            AutoML: Returns *self* for method chaining::

                pdf = AutoML().fit(X, y, optuna_trials=10).generate_report()

        Raises:
            TypeError: If X is not a DataFrame or y is not a Series.
            ValueError: If X and y have different row counts, or are empty.

        Examples:
            Quick run — no Optuna, 2 models::

                automl.fit(X, y, use_optuna=False, n_models=2)

            Full run with custom Optuna budget::

                automl.fit(X, y, optuna_trials=50, optuna_timeout=300)

            Custom preprocessing::

                automl.fit(X, y,
                           imputer_strategy={'numeric': 'median'},
                           scaler='robust')
        """
        # ── Apply per-call overrides non-destructively ────────────────────
        # We snapshot the original values and restore them after the run so
        # the stored config objects are never mutated by fit() kwargs.
        _orig = {}

        if optuna_trials is not None:
            _orig['optuna_trials_per_model'] = self.optimization_config.optuna_trials_per_model
            self.optimization_config.optuna_trials_per_model = optuna_trials

        if optuna_timeout is not None:
            _orig['optuna_timeout_seconds'] = self.optimization_config.optuna_timeout_seconds
            self.optimization_config.optuna_timeout_seconds = optuna_timeout

        if use_optuna is not None:
            _orig['use_optuna'] = self.optimization_config.use_optuna
            self.optimization_config.use_optuna = use_optuna

        if test_size is not None:
            if not 0.05 <= test_size <= 0.5:
                raise ValueError(f"test_size must be between 0.05 and 0.5, got {test_size}")
            _orig['test_size'] = self.data_config.test_size
            self.data_config.test_size = test_size

        if random_state is not None:
            _orig['random_state'] = self.data_config.random_state
            self.data_config.random_state = random_state

        if models is not None:
            _orig['models_to_train'] = self.modeling_config.models_to_train
            self.modeling_config.models_to_train = models

        if n_models is not None:
            _orig['n_models'] = self.modeling_config.n_models
            self.modeling_config.n_models = n_models

        if evaluation_metric is not None:
            _orig['evaluation_metric'] = self.modeling_config.evaluation_metric
            self.modeling_config.evaluation_metric = evaluation_metric

        if imputer_strategy is not None:
            _orig['imputer_strategy'] = self.preprocessing_config.imputer_strategy
            self.preprocessing_config.imputer_strategy = imputer_strategy

        if scaler is not None:
            _orig['scaler'] = self.preprocessing_config.scaler
            self.preprocessing_config.scaler = scaler

        try:
            # Step 1: Validate inputs
            self._validate_inputs(X, y)

            # Step 2-10: Execute pipeline
            self._execute_pipeline(X, y)

            if self.show_progress:
                logger.info("AutoML pipeline complete! [OK]")

        finally:
            # ── Restore original config values ────────────────────────────
            for attr, val in _orig.items():
                if attr in ('optuna_trials_per_model', 'optuna_timeout_seconds', 'use_optuna'):
                    setattr(self.optimization_config, attr, val)
                elif attr in ('test_size', 'random_state'):
                    setattr(self.data_config, attr, val)
                elif attr in ('models_to_train', 'n_models', 'evaluation_metric'):
                    setattr(self.modeling_config, attr, val)
                elif attr in ('imputer_strategy', 'scaler'):
                    setattr(self.preprocessing_config, attr, val)

        return self
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Comprehensively validate input X and y.
        
        Performs detailed validation including type checking, shape compatibility,
        column name validation, and edge case detection.
        
        Args:
            X (pd.DataFrame): Feature matrix to validate
            y (pd.Series or array-like): Target variable to validate
        
        Raises:
            TypeError: If inputs are wrong type
            ValueError: If data is invalid or incompatible
        """
        # Type checking
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X).__name__}. "
                f"Hint: Use pd.DataFrame(X) to convert numpy arrays or lists."
            )
        
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                f"y must be pandas Series/DataFrame or numpy array, got {type(y).__name__}"
            )
        
        # Convert y to Series if needed
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        elif isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    f"y has {y.shape[1]} columns, but must be 1D. "
                    f"Hint: Use y = y.iloc[:, 0] to extract single column."
                )
            y = y.squeeze()
        
        # Emptiness checks
        if X.empty:
            raise ValueError("X is empty. Please check your input data.")
        
        if len(y) == 0:
            raise ValueError("y is empty. Please check your target variable.")
        
        # Shape compatibility
        if len(X) != len(y):
            raise ValueError(
                f"X and y have different number of rows: {len(X)} vs {len(y)}. "
                f"Hint: Ensure both have the same number of samples."
            )
        
        # Column name validation
        non_string_cols = [col for col in X.columns if not isinstance(col, str)]
        if non_string_cols:
            logger.warning(
                f"Found {len(non_string_cols)} non-string column names. "
                f"Converting to strings for compatibility."
            )
            X.columns = [str(col) for col in X.columns]
        
        # All-NaN column detection
        all_nan_cols = X.columns[X.isnull().all()].tolist()
        if all_nan_cols:
            raise ValueError(
                f"Found {len(all_nan_cols)} columns with all null values: {all_nan_cols}. "
                f"Hint: Remove these columns before passing to AutoML."
            )
        
        # Target validation
        if y.isnull().all():
            raise ValueError(
                "Target variable (y) is entirely null/NaN. "
                "Hint: Check your target variable for valid values."
            )
        
        # Classification-specific checks
        n_unique = y.nunique()
        if n_unique == 1:
            logger.warning(
                "Target variable has only 1 unique value. "
                "This is not a valid classification problem. "
                "Hint: Check your target variable or consider it as regression."
            )
        elif n_unique > 20 and y.dtype in ['object', 'category']:
            logger.warning(
                f"Target variable has {n_unique} unique classes (multi-class classification). "
                "This can be challenging. Consider reducing target classes if possible."
            )
        
        if self.show_progress:
            logger.info(f"[OK] Input validation passed: X{X.shape}, y{len(y)}")
    
    def _execute_pipeline(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Execute all stages of the AutoML pipeline.
        
        This is the main orchestration method that coordinates all pipeline stages.
        """
        # Make copies to avoid modifying original data
        X = X.copy()
        y = y.copy()
        self.X_raw_ = X.copy()  # Save raw for risk scoring
        
        if not self.data_config.use_full_data and len(X) > self.data_config.sample_size:
            if self.show_progress:
                logger.info(f"Performance Optimization: Sampling {self.data_config.sample_size} rows from {len(X)} total...")
            
            # Stratified sampling if possible
            stratify = None
            if self.data_config.stratify_target:
                n_unique = y.nunique()
                # Stratify for categorical targets or integer targets with few classes
                if y.dtype.kind in ('O', 'U', 'S') or y.dtype.name == 'category' or (
                    y.dtype.kind in ('i', 'u') and n_unique < 20
                ):
                    stratify = y

            # Use sklearn's train_test_split for stratified sampling, but we just want one chunk
            try:
                from sklearn.model_selection import train_test_split
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, 
                    train_size=self.data_config.sample_size,
                    random_state=self.data_config.random_state,
                    stratify=stratify
                )
                X = X_sample.copy()
                y = y_sample.copy()
            except Exception as e:
                # Fallback to random choice if stratification fails (e.g. rare classes)
                logger.warning(f"Stratified sampling failed ({e}), falling back to random sampling.")
                sample_idx = np.random.RandomState(self.data_config.random_state).choice(
                    len(X), size=self.data_config.sample_size, replace=False
                )
                X = X.iloc[sample_idx].copy()
                y = y.iloc[sample_idx].copy()
        
        self.X_raw_ = X.copy()  # Save raw for risk scoring (now sampled)
        
        # PHASE 1: Raw data profiling
        if self.show_progress:
            logger.info("\nPHASE 1: Profiling raw data...")
        # Since we already sampled globally if needed, we pass the (potentially sampled) data
        self._profile_raw_data(X, y)
        
        # PHASE 2: Train/test split (BEFORE cleaning to prevent leakage)
        if self.show_progress:
            logger.info("PHASE 2: Train/test split...")
        self._split_data(X, y)
        
        # PHASE 3: Data cleaning
        if self.show_progress:
            logger.info("PHASE 3: Data cleaning...")
        self._clean_data()
        
        # PHASE 4: Cleaned data profiling
        if self.show_progress:
            logger.info("PHASE 4: Profiling cleaned data...")
        self._profile_clean_data()
        
        # PHASE 5: Feature engineering
        if self.show_progress:
            logger.info("PHASE 5: Feature engineering...")
        self._feature_engineering()
        
        # PHASE 6: Model training
        if self.modeling_config.train_models:
            if self.show_progress:
                logger.info("PHASE 6: Model training...")
            self._train_models()
    
    def _profile_raw_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Profile raw data before cleaning."""
        # Sample logic moved to _execute_pipeline for global effect.
        # Here we just use the passed X/y which might already be sampled.
        X_for_profile = X
        y_for_profile = y
        
        # Legacy: if use_full_data=True but we still want to profile a sample?
        # Actually, if use_full_data=True, we profile everything.
        # If use_full_data=False, we already sampled in _execute_pipeline.
        # So we can just use X/y directly.
        pass
        
        # Profile
        self.raw_profile_ = self.profiler.profile(X_for_profile, y_for_profile)
        
        # Generate preprocessing suggestions
        if self.show_progress:
            logger.info("  Generating preprocessing suggestions...")
        self.preprocessing_suggester_ = PreprocessingSuggester(self.raw_profile_, X)
        self.preprocessing_suggestions_ = self.preprocessing_suggester_.generate_suggestions()
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split data into train/test sets."""
        # Determine stratification
        stratify = None
        if self.data_config.stratify_target:
            n_unique = y.nunique()
            # Stratify for categorical targets or integer targets with few classes
            if y.dtype.kind in ('O', 'U', 'S') or y.dtype.name == 'category' or (
                y.dtype.kind in ('i', 'u') and n_unique < 20
            ):
                stratify = y
        
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X, y,
            test_size=self.data_config.test_size,
            random_state=self.data_config.random_state,
            stratify=stratify
        )
        
        if self.show_progress:
            logger.info(f"  Train: {len(self.X_train_)} rows | Test: {len(self.X_test_)} rows")
    
    def _clean_data(self) -> None:
        """Clean training and test data."""
        if self.preprocessing_config.auto_clean:
            # Remove duplicates ONLY from training set
            initial_train_size = len(self.X_train_)
            dup_mask = self.X_train_.duplicated()
            dup_count = dup_mask.sum()
            if dup_count > 0:
                self.X_train_ = self.X_train_[~dup_mask]
                self.y_train_ = self.y_train_[~dup_mask]
                if self.show_progress:
                    logger.info(f"  Removed {dup_count} duplicate rows from training set")
            
            # Clean data
            try:
                self.cleaner_ = AutoCleaner(
                    profile=self.raw_profile_,
                    imputer_strategy=self.preprocessing_config.imputer_strategy,
                    encoder_strategy=self.preprocessing_config.encoder_strategy,
                    scaler=self.preprocessing_config.scaler,
                    id_columns=self.preprocessing_config.id_columns
                )
                
                self.X_train_, self.y_train_, self.cleaning_log_ = self.cleaner_.fit_transform(
                    self.X_train_, self.y_train_
                )
                self.X_test_ = self.cleaner_.transform(self.X_test_)
                
                if self.show_progress:
                    logger.info("  Data cleaning complete [OK]")
                
                # Manually inject duplicate removal stats since it was done outside AutoCleaner
                if self.cleaning_log_:
                    self.cleaning_log_['duplicates_removed'] = int(dup_count)
                    
            except Exception as e:
                logger.error(f"Data cleaning failed: {str(e)}")
                raise ValueError(f"Data cleaning failed. Error: {str(e)}") from e
        
        # Combine for analysis
        self.X_ = pd.concat([self.X_train_, self.X_test_], axis=0).sort_index()
        self.y_ = pd.concat([self.y_train_, self.y_test_], axis=0).sort_index()
    
    def _profile_clean_data(self) -> None:
        """Profile cleaned data."""
        self.clean_profile_ = self.profiler.profile(self.X_, self.y_)
    
    def _feature_engineering(self) -> None:
        """Perform intelligent feature engineering."""
        # 1. Feature Generation (New in v0.8.0)
        # We use analyzing_interactions flag as proxy, or it should be enabled by default
        if self.profiling_config.analyze_interactions:
             try:
                 generator = FeatureGenerator(self.clean_profile_)
                 # Fit on training data ONLY to prevent leakage
                 generator.fit(self.X_train_, self.y_train_)
                 
                 # Transform both
                 self.X_train_ = generator.transform(self.X_train_)
                 self.X_test_ = generator.transform(self.X_test_)
                 
                 # Update combined data for downstream tasks
                 self.X_ = pd.concat([self.X_train_, self.X_test_], axis=0).sort_index()
                 
                 # Re-profile if features were added
                 if generator.skewed_feats_ or generator.date_cols_ or generator.interaction_names_:
                     logger.info("Features generated. Re-profiling...")
                     self.clean_profile_ = self.profiler.profile(self.X_, self.y_)
                     
             except Exception as e:
                 logger.warning(f"Feature generation failed: {str(e)}")

        # 2. Outlier Detection
        if self.profiling_config.detect_outliers:
            try:
                outlier_detector = OutlierDetector(self.X_, self.clean_profile_)
                self.outlier_results_ = outlier_detector.detect()
            except Exception as e:
                logger.warning(f"Outlier detection failed: {str(e)}")
        
        # 3. Interaction Analysis (for reporting)
        if self.profiling_config.analyze_interactions:
            try:
                interaction_analyzer = FeatureInteractionAnalyzer(self.X_, self.y_, self.clean_profile_)
                self.interaction_results_ = interaction_analyzer.analyze()
            except Exception as e:
                logger.warning(f"Interaction analysis failed: {str(e)}")
    
    def _train_models(self) -> None:
        """Train machine learning models."""
        if not self.modeling_config.train_models:
            logger.info("Skipping model training (train_models=False)")
            return
        
        try:
            metric = self.modeling_config.evaluation_metric
            if metric is None:
                metric = 'f1' if self.clean_profile_.task_type == 'classification' else 'rmse'
            
            trainer = ModelTrainer(
                X=None, y=None,
                profile=self.clean_profile_,
                task_type=self.clean_profile_.task_type,
                evaluation_metric=metric,
                X_train=self.X_train_,
                X_test=self.X_test_,
                y_train=self.y_train_,
                y_test=self.y_test_,
                # New params
                enable_gpu=self.parallel_config.enable_gpu,
                early_stopping_rounds=self.optimization_config.early_stopping_rounds,
                hyperparameter_overrides=self.optimization_config.hyperparameter_overrides,
                n_trials=self.optimization_config.optuna_trials_per_model if self.optimization_config.use_optuna else None,
                timeout_seconds=self.optimization_config.optuna_timeout_seconds if self.optimization_config.use_optuna else None
            )
            
            results = trainer.train_all_models()
            
            self.trained_models_ = trainer.trained_models
            self.best_model_ = trainer.best_model
            self.model_benchmarks_ = getattr(trainer, 'model_benchmarks', [])
            
            if self.optimization_config.use_registry:
                self.registry_ = ModelRegistry()
                for model_name, model in trainer.trained_models.items():
                    score = trainer.model_scores.get(model_name, 0)
                    params = trainer.best_hp_params.get(model_name, {})
                    self.registry_.register_model(
                        name=model_name, model=model,
                        task_type=self.clean_profile_.task_type,
                        metrics={'score': score},
                        parameters=params
                    )
            
            if self.show_progress:
                logger.info(f"Model training complete [OK]")
                logger.info(f"  Best model: {results.get('best_model')}")
                logger.info(f"  Best score: {results.get('best_score'):.4f}")
                
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using the best trained model.
        
        Automatically applies the same preprocessing that was used during training,
        including imputation, encoding, and scaling.
        
        Args:
            X_new (pd.DataFrame): New data to make predictions on.
                Must have the same features as training data (after cleaning).
        
        Returns:
            np.ndarray: Predictions of same length as X_new.
                For classification: class labels
                For regression: continuous predictions
        
        Raises:
            ValueError: If model hasn't been trained yet
            Exception: If preprocessing or prediction fails
        
        Example:
            >>> automl = AutoML()
            >>> automl.fit(X_train, y_train)
            >>> y_pred = automl.predict(X_test)
            >>> # y_pred contains predictions in same format as y_train
        """
        if self.best_model_ is None:
            raise ValueError(
                "No trained model available. Run fit() and train_models=True first."
            )
        
        # Apply same preprocessing as training
        if self.cleaner_ is None:
            X_clean = X_new.copy()
        else:
            X_clean = self.cleaner_.transform(X_new)
        
        return self.best_model_.predict(X_clean)
    
    def generate_report(self, filename: Optional[str] = None) -> str:
        """
        Generate comprehensive PDF report.
        
        Creates a professional, detailed PDF report including:
        - Executive summary with key insights
        - Data quality assessment and risk score
        - Complete data transformation journey (showing before/after)
        - Preprocessing strategy and rationale
        - Feature importance and SHAP analysis
        - Model comparison and selection
        - Actionable recommendations
        - All visualizations and metrics
        
        The report serves as documentation for stakeholders and technical audience.
        
        Returns:
            str: Path to generated PDF file
        
        Raises:
            ValueError: If fit() hasn't been called yet
            Exception: If PDF generation fails
        
        Example:
            >>> automl = AutoML()
            >>> automl.fit(X, y)
            >>> pdf_path = automl.generate_report()
            >>> print(f"Report saved to: {pdf_path}")
            # Opens PDF to view results
        
        Note:
            Report generation can take 1-5 minutes depending on:
            - Dataset size
            - Number of features
            - Detail level (brief vs detailed)
            - Whether to include SHAP analysis
        """
        if self.raw_profile_ is None:
            raise ValueError("Run fit() before generating report.")
        
        if self.show_progress:
            logger.info("\nGenerating comprehensive PDF report...")
        
        # Generate all report components
        results = self._generate_report_components()
        
        # Determine report mode from config
        report_mode = getattr(self.reporting_config, 'report_detail', 'detailed')
        mode = 'brief' if report_mode == 'brief' else 'detailed'
        
        generator = ReportGenerator(
            raw_profile=self.raw_profile_,
            clean_profile=self.clean_profile_,
            mode=mode,
            dist_plots=results.get("dist_paths"),
            heatmap_plot=results.get("heatmap_path"),
            corr_summary=results.get("corr_summary", {}),
            recommendations=results.get("recommendations"),
            risk_score=results.get("risk_score"),
            risk_category=results.get("risk_category"),
            risk_factors=results.get("risk_factors"),
            preprocessing_suggestions=self.preprocessing_suggestions_,
            feature_importance=results.get("feature_importance"),
            shap_path=results.get("shap_path"),
            model_benchmarks=self.model_benchmarks_,
            best_model_name=self.best_model_.__class__.__name__ if self.best_model_ else None,
            cleaning_log=self.cleaning_log_,
            outlier_results=getattr(self, 'outlier_results_', {}),
            interaction_results=getattr(self, 'interaction_results_', {}),
            logo_path=getattr(self, 'logo_path', None) or str(Path(__file__).parent / 'images' / 'logo.png'),
            title=getattr(self, 'report_title', 'OctoLearn Intelligence Report'),
            author=getattr(self, 'author', 'OctoLearn AutoML'),
            company=getattr(self, 'report_company', 'Data Science Team'),
            # Pass raw and clean DataFrames for before/after distribution plots
            raw_X=self.X_raw_,
            clean_X=self.X_train_,
        )
        if self.show_progress:
            logger.info("  Composing PDF...")
        
        pdf_file = generator.generate(filename=filename)
        
        if self.show_progress:
            logger.info(f"[OK] Report saved: {pdf_file}")
        
        return pdf_file
    
    def _generate_report_components(self) -> Dict[str, Any]:
        """Generate all components needed for the report."""
        results = {}
        
        # Visualizations
        plotter = PlotGenerator(
            self.X_, 
            self.y_, 
            self.clean_profile_, 
            mode=self.reporting_config.plot_mode,
            theme=self.reporting_config.color_scheme
        )
        
        try:
            results["dist_paths"] = plotter.generate_smart_visuals(limit=self.reporting_config.visuals_limit)
        except Exception as e:
            logger.warning(f"Distribution plots failed: {e}")
        
        try:
            heatmap_result = plotter.generate_correlation_heatmap(
                corr_top_n=getattr(self.reporting_config, 'corr_top_n', 15)
            )
            # generate_correlation_heatmap returns (path, corr_summary)
            if isinstance(heatmap_result, tuple):
                results["heatmap_path"], results["corr_summary"] = heatmap_result
            else:
                results["heatmap_path"] = heatmap_result
                results["corr_summary"] = {}
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")
            results["corr_summary"] = {}
        
        if self.reporting_config.include_shap:
            try:
                results["shap_path"] = plotter.generate_shap_plot()
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
        
        # Feature importance
        if self.profiling_config.calculate_feature_importance:
            try:
                results["feature_importance"] = BaselineImportance(
                    self.X_, self.y_, self.clean_profile_
                ).calculate_importance()
            except Exception as e:
                logger.warning(f"Feature importance failed: {e}")
        
        # Risk score
        if self.profiling_config.generate_risk_score:
            try:
                scorer = RiskScorer(self.raw_profile_, self.X_raw_)
                score, category, factors = scorer.calculate_risk_score()
                results["risk_score"] = score
                results["risk_category"] = category
                results["risk_factors"] = factors
            except Exception as e:
                logger.warning(f"Risk scoring failed: {e}")
        
        # Recommendations
        if self.profiling_config.generate_recommendations:
            try:
                recommender = RecommendationEngine(
                    self.clean_profile_,
                    raw_profile=self.raw_profile_,  # Use raw count for sample-size messages
                )
                results["recommendations"] = recommender.generate()
            except Exception as e:
                logger.warning(f"Recommendations failed: {e}")
        
        return results
    
    def get_risk_score(self) -> Dict[str, Any]:
        """Get data quality risk assessment (0-100 scale)."""
        if self.raw_profile_ is None:
            raise ValueError("Run fit() first.")
        
        scorer = RiskScorer(self.raw_profile_, self.X_raw_)
        score, category, factors = scorer.calculate_risk_score()
        
        return {"score": score, "category": category, "factors": factors}
    
    def get_preprocessing_suggestions(self) -> Dict[str, Any]:
        """Get preprocessing recommendations for your data."""
        if self.raw_profile_ is None:
            raise ValueError("Run fit() first.")
        
        return self.preprocessing_suggestions_
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.clean_profile_ is None:
            raise ValueError("Run fit() first.")
        
        return BaselineImportance(self.X_, self.y_, self.clean_profile_).calculate_importance()
    
    def get_recommendations(self) -> Dict[str, List[str]]:
        """Get ML recommendations based on data analysis."""
        if self.clean_profile_ is None:
            raise ValueError("Run fit() first.")

        engine = RecommendationEngine(
            self.clean_profile_,
            raw_profile=self.raw_profile_,  # Use raw count for sample-size messages
        )
        return engine.generate()
    
    def get_model_benchmarks(self) -> List[Dict]:
        """Get performance metrics for all trained models."""
        if self.model_benchmarks_ is None:
            logger.warning("No models trained. Set train_models=True in fit().")
            return []
        
        return self.model_benchmarks_
    
    def __repr__(self) -> str:
        """String representation showing pipeline state."""
        status = "Not fitted" if self.raw_profile_ is None else "Fitted"
        models = len(self.trained_models_ or {})
        return f"AutoML(status='{status}', models_trained={models})"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [
            "OctoLearn AutoML Pipeline",
            "=" * 50,
            f"Status: {'Fitted' if self.raw_profile_ is not None else 'Not fitted'}",
        ]
        
        if self.raw_profile_ is not None:
            lines.extend([
                f"Raw data shape: {self.raw_profile_.shape}",
                f"Cleaned data shape: {self.clean_profile_.shape if self.clean_profile_ else 'N/A'}",
                f"Task type: {self.raw_profile_.task_type}",
                f"Models trained: {len(self.trained_models_ or {})}",
                f"Best model: {self.best_model_.__class__.__name__ if self.best_model_ else 'None'}",
            ])
        
        return "\n".join(lines)


# Convenience alias for shorter imports
__all__ = ['AutoML', 'DataConfig', 'ProfilingConfig', 'PreprocessingConfig',
           'ModelingConfig', 'OptimizationConfig', 'ReportingConfig', 'ParallelConfig']