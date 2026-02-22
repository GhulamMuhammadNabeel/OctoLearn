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
    0.9.0
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
from .preprocessing.pipeline_builder import PipelineBuilder

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
    Configuration for data ingestion, sampling, and validation strategy.

    Parameters
    ----------
    use_full_data : bool, default=False
        If True, the entire dataset will be used for training. For large datasets,
        this may increase memory usage significantly.
    sample_size : int, default=500
        The number of rows to sample if `use_full_data` is False. This is useful
        for quick prototyping on large datasets.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split (0.0 < size < 1.0).
    random_state : int, default=42
        Determines random number generation for reproducibility across runs.
    stratify_target : bool, default=True
        If True, data is split in a stratified fashion using the target labels.
        Only applicable for classification tasks.

    Examples
    --------
    >>> from octolearn import AutoML, DataConfig
    >>> config = DataConfig(sample_size=1000, test_size=0.3)
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
    Configuration for dataset profiling and structural analysis.

    Parameters
    ----------
    detect_outliers : bool, default=True
        Whether to run outlier detection using multiple methods (IQR, Isolation Forest, Z-Score).
    analyze_interactions : bool, default=False
        Whether to perform pairwise feature interaction analysis. This can be
        computationally expensive for high-dimensional data.
    generate_risk_score : bool, default=True
        Whether to calculate an overall data health and reliability score.
    calculate_feature_importance : bool, default=True
        Whether to calculate baseline feature importance scores before modeling.
    generate_recommendations : bool, default=True
        Whether the engine should generate actionable advice for improving data/models.
    include_duplicates_analysis : bool, default=True
        Whether to scan and report on duplicate rows or highly correlated (leaked) features.

    Examples
    --------
    >>> from octolearn import ProfilingConfig
    >>> config = ProfilingConfig(analyze_interactions=True)
    """
    detect_outliers: bool = True
    analyze_interactions: bool = False
    generate_risk_score: bool = True
    calculate_feature_importance: bool = True
    generate_recommendations: bool = True
    include_duplicates_analysis: bool = True


@dataclass
class PreprocessingConfig:
    """
    Configuration for automated data cleaning, imputation, and encoding.

    Parameters
    ----------
    auto_clean : bool, default=True
        Whether to automatically apply data sanitation (ID removal, constant removal, etc.).
    imputer_strategy : dict, optional
        Custom mapping for imputation strategies (e.g., {'age': 'median'}).
        If None, the global default from `config.py` is used.
    encoder_strategy : dict, optional
        Custom mapping for encoding categorical columns.
    scaler : str, default='standard'
        The scaling method to apply to numeric features. Options: 'standard', 'robust', 'minmax', None.
    id_columns : list of str, optional
        Explicit list of columns to treat as IDs and exclude from modeling.

    Examples
    --------
    >>> config = PreprocessingConfig(scaler='robust', auto_clean=True)
    """
    auto_clean: bool = True
    imputer_strategy: Dict[str, str] = None
    encoder_strategy: Dict[str, List[str]] = None
    scaler: Optional[str] = 'standard'
    id_columns: Optional[List[str]] = None


@dataclass
class ModelingConfig:
    """
    Configuration for model selection, training, and ensemble strategy.

    Parameters
    ----------
    train_models : bool, default=True
        Whether to perform model training. If False, the pipeline will stop
        after the profiling/preprocessing phase.
    models_to_train : list of str, optional
        List of specific algorithms to train (e.g., ['xgboost', 'lightgbm']).
        If None, the defaults from `config.py` are used.
    evaluation_metric : str, optional
        The metric used to select the 'Champion' model. If None, defaults to
        'f1' for classification and 'r2' for regression.
    n_models : int, default=5
        Number of top-performing models to consider for the final leaderboard.
    test_size : float, default=0.2
        Percentage of data to withhold for model evaluation.
    use_stacking : bool, default=True
        Whether to train a Stacking Ensemble (combiner) of the top base models.

    Examples
    --------
    >>> config = ModelingConfig(models_to_train=['xgboost'], use_stacking=False)
    """
    train_models: bool = True
    models_to_train: Optional[List[str]] = None
    evaluation_metric: Optional[str] = None
    n_models: int = 5
    test_size: float = 0.2
    use_stacking: bool = True


@dataclass
class OptimizationConfig:
    """
    Configuration for hyperparameter optimization and model tracking.

    Parameters
    ----------
    use_optuna : bool, default=True
        Whether to use the Optuna optimizer for automated hyperparameter tuning.
    optuna_trials_per_model : int, default=20
        Maximum number of optimization trials per unique algorithm.
    optuna_timeout_seconds : int, default=300
        Maximum time spent optimizing a single algorithm.
    optuna_parallel_jobs : int, default=-1
        Number of parallel trials to run (-1 uses all cores).
    use_registry : bool, default=True
        Whether to store trained models and experiments in the local Model Registry.
    early_stopping_rounds : int, optional
        Rounds of non-improvement before stopping boosting iterations.
    hyperparameter_overrides : dict, optional
        Manual hyperparameter constraints to pass to specific models.

    Examples
    --------
    >>> config = OptimizationConfig(optuna_trials_per_model=50)
    """
    use_optuna: bool = True
    optuna_trials_per_model: int = 20
    optuna_timeout_seconds: int = 300
    optuna_parallel_jobs: int = -1
    use_registry: bool = True
    early_stopping_rounds: int = None
    hyperparameter_overrides: Dict[str, Dict] = None


@dataclass
class ReportingConfig:
    """
    Configuration for final PDF report generation and visualization style.

    Parameters
    ----------
    generate_report : bool, default=True
        Whether to produce the professional PDF intelligence report.
    report_title : str, default='OctoLearn Intelligence Report'
        The header title displayed on the cover page.
    report_detail : str, default='detailed'
        The level of complexity in the metrics/narrative ('brief' or 'detailed').
    include_data_journey : bool, default=True
        Whether to include the 'Before & After' cleaning distribution visuals.
    include_model_comparison : bool, default=True
        Whether to include the 'Model Arena' leaderboard.
    include_recommendations : bool, default=True
        Whether to include the narrative insights section.
    visuals_limit : int, default=10
        Maximum number of feature distribution plots to generate.
    plot_mode : str, default='simple'
        The complexity of the matplotlib visuals.
    include_shap : bool, default=True
        Whether to calculate and display SHAP global importance values.
    color_scheme : str, default='light'
        The aesthetic theme of the report ('light', 'dark', or 'neon').

    Examples
    --------
    >>> config = ReportingConfig(report_title="Q4 Churn Prediction Analysis")
    """
    generate_report: bool = True
    report_title: str = 'OctoLearn Intelligence Report'
    report_detail: str = 'detailed'
    include_data_journey: bool = True
    include_model_comparison: bool = True
    include_recommendations: bool = True
    visuals_limit: int = 10
    plot_mode: str = 'simple'
    include_shap: bool = True
    color_scheme: str = 'light'


@dataclass
class ParallelConfig:
    """
    Configuration for multi-core parallel processing and hardware acceleration.

    Parameters
    ----------
    parallel_processing : bool, default=True
        Whether to enable parallel execution of training and profiling tasks.
    n_jobs : int, default=-1
        The number of worker threads/processes. -1 uses all available CPU cores.
    backend : str, default='threading'
        The parallelization engine ('threading', 'loky', or 'mulitprocessing').
    verbose : int, default=0
        Logging verbosity for parallel workers.
    enable_gpu : bool, default=False
        Whether to attempt hardware acceleration for compatible models (XGBoost/LightGBM).

    Examples
    --------
    >>> config = ParallelConfig(n_jobs=4, enable_gpu=True)
    """
    parallel_processing: bool = True
    n_jobs: int = -1
    backend: str = 'threading'
    verbose: int = 0
    enable_gpu: bool = False


class AutoML:
    """
    Enterprise-Grade AutoML Pipeline Orchestrator.

    This is the main entry point for the OctoLearn library. It orchestrates the
    entire machine learning lifecycle—from data profiling and automated cleaning
    to hyperparameter optimization and professional report generation.

    Attributes
    ----------
    best_model_ : object
        The top-performing estimator selected after training and optional optimization.
    model_benchmarks_ : dict
        Scores for all trained models across all evaluation metrics.
    X_train_ : pd.DataFrame
        The final cleaned and preprocessed training data.
    registry_ : ModelRegistry
        The local tracking system used for model versioning and artifact storage.

    Notes
    -----
    OctoLearn follows a 'fit-then-report' pattern. Calling `.fit()` executes the
    entire internal pipeline, while `.generate_report()` produces a PDF summarizing
    the intelligence gathered.

    Examples
    --------
    >>> from octolearn import AutoML
    >>> import pandas as pd
    >>> data = pd.read_csv('data.csv')
    >>> X, y = data.drop('target', axis=1), data['target']
    >>> automl = AutoML()
    >>> automl.fit(X, y)
    >>> pdf_path = automl.generate_report()
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
        Initialize the AutoML pipeline with specific configuration modules.

        Parameters
        ----------
        data_config : DataConfig, optional
            Settings for sampling and data splitting.
        profiling_config : ProfilingConfig, optional
            Settings for dataset health analysis and outlier detection.
        preprocessing_config : PreprocessingConfig, optional
            Settings for imputation, encoding, and scaling.
        modeling_config : ModelingConfig, optional
            Settings for algorithm selection and training.
        optimization_config : OptimizationConfig, optional
            Settings for Bayesian hyperparameter tuning (Optuna).
        reporting_config : ReportingConfig, optional
            Settings for the PDF intelligence report and plot themes.
        parallel_config : ParallelConfig, optional
            Settings for multi-core processing Backend.
        show_progress : bool, default=True
            Whether to output pipeline status to the console.
        save_artifacts : bool, default=True
            Whether to persist models and logs to the local filesystem.
        artifact_dir : str, default='./octolearn_artifacts/'
            Base directory for all saved artifacts.
        **kwargs : dict
            Deprecated legacy parameters for backward compatibility.
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
        self.y_train_ = None
        self.y_test_ = None
        
        # New: Track original dataset size before sampling/cleaning
        self.original_rows_ = None
        
        # New: Target encoder for string classification targets
        self.target_encoder_ = None

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
            logger.info(f"AutoML initialized (v0.9.0)")
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
        # Standard flags
        train_models: Optional[bool] = None,
    ):
        """
        Execute the complete AutoML pipeline on the provided dataset.

        This method orchestrates data validation, profiling, cleaning,
        feature engineering, and model training. By default, it uses the
        configuration provided during initialization.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix. Must be a pandas DataFrame.
        y : pd.Series
            The target vector. Must be a pandas Series.
        optuna_trials : int, optional
            Override for `optimization_config.optuna_trials_per_model`.
        optuna_timeout : int, optional
            Override for `optimization_config.optuna_timeout_seconds`.
        use_optuna : bool, optional
            Override for `optimization_config.use_optuna`.
        test_size : float, optional
            Override for `data_config.test_size`.
        random_state : int, optional
            Override for `data_config.random_state`.
        models : list of str, optional
            Override for `modeling_config.models_to_train`.
        n_models : int, optional
            Override for `modeling_config.n_models`.
        evaluation_metric : str, optional
            Override for `modeling_config.evaluation_metric`.
        imputer_strategy : dict, optional
            Override for `preprocessing_config.imputer_strategy`.
        scaler : str, optional
            Override for `preprocessing_config.scaler`.
        train_models : bool, optional
            Override for `modeling_config.train_models`.

        Returns
        -------
        self : AutoML
            Returns the instance itself after fitting.

        Raises
        ------
        ValueError
            If input dimensions are incompatible or data quality is too poor.
        TypeError
            If X or y are not pandas objects.
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

        if train_models is not None:
            _orig['train_models'] = self.modeling_config.train_models
            self.modeling_config.train_models = train_models

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
                elif attr in ('models_to_train', 'n_models', 'evaluation_metric', 'train_models'):
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
            logger.warning(
                f"Found {len(all_nan_cols)} columns with all null values: {all_nan_cols}. "
                f"Dropping them automatically."
            )
            X.drop(columns=all_nan_cols, inplace=True)
            # Update self.X_raw_ if it was already assigned? No, _validate_inputs is called before pipeline execution starts (usually).
            # But wait, X is passed by reference? 
            # If I modify X in place, caller sees it?
            # Pandas functions usually don't modify in place unless specified.
            # But `drop(inplace=True)` modifies the object.
            # `_validate_inputs` takes `X`.
            # If `X` is modified here, does it affect `_execute_pipeline`'s `X`?
            # `fit(X, y)` calls `_validate_inputs(X, y)`.
            # If `_validate_inputs` modifies `X` in place, `fit` sees it?
            # Validating inputs should generally NOT modify inputs.
            # But here we want to proceed.
            # Better to return cleaned X, y from validate?
            # Or just warn and let pipeline handle it if pipeline handles it?
            # `AutoML` pipeline starts with `profile`. Profiler handles NaN.
            # Then `cleaner`. Cleaner handles NaN (imputes).
            # But `cleaner` might fail on all-NaN column if strategy is 'mean' (all NaN -> mean=NaN).
            # So dropping is safer.
            # But I should probably do it in `fit` before calling `validate` or let `validate` return.
            # Refactoring: allow `validate` to modify? 
            # Or just warn here and let `AutoCleaner` handle it?
            # But if `AutoCleaner` fails...
            # The previous error was explicit `raise ValueError`.
            # I will drop in place and warn.

        
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
        self.original_rows_ = len(X)  # Save original row count for recommendations
        
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
            
            # PHASE 7: Feature Importance Analysis
            if self.show_progress:
                logger.info("PHASE 7: Feature Importance Analysis...")
            self._analyze_feature_importance()
    
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
        
        # Handle Target Encoding for Classification (if strings)
        # ModelTrainer (XGBoost/LightGBM) requires numeric targets
        if self.modeling_config.train_models:  # Only needed if training
             # Access task type from profile or detect it
             task = self.raw_profile_.task_type if self.raw_profile_ else "unknown"
             if task == 'classification':
                 # Check if target is not numeric
                 # Use y_train_ to check dtype
                 if not pd.api.types.is_numeric_dtype(self.y_train_):
                     try:
                         from sklearn.preprocessing import LabelEncoder
                         self.target_encoder_ = LabelEncoder()
                         # Fit on all known target values (train + test to be safe? No, only train usually)
                         # But wait, we split already.
                         # Best to fit on train, but handle unknown in test?
                         # LabelEncoder doesn't handle unknown.
                         # Better to fit on raw y before split?
                         # But we are in _clean_data, after split.
                         # Let's fit on concatenated y_train and y_test to ensure all classes are covered if possible
                         # (Leakage is minimal for target encoding labels)
                         self.target_encoder_.fit(pd.concat([self.y_train_, self.y_test_]))
                         
                         self.y_train_ = pd.Series(
                             self.target_encoder_.transform(self.y_train_),
                             index=self.y_train_.index,
                             name=self.y_train_.name
                         )
                         self.y_test_ = pd.Series(
                             self.target_encoder_.transform(self.y_test_),
                             index=self.y_test_.index,
                             name=self.y_test_.name
                         )
                         if self.show_progress:
                             logger.info(f"  Encoded string target labels to integers. Classes: {len(self.target_encoder_.classes_)}")
                     except Exception as e:
                         logger.warning(f"Target encoding failed: {e}")
        
        # Combine for analysis
        self.X_ = pd.concat([self.X_train_, self.X_test_], axis=0).sort_index()
        self.y_ = pd.concat([self.y_train_, self.y_test_], axis=0).sort_index()
    
    def _profile_clean_data(self) -> None:
        """Profile cleaned data."""
        self.clean_profile_ = self.profiler.profile(self.X_, self.y_)
    
    def _feature_engineering(self) -> None:
        """Perform intelligent feature engineering."""
        # 1. Feature Generation (New in v0.9.0)
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

    def _analyze_feature_importance(self) -> None:
        """Extract feature importance from the best model."""
        self.feature_importance_ = {}
        
        if self.best_model_ is None:
            return
            
        try:
            # Tree-based
            if hasattr(self.best_model_, 'feature_importances_'):
                importances = self.best_model_.feature_importances_
                feature_names = self.X_train_.columns
                self.feature_importance_ = dict(zip(feature_names, importances))
                
            # Linear models
            elif hasattr(self.best_model_, 'coef_'):
                importances = np.abs(self.best_model_.coef_)
                if importances.ndim > 1:
                    importances = importances[0] # Take first class for multiclass or flatten
                feature_names = self.X_train_.columns
                self.feature_importance_ = dict(zip(feature_names, importances))
                
            else:
                logger.info("Best model does not support native feature importance.")
                
        except Exception as e:
            logger.warning(f"Failed to extract feature importance: {e}")
    
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
            self.best_model_predictions_ = getattr(trainer, 'best_model_predictions', None)
            self.best_model_probabilities_ = getattr(trainer, 'best_model_probabilities', None)
            
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
        Predict target values for new, unseen data.

        Automatically applies the exact same preprocessing pipeline (imputation,
        encoding, scaling) that was fitted on the training data.

        Parameters
        ----------
        X_new : pd.DataFrame
            The new feature matrix to predict on.

        Returns
        -------
        y_pred : np.ndarray
            Predicted values (class labels or regression targets).

        Notes
        -----
        If the target was string-encoded during training, the predictions will
        be automatically decoded back to their original labels.
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
            best_model_name=self.model_benchmarks_[0]['model'] if self.model_benchmarks_ else (self.best_model_.__class__.__name__ if self.best_model_ else None),
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
            # Performance visuals
            confusion_matrix_plot=results.get('confusion_matrix'),
            roc_curve_plot=results.get('roc_curve'),
            residual_plot=results.get('residuals'),
            feature_importance_plot=results.get('feature_importance_plot'),
        )
        if self.show_progress:
            logger.info("  Composing PDF...")
        
        pdf_file = generator.generate(filename=filename)
        
        if self.show_progress:
            logger.info(f"[OK] Report saved: {pdf_file}")
        
        return pdf_file
    
    def get_pipeline(self) -> Any:
        """
        Get the complete standalone scikit-learn Pipeline (Preprocessing + Model).
        
        This returns a unified pipeline object that can be used for deployment
        without the OctoLearn library.
        
        Returns
        -------
        pipeline : sklearn.pipeline.Pipeline
            The complete ML pipeline containing all cleaning steps and the best model.
            
        Examples
        --------
        >>> automl.fit(X, y)
        >>> pipeline = automl.get_pipeline()
        >>> pipeline.predict(X_new)  # Standard sklearn API
        """
        if self.best_model_ is None:
            raise ValueError("Run fit() first to generate a model.")
            
        from sklearn.pipeline import Pipeline
        
        # 1. Build preprocessing part
        builder = PipelineBuilder(self.cleaner_, scaler=self.preprocessing_config.scaler)
        preprocessor = builder.build()
        
        # 2. Combine with best model
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', self.best_model_)
        ])
        
        return pipeline

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

        try:
            results["feature_importance_plot"] = plotter.generate_feature_importance_plot(
                getattr(self, 'feature_importance_', {})
            )
        except Exception as e:
            logger.warning(f"Feature importance plot failed: {e}")
            
        # Performance Plots
        try:
            # Need predictions and true metrics
            # Note: y_test_ is what we evaluated on. 
            if hasattr(self, 'y_test_') and hasattr(self, 'best_model_predictions_'):
                perf_paths = plotter.generate_performance_plots(
                    y_true=self.y_test_,
                    y_pred=np.array(self.best_model_predictions_) if self.best_model_predictions_ is not None else None,
                    y_proba=np.array(self.best_model_probabilities_) if self.best_model_probabilities_ is not None else None
                )
                results.update(perf_paths)
        except Exception as e:
            logger.warning(f"Performance plots failed: {e}")
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
                    raw_profile=self.raw_profile_,
                    original_row_count=self.original_rows_
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
            raw_profile=self.raw_profile_,
            original_row_count=self.original_rows_
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

    @classmethod
    def surprise_me(cls, task: str = 'classification') -> Tuple[Optional[str], Any]:
        """
        Automatically fetch a dataset, run the full OctoLearn pipeline, 
        and generate a comprehensive PDF intelligence report.

        This API demonstrates the full power of the OctoLearn library with 
        a single function call. It uses Bayesian Search (Optuna) to find 
        the best model and hyperparameters for the dataset.

        Parameters
        ----------
        task : str, default='classification'
            The type of ML task to demonstrate. Options: 'classification', 'regression'.

        Returns
        -------
        pdf_path : str
            The file path to the generated PDF report.
        best_model : object
            The optimized, trained model object pipeline.
        """
        import pandas as pd
        from sklearn.datasets import load_breast_cancer, fetch_california_housing
        import tempfile
        import os

        # 1. Fetch Dataset
        print(f"\n[OctoLearn Surprise Me] Fetching {task} dataset...")
        if task == 'classification':
            data = load_breast_cancer(as_frame=True)
            X = data.data
            y = data.target
            dataset_name = "Breast_Cancer"
        elif task == 'regression':
            data = fetch_california_housing(as_frame=True)
            X = data.data
            y = data.target
            dataset_name = "California_Housing"
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")
            
        print(f"[OctoLearn Surprise Me] Dataset shape: X{X.shape}, y{y.shape}")

        # 2. Configure AutoML for MAXIMUM power
        # We enable Optuna with a healthy number of trials to show off the Bayesian search
        opt_config = OptimizationConfig(
            use_optuna=True, 
            optuna_trials_per_model=30,  # 30 trials per model to find best params
            optuna_timeout_seconds=300   # 5 mins max per model
        )
        report_config = ReportingConfig(
            generate_report=True,
            report_detail='detailed',
            include_shap=True
        )

        automl = cls(
            optimization_config=opt_config,
            reporting_config=report_config
        )

        # 3. Run Pipeline
        print("\n[OctoLearn Surprise Me] Engaging AutoML Pipeline. Please wait...")
        automl.fit(X, y)

        # 4. Generate Report
        print("\n[OctoLearn Surprise Me] Generating Intelligence Report...")
        report_filename = f"OctoLearn_{dataset_name}_Intelligence_Report.pdf"
        try:
            pdf_path = automl.generate_report(filename=report_filename)
            absolute_path = os.path.abspath(pdf_path)
            print(f"\n[Success] View your Intelligence Report here: {absolute_path}")
        except Exception as e:
            print(f"\n[OctoLearn Surprise Me] Report generation failed: {e}")
            absolute_path = None

        print(f"[Success] Best model found: {automl.best_model_.__class__.__name__}")
        
        return absolute_path, automl.best_model_


# Convenience alias for shorter imports
__all__ = ['AutoML', 'DataConfig', 'ProfilingConfig', 'PreprocessingConfig',
           'ModelingConfig', 'OptimizationConfig', 'ReportingConfig', 'ParallelConfig']