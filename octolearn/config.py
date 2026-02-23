"""
Configuration Constants for OctoLearn.

This module contains all threshold constants, dictionary configurations, and 
hyperparameter search spaces for the OctoLearn orchestrator. Centralizing 
these settings ensures consistency across profiling, cleaning, modeling, 
and reporting phases.
"""

# ============================================================================
# DATASET PROFILING THRESHOLDS
# ============================================================================

PROFILING_CONFIG = {
    """Settings for automated feature type detection and structural analysis."""
    # Feature type detection
    'binary_cardinality_threshold': 2,                  # Cardinality for binary features
    'low_cardinality_threshold': 10,                    # Threshold for categorical
    'low_cardinality_ratio': 0.01,                      # Unique ratio for categorical conversion
    'high_cardinality_ratio': 0.3,                      # High cardinality (>30% unique)
    
    # Skewness and variance
    'skewness_threshold': 1.0,                          # |skew| > 1 = skewed
    'low_variance_threshold': 1e-5,                     # Variance < 1e-5 = low variance
    
    # Duplicate and leakage detection
    'leakage_correlation_threshold': 0.95,              # |corr| > 0.95 = leakage suspect
    
    # Sampling
    'smart_sample_threshold': 100_000,                  # Sample if > 100k rows
    'hash_sample_size': 1000,                           # Rows for hash generation
}

# ============================================================================
# RISK SCORING THRESHOLDS
# ============================================================================

RISK_SCORING_CONFIG = {
    """Scoring weights and logic for the Dataset Risk Assessment engine."""
    'thresholds': {
        'low_risk_max': 30,                             # 0-30 = Low
        'moderate_risk_max': 60,                        # 31-60 = Moderate
        'high_risk_max': 100,                           # 61-100 = High
    },
    
    'risk_factors': {
        'id_like_columns': 10,                          # Points for ID-like columns
        'potential_leakage': 25,                        # Points for leakage suspects
        'low_variance': 5,                              # Points for low variance
        'duplicate_rows': 5,                            # Points for duplicates
        'class_imbalance': 10,                          # Points for imbalance >85%
    },
    
    'imbalance_threshold': 0.85,                        # Class imbalance threshold
}

# ============================================================================
# OUTLIER DETECTION CONFIG
# ============================================================================

OUTLIER_CONFIG = {
    """Configuration for statistical and ML-based outlier detection."""
    'methods': ['iqr', 'isolation_forest', 'zscore'],  # Detection methods
    
    'iqr': {
        'multiplier': 1.5,                              # IQR * 1.5
    },
    
    'isolation_forest': {
        'contamination': 0.1,                           # Expected outlier %
        'n_estimators': 100,
        'random_state': 42,
    },
    
    'zscore': {
        'threshold': 3,                                 # Z-score threshold
    },
    
    'numeric_only': True,                              # Outliers for numeric only
    'report_percentile': 95,                           # Outlier % report
}

# ============================================================================
# FEATURE INTERACTION CONFIG
# ============================================================================

INTERACTION_CONFIG = {
    """Settings for automatic creation of polynomial and ratio interactions."""
    'enabled': True,
    'types': ['polynomial', 'interaction', 'ratio'],   # Interaction types
    
    'polynomial': {
        'degree': 2,                                    # Polynomial degree
        'interaction_only': False,                      # Include self terms
        'include_bias': False,                          # Include bias term
    },
    
    'interaction': {
        'max_pairs': 10,                                # Top N feature pairs
        'selection_method': 'correlation',              # By correlation to target
    },
    
    'ratio': {
        'enabled': True,
        'max_ratios': 5,                                # Top N ratios
    },
    
    'min_corr_interaction': 0.3,                       # Min corr to select
}

# ============================================================================
# FEATURE GENERATION CONFIG
# ============================================================================

FEATURE_GENERATION_CONFIG = {
    """Settings for semantic feature synthesis (Dates, Skew correction, etc.)."""
    'enabled': True,
    
    'date_features': {
        'enabled': True,
        'parts': ['year', 'month', 'day', 'weekday', 'is_weekend'], 
        'drop_original': True
    },
    
    'skewed_features': {
        'enabled': True,
        'threshold': 1.0,           # Skew threshold for log transform
        'method': 'log1p',          # Transformation method
        'drop_original': False      # Keep original features?
    },
    
    'interaction_features': {
        'enabled': True,            # Add top interactions
        'top_n': 5,                 # Number of interactions to add
    }
}

# ============================================================================
# AUTO CLEANING CONFIG
# ============================================================================

AUTO_CLEAN_CONFIG = {
    """Settings for the AutoCleaner module, including imputation and feature removal."""
    'enabled': True,
    
    'actions': {
        'remove_id_columns': True,                      # Remove ID-like columns
        'remove_constants': True,                       # Remove constant columns
        'remove_low_variance': True,                    # Remove low variance
        'remove_duplicates': True,                      # Remove duplicate rows
        'impute_missing': True,                         # Impute missing values
        'remove_high_cardinality': False,               # Don't auto-remove high card
    },
    
    'imputation': {
        'numeric': 'mean',                              # mean, median, knn
        'categorical': 'mode',                          # mode, constant
        'missing_threshold': 0.5,                       # Drop if > 50% missing
    },
}

# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

MODEL_TRAINING_CONFIG = {
    """Global settings for model selection, training splits, and parallelisms."""
    'enabled': True,
    'parallel_processing': True,                        # Use parallel models
    'n_jobs': -1,                                       # -1 = all cores
    
    'classification_models': [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'xgboost',
        'lightgbm',
        'svm',
    ],
    
    'regression_models': [
        'linear_regression',
        'random_forest',
        'gradient_boosting',
        'xgboost',
        'lightgbm',
        'svr',
    ],
    
    'test_split': 0.2,                                  # Train/test split
    'random_state': 42,
}

# ============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION CONFIG
# ============================================================================

OPTUNA_CONFIG = {
    """Configuration for hyperparameter optimization and search spaces."""
    'enabled': True,
    'study_name': 'octolearn_hpo',
    'baseline_score': None,
    
    'optimization': {
        'n_trials': 20,                                 # Number of trials (reduced for speed)
        'n_jobs': 1,                                    # Sequential trials (avoids CPU oversubscription on Windows)
        'timeout': 120,                                 # 2 min per-model timeout
        'sampler': 'TPESampler',                        # Sampler type
    },
    
    'pruner': 'MedianPruner',                          # Early stopping
    'cv_folds': 5,                                      # Cross-validation folds
    
    'hyperparameters': {
        'logistic_regression': {
            'C': [0.001, 100],                           # Expanded regularization
            'solver': ['lbfgs', 'liblinear'],
        },
        'linear_regression': {
            'fit_intercept': [True, False],
        },
        'random_forest': {
            'n_estimators': [50, 300],                  # Increased upper bound
            'max_depth': [3, 20],                       # Deeper trees
            'min_samples_split': [2, 15],
            'min_samples_leaf': [1, 10],
            'max_features': ['sqrt', 'log2', None],     # Added max_features
        },
        'gradient_boosting': {
            'n_estimators': [50, 300],
            'max_depth': [2, 10],
            'learning_rate': [0.005, 0.3],              # Lower learning rate bound
            'subsample': [0.5, 1.0],
            'min_samples_split': [2, 10],
        },
        'xgboost': {
            'n_estimators': [50, 400],                  # More estimators for XGBoost
            'max_depth': [3, 12],                       # Deeper trees
            'learning_rate': [0.005, 0.3],
            'subsample': [0.5, 1.0],
            'colsample_bytree': [0.5, 1.0],
            'gamma': [0, 5],                            # Added minimum loss reduction
            'min_child_weight': [1, 7],                 # Added child weight tuning
        },
        'lightgbm': {
            'n_estimators': [50, 400],
            'max_depth': [3, 12],
            'learning_rate': [0.005, 0.3],
            'num_leaves': [20, 150],                    # More leaves
            'subsample': [0.5, 1.0],                    # Added bagging
            'colsample_bytree': [0.5, 1.0],             # Added feature fraction
        },
        'svm': {
            'C': [0.01, 100],
            'kernel': ['rbf', 'linear', 'poly'],        # Added poly kernel
            'gamma': ['scale', 'auto'],
        },
        'svr': {
            'C': [0.01, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'epsilon': [0.01, 1.0],
        },
    },
}

# ============================================================================
# MODEL REGISTRY CONFIG
# ============================================================================

MODEL_REGISTRY_CONFIG = {
    """Settings for tracking and versioning models and performance artifacts."""
    'enabled': True,
    'storage': 'json',                                  # json (default, no deps), sqlite (optional), csv (readable)
    'db_path': '.octolearn/model_registry.json',       # Registry database/file
    
    'tracking': {
        'track_parameters': True,
        'track_metrics': True,
        'track_models': True,
        'track_artifacts': True,
    },
    
    'versioning': {
        'auto_version': True,
        'max_versions': 10,                             # Keep last N versions
    },
}

# ============================================================================
# EVALUATION METRICS CONFIG
# ============================================================================

EVALUATION_CONFIG = {
    """Metrics and cross-validation settings for model evaluation."""
    'classification_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'confusion_matrix',
        'classification_report',
    ],
    
    'regression_metrics': [
        'mse',
        'rmse',
        'mae',
        'r2',
        'mape',
    ],
    
    'cv_folds': 5,
    'random_state': 42,
}

# ============================================================================
# PARALLEL PROCESSING CONFIG
# ============================================================================

PARALLEL_CONFIG = {
    """Hardware acceleration and multi-core processing settings."""
    'enabled': True,
    'n_jobs': -1,                                       # -1 = all cores
    'backend': 'threading',                             # threading, loky, dask
    'batch_size': 'auto',
    'verbose': 0,
    
    'task_workers': {
        'profiling': 1,                                 # Sequential
        'outlier_detection': 4,
        'interaction_analysis': 4,
        'auto_cleaning': 1,
        'model_training': -1,                           # All cores
        'hyperparameter_tuning': -1,
    },
}

# ============================================================================
# REPORT GENERATION CONFIG
# ============================================================================

REPORT_CONFIG = {
    """Aesthetics, fonts, and section visibility for the PDF Intelligence Report."""
    'mode': 'detailed',
    'include_sections': {
        'executive_summary': True,
        'risk_analysis': True,
        'preprocessing_details': True,
        'feature_analysis': True,
        'model_benchmarking': True,
        'visual_insights': True,
        'recommendations': True,
    },
    'fonts': {
        'title': 'ShantellSans-ExtraBold',
        'section': 'ShantellSans-Bold',
        'normal': 'ShantellSans-Regular',
    },
    'colors': {
        'primary': '#00F0FF',        # Neon Cyan
        'risk_low': '#00FF9F',       # Neon Green
        'risk_moderate': '#FFB800',  # Neon Orange
        'risk_high': '#FF0055',      # Neon Pink/Red
        'background': '#0D0D15',     # Cyberpunk Dark
        'text': '#E0E0E0',
    },
}

# ============================================================================
# LOGGING CONFIG
# ============================================================================

LOGGING_CONFIG = {
    """Central logging configuration for the entire library."""
    'enabled': True,
    'level': 'INFO',                                    # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'octolearn.log',
}

# ============================================================================
# ERROR HANDLING CONFIG
# ============================================================================

ERROR_CONFIG = {
    """Fault tolerance and retry logic for the pipeline orchestration."""
    'raise_on_error': False,                            # Continue on errors
    'log_errors': True,
    'max_retries': 3,                                   # Retry failed tasks
    'retry_info': 'exponential',                        # Linear, exponential
}
