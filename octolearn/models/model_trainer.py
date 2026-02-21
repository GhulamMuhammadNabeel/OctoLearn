"""
Model Training Module with Optuna Hyperparameter Optimization

Trains multiple models with automated hyperparameter tuning
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# suppress parallel-related sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

from ..config import MODEL_TRAINING_CONFIG, OPTUNA_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class ModelTrainer:
    """
    Orchestrator for automated model training and hyperparameter tuning.

    This class manages the lifecycle of multiple machine learning models,
    handling data splitting, Optuna-based hyperparameter optimization, and
    stacking ensembles.

    Attributes
    ----------
    trained_models : dict of str: object
        Dictionary mapping model names (e.g., 'xgboost') to their fitted
        scikit-learn compatible estimator objects.
    model_scores : dict of str: float
        Dictionary mapping model names to their primary evaluation scores.
    best_model : object
        The top-performing estimator object after all training and optimization.
    best_hp_params : dict of str: dict
        Optimal hyperparameters found by Optuna for each model.
    model_benchmarks : pd.DataFrame
        A comparative table of all models across various performance metrics.
    """

    def __init__(
        self,
        X: Optional[pd.DataFrame],
        y: Optional[pd.Series],
        profile,
        task_type: str = None,
        evaluation_metric: str = None,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        y_test: Optional[pd.Series] = None,
        enable_gpu: bool = False,
        early_stopping_rounds: Optional[int] = None,
        hyperparameter_overrides: Optional[Dict] = None,
        n_trials: Optional[int] = None,
        timeout_seconds: Optional[int] = None
    ):
        """
        Initialize the ModelTrainer with data and optimization settings.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The full feature matrix.
        y : pd.Series, optional
            The full target vector.
        profile : DatasetProfile
            A profile object detailing data characteristics and task type.
        task_type : str, optional
            Manual override for the task ('classification' or 'regression').
        evaluation_metric : str, optional
            The metric to optimize during hyperparameter tuning.
        X_train, X_test, y_train, y_test : pandas objects, optional
            Pre-split datasets to avoid internal splitting and prevent leakage.
        enable_gpu : bool, default=False
            Whether to enable GPU acceleration for compatible models.
        early_stopping_rounds : int, optional
            The number of rounds for early stopping in gradient boosting models.
        hyperparameter_overrides : dict, optional
            Custom search spaces or fixed parameters for specific models.
        n_trials : int, optional
            The number of Optuna trials to run per model.
        timeout_seconds : int, optional
            The maximum time in seconds for the entire optimization process.
        """
        # store original
        self.X = X
        self.y = y
        self.profile = profile
        self.task_type = task_type or getattr(profile, 'task_type', None)
        self.evaluation_metric = evaluation_metric
        
        self.enable_gpu = enable_gpu
        self.early_stopping_rounds = early_stopping_rounds
        self.hyperparameter_overrides = hyperparameter_overrides or {}
        # User-supplied Optuna settings (override global config)
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds

        # containers
        self.trained_models: Dict[str, object] = {}
        self.model_scores: Dict[str, float] = {}
        self.best_model = None
        self.best_hp_params: Dict[str, Dict] = {}
        self.model_benchmarks = None
        
        # New: Store predictions for report visualizations
        self.best_model_predictions = None
        self.best_model_probabilities = None

        # --- Handle provided splits ---
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
            logger.info("ModelTrainer: using externally provided train/test split (no internal split).")
        else:
            # --- Prepare split internally if full X/y is available ---
            if self.X is None or self.y is None:
                raise ValueError(
                    "X and y must be provided if train/test splits are not supplied."
                )
            self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()

    def _prepare_data(self) -> Tuple:
        """Prepare train-test split for model training and evaluation."""
        test_split = MODEL_TRAINING_CONFIG.get('test_split', 0.2)
        random_state = MODEL_TRAINING_CONFIG.get('random_state', 42)

        stratify = None
        if self.task_type == 'classification':
             stratify = self.y
             # Check if stratification is possible (min 2 samples per class)
             if self.y.value_counts().min() < 2:
                 logger.warning("Target class has <2 samples. Disabling stratified split.")
                 stratify = None
        
        try:
            return train_test_split(
                self.X, self.y,
                test_size=test_split,
                random_state=random_state,
                stratify=stratify
            )
        except ValueError as e:
            logger.warning(f"Train/Test split failed (likely too small data): {e}. Falling back to non-stratified.")
            return train_test_split(
                self.X, self.y,
                test_size=test_split,
                random_state=random_state,
                stratify=None
            )

    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple preprocessing fallback: numeric fill + label encoding for objects."""
        X_proc = X.copy()
        # numeric only mean fill
        X_proc = X_proc.fillna(X_proc.mean(numeric_only=True))
        # label-encode remaining object/category columns (deterministic fallback)
        for col in X_proc.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))
        return X_proc

    def _force_single_threading_env(self):
        """
        Set environment variables to limit native BLAS/OMP threading inside Optuna worker processes.
        Helps avoid CPU oversubscription when running parallel trials.
        """
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    @log_execution(logger_obj=logger)
    def train_all_models(self) -> Dict[str, float]:
        """
        Execute the automated training pipeline for all selected algorithms.

        Orchestrates the training process, including hyperparameter optimization via
        Optuna and final model evaluation. If multiple models are trained, it also
        fits a stacking ensemble.

        Returns
        -------
        scores : dict of str: float
            Dictionary mapping algorithm names to their optimized performance scores.

        Notes
        -----
        This method handles parallel execution and ensures that each model is
        evaluated on the same test set to ensure fair comparison.
        """
        logger.info(f"Starting training for task: {self.task_type}")

        # choose model list
        if self.task_type == 'classification':
            models = MODEL_TRAINING_CONFIG.get('classification_models', [])
        else:
            models = MODEL_TRAINING_CONFIG.get('regression_models', [])

        results = {
            'trained_models': {},
            'model_scores': {},
            'best_model': None,
            'best_score': None,
            'model_benchmarks': []
        }

        best_score: Optional[float] = None
        best_model_name: Optional[str] = None

        # Preprocess train/test once (used for final fitting & evaluation)
        X_train_proc = self._preprocess_data(self.X_train)
        X_test_proc = self._preprocess_data(self.X_test)

        for model_name in models:
            try:
                logger.info(f"Training model: {model_name}")

                # 1) Hyperparameter optimization with Optuna (returns best params or {})
                best_params = self._optimize_hyperparameters(model_name)

                # 2) Build final model (force single-thread for safety)
                model = self._build_model(model_name, best_params, force_single_thread=True)

                # 3) Fit on training data
                start_time = time.time()
                model.fit(X_train_proc, self.y_train)
                train_time = time.time() - start_time

                # 4) Evaluate using ModelEvaluator for consistent metrics output
                from ..evaluation.metrics import ModelEvaluator
                evaluator = ModelEvaluator(model, X_test_proc, self.y_test, self.task_type)
                eval_results = evaluator.evaluate()

                # Determine metric key
                metric = self.evaluation_metric or ('f1' if self.task_type == 'classification' else 'rmse')

                # Extract metric value (fallback to model.score if necessary)
                metric_value = eval_results.get('metrics', {}).get(metric, None)
                if metric_value is None:
                    try:
                        score = float(model.score(X_test_proc, self.y_test))
                    except Exception:
                        score = float(0.0)
                else:
                    score = float(metric_value)

                # Save artifacts
                self.trained_models[model_name] = model
                self.model_scores[model_name] = score
                self.best_hp_params[model_name] = best_params

                results['trained_models'][model_name] = str(model)
                results['model_scores'][model_name] = round(score, 4)
                results['model_benchmarks'].append({
                    'model': model_name,
                    'score': round(score, 4),
                    'params': best_params,
                    'metrics': eval_results.get('metrics', {}),
                    'training_time': round(train_time, 4)
                })

                logger.info(f"Model {model_name} result -> {score:.4f} ({metric})")

                # Compare to best (safe None handling)
                is_better = False
                if best_score is None:
                    is_better = True
                else:
                    if self.task_type == 'regression':
                        # For common error metrics lower is better
                        if metric in ['rmse', 'mse', 'mae', 'mape']:
                            if score < best_score:
                                is_better = True
                        else:
                            if score > best_score:
                                is_better = True
                    else:
                        # classification maximize
                        if score > best_score:
                            is_better = True

                if is_better:
                    best_score = score
                    best_model_name = model_name
                    self.best_model = model

                    # Capture predictions for report visual
                    self.best_model_predictions = eval_results.get('predictions')
                    self.best_model_probabilities = eval_results.get('probabilities')

            except Exception as e:
                logger.warning(f"Training failed for {model_name}: {e}")
                continue

        # 5) Stacking Ensemble (Champion Search)
        if len(self.trained_models) >= 2:
            try:
                # Set preliminary best results for stacking check
                results['best_model'] = best_model_name
                results['best_score'] = best_score
                
                self._train_stacking_ensemble(results, X_train_proc, X_test_proc)
                
                # Update locals after potential stacking update
                best_model_name = results['best_model']
                best_score = results['best_score']

            except Exception as e:
                logger.warning(f"Stacking Ensemble failed: {e}")

        # finalize results ordering
        chosen_metric = self.evaluation_metric or ('f1' if self.task_type == 'classification' else 'rmse')
        reverse_sort = True
        if self.task_type == 'regression' and chosen_metric in ['rmse', 'mse', 'mae', 'mape']:
            reverse_sort = False

        results['model_benchmarks'] = sorted(results['model_benchmarks'], key=lambda x: x['score'], reverse=reverse_sort)
        results['best_model'] = best_model_name
        results['best_score'] = best_score
        self.model_benchmarks = results['model_benchmarks']

        return results

    def _optimize_hyperparameters(self, model_name: str) -> Dict:
        """Optimize hyperparameters using Optuna with guarded single-threading in trials."""

        X_train_proc = self._preprocess_data(self.X_train)

        def objective(trial):
            try:
                # reduce native thread usage inside this trial to avoid oversubscription
                self._force_single_threading_env()

                # Check for overrides first
                hp_config = self.hyperparameter_overrides.get(model_name)
                if not hp_config:
                    if model_name not in OPTUNA_CONFIG.get('hyperparameters', {}):
                        return 0.0
                    hp_config = OPTUNA_CONFIG['hyperparameters'].get(model_name, {})
                
                params = self._create_trial_params(trial, model_name, hp_config)

                # Defensive: remove user-supplied thread keys to avoid collisions
                params_for_build = params.copy()
                params_for_build.pop('n_jobs', None)
                params_for_build.pop('nthread', None)

                # Force single-thread at model-level (defensive)
                params_for_build['n_jobs'] = 1
                params_for_build['nthread'] = 1

                # Build model with forced single thread behavior
                model = self._build_model(model_name, params_for_build, force_single_thread=True)

                scoring_metric = 'accuracy' if self.task_type == 'classification' else 'r2'
                cv_scores = cross_val_score(
                    model, X_train_proc, self.y_train,
                    cv=OPTUNA_CONFIG.get('cv_folds', 3),
                    scoring=scoring_metric,
                    n_jobs=1  # IMPORTANT: keep CV single-threaded inside each optuna worker
                )
                return float(np.mean(cv_scores))
            except Exception as e:
                logger.debug(f"Optuna trial for {model_name} failed: {e}")
                return 0.0

        try:
            sampler = TPESampler(seed=42)
            pruner = MedianPruner()
            study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')

            # Prefer instance-level settings (from user's OptimizationConfig), fall back to global config
            n_trials = self.n_trials if self.n_trials is not None else OPTUNA_CONFIG.get('optimization', {}).get('n_trials', 20)
            timeout = self.timeout_seconds if self.timeout_seconds is not None else OPTUNA_CONFIG.get('optimization', {}).get('timeout', 120)
            n_jobs_config = OPTUNA_CONFIG.get('optimization', {}).get('n_jobs', 1)
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs_config,
                show_progress_bar=False
            )

            return study.best_params or {}
        except Exception as e:
            logger.warning(f"Hyperopt failed for {model_name}: {e}")
            return {}

    def _create_trial_params(self, trial, model_name: str, hp_config: Dict) -> Dict:
        """Map hyperparameter config to Optuna trial suggestions."""
        params = {}
        for hp_name, hp_range in hp_config.items():
            try:
                if isinstance(hp_range, list) and len(hp_range) == 2:
                    if isinstance(hp_range[0], int):
                        params[hp_name] = trial.suggest_int(hp_name, hp_range[0], hp_range[1])
                    else:
                        params[hp_name] = trial.suggest_float(hp_name, hp_range[0], hp_range[1])
                elif isinstance(hp_range, list):
                    params[hp_name] = trial.suggest_categorical(hp_name, hp_range)
            except Exception as e:
                logger.debug(f"Failed to suggest {hp_name}: {e}")
        return params

    def _build_model(self, model_name: str, params: Dict = None, force_single_thread: bool = False):
        """
        Build a model with specified hyperparameters.

        This method removes potential threading keys from params (n_jobs/nthread)
        to avoid collisions with explicit keyword arguments, and builds model-specific kwargs.
        """
        params = params or {}
        params = params.copy()  # avoid mutating caller

        # Defensive removal to avoid "multiple values for argument" errors
        params.pop('n_jobs', None)
        params.pop('nthread', None)

        # Prepare model-specific kwargs (we will pass these explicitly)
        if force_single_thread:
            # enforce single-thread defaults
            base_thread_kwargs = {'n_jobs': 1, 'nthread': 1}
        else:
            base_thread_kwargs = {}

        # Classification
        if self.task_type == 'classification':
            if model_name == 'logistic_regression':
                kwargs = {'max_iter': 1000, 'random_state': 42}
                kwargs.update(params)
                return LogisticRegression(**kwargs)

            if model_name == 'random_forest':
                kwargs = {'n_jobs': base_thread_kwargs.get('n_jobs', 1), 'random_state': 42}
                kwargs.update(params)
                return RandomForestClassifier(**kwargs)

            if model_name == 'gradient_boosting':
                kwargs = {'random_state': 42}
                kwargs.update(params)
                return GradientBoostingClassifier(**kwargs)

            if model_name == 'xgboost':
                # xgboost accepts n_jobs; older versions accept nthread. pass explicit kwargs.
                kwargs = {'random_state': 42, 'verbosity': 0}
                if self.enable_gpu:
                    kwargs['tree_method'] = 'gpu_hist'
                    kwargs['gpu_id'] = 0  # Default 0
                
                if force_single_thread:
                    kwargs['n_jobs'] = 1
                    # some xgb versions accept nthread
                    kwargs['nthread'] = 1
                kwargs.update(params)
                
                # Create validation set if needed for early stopping
                # Note: XGBClassifier.fit handles early_stopping_rounds via eval_set
                # but sklearn wrapping can be tricky. We pass it here just in case.
                if self.early_stopping_rounds:
                    kwargs['early_stopping_rounds'] = self.early_stopping_rounds
                    
                return xgb.XGBClassifier(**kwargs)

            if model_name == 'lightgbm':
                kwargs = {'random_state': 42, 'verbose': -1}
                if self.enable_gpu:
                    kwargs['device'] = 'gpu'
                if force_single_thread:
                    kwargs['n_jobs'] = 1
                kwargs.update(params)
                return lgb.LGBMClassifier(**kwargs)

            if model_name == 'svm':
                kwargs = {'kernel': 'rbf', 'probability': True, 'random_state': 42}
                kwargs.update(params)
                return SVC(**kwargs)

        # Regression
        else:
            if model_name == 'linear_regression':
                kwargs = {}
                kwargs.update(params)
                return LinearRegression(**kwargs)

            if model_name == 'random_forest':
                kwargs = {'n_jobs': base_thread_kwargs.get('n_jobs', 1), 'random_state': 42}
                kwargs.update(params)
                return RandomForestRegressor(**kwargs)

            if model_name == 'gradient_boosting':
                kwargs = {'random_state': 42}
                kwargs.update(params)
                return GradientBoostingRegressor(**kwargs)

            if model_name == 'xgboost':
                kwargs = {'random_state': 42, 'verbosity': 0}
                if self.enable_gpu:
                    kwargs['tree_method'] = 'gpu_hist'
                    kwargs['gpu_id'] = 0
                if force_single_thread:
                    kwargs['n_jobs'] = 1
                    kwargs['nthread'] = 1
                kwargs.update(params)
                if self.early_stopping_rounds:
                    kwargs['early_stopping_rounds'] = self.early_stopping_rounds
                return xgb.XGBRegressor(**kwargs)

            if model_name == 'lightgbm':
                kwargs = {'random_state': 42, 'verbose': -1}
                if force_single_thread:
                    kwargs['n_jobs'] = 1
                kwargs.update(params)
                return lgb.LGBMRegressor(**kwargs)

            if model_name == 'svr':
                kwargs = {}
                kwargs.update(params)
                return SVR(**kwargs)

        raise ValueError(f"Unknown model: {model_name}")

    def _train_stacking_ensemble(self, results: Dict, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """Train a stacking ensemble using the top 3 trained models."""
        logger.info("Training Stacking Ensemble (Champion Search)...")
        
        # Get top 3 models
        benchmarks = results['model_benchmarks']
        top_n = min(3, len(benchmarks))
        top_models_info = benchmarks[:top_n]
        
        base_estimators = []
        for info in top_models_info:
            name = info['model']
            model = self.trained_models[name]
            base_estimators.append((name, model))
            
        if self.task_type == 'classification':
            stack = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(),
                cv=3,
                n_jobs=1
            )
        else:
            stack = StackingRegressor(
                estimators=base_estimators,
                final_estimator=Ridge(),
                cv=3,
                n_jobs=1
            )
            
        start_time = time.time()
        stack.fit(X_train, self.y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        from ..evaluation.metrics import ModelEvaluator
        evaluator = ModelEvaluator(stack, X_test, self.y_test, self.task_type)
        eval_results = evaluator.evaluate()
        
        metric = self.evaluation_metric or ('f1' if self.task_type == 'classification' else 'rmse')
        score = float(eval_results.get('metrics', {}).get(metric, 0.0))
        
        model_name = 'stacking_ensemble'
        self.trained_models[model_name] = stack
        self.model_scores[model_name] = score
        
        results['trained_models'][model_name] = "StackingEnsemble"
        results['model_scores'][model_name] = round(score, 4)
        results['model_benchmarks'].append({
            'model': model_name,
            'score': round(score, 4),
            'params': {'base_models': [n for n, _ in base_estimators]},
            'metrics': eval_results.get('metrics', {}),
            'training_time': round(train_time, 4)
        })
        
        logger.info(f"Stacking Ensemble result -> {score:.4f} ({metric})")
        
        # Update best model if stacking is better
        is_better = False
        best_score = results.get('best_score')
        if best_score is None:
            is_better = True
        else:
            if self.task_type == 'regression' and metric in ['rmse', 'mse', 'mae', 'mape']:
                if score < best_score: is_better = True
            else:
                if score > best_score: is_better = True
                
        if is_better:
            results['best_score'] = score
            results['best_model'] = model_name
            self.best_model = stack
            self.best_model_predictions = eval_results.get('predictions')
            self.best_model_probabilities = eval_results.get('probabilities')

    def get_best_model(self) -> Optional[object]:
        """
        Retrieve the top-performing model object.

        Returns
        -------
        model : object, optional
            The best fitted estimator found during training, or None if no
            models have been trained yet.
        """
        return self.best_model

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Generate a comparative leaderboard of all trained models.

        Returns
        -------
        comparison : pd.DataFrame
            A DataFrame containing scores, training times, and rank for each
            algorithm, sorted by performance.
        """
        if self.model_benchmarks is None:
            return pd.DataFrame()
        return self.model_benchmarks
