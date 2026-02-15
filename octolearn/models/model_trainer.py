"""
Model Training Module with Optuna Hyperparameter Optimization

Trains multiple models with automated hyperparameter tuning
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ..config import MODEL_TRAINING_CONFIG, OPTUNA_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class ModelTrainer:
    """
    Trains multiple models with hyperparameter optimization using Optuna.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, profile, task_type: str = None):
        """
        Initialize ModelTrainer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        profile : DatasetProfile
            Dataset profile object
        task_type : str, optional
            'classification' or 'regression'. Auto-detected if None
        """
        self.X = X
        self.y = y
        self.profile = profile
        self.task_type = task_type or profile.task_type
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_hp_params = {}
        
        # Prepare data
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
    
    def _prepare_data(self) -> Tuple:
        """Prepare train-test split."""
        test_split = MODEL_TRAINING_CONFIG['test_split']
        random_state = MODEL_TRAINING_CONFIG['random_state']
        
        return train_test_split(
            self.X, self.y,
            test_size=test_split,
            random_state=random_state,
            stratify=self.y if self.task_type == 'classification' else None
        )
    
    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data: handle categorical, missing values."""
        X_proc = X.copy()
        
        # Handle missing values
        X_proc = X_proc.fillna(X_proc.mean(numeric_only=True))
        
        # Encode categorical features
        for col in X_proc.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))
        
        return X_proc
    
    @log_execution(logger_obj=logger)
    def train_all_models(self) -> Dict:
        """
        Train all models with hyperparameter optimization.
        
        Returns
        -------
        dict
            Training results
        """
        logger.info(f"Starting training for {self.task_type} task...")
        
        # Get model list
        if self.task_type == 'classification':
            models = MODEL_TRAINING_CONFIG['classification_models']
        else:
            models = MODEL_TRAINING_CONFIG['regression_models']
        
        results = {
            'trained_models': {},
            'model_scores': {},
            'best_model': None,
            'best_score': 0
        }
        
        for model_name in models:
            try:
                logger.info(f"Training {model_name}...")
                
                # Optimize hyperparameters
                best_params = self._optimize_hyperparameters(model_name)
                
                # Train model with best params
                model = self._build_model(model_name, best_params)
                X_train_proc = self._preprocess_data(self.X_train)
                X_test_proc = self._preprocess_data(self.X_test)
                
                model.fit(X_train_proc, self.y_train)
                
                # Evaluate
                score = model.score(X_test_proc, self.y_test)
                
                self.trained_models[model_name] = model
                self.model_scores[model_name] = score
                self.best_hp_params[model_name] = best_params
                
                results['trained_models'][model_name] = str(model)
                results['model_scores'][model_name] = round(score, 4)
                
                logger.info(f"{model_name}: {score:.4f}")
                
                # Track best model
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_model'] = model_name
                    self.best_model = model
            
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return results
    
    def _optimize_hyperparameters(self, model_name: str) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Parameters
        ----------
        model_name : str
            Model name
            
        Returns
        -------
        dict
            Best hyperparameters
        """
        X_train_proc = self._preprocess_data(self.X_train)
        
        def objective(trial):
            try:
                # Get hyperparameters for this model
                if model_name not in OPTUNA_CONFIG['hyperparameters']:
                    return 0.0
                
                hp_config = OPTUNA_CONFIG['hyperparameters'].get(model_name, {})
                params = self._create_trial_params(trial, model_name, hp_config)
                
                # Build and train model
                model = self._build_model(model_name, params)
                model.fit(X_train_proc, self.y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train_proc, self.y_train,
                    cv=OPTUNA_CONFIG['cv_folds'],
                    scoring='accuracy' if self.task_type == 'classification' else 'r2',
                    n_jobs=1
                )
                
                return cv_scores.mean()
            
            except Exception as e:
                logger.debug(f"Trial failed: {str(e)}")
                return 0.0
        
        try:
            sampler = TPESampler(seed=42)
            pruner = MedianPruner()
            
            study = optuna.create_study(
                sampler=sampler,
                pruner=pruner,
                direction='maximize'
            )
            
            study.optimize(
                objective,
                n_trials=OPTUNA_CONFIG['optimization']['n_trials'],
                show_progress_bar=False
            )
            
            return study.best_params
        
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed for {model_name}: {str(e)}")
            return {}
    
    def _create_trial_params(self, trial, model_name: str, hp_config: Dict) -> Dict:
        """
        Create trial parameters for Optuna.
        
        Parameters
        ----------
        trial : optuna.trial.Trial
            Optuna trial
        model_name : str
            Model name
        hp_config : dict
            Hyperparameter configuration
            
        Returns
        -------
        dict
            Trial parameters
        """
        params = {}
        
        for hp_name, hp_range in hp_config.items():
            if isinstance(hp_range, list) and len(hp_range) == 2:
                # Numeric range
                if isinstance(hp_range[0], int):
                    params[hp_name] = trial.suggest_int(hp_name, hp_range[0], hp_range[1])
                else:
                    params[hp_name] = trial.suggest_float(hp_name, hp_range[0], hp_range[1])
            elif isinstance(hp_range, list):
                # Categorical choice
                params[hp_name] = trial.suggest_categorical(hp_name, hp_range)
        
        return params
    
    def _build_model(self, model_name: str, params: Dict = None):
        """
        Build a model with specified hyperparameters.
        
        Parameters
        ----------
        model_name : str
            Model name
        params : dict, optional
            Hyperparameters
            
        Returns
        -------
        model
            Trained model
        """
        params = params or {}
        
        if self.task_type == 'classification':
            if model_name == 'logistic_regression':
                return LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    **params
                )
            elif model_name == 'random_forest':
                return RandomForestClassifier(
                    n_jobs=MODEL_TRAINING_CONFIG['n_jobs'],
                    random_state=42,
                    **params
                )
            elif model_name == 'gradient_boosting':
                return GradientBoostingClassifier(
                    random_state=42,
                    **params
                )
            elif model_name == 'xgboost':
                return xgb.XGBClassifier(
                    random_state=42,
                    verbosity=0,
                    **params
                )
            elif model_name == 'lightgbm':
                return lgb.LGBMClassifier(
                    random_state=42,
                    verbose=-1,
                    **params
                )
            elif model_name == 'svm':
                return SVC(
                    kernel='rbf',
                    random_state=42,
                    **params
                )
        else:
            if model_name == 'linear_regression':
                return LinearRegression(**params)
            elif model_name == 'random_forest':
                return RandomForestRegressor(
                    n_jobs=MODEL_TRAINING_CONFIG['n_jobs'],
                    random_state=42,
                    **params
                )
            elif model_name == 'gradient_boosting':
                return GradientBoostingRegressor(
                    random_state=42,
                    **params
                )
            elif model_name == 'xgboost':
                return xgb.XGBRegressor(
                    random_state=42,
                    verbosity=0,
                    **params
                )
            elif model_name == 'lightgbm':
                return lgb.LGBMRegressor(
                    random_state=42,
                    verbose=-1,
                    **params
                )
            elif model_name == 'svr':
                return SVR(**params)
        
        raise ValueError(f"Unknown model: {model_name}")
    
    def get_best_model(self):
        """Get the best trained model."""
        return self.best_model
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all trained models.
        
        Returns
        -------
        pd.DataFrame
            Model comparison table
        """
        comparison_data = {
            'Model': list(self.model_scores.keys()),
            'Score': list(self.model_scores.values()),
        }
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Score', ascending=False)
        
        return df
