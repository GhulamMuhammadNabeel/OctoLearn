"""
Advanced Model Evaluation Module

Comprehensive evaluation with multiple metrics and visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
)
import warnings

warnings.filterwarnings('ignore')

from ..config import EVALUATION_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics.
    """
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, task_type: str):
        """
        Initialize ModelEvaluator.
        
        Parameters
        ----------
        model : sklearn model
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
        task_type : str
            'classification' or 'regression'
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type
        self.evaluation_results = {}
    
    @log_execution(logger_obj=logger)
    def evaluate(self) -> Dict:
        """
        Perform comprehensive evaluation.
        
        Returns
        -------
        dict
            Evaluation results
        """
        logger.info(f"Evaluating {self.task_type} model...")
        
        if self.task_type == 'classification':
            return self._evaluate_classification()
        else:
            return self._evaluate_regression()
    
    def _evaluate_classification(self) -> Dict:
        """Evaluate classification model."""
        y_pred = self.model.predict(self.X_test)
        results = {
            'task': 'classification',
            'metrics': {},
            'confusion_matrix': None,
            'classification_report': None
        }
        
        # Basic metrics
        metrics_dict = EVALUATION_CONFIG['classification_metrics']
        
        if 'accuracy' in metrics_dict:
            acc = accuracy_score(self.y_test, y_pred)
            results['metrics']['accuracy'] = round(acc, 4)
        
        if 'precision' in metrics_dict:
            try:
                prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                results['metrics']['precision'] = round(prec, 4)
            except:
                pass
        
        if 'recall' in metrics_dict:
            try:
                rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                results['metrics']['recall'] = round(rec, 4)
            except:
                pass
        
        if 'f1' in metrics_dict:
            try:
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                results['metrics']['f1'] = round(f1, 4)
            except:
                pass
        
        if 'roc_auc' in metrics_dict:
            try:
                if len(np.unique(self.y_test)) == 2:
                    y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
                    auc = roc_auc_score(self.y_test, y_pred_proba)
                    results['metrics']['roc_auc'] = round(auc, 4)
            except:
                pass
        
        if 'confusion_matrix' in metrics_dict:
            cm = confusion_matrix(self.y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
        
        if 'classification_report' in metrics_dict:
            cr = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
            results['classification_report'] = cr
        
        return results
    
    def _evaluate_regression(self) -> Dict:
        """Evaluate regression model."""
        y_pred = self.model.predict(self.X_test)
        results = {
            'task': 'regression',
            'metrics': {}
        }
        
        # Basic metrics
        metrics_dict = EVALUATION_CONFIG['regression_metrics']
        
        if 'mse' in metrics_dict:
            mse = mean_squared_error(self.y_test, y_pred)
            results['metrics']['mse'] = round(mse, 4)
        
        if 'rmse' in metrics_dict:
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            results['metrics']['rmse'] = round(rmse, 4)
        
        if 'mae' in metrics_dict:
            mae = mean_absolute_error(self.y_test, y_pred)
            results['metrics']['mae'] = round(mae, 4)
        
        if 'r2' in metrics_dict:
            r2 = r2_score(self.y_test, y_pred)
            results['metrics']['r2'] = round(r2, 4)
        
        if 'mape' in metrics_dict:
            try:
                mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
                results['metrics']['mape'] = round(mape, 4)
            except:
                pass
        
        return results
    
    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = None) -> Dict:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        cv : int, optional
            CV folds
            
        Returns
        -------
        dict
            Cross-validation results
        """
        cv_folds = cv or EVALUATION_CONFIG['cv_folds']
        
        if self.task_type == 'classification':
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        try:
            cv_results = cross_validate(
                self.model, X_train, y_train,
                cv=cv_folds,
                scoring=scoring,
                return_train_score=True
            )
            
            # Process results
            results = {
                'cv_folds': cv_folds,
                'fold_results': [],
                'summary': {}
            }
            
            # Get fold scores
            for fold in range(cv_folds):
                fold_data = {}
                for score_name in scoring:
                    test_key = f'test_{score_name}'
                    if test_key in cv_results:
                        fold_data[score_name] = cv_results[test_key][fold]
                results['fold_results'].append(fold_data)
            
            # Summary statistics
            for score_name in scoring:
                test_key = f'test_{score_name}'
                train_key = f'train_{score_name}'
                
                if test_key in cv_results:
                    test_scores = cv_results[test_key]
                    results['summary'][f'{score_name}_mean'] = round(test_scores.mean(), 4)
                    results['summary'][f'{score_name}_std'] = round(test_scores.std(), 4)
            
            return results
        
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return {'error': str(e)}
    
    def feature_importance(self) -> Dict:
        """
        Extract feature importance from model.
        
        Returns
        -------
        dict
            Feature importance scores
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.X_test.columns.tolist()
                
                importance_dict = dict(zip(feature_names, importances))
                sorted_importance = dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
                
                return {
                    'method': 'built-in',
                    'importances': {
                        k: round(v, 4) for k, v in sorted_importance.items()
                    }
                }
            
            elif hasattr(self.model, 'coef_'):
                coef = np.abs(self.model.coef_).flatten()
                feature_names = self.X_test.columns.tolist()
                
                importance_dict = dict(zip(feature_names, coef))
                sorted_importance = dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
                
                return {
                    'method': 'coefficients',
                    'importances': {
                        k: round(v, 4) for k, v in sorted_importance.items()
                    }
                }
            
            else:
                return {'error': 'Model does not support feature importance'}
        
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {str(e)}")
            return {'error': str(e)}
    
    def get_evaluation_report(self) -> str:
        """
        Get formatted evaluation report.
        
        Returns
        -------
        str
            Formatted report
        """
        report = [
            "=" * 60,
            "MODEL EVALUATION REPORT",
            "=" * 60,
            f"\nTask Type: {self.task_type}",
            f"Test Set Size: {len(self.X_test)}",
            "\nMETRICS:",
            "-" * 60,
        ]
        
        evaluation = self.evaluate()
        
        if 'metrics' in evaluation:
            for metric, value in evaluation['metrics'].items():
                report.append(f"{metric.upper()}: {value}")
        
        if 'confusion_matrix' in evaluation and evaluation['confusion_matrix']:
            report.append("\nCONFUSION MATRIX:")
            cm = np.array(evaluation['confusion_matrix'])
            report.append(str(cm))
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
