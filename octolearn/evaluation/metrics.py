"""
Advanced Model Evaluation Module

Comprehensive evaluation with multiple metrics and visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
import warnings

warnings.filterwarnings('ignore')

from ..config import EVALUATION_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    """

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, task_type: str):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.task_type = task_type
        self.evaluation_results = {}

    @log_execution(logger_obj=logger)
    def evaluate(self) -> Dict:
        logger.info(f"Evaluating {self.task_type} model...")

        if self.task_type == 'classification':
            return self._evaluate_classification()
        else:
            return self._evaluate_regression()

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------
    def _evaluate_classification(self) -> Dict:

        y_pred = self.model.predict(self.X_test)

        results = {
            'task': 'classification',
            'metrics': {},
            'confusion_matrix': None,
            'classification_report': None,
            'predictions': y_pred.tolist(),
            'probabilities': None
        }

        metrics_dict = EVALUATION_CONFIG['classification_metrics']

        # Accuracy
        if 'accuracy' in metrics_dict:
            acc = accuracy_score(self.y_test, y_pred)
            results['metrics']['accuracy'] = round(acc, 4)

        # Precision
        if 'precision' in metrics_dict:
            try:
                prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                results['metrics']['precision'] = round(prec, 4)
            except:
                pass

        # Recall
        if 'recall' in metrics_dict:
            try:
                rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                results['metrics']['recall'] = round(rec, 4)
            except:
                pass

        # F1
        if 'f1' in metrics_dict:
            try:
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                results['metrics']['f1'] = round(f1, 4)
            except:
                pass

        # ✅ UPDATED ROC-AUC (Binary + Multiclass)
        if 'roc_auc' in metrics_dict:
            try:
                if hasattr(self.model, "predict_proba"):
                    n_classes = len(np.unique(self.y_test))

                    if n_classes == 2:
                        # Binary
                        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
                        auc = roc_auc_score(self.y_test, y_pred_proba)

                    else:
                        # Multiclass
                        y_pred_proba = self.model.predict_proba(self.X_test)
                        auc = roc_auc_score(
                            self.y_test,
                            y_pred_proba,
                            multi_class='ovr',
                            average='weighted'
                        )

                    results['metrics']['roc_auc'] = round(auc, 4)
                    results['probabilities'] = y_pred_proba.tolist()
            except:
                pass

        # Confusion Matrix
        if 'confusion_matrix' in metrics_dict:
            cm = confusion_matrix(self.y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()

        # Classification Report
        if 'classification_report' in metrics_dict:
            cr = classification_report(
                self.y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )
            results['classification_report'] = cr

        return results

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------
    def _evaluate_regression(self) -> Dict:

        y_pred = self.model.predict(self.X_test)

        results = {
            'task': 'regression',
            'metrics': {},
            'predictions': y_pred.tolist()
        }

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

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------
    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = None) -> Dict:

        cv_folds = cv or EVALUATION_CONFIG['cv_folds']

        if self.task_type == 'classification':
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        try:
            cv_results = cross_validate(
                self.model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring=scoring,
                return_train_score=True
            )

            results = {
                'cv_folds': cv_folds,
                'fold_results': [],
                'summary': {}
            }

            for fold in range(cv_folds):
                fold_data = {}
                for score_name in scoring:
                    test_key = f'test_{score_name}'
                    if test_key in cv_results:
                        fold_data[score_name] = cv_results[test_key][fold]
                results['fold_results'].append(fold_data)

            for score_name in scoring:
                test_key = f'test_{score_name}'
                if test_key in cv_results:
                    scores = cv_results[test_key]
                    results['summary'][f'{score_name}_mean'] = round(scores.mean(), 4)
                    results['summary'][f'{score_name}_std'] = round(scores.std(), 4)

            return results

        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return {'error': str(e)}

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------
    def feature_importance(self) -> Dict:

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
                    'importances': {k: round(v, 4) for k, v in sorted_importance.items()}
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
                    'importances': {k: round(v, 4) for k, v in sorted_importance.items()}
                }

            else:
                return {'error': 'Model does not support feature importance'}

        except Exception as e:
            logger.error(f"Feature importance extraction failed: {str(e)}")
            return {'error': str(e)}

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------
    def get_evaluation_report(self) -> str:

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
