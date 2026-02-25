"""
AutoSampler Module for Handling Imbalanced Datasets

This module provides automated sampling strategies for classification tasks.
It supports oversampling (SMOTE, ADASYN), undersampling (RandomUnderSampler),
and combined approaches to balance class distributions before model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Union
import warnings

from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)

class AutoSampler:
    """
    Automated resampling engine for imbalanced classification datasets.
    
    Parameters
    ----------
    strategy : str, default='auto'
        The sampling strategy to apply. Options:
        - 'auto': Automatically decides based on imbalance ratio and dataset size.
        - 'smote': Synthetic Minority Over-sampling Technique.
        - 'adasyn': Adaptive Synthetic Sampling.
        - 'undersample': Randomly removes majority class samples.
        - 'combine': SMOTE combined with Tomek links (cleans boundary).
        - 'none': Skips sampling entirely.
    random_state : int, default=42
        Seed for reproducibility.
    """
    
    def __init__(
        self,
        strategy: str = 'auto',
        random_state: int = 42
    ):
        self.strategy = strategy.lower()
        self.random_state = random_state
        self.sampler_ = None
        self.is_fitted_ = False
        self._check_dependencies()
        
    def _check_dependencies(self) -> None:
        """Verify imbalanced-learn is installed."""
        if self.strategy == 'none':
            return
            
        try:
            import imblearn
        except ImportError:
            logger.warning(
                "imbalanced-learn is not installed. Sampling features will be disabled. "
                "Install via: pip install imbalanced-learn"
            )
            self.strategy = 'none'

    def _determine_auto_strategy(self, y: pd.Series, X_shape: Tuple[int, int]) -> str:
        """
        Determine the optimal sampling strategy based on data characteristics.
        
        Logic:
        - Strict check: if no minority class has >= 6 samples, fallback to undersample or none.
        - If dataset is huge (>100k rows), prefer undersampling for speed.
        - If dataset is small/medium and moderately imbalanced, use SMOTE.
        - If highly imbalanced and noisy boundaries are suspected, use combine.
        """
        n_samples, _ = X_shape
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        
        imbalance_ratio = max_class_count / max(min_class_count, 1)
        
        if imbalance_ratio < 1.5:
            # Not imbalanced enough to warrant sampling
            return 'none'
            
        if min_class_count < 6:
            # SMOTE requires at least k_neighbors=5 by default, meaning 6 samples min
            logger.info(f"Minority class has only {min_class_count} samples. Fallback to random undersampling.")
            return 'undersample'
            
        if n_samples > 100_000:
            # SMOTE can be very slow and memory intensive on huge datasets
            return 'undersample'
            
        if imbalance_ratio > 10:
            # Severe imbalance, combined approach is often best
            return 'combine'
            
        # Default good choice for moderate imbalance
        return 'smote'

    def _initialize_sampler(self, strategy: str):
        """Instantiate the correct imblearn object."""
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTETomek
        
        if strategy == 'smote':
            return SMOTE(random_state=self.random_state)
        elif strategy == 'adasyn':
            return ADASYN(random_state=self.random_state)
        elif strategy == 'undersample':
            return RandomUnderSampler(random_state=self.random_state)
        elif strategy == 'combine':
            return SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    @log_execution(logger_obj=logger)
    def fit_resample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        task_type: str = 'classification'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply the selected sampling strategy to the dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must be numeric/encoded prior to this step).
        y : pd.Series
            Target vector.
        task_type : str
            Only valid for 'classification'. Does nothing for 'regression'.
            
        Returns
        -------
        X_res, y_res : Tuple[pd.DataFrame, pd.Series]
            The re-sampled dataset.
        """
        # We only resample classification tasks
        if task_type != 'classification' or self.strategy == 'none':
            return X, y
            
        # Handle 'auto' strategy
        actual_strategy = self.strategy
        if actual_strategy == 'auto':
            actual_strategy = self._determine_auto_strategy(y, X.shape)
            logger.info(f"Auto-selected sampling strategy: {actual_strategy}")
            
        if actual_strategy == 'none':
            logger.info("Skipping sampling (imbalance ratio is low or unsupported).")
            return X, y
            
        try:
            self.sampler_ = self._initialize_sampler(actual_strategy)
            
            # Record original distribution
            orig_counts = y.value_counts().to_dict()
            logger.info(f"Original class distribution: {orig_counts}")
            
            # Apply sampling
            X_res, y_res = self.sampler_.fit_resample(X, y)
            
            # Ensure types remain pandas
            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X.columns)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y.name)
                
            # Record new distribution
            new_counts = y_res.value_counts().to_dict()
            logger.info(f"Resampled class distribution: {new_counts}")
            
            self.is_fitted_ = True
            return X_res, y_res
            
        except Exception as e:
            logger.warning(f"Sampling with strategy '{actual_strategy}' failed: {e}. Returning original data.")
            return X, y
