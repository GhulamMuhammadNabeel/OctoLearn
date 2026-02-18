"""
Feature Generation Module

Intelligent feature engineering:
- Date parts extraction
- Log transformation for skewed features
- Interaction features
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional
import warnings

from ..config import FEATURE_GENERATION_CONFIG, INTERACTION_CONFIG
from ..utils.helpers import setup_logger, log_execution
from .interaction_analyzer import FeatureInteractionAnalyzer

logger = setup_logger(__name__)

class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Intelligently generates new features from existing data.
    """
    
    def __init__(self, profile=None):
        self.profile = profile
        self.config = FEATURE_GENERATION_CONFIG
        self.skewed_feats_ = []
        self.date_cols_ = []
        self.interaction_names_ = []
        self.analyzer_ = None
        
    @log_execution(logger_obj=logger)
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Learn feature engineering parameters.
        """
        if not self.config['enabled']:
            return self
            
        # 1. Identify Date Columns (if not already handled by profiler)
        # Simple heuristic: object/string columns that look like dates
        # Or check profile if available
        self._find_date_columns(X)
        
        # 2. Identify Skewed Features
        if self.config['skewed_features']['enabled']:
            self._find_skewed_features(X)
            
        # 3. Identify Interactions (using existing Analyzer)
        if self.config['interaction_features']['enabled'] and y is not None:
             self._find_interactions(X, y)
             
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        """
        if not self.config['enabled']:
            return X
            
        X_new = X.copy()
        
        # 1. Date Features
        if self.date_cols_:
            X_new = self._process_date_features(X_new)
            
        # 2. Skewed Features (Log Transform)
        if self.skewed_feats_:
            X_new = self._process_skewed_features(X_new)
            
        # 3. Interaction Features
        if self.interaction_names_ and self.analyzer_:
             # Re-use analyzer logic to create features
             try:
                 interactions_df = self.analyzer_.create_interaction_features(self.interaction_names_)
                 # Join with X_new (align index)
                 interactions_df.index = X_new.index
                 X_new = pd.concat([X_new, interactions_df], axis=1)
             except Exception as e:
                 logger.warning(f"Failed to generate interaction features: {e}")
                 
        return X_new

    def _find_date_columns(self, X: pd.DataFrame):
        """Identify potential date columns"""
        # Checks object/string cols and tries pd.to_datetime
        for col in X.select_dtypes(include=['object', 'datetime']).columns:
            try:
                # If already datetime, adds to list
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    self.date_cols_.append(col)
                    continue
                    
                # If object, try converting sample
                sample = X[col].dropna().sample(min(100, len(X[col])), random_state=42)
                pd.to_datetime(sample, errors='raise') # Will fail if not date
                self.date_cols_.append(col)
            except Exception:
                pass

    def _find_skewed_features(self, X: pd.DataFrame):
        """Identify numeric features with high skew"""
        threshold = self.config['skewed_features']['threshold']
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Exclude binary
        numeric_cols = [c for c in numeric_cols if X[c].nunique() > 2]
        
        if not numeric_cols:
            return

        skew_vals = X[numeric_cols].skew().abs()
        self.skewed_feats_ = skew_vals[skew_vals > threshold].index.tolist()
        if self.skewed_feats_:
            logger.info(f"Identified {len(self.skewed_feats_)} skewed features for log transform")

    def _find_interactions(self, X: pd.DataFrame, y: pd.Series):
        """Find top interactions using Analyzer"""
        try:
            self.analyzer_ = FeatureInteractionAnalyzer(X, y, self.profile)
            results = self.analyzer_.analyze()
            
            # extract top N interactions from all types
            top_n = self.config['interaction_features']['top_n']
            candidates = {}
            
            # Collect from Pairwise
            if 'pairwise_interactions' in results and 'top_interactions' in results['pairwise_interactions']:
                candidates.update(results['pairwise_interactions']['top_interactions'])
                
            # Collect from Ratio
            if 'ratio_interactions' in results and 'top_ratios' in results['ratio_interactions']:
                 candidates.update(results['ratio_interactions']['top_ratios'])

            # Sort by score and take top N
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
            self.interaction_names_ = [name for name, score in sorted_candidates]
            
            if self.interaction_names_:
                logger.info(f"Selected top {len(self.interaction_names_)} interactions: {self.interaction_names_}")
                
        except Exception as e:
            logger.warning(f"Interaction discovery failed: {e}")

    def _process_date_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract date parts"""
        parts = self.config['date_features']['parts']
        drop = self.config['date_features']['drop_original']
        
        for col in self.date_cols_:
            try:
                # Convert to datetime if not
                series = pd.to_datetime(X[col], errors='coerce')
                
                if 'year' in parts:
                    X[f"{col}_year"] = series.dt.year
                if 'month' in parts:
                    X[f"{col}_month"] = series.dt.month
                if 'day' in parts:
                    X[f"{col}_day"] = series.dt.day
                if 'weekday' in parts:
                    X[f"{col}_weekday"] = series.dt.weekday
                if 'is_weekend' in parts:
                    X[f"{col}_is_weekend"] = (series.dt.weekday >= 5).astype(int)
                    
                if drop:
                    X.drop(col, axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"Date extraction failed for {col}: {e}")
                
        return X

    def _process_skewed_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform"""
        drop = self.config['skewed_features']['drop_original']
        
        for col in self.skewed_feats_:
            try:
                # Basic validation: must be positive for log
                min_val = X[col].min()
                if min_val <= 0:
                     # Shift to positive
                     offset = abs(min_val) + 1
                     X[f"log_{col}"] = np.log(X[col] + offset)
                else:
                     X[f"log_{col}"] = np.log1p(X[col])
                     
                if drop:
                    X.drop(col, axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"Log transform failed for {col}: {e}")
        return X
