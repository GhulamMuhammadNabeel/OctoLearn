"""
Outlier Detection Module for Octolearn

Detects outliers using multiple methods: IQR, Isolation Forest, Z-score
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

from ..config import OUTLIER_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class OutlierDetector:
    """
    Detects outliers using multiple methods on numeric features.
    """
    
    def __init__(self, X: pd.DataFrame, profile):
        """
        Initialize OutlierDetector.
        """
        self.X = X
        self.profile = profile
        # FIX: Attribute name updated to match DataProfiler
        self.numeric_columns = profile.numeric_columns
        self.outliers = {}
        self.outlier_counts = {}
        self.outlier_percentages = {}
    
    @log_execution(logger_obj=logger)
    def detect(self) -> Dict:
        """
        Detect outliers using multiple methods.
        """
        if not self.numeric_columns:
            logger.info("No numeric features for outlier detection")
            return {'error': 'No numeric features found'}
        
        results = {
            'methods': {},
            'summary': {},
            'affected_features': {}
        }
        
        # IQR method
        if 'iqr' in OUTLIER_CONFIG['methods']:
            try:
                iqr_results = self._detect_iqr()
                results['methods']['iqr'] = iqr_results
            except Exception as e:
                logger.warning(f"IQR outlier detection failed: {e}")
        
        # Isolation Forest method
        if 'isolation_forest' in OUTLIER_CONFIG['methods']:
            try:
                if_results = self._detect_isolation_forest()
                results['methods']['isolation_forest'] = if_results
            except Exception as e:
                logger.warning(f"Isolation Forest detection failed: {e}")
        
        # Z-score method
        if 'zscore' in OUTLIER_CONFIG['methods']:
            try:
                zscore_results = self._detect_zscore()
                results['methods']['zscore'] = zscore_results
            except Exception as e:
                logger.warning(f"Z-Score detection failed: {e}")
        
        # Summary statistics
        results['summary'] = self._summarize_outliers()
        results['affected_features'] = self._get_affected_features()
        
        return results
    
    def _detect_iqr(self) -> Dict:
        """Detect outliers using Interquartile Range (IQR) method."""
        results = {}
        multiplier = OUTLIER_CONFIG['iqr']['multiplier']
        
        for col in self.numeric_columns:
            try:
                series = self.X[col].dropna()
                if series.empty: continue

                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - (multiplier * IQR)
                upper_bound = Q3 + (multiplier * IQR)
                
                outliers_mask = (self.X[col] < lower_bound) | (self.X[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                outlier_pct = (outlier_count / len(self.X)) * 100
                
                results[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2),
                    'bounds': {'lower': round(lower_bound, 2), 'upper': round(upper_bound, 2)},
                    'indices': self.X[outliers_mask].index.tolist()[:50]
                }
                
                if outlier_count > 0:
                    self.outliers[col] = outliers_mask
                    self.outlier_counts[col] = outlier_count
                    self.outlier_percentages[col] = outlier_pct
            
            except Exception as e:
                continue
        
        return results
    
    def _detect_isolation_forest(self) -> Dict:
        """Detect outliers using Isolation Forest."""
        results = {}
        
        if len(self.X) > 100_000:
            sample_idx = self.X.sample(100_000, random_state=42).index
            X_sample = self.X.loc[sample_idx]
        else:
            X_sample = self.X
        
        # Prepare numeric data
        X_numeric = X_sample[self.numeric_columns].fillna(X_sample[self.numeric_columns].mean())
        
        iso_forest = IsolationForest(
            contamination=OUTLIER_CONFIG['isolation_forest']['contamination'],
            n_estimators=OUTLIER_CONFIG['isolation_forest']['n_estimators'],
            random_state=OUTLIER_CONFIG['isolation_forest']['random_state']
        )
        
        predictions = iso_forest.fit_predict(X_numeric)
        outliers_mask = predictions == -1
        
        outlier_count = outliers_mask.sum()
        outlier_pct = (outlier_count / len(self.X)) * 100
        
        results['overall'] = {
            'count': int(outlier_count),
            'percentage': round(outlier_pct, 2),
            'indices': self.X[outliers_mask].index.tolist()[:50] if len(self.X) <= 100000 else []
        }
        
        return results
    
    def _detect_zscore(self) -> Dict:
        """Detect outliers using Z-score method."""
        results = {}
        threshold = OUTLIER_CONFIG['zscore']['threshold']
        
        for col in self.numeric_columns:
            try:
                mean = self.X[col].mean()
                std = self.X[col].std()
                
                if std == 0: continue
                
                z_scores = np.abs((self.X[col] - mean) / std)
                outliers_mask = z_scores > threshold
                
                outlier_count = outliers_mask.sum()
                outlier_pct = (outlier_count / len(self.X)) * 100
                
                results[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2),
                    'threshold': threshold,
                    'indices': self.X[outliers_mask].index.tolist()[:50]
                }
                
                if outlier_count > 0 and col not in self.outliers:
                    self.outliers[col] = outliers_mask
                    self.outlier_counts[col] = outlier_count
                    self.outlier_percentages[col] = outlier_pct
            except:
                continue
        
        return results
    
    def _summarize_outliers(self) -> Dict:
        """Summarize outlier statistics."""
        if not self.outlier_counts:
            return {
                'total_outlier_features': 0,
                'total_outlier_rows': 0,
                'severity': 'low'
            }
        
        total_outliers = sum(self.outlier_counts.values())
        total_rows = len(self.X)
        outlier_pct = (total_outliers / total_rows) * 100
        
        if outlier_pct < 1: severity = 'low'
        elif outlier_pct < 5: severity = 'moderate'
        else: severity = 'high'
        
        return {
            'total_outlier_features': len(self.outlier_counts),
            'total_outlier_rows': int(total_outliers),
            'outlier_percentage': round(outlier_pct, 2),
            'severity': severity,
            'features_with_outliers': list(self.outlier_counts.keys())
        }
    
    def _get_affected_features(self) -> Dict:
        affected = {}
        for col, count in self.outlier_counts.items():
            pct = self.outlier_percentages[col]
            if pct > 5: recommendation = "Investigate and potentially remove outliers"
            elif pct > 1: recommendation = "Consider robust scaling or transformation"
            else: recommendation = "Monitor but likely acceptable"
            
            affected[col] = {
                'outlier_count': count,
                'outlier_percentage': round(pct, 2),
                'recommendation': recommendation
            }
        return affected