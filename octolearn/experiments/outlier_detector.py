"""
Outlier Detection Module for OctoLearn

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
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        profile : DatasetProfile
            Dataset profile object
        """
        self.X = X
        self.profile = profile
        self.numeric_features = profile.numeric_features
        self.outliers = {}
        self.outlier_counts = {}
        self.outlier_percentages = {}
    
    @log_execution(logger_obj=logger)
    def detect(self) -> Dict:
        """
        Detect outliers using multiple methods.
        
        Returns
        -------
        dict
            Outlier detection results
        """
        if not self.numeric_features:
            logger.info("No numeric features for outlier detection")
            return {'error': 'No numeric features found'}
        
        results = {
            'methods': {},
            'summary': {},
            'affected_features': {}
        }
        
        # IQR method
        if 'iqr' in OUTLIER_CONFIG['methods']:
            iqr_results = self._detect_iqr()
            results['methods']['iqr'] = iqr_results
        
        # Isolation Forest method
        if 'isolation_forest' in OUTLIER_CONFIG['methods']:
            if_results = self._detect_isolation_forest()
            results['methods']['isolation_forest'] = if_results
        
        # Z-score method
        if 'zscore' in OUTLIER_CONFIG['methods']:
            zscore_results = self._detect_zscore()
            results['methods']['zscore'] = zscore_results
        
        # Summary statistics
        results['summary'] = self._summarize_outliers()
        results['affected_features'] = self._get_affected_features()
        
        return results
    
    def _detect_iqr(self) -> Dict:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Returns
        -------
        dict
            IQR outlier detection results
        """
        results = {}
        multiplier = OUTLIER_CONFIG['iqr']['multiplier']
        
        for col in self.numeric_features:
            try:
                Q1 = self.X[col].quantile(0.25)
                Q3 = self.X[col].quantile(0.75)
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
                    'indices': self.X[outliers_mask].index.tolist()[:50]  # Top 50
                }
                
                if outlier_count > 0:
                    self.outliers[col] = outliers_mask
                    self.outlier_counts[col] = outlier_count
                    self.outlier_percentages[col] = outlier_pct
            
            except Exception as e:
                logger.warning(f"IQR detection failed for {col}: {str(e)}")
                continue
        
        return results
    
    def _detect_isolation_forest(self) -> Dict:
        """
        Detect outliers using Isolation Forest.
        
        Returns
        -------
        dict
            Isolation Forest outlier detection results
        """
        results = {}
        
        try:
            # Sample data if too large
            if len(self.X) > 100_000:
                sample_idx = self.X.sample(100_000, random_state=42).index
                X_sample = self.X.loc[sample_idx]
            else:
                X_sample = self.X
            
            # Get numeric data
            X_numeric = X_sample[self.numeric_features].fillna(X_sample[self.numeric_features].mean())
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=OUTLIER_CONFIG['isolation_forest']['contamination'],
                n_estimators=OUTLIER_CONFIG['isolation_forest']['n_estimators'],
                random_state=OUTLIER_CONFIG['isolation_forest']['random_state']
            )
            
            predictions = iso_forest.fit_predict(X_numeric)
            outliers_mask = predictions == -1
            
            # Map back to original indices
            if len(self.X) > 100_000:
                original_mask = pd.Series(False, index=self.X.index)
                original_mask.loc[sample_idx] = outliers_mask
                outliers_mask = original_mask
            
            outlier_count = outliers_mask.sum()
            outlier_pct = (outlier_count / len(self.X)) * 100
            
            results['overall'] = {
                'count': int(outlier_count),
                'percentage': round(outlier_pct, 2),
                'indices': self.X[outliers_mask].index.tolist()[:50]
            }
        
        except Exception as e:
            logger.warning(f"Isolation Forest detection failed: {str(e)}")
        
        return results
    
    def _detect_zscore(self) -> Dict:
        """
        Detect outliers using Z-score method.
        
        Returns
        -------
        dict
            Z-score outlier detection results
        """
        results = {}
        threshold = OUTLIER_CONFIG['zscore']['threshold']
        
        for col in self.numeric_features:
            try:
                mean = self.X[col].mean()
                std = self.X[col].std()
                
                if std == 0:
                    continue
                
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
            
            except Exception as e:
                logger.warning(f"Z-score detection failed for {col}: {str(e)}")
                continue
        
        return results
    
    def _summarize_outliers(self) -> Dict:
        """
        Summarize outlier statistics.
        
        Returns
        -------
        dict
            Outlier summary statistics
        """
        if not self.outlier_counts:
            return {
                'total_outlier_features': 0,
                'total_outlier_rows': 0,
                'severity': 'low'
            }
        
        total_outliers = sum(self.outlier_counts.values())
        total_rows = len(self.X)
        outlier_pct = (total_outliers / total_rows) * 100
        
        # Classify severity
        if outlier_pct < 1:
            severity = 'low'
        elif outlier_pct < 5:
            severity = 'moderate'
        else:
            severity = 'high'
        
        return {
            'total_outlier_features': len(self.outlier_counts),
            'total_outlier_rows': int(total_outliers),
            'outlier_percentage': round(outlier_pct, 2),
            'severity': severity,
            'features_with_outliers': list(self.outlier_counts.keys())
        }
    
    def _get_affected_features(self) -> Dict:
        """
        Get features affected by outliers and recommendations.
        
        Returns
        -------
        dict
            Affected features and recommendations
        """
        affected = {}
        
        for col, count in self.outlier_counts.items():
            pct = self.outlier_percentages[col]
            
            # Recommendation based on percentage
            if pct > 5:
                recommendation = "Investigate and potentially remove outliers"
            elif pct > 1:
                recommendation = "Consider robust scaling or transformation"
            else:
                recommendation = "Monitor but likely acceptable"
            
            affected[col] = {
                'outlier_count': count,
                'outlier_percentage': round(pct, 2),
                'recommendation': recommendation
            }
        
        return affected
    
    def get_clean_data(self, method: str = 'iqr', keep_pct: float = 0.95) -> Tuple:
        """
        Remove outliers and return clean data.
        
        Parameters
        ----------
        method : str
            Detection method: 'iqr', 'isolation_forest', 'zscore'
        keep_pct : float
            Percentage of data to keep (0-1)
            
        Returns
        -------
        X_clean : pd.DataFrame
            Data with outliers removed
        removed_indices : list
            Indices of removed rows
        """
        if method == 'iqr' and self.outliers:
            mask = pd.Series(False, index=self.X.index)
            for col, col_mask in self.outliers.items():
                mask = mask | col_mask
        else:
            return self.X, []
        
        X_clean = self.X[~mask]
        removed_indices = self.X[mask].index.tolist()
        
        logger.info(f"Removed {len(removed_indices)} outlier rows ({(len(removed_indices)/len(self.X)*100):.2f}%)")
        
        return X_clean, removed_indices
