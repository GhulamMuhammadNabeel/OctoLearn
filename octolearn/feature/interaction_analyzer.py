"""
Feature Interaction Analysis Module

Creates and analyzes feature interactions: polynomial features, pairwise interactions, ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')

from ..config import INTERACTION_CONFIG
from ..utils.helpers import setup_logger, log_execution

logger = setup_logger(__name__)


class FeatureInteractionAnalyzer:
    """
    Analyzes and creates feature interactions.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, profile):
        """
        Initialize FeatureInteractionAnalyzer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target variable
        profile : DatasetProfile
            Dataset profile object
        """
        self.X = X
        self.y = y
        self.profile = profile
        self.numeric_features = profile.numeric_features
        self.interactions = {}
        self.interaction_importance = {}
    
    @log_execution(logger_obj=logger)
    def analyze(self) -> Dict:
        """
        Analyze all types of feature interactions.
        
        Returns
        -------
        dict
            Analysis results for all interaction types
        """
        if len(self.numeric_features) < 2:
            logger.info("Need at least 2 numeric features for interaction analysis")
            return {'error': 'Insufficient numeric features'}
        
        results = {
            'polynomial_interactions': {},
            'pairwise_interactions': {},
            'ratio_interactions': {},
            'interaction_summary': {}
        }
        
        # Polynomial features
        if INTERACTION_CONFIG['types']:
            if 'polynomial' in INTERACTION_CONFIG['types']:
                poly_results = self._analyze_polynomial_features()
                results['polynomial_interactions'] = poly_results
            
            # Pairwise interactions
            if 'interaction' in INTERACTION_CONFIG['types']:
                pair_results = self._analyze_pairwise_interactions()
                results['pairwise_interactions'] = pair_results
            
            # Ratio interactions
            if 'ratio' in INTERACTION_CONFIG['types']:
                ratio_results = self._analyze_ratio_interactions()
                results['ratio_interactions'] = ratio_results
        
        # Summary
        results['interaction_summary'] = self._summarize_interactions(results)
        
        return results
    
    def _analyze_polynomial_features(self) -> Dict:
        """
        Analyze polynomial feature interactions.
        
        Returns
        -------
        dict
            Polynomial interaction details
        """
        try:
            X_numeric = self.X[self.numeric_features].fillna(self.X[self.numeric_features].mean())
            
            degree = INTERACTION_CONFIG['polynomial']['degree']
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=INTERACTION_CONFIG['polynomial']['interaction_only'],
                include_bias=INTERACTION_CONFIG['polynomial']['include_bias']
            )
            
            X_poly = poly.fit_transform(X_numeric)
            feature_names = poly.get_feature_names_out(self.numeric_features)
            
            # Get interaction feature names (exclude original features)
            new_features = []
            for name in feature_names:
                if name not in self.numeric_features:
                    new_features.append(name)
            
            # Calculate correlation with target
            correlations = {}
            for i, name in enumerate(feature_names):
                if name in new_features:
                    try:
                        corr = np.abs(np.corrcoef(X_poly[:, i], self.y)[0, 1])
                        if not np.isnan(corr):
                            correlations[name] = round(corr, 4)
                    except:
                        pass
            
            # Sort by correlation
            sorted_interactions = dict(sorted(
                correlations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            return {
                'degree': degree,
                'total_new_features': len(new_features),
                'top_interactions': sorted_interactions,
                'feature_count': X_poly.shape[1]
            }
        
        except Exception as e:
            logger.warning(f"Polynomial feature analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_pairwise_interactions(self) -> Dict:
        """
        Analyze pairwise feature interactions.
        
        Returns
        -------
        dict
            Pairwise interaction details
        """
        try:
            interactions = {}
            X_numeric = self.X[self.numeric_features].fillna(self.X[self.numeric_features].mean())
            
            # Create all pairwise interactions
            for i, feat1 in enumerate(self.numeric_features):
                for feat2 in self.numeric_features[i+1:]:
                    # Interaction: multiplication
                    interaction = X_numeric[feat1] * X_numeric[feat2]
                    
                    # Correlation with target
                    try:
                        corr = np.abs(np.corrcoef(interaction, self.y)[0, 1])
                        if not np.isnan(corr) and corr > INTERACTION_CONFIG['min_corr_interaction']:
                            pair_name = f"{feat1}_x_{feat2}"
                            interactions[pair_name] = round(corr, 4)
                    except:
                        pass
            
            # Sort and select top
            top_interactions = dict(sorted(
                interactions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:INTERACTION_CONFIG['interaction']['max_pairs']])
            
            return {
                'total_pairs': len(self.numeric_features) * (len(self.numeric_features) - 1) // 2,
                'significant_pairs': len(top_interactions),
                'top_interactions': top_interactions,
                'selection_method': INTERACTION_CONFIG['interaction']['selection_method']
            }
        
        except Exception as e:
            logger.warning(f"Pairwise interaction analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_ratio_interactions(self) -> Dict:
        """
        Analyze ratio-based feature interactions.
        
        Returns
        -------
        dict
            Ratio interaction details
        """
        try:
            interactions = {}
            X_numeric = self.X[self.numeric_features].fillna(self.X[self.numeric_features].mean())
            
            # Create ratio features
            for i, feat1 in enumerate(self.numeric_features):
                for feat2 in self.numeric_features[i+1:]:
                    # Avoid division by zero
                    if (X_numeric[feat2] != 0).all():
                        ratio = X_numeric[feat1] / (X_numeric[feat2] + 1e-8)
                        
                        # Correlation with target
                        try:
                            corr = np.abs(np.corrcoef(ratio, self.y)[0, 1])
                            if not np.isnan(corr) and corr > INTERACTION_CONFIG['min_corr_interaction']:
                                ratio_name = f"{feat1}_/{feat2}"
                                interactions[ratio_name] = round(corr, 4)
                        except:
                            pass
            
            # Sort and select top
            top_ratios = dict(sorted(
                interactions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:INTERACTION_CONFIG['ratio']['max_ratios']])
            
            return {
                'total_ratio_pairs': len(self.numeric_features) * (len(self.numeric_features) - 1) // 2,
                'significant_ratios': len(top_ratios),
                'top_ratios': top_ratios
            }
        
        except Exception as e:
            logger.warning(f"Ratio interaction analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _summarize_interactions(self, results: Dict) -> Dict:
        """
        Summarize all interaction analysis results.
        
        Returns
        -------
        dict
            Interaction summary
        """
        summary = {
            'total_features': len(self.numeric_features),
            'interaction_types_enabled': INTERACTION_CONFIG['types'],
            'recommendations': []
        }
        
        # Count significant interactions
        total_significant = 0
        
        if results['polynomial_interactions'] and 'top_interactions' in results['polynomial_interactions']:
            poly_count = len(results['polynomial_interactions']['top_interactions'])
            total_significant += poly_count
        
        if results['pairwise_interactions'] and 'top_interactions' in results['pairwise_interactions']:
            pair_count = len(results['pairwise_interactions']['top_interactions'])
            total_significant += pair_count
        
        if results['ratio_interactions'] and 'top_ratios' in results['ratio_interactions']:
            ratio_count = len(results['ratio_interactions']['top_ratios'])
            total_significant += ratio_count
        
        summary['total_significant_interactions'] = total_significant
        
        # Recommendations
        if total_significant > 10:
            summary['recommendations'].append(
                f"Strong interaction patterns detected ({total_significant} significant). Consider including in feature engineering."
            )
        elif total_significant > 0:
            summary['recommendations'].append(
                f"Moderate interaction patterns detected ({total_significant} significant). Include top interactions in model training."
            )
        else:
            summary['recommendations'].append(
                "Few interaction patterns found. Focus on original features."
            )
        
        return summary
    
    def create_interaction_features(self, interaction_names: List[str]) -> pd.DataFrame:
        """
        Create specified interaction features.
        
        Parameters
        ----------
        interaction_names : list
            List of interaction names (e.g., ['feat1_x_feat2', 'feat3_/feat4'])
            
        Returns
        -------
        pd.DataFrame
            DataFrame with interaction features
        """
        X_numeric = self.X[self.numeric_features].fillna(self.X[self.numeric_features].mean())
        interaction_features = {}
        
        for name in interaction_names:
            try:
                if '_x_' in name:
                    feat1, feat2 = name.split('_x_')
                    interaction_features[name] = X_numeric[feat1] * X_numeric[feat2]
                
                elif '_/_' in name:
                    feat1, feat2 = name.split('_/_')
                    interaction_features[name] = X_numeric[feat1] / (X_numeric[feat2] + 1e-8)
            
            except Exception as e:
                logger.warning(f"Failed to create interaction {name}: {str(e)}")
                continue
        
        return pd.DataFrame(interaction_features, index=self.X.index)
